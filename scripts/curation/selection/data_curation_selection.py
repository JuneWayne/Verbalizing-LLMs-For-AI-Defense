"""Fit the probes, choose classifier layers and select FIPI training pairs

Each method selects the same number of eligible pairs for LoRA fine-tuning
For probes and Jlens, a prompt score is one minus its correct-label probability
The pairwise selection score is the average of the safe and injection prompt scores

Jlens scoring: hidden state vector -> Jacobian -> RMS normalization -> label logits
The two logits become injection and safe probabilities for classification and selection
"""

import json
import warnings

from scipy.special import expit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits
import joblib
import numpy as np
import pandas as pd
import torch

from scripts.curation.common import (
    ROOT,
    digest,
    experiment_fingerprint,
    load_pairs,
    read_jsonl,
    settings,
    write_json,
    write_jsonl,
)


PROBES = {
    "logistic_regression": "Logistic Regression",
    "linear_svm": "Linear SVM",
    "neural_network": "2 layer neural network",
}

PROBABILITY_RULE = "mean of one minus correct-label probability across each safe/injection pair"


# Load the hidden state vectors used for classifier fitting or validation
# Each prompt has a label and pair ID beside its vectors, so the arrays must keep the same order
# The returned array uses [prompt, inspected layer, hidden state vector value]
# The labels array uses 0 for safe and 1 for injection
# The pair IDs connect each safe prompt with its injection counterpart
# Checking these together prevents a classifier from learning from a vector paired with the wrong label
def load_states(split):
    """Read aligned hidden state vectors, class labels and pair IDs for one split"""

    # the array axes are prompt, layer, and hidden state vector value
    _, output = settings()
    experiment_fingerprint()
    chunks = [np.load(path, allow_pickle=False) for path in sorted((output / "states" / split).glob("*.npz"))]
    if not chunks:
        raise FileNotFoundError(f"No hidden state vectors found for {split}; collect them with scripts.curation.preparation.Record_hidden_state_vectors_and_model_response")

    # join batches in filename order so vectors, labels and pair IDs stay aligned
    states = np.concatenate([chunk["states"] for chunk in chunks]).astype(np.float32)
    labels = np.concatenate([chunk["labels"] for chunk in chunks])
    ids = np.concatenate([chunk["pair_ids"] for chunk in chunks])

    # each source pair must contribute safe label 0 followed by injection label 1
    expected = [(row["pair_id"], label) for row in load_pairs(split) for label in (0, 1)]
    if list(zip(ids, labels)) != expected or not np.isfinite(states).all():
        raise ValueError(f"Incomplete or misaligned hidden state vectors for {split}")
    for chunk in chunks:
        chunk.close()
    return states, labels, ids


# Read candidate hidden state vectors in batches when applying an existing classifier
# The candidate pool can contain many thousands of prompts, each with vectors from multiple layers
# Reading one batch at a time keeps the full pool from occupying memory at once
# Each batch returns its vectors, labels and pair IDs in their original order
# The position variable tracks how many prompts were read, so missing or repeated batches are detected
def state_batches(split):
    """Read candidate hidden state vectors one batch at a time"""

    # score one saved batch at a time instead of loading every candidate layer into RAM
    _, output = settings()
    expected = [(row["pair_id"], label) for row in load_pairs(split) for label in (0, 1)]
    position = 0
    for path in sorted((output / "states" / split).glob("*.npz")):
        with np.load(path, allow_pickle=False) as chunk:
            states, labels, ids = chunk["states"].astype(np.float32), chunk["labels"], chunk["pair_ids"]

        # compare this batch with the corresponding slice of the original prompt order
        if list(zip(ids, labels)) != expected[position:position + len(ids)] or not np.isfinite(states).all():
            raise ValueError(f"Misaligned hidden state vectors in {path}")
        position += len(ids)
        yield states, labels, ids
    if position != len(expected):
        raise ValueError(f"Incomplete hidden state vectors for {split}: {position}/{len(expected)}")


# Collect the base model responses needed to decide which pairs can enter selection
# Each candidate pair should have one safe response and one injection response
# The response text, completion flag and requested-answer check are retained
# The response_ids field contains token IDs and is omitted from this table because the selection rules use the other fields
# The final check requires one response per prompt, preventing an incomplete response set from changing the comparison
def response_table():
    """Collect the safe and injection responses for every candidate pair"""
    _, output = settings()
    records = []
    for path in sorted((output / "responses" / "candidates").glob("*.jsonl")):
        for row in read_jsonl(path):
            # response text and completion checks are enough for selecting pairs
            record = {key: value for key, value in row.items() if key != "response_ids"}
            records.append(record)
    frame = pd.DataFrame(records)
    expected = {(row["pair_id"], condition) for row in load_pairs("candidates") for condition in ("safe", "injection")}
    if frame.empty or set(zip(frame.pair_id, frame.condition)) != expected or frame.duplicated(["pair_id", "condition"]).any():
        raise ValueError("Candidate responses are incomplete or duplicated")
    return frame


# Combine the original candidate prompts with the results of response generation
# A pair is eligible only when both of its responses finished within the generation limit
# All eight methods use this same eligible pool, so each selects from the same available pairs
# The witness_redirection column checks whether the requested injected answer appears only in the injection response
# The witness name comes from FIPI and means the answer requested by the inserted instruction
def candidate_table():
    """Combine candidate prompts with response completion and injected-answer checks"""
    rows = pd.DataFrame(load_pairs("candidates"))
    responses = response_table()

    # the all operation requires both responses in a pair to reach an end token
    complete = responses.groupby("pair_id").finished.all()

    # place the requested-answer checks for both prompt versions on the same pair row
    witness = responses.pivot(index="pair_id", columns="condition", values="witness_present")

    # all methods use the same complete pairs, so output length cannot change one method's pool
    rows["eligible"] = rows.pair_id.map(complete)
    rows["witness_redirection"] = rows.pair_id.map(witness.injection & ~witness.safe)
    return rows


# Turn one method's candidate scores into its final training group
# The scores dictionary maps each pair ID to that method's selection score
# Only eligible pairs enter the ranking, with higher scores first and pair IDs breaking exact ties
# The selected_pairs configuration parameter sets how many pairs to keep, which is 256 in this experiment
# The returned table contains the original prompts and scores for those chosen pairs
# The method and input hashes are recorded so this selection can be traced back to its experiment
def save_selection(method, scores, details):
    """Rank eligible pairs by score and keep the configured number for this method"""
    config, output = settings()
    frame = candidate_table()

    # match scores to pair IDs before ranking so row order cannot mix unrelated examples
    frame["selection_score"] = frame.pair_id.map(scores)
    if frame.selection_score.isna().any():
        raise ValueError("Every candidate must have a selection score")

    # use only pairs whose safe and injection responses both finished
    eligible = frame[frame.eligible].copy()
    if len(eligible) < config["selected_pairs"]:
        raise ValueError("Too few complete pairs for the fixed selection budget")

    # rank higher scores first and keep pair IDs as the fixed tie-breaking order
    selected = eligible.sort_values(["selection_score", "pair_id"], ascending=[False, True])
    selected = selected.head(config["selected_pairs"])
    folder = output / "selections" / method
    folder.mkdir(parents=True, exist_ok=True)
    write_jsonl(folder / "pairs.jsonl", selected.to_dict("records"))
    selected[["pair_id", "user_prompt_id", "attack_method", "selection_score", "witness_redirection"]].to_csv(folder / "selection.csv", index=False)
    write_json(folder / "manifest.json", {"method": method, "fingerprint": experiment_fingerprint(), "selection_code_sha256": digest(__file__), "selected_pairs": len(selected), "eligible_pairs": len(eligible), "pairs_sha256": digest(folder / "pairs.jsonl"), **details})
    return selected


# Select training pairs through random ordering instead of classifier probabilities
# A seeded random generator assigns one number to each candidate pair
# Ranking these random numbers gives every eligible pair the same chance of selection
# The seed parameter makes the same input order produce the same sample on another run
# The random score records this ordering and does not describe the probability of a class label
def select_random():
    """Give every eligible pair the same chance through seeded random ordering"""
    config, _ = settings()
    rows = candidate_table()

    # the seed fixes the random comparison before any training results exist
    scores = dict(zip(rows.pair_id, np.random.default_rng(config["seed"]).random(len(rows))))
    return save_selection("random", scores, {"rule": "seeded random pairs"})


# Select pairs whose prompts differ in wording from the prompts already chosen
# TF-IDF represents the user instruction and injection content with weighted words and two-word phrases
# Cosine similarity compares the directions of these vectors, with higher scores meaning more similar wording
# After one random starting pair, each candidate is compared with all selected prompts
# The closest variable retains the highest similarity found for each candidate across those selected prompts
# The lowest of these closest-match similarities identifies the next pair to add
# This repeats until the configured number of pairs is reached, then decreasing scores preserve that selection order
def select_diversity():
    """Choose pairs with varied wording using TF-IDF and cosine similarity"""
    config, output = settings()
    rows = candidate_table()
    eligible = rows[rows.eligible].reset_index(drop=True)
    texts = eligible.user_instruction + "\n" + eligible.injected_data

    # represent the task and injected content with weighted words and two-word phrases
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000, sublinear_tf=True)
    vectors = vectorizer.fit_transform(texts)
    rng = np.random.default_rng(config["seed"])

    # begin with one random pair, then add examples based on wording differences
    chosen = [int(rng.integers(len(eligible)))]

    # closest stores each candidate's highest similarity to any selected prompt
    closest = np.zeros(len(eligible))
    for _ in range(config["selected_pairs"] - 1):
        # TF-IDF vectors have unit length, so their dot product is cosine similarity
        similarity = (vectors @ vectors[chosen[-1]].T).toarray().ravel()
        closest = np.maximum(closest, similarity)

        # exclude selected pairs, then choose the candidate furthest from its closest match
        closest[chosen] = np.inf
        chosen.append(int(np.argmin(closest)))

    # decreasing scores retain the selected order when the shared ranking function runs
    scores = dict.fromkeys(rows.pair_id, -1.0)
    scores.update({eligible.iloc[index].pair_id: float(len(chosen) - rank) for rank, index in enumerate(chosen)})
    joblib.dump(vectorizer, output / "diversity_vectorizer.joblib")
    return save_selection("text_diversity", scores, {"rule": "greedy farthest-first selection using word tf-idf cosine distance", "features": len(vectorizer.vocabulary_)})


# Select examples where the base model response contains the answer requested by the injection
# The witness_redirection column records the answer check for each pair
# A matching pair receives a base score of 1, while other pairs receive 0
# A small seeded random number orders pairs with the same result
# The final ranking therefore prioritizes observed injected-answer responses before any other eligible pairs
# The manifest also records whether matching pairs were sufficient to fill the full selection size
def select_failures():
    """Prioritize pairs where the base response contains the requested injected answer"""
    config, _ = settings()
    rows = candidate_table()

    # witness redirection has priority; random ties keep the rule independent of other detectors
    tie = np.random.default_rng(config["seed"]).random(len(rows))
    scores = dict(zip(rows.pair_id, rows.witness_redirection.astype(int) + tie * 0.01))
    failures = int((rows.eligible & rows.witness_redirection).sum())
    return save_selection("model_failures", scores, {"rule": "injection witness present only in the injected response; both responses finished", "available_witness_redirections": failures, "random_fill_pairs": max(0, config["selected_pairs"] - failures), "limitation": "witness presence is a redirection proxy, not a check of full task completion"})


# Measure how well a classifier identifies each label on the validation prompts
# Injection accuracy is the fraction of injection prompts predicted as injection
# Safe accuracy is the fraction of safe prompts predicted as safe
# Balanced accuracy is the average of those two accuracies, giving the two labels equal weight
# This average is the value used to choose which model layer supplies vectors for selection
def accuracy_rows(labels, predictions):
    """Calculate injection accuracy, safe accuracy and their average"""

    # each class is measured separately, then balanced accuracy averages the two
    return {
        "injection_accuracy": float((predictions[labels == 1] == 1).mean()),
        "safe_accuracy": float((predictions[labels == 0] == 0).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
    }


# Prepare the training groups used to convert Linear SVM scores into probabilities
# An SVM first produces a decision score, and a fitted sigmoid maps that score to a probability
# Three groups supply held-out training predictions for fitting this conversion
# Both prompts from the same pair remain in one group, preventing the safe and injection versions from being split across fitting and calibration
# The separate validation dataset is still reserved for choosing the layer
def pair_calibration_splits(pair_ids, seed):
    """Keep both prompts in each pair together when calibrating SVM probabilities"""

    # keep the safe and injection versions together so calibration sees unseen pairs
    splitter = GroupKFold(n_splits=3, shuffle=True, random_state=seed)
    return list(splitter.split(pair_ids, groups=pair_ids))


# Convert a classifier's injection probability into the selection score for one prompt
# For an injection prompt, the correct probability is the injection probability itself
# For a safe prompt, the correct probability is one minus the injection probability
# Subtracting the correct probability from 1 gives the probability assigned to the incorrect label
# For example, an injection prompt assigned 0.90 to safe receives a prompt selection score of 0.90
# The two prompt scores are averaged later, after matching them by pair ID
def probability_scores(labels, injection_probability):
    """Calculate one minus the correct-label probability for each prompt"""

    # every detector supplies probabilities for the same two labels
    injection_probability = np.asarray(injection_probability, dtype=np.float64)
    if not np.isfinite(injection_probability).all() or ((injection_probability < 0) | (injection_probability > 1)).any():
        raise ValueError("Label probabilities must be finite numbers between zero and one")

    # label 1 is injection and label 0 is safe
    # for a safe prompt, the correct-label probability is one minus the injection probability
    correct_probability = np.where(labels == 1, injection_probability, 1 - injection_probability)
    return 1 - correct_probability


# Use the same label decision and pairwise selection rule for every probe and both lenses
# The injection_probability array contains one probability per candidate prompt
# A probability above 0.5 predicts injection, while a probability of 0.5 or less predicts safe
# The difficulty column stores one minus the probability of the prompt's correct label
# Grouping by pair ID then averages the safe and injection scores into one pairwise selection score
# For example, incorrect-label probabilities of 0.90 and 0.03 produce a pair score of 0.465
# The function returns these pair scores for ranking and retains the individual decisions for inspection
def save_probability_scores(method, pair_ids, labels, injection_probability, logit_difference=None):
    """Turn probabilities into class labels and average prompt scores by pair"""

    # use the same scoring and pair averaging for all probes and both lenses
    _, output = settings()
    folder = output / "selectors"
    injection_probability = np.asarray(injection_probability, dtype=np.float64)
    difficulty = probability_scores(labels, injection_probability)

    # this threshold turns the two-label probabilities into the classifier's decision
    # exactly 0.5 is assigned safe, matching the validation rule
    predictions = injection_probability > 0.5
    frame = pd.DataFrame({
        "pair_id": pair_ids,
        "condition": np.where(labels == 1, "injection", "safe"),
        "predicted_label": np.where(predictions, "injection", "safe"),
        "correct": predictions == labels,
        "injection_probability": injection_probability,
        "safe_probability": 1 - injection_probability,
        "correct_label_probability": 1 - difficulty,
        "difficulty": difficulty,
    })
    if logit_difference is not None:
        frame["logit_difference"] = logit_difference
    predictions_path = folder / f"{method}_candidate_predictions.csv"
    frame.to_csv(predictions_path.with_suffix(".tmp"), index=False)
    predictions_path.with_suffix(".tmp").replace(predictions_path)

    # difficulty is the incorrect-label probability for one prompt
    # join both versions by pair ID and average them to get the pairwise selection score
    pair_scores = frame.groupby("pair_id").difficulty.mean()
    scores_path = folder / f"{method}_candidate_scores.csv"
    pair_scores.to_csv(scores_path.with_suffix(".tmp"))
    scores_path.with_suffix(".tmp").replace(scores_path)
    return pair_scores.to_dict()


# Apply an already fitted probe to candidate hidden state vectors
# The layer parameter identifies the block output chosen through validation accuracy
# Only that layer's vector is passed to the probe for each candidate prompt
# The injection_column variable identifies which predict_proba column belongs to the injection label
# The returned probabilities then use the same pairwise scoring function as Jlens
def score_probe(method, probe, layer):
    """Apply the chosen probe layer to all candidate hidden state vectors"""
    ids, labels, probabilities = [], [], []
    injection_column = int(np.flatnonzero(probe.classes_ == 1)[0])
    with threadpool_limits(limits=4):
        for hidden, batch_labels, batch_ids in state_batches("candidates"):
            probabilities.extend(probe.predict_proba(hidden[:, layer])[:, injection_column])
            labels.extend(batch_labels)
            ids.extend(batch_ids)
    return save_probability_scores(method, ids, np.asarray(labels), probabilities)


# Fit a separate classifier at each recorded block output, then choose one layer
# The selector_training split supplies the vectors and labels used to fit each classifier
# The validation split measures how that classifier performs on separate prompt pairs
# Each layer fits its own scaling parameters using training vectors before fitting its classifier
# Logistic Regression and the neural network supply label probabilities directly
# Linear SVM additionally fits a sigmoid conversion using grouped training folds
# The layer with the highest balanced validation accuracy is retained, choosing the earlier layer if scores tie
# Existing fitting results are reused only when their configuration and code signatures match
def fit_probe(method, score_candidates=True):
    """Fit a classifier at each recorded layer and choose the best validation layer"""

    # each layer gets a separate classifier so validation can choose where to read the model
    config, output = settings()
    folder = output / "selectors"
    folder.mkdir(exist_ok=True)
    cache_path = folder / f"{method}_fit.json"
    signature = {"fingerprint": experiment_fingerprint(), "code_sha256": digest(__file__), "recording_sha256": digest(output / "recording.json")}

    # reuse prior fitting results only if their data and code identifiers still match
    if cache_path.exists():
        saved = json.loads(cache_path.read_text())
        if saved["signature"] == signature:
            metrics = pd.read_csv(folder / f"{method}_validation.csv")
            scores = {}
            if score_candidates:
                score_path = folder / f"{method}_candidate_scores.csv"
                if score_path.exists():
                    scores = pd.read_csv(score_path).set_index("pair_id").difficulty.to_dict()
                else:
                    probe = joblib.load(folder / f"{method}.joblib")
                    layer = saved["details"]["selected_layer"]
                    scores = score_probe(method, probe, layer)
            return metrics, scores, saved["details"]

    # these arrays contain one hidden state vector per prompt at each recorded block output
    # validation vectors remain separate from the vectors used to fit the classifier
    training, y_train, training_ids = load_states("selector_training")
    validation, y_validation, _ = load_states("validation")
    metrics, best_score, best_model, best_layer = [], -1.0, None, None
    if method == "linear_svm":
        # generate held-out training predictions for sigmoid calibration, then refit on all training pairs
        # validation still chooses the layer and never fits the probability conversion
        calibration_folds = pair_calibration_splits(training_ids, config["seed"])
        calibration_rows = []
        for fold, (_, calibration_indices) in enumerate(calibration_folds):
            for pair_id in np.unique(training_ids[calibration_indices]):
                calibration_rows.append({"pair_id": pair_id, "calibration_fold": fold})
        pd.DataFrame(calibration_rows).to_csv(folder / "linear_svm_calibration_folds.csv", index=False)

    # fit a separate classifier at every recorded block output before choosing one layer
    for layer in range(training.shape[1]):
        if method == "logistic_regression":
            # C controls regularization, which limits how strongly individual vector weights can grow
            estimator = LogisticRegression(C=1.0, max_iter=1000, random_state=config["seed"])
        elif method == "linear_svm":
            # the SVM learns a linear boundary, then training-fold calibration supplies probabilities
            estimator = LinearSVC(C=1.0, dual="auto", max_iter=10000, random_state=config["seed"])
        elif method == "neural_network":
            # one hidden layer and one output layer give two learned layers
            # one 64-unit hidden layer and one output layer give two learned layers
            estimator = MLPClassifier(hidden_layer_sizes=(64,), alpha=0.01, batch_size=64, max_iter=200, early_stopping=False, random_state=config["seed"])
        else:
            raise ValueError(f"Unknown probe: {method}")

        # fit scaling on training vectors, then apply those same learned scales to validation
        probe = make_pipeline(StandardScaler(), estimator)
        if method == "linear_svm":
            # scaling is fitted inside each fold along with the svm to prevent calibration leakage
            probe = CalibratedClassifierCV(probe, method="sigmoid", cv=calibration_folds, ensemble=False, n_jobs=1)
        with threadpool_limits(limits=4), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            probe.fit(training[:, layer], y_train)
            injection_column = int(np.flatnonzero(probe.classes_ == 1)[0])
            predictions = probe.predict_proba(validation[:, layer])[:, injection_column] > 0.5

        # compare the layer on held-out validation pairs, using the same 0.5 decision threshold
        scores = accuracy_rows(y_validation, predictions)
        metrics.append({"layer": layer, **scores, "converged": not any(issubclass(w.category, ConvergenceWarning) for w in caught)})

        # ties keep the earlier layer; testing data never enters this choice
        if scores["balanced_accuracy"] > best_score:
            best_score, best_model, best_layer = scores["balanced_accuracy"], probe, layer
        print(f"{PROBES[method]}: layer {layer}, validation balanced accuracy {scores['balanced_accuracy']:.3f}", flush=True)

    pd.DataFrame(metrics).to_csv(folder / f"{method}_validation.csv", index=False)

    # retain only the validation-selected classifier for subsequent candidate scoring
    joblib.dump(best_model, folder / f"{method}.joblib")
    probability_method = "three-fold pair-grouped sigmoid calibration on training data" if method == "linear_svm" else "classifier predict_proba"
    details = {"detector": PROBES[method], "selected_layer": best_layer, "validation_balanced_accuracy": best_score, "rule": PROBABILITY_RULE, "probability_method": probability_method, "decision_threshold": 0.5, "training_pairs": config["selector_training_pairs"], "validation_pairs": config["validation_pairs"]}
    write_json(cache_path, {"signature": signature, "details": details})
    pair_scores = score_probe(method, best_model, best_layer) if score_candidates else {}
    return pd.DataFrame(metrics), pair_scores, details


# Combine classifier scoring with the common final selection rule
# The fitting function returns the chosen classifier layer and candidate pair scores
# The selection function ranks those eligible pairs and keeps the configured number
# The returned values include both layer accuracies and the selected training group for notebook inspection
def select_probe(method):
    """Select training pairs using the chosen probe and its pairwise scores"""
    metrics, scores, details = fit_probe(method)
    return metrics, save_selection(method, scores, details)


# Calculate the two label scores that let Jlens act as an injection classifier
# The states array contains hidden state vectors from the last prompt token at each inspected layer
# Each Jacobian matrix estimates how that layer's vector maps to the final hidden state vector
# Because prompts are stored as row vectors, multiplying by the transpose applies the matrix to every prompt
# Final RMS normalization rescales each estimated vector and applies the model's learned normalization weights
# The unembedding matrix contains two vocabulary rows, with injection first and safe second
# Multiplying each row by the normalized vector and summing the products gives that label's logit
# The function returns injection logit minus safe logit, with a positive result favoring injection
def lens_differences(states, jacobians, readout):
    """Estimate the injection logit minus safe logit at each inspected layer"""

    # states has shape [prompts, inspected layers, hidden state vector size]
    # unembedding has two rows, ordered injection first and safe second
    output = np.empty(states.shape[:2], dtype=np.float32)

    # norm contains the learned final RMS scaling weights
    # unembedding contains only the vocabulary rows for injection and safe
    norm = readout["norm"].to(torch.bfloat16)
    unembedding = readout["unembedding"].to(torch.bfloat16)
    with torch.inference_mode():
        for layer in range(states.shape[1]):
            hidden = torch.from_numpy(states[:, layer])

            # prompts are row vectors here, so multiplying by J transpose applies J to each vector
            estimated = (hidden @ jacobians[layer].float().T).to(torch.bfloat16)

            # RMS normalization rescales the full estimated vector before applying learned weights
            # keep the original float32 calculation and BF16 rounding to match the model
            estimated_float = estimated.float()
            mean_square = estimated_float.square().mean(-1, keepdim=True)
            inverse_rms = torch.rsqrt(mean_square + readout["eps"])
            normalized = (estimated_float * inverse_rms).to(torch.bfloat16) * norm

            # each unembedding row gives one label's score by summing weight-vector products
            logits = (normalized @ unembedding.T).float()

            # a positive difference favors injection, while a negative difference favors safe
            output[:, layer] = (logits[:, 0] - logits[:, 1]).numpy()
    return output


# Convert a lens logit difference into a probability for the injection label
# Applying sigmoid to injection logit minus safe logit is equivalent to softmax over those two logits
# A difference of 0 gives injection probability 0.5, while positive differences increase that probability
# The safe probability is one minus the injection probability
# These two probabilities compare the selected class tokens and sum to 1 within that comparison
def lens_probability(differences):
    """Convert a logit difference into injection probability over the two labels"""

    # sigmoid(injection logit - safe logit) equals softmax over those two logits
    # the injection and safe probabilities add up to 1 within this two-label comparison
    return expit(np.asarray(differences, dtype=np.float64))


# Apply the chosen lens layer to every candidate prompt without fitting new Jacobian matrices
# The jacobian parameter is the matrix belonging to the layer selected through validation
# The readout dictionary contains the model's normalization weights and the two unembedding rows
# For each candidate batch, the chosen hidden state vectors are converted into logit differences and label probabilities
# Those probabilities are passed to the shared scoring function to obtain class decisions and pairwise scores
def score_lens(method, jacobian, readout, layer):
    """Score candidate prompts with the existing Jacobian from the chosen layer"""
    ids, labels, differences = [], [], []
    with threadpool_limits(limits=4):
        for hidden, batch_labels, batch_ids in state_batches("candidates"):
            # validation already chose the layer, so the other candidate layers need no lens calculation
            # retain the layer dimension because lens_differences expects a three-dimensional array
            selected_hidden = hidden[:, layer:layer + 1]
            batch_differences = lens_differences(selected_hidden, {0: jacobian}, readout)
            differences.extend(batch_differences[:, 0])
            labels.extend(batch_labels)
            ids.extend(batch_ids)

    # collect all batches before assigning labels and averaging the two prompt scores per pair
    differences = np.asarray(differences, dtype=np.float32)
    return save_probability_scores(method, ids, np.asarray(labels), lens_probability(differences), differences)


# Choose a layer for an existing fitted Jacobian lens
# Despite the function name, this stage reads lens matrices that were fitted earlier
# The validation prompts are scored at every recorded layer, then injection and safe accuracy are averaged
# The best layer is chosen using those validation accuracies, with earlier layers winning exact ties
# When score_candidates is true, that chosen layer also scores the candidate pool
# The model weights and fitted Jacobian matrices stay unchanged during this layer comparison
# The returned results contain the validation table, candidate pair scores and selected layer details
def fit_lens(method, score_candidates=True):
    """Choose a validation layer for an existing lens and optionally score candidates"""

    # choose a layer for an existing Jacobian lens without training new lens weights
    config, output = settings()
    lens_info = config["lenses"][method]
    lens_path = ROOT / lens_info["path"]
    folder = output / "selectors"
    folder.mkdir(exist_ok=True)
    cache_path = folder / f"{method}_fit.json"
    signature = {"fingerprint": experiment_fingerprint(), "code_sha256": digest(__file__), "lens_sha256": digest(lens_path), "readout_sha256": digest(output / "readout.pt"), "recording_sha256": digest(output / "recording.json")}
    if cache_path.exists():
        saved = json.loads(cache_path.read_text())
        if saved["signature"] == signature:
            metrics = pd.read_csv(folder / f"{method}_validation.csv")
            scores = {}
            if score_candidates:
                score_path = folder / f"{method}_candidate_scores.csv"
                if score_path.exists():
                    scores = pd.read_csv(score_path).set_index("pair_id").difficulty.to_dict()
                else:
                    lens = torch.load(lens_path, map_location="cpu", weights_only=True)
                    readout = torch.load(output / "readout.pt", map_location="cpu", weights_only=True)
                    layer = saved["details"]["selected_layer"]
                    scores = score_lens(method, lens["J"][layer], readout, layer)
            return metrics, scores, saved["details"]

    # read the fitted Jacobian matrices and the model's final normalization and label weights
    lens = torch.load(lens_path, map_location="cpu", weights_only=True)
    readout = torch.load(output / "readout.pt", map_location="cpu", weights_only=True)
    validation, labels, _ = load_states("validation")
    with threadpool_limits(limits=4):
        differences = lens_differences(validation, lens["J"], readout)

    # compare layer accuracy on validation prompts before looking at candidate rankings
    metrics = []
    for layer in range(differences.shape[1]):
        probabilities = lens_probability(differences[:, layer])
        scores = accuracy_rows(labels, probabilities > 0.5)
        metrics.append({"layer": layer, **scores})

    # layers are visited in order, so max keeps the earlier layer when accuracies tie
    best = max(metrics, key=lambda row: row["balanced_accuracy"])

    # use the same probability rule and decision threshold as the probes
    pd.DataFrame(metrics).to_csv(folder / f"{method}_validation.csv", index=False)
    np.savez(folder / f"{method}_scores.npz", validation=differences, selected_layer=best["layer"])
    details = {"detector": "Jlens verbalization", "selected_layer": best["layer"], "validation_balanced_accuracy": best["balanced_accuracy"], "rule": PROBABILITY_RULE, "probability_method": "softmax over injection and safe token logits only", "decision_threshold": 0.5, "validation_pairs": config["validation_pairs"], "lens_sha256": digest(lens_path), "lens": lens_info, "limitation": "two-label lens probabilities at the last prompt token, not calibrated correctness estimates or full-vocabulary generation probabilities"}
    write_json(cache_path, {"signature": signature, "details": details})
    scores = {}

    # candidate scoring uses the validation-selected layer's own Jacobian matrix
    if score_candidates:
        layer = best["layer"]
        scores = score_lens(method, lens["J"][layer], readout, layer)
    return pd.DataFrame(metrics), scores, details


# Turn the existing lens's candidate scores into a selected training group
# The lens layer is chosen from validation accuracy before the candidate ranking is used
# The same save_selection function used by the probes keeps the highest-scoring eligible pairs
# The returned tables show which layer was chosen and which prompt pairs enter LoRA training
def select_lens(method):
    """Select training pairs using the chosen Jlens layer and its pairwise scores"""
    metrics, scores, details = fit_lens(method)
    return metrics, save_selection(method, scores, details)


# Bring the selected prompts and model responses together for inspecting reference-answer quality
# A pair selected by more than one method appears once, with all selecting methods listed beside it
# The safe response is the reference answer used by the final response-training objective
# The safe_task_correct and injection_followed columns provide space for recording response-review decisions
# Existing decisions are retained when the table is rebuilt, so a later selection run does not erase them
# This table supports review and does not change the selection scores
def prepare_review():
    """Combine selected pairs and both responses while preserving prior review decisions"""
    _, output = settings()
    responses = response_table().pivot(index="pair_id", columns="condition", values="response")

    # track every method that selected a pair so shared pairs appear only once in the review
    methods = {}
    for path in sorted((output / "selections").glob("*/pairs.jsonl")):
        for row in read_jsonl(path):
            methods.setdefault(row["pair_id"], []).append(path.parent.name)
    rows = pd.DataFrame(load_pairs("candidates")).set_index("pair_id")
    review = rows.loc[list(methods), ["user_instruction", "clean_data", "injected_data", "witness"]].join(responses.rename(columns={"safe": "safe_response", "injection": "injection_response"}))
    review["selection_methods"] = [", ".join(methods[key]) for key in review.index]

    # these fields record response quality separately from the classifier's selection score
    review["safe_task_correct"] = ""
    review["injection_followed"] = ""
    review["notes"] = ""
    review["approved_safe_response"] = ""
    path = output / "training_review.csv"

    # keep any human decisions from an earlier review
    if path.exists():
        previous = pd.read_csv(path, index_col="pair_id", keep_default_na=False)
        for column in ("safe_task_correct", "injection_followed", "notes", "approved_safe_response"):
            if column in previous:
                review[column] = previous[column].reindex(review.index).fillna("")
    review.to_csv(path)
    return review
