"""Shared probe training and validation selection, with historical file-key compatibility."""

import numpy as np
import pandas as pd
import torch

# Use the same initialization and training budget for every model and layer.
SEED = 0
EPOCHS = 300
HIDDEN_UNITS = 64
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.01
# These keys also appear in saved historical results, so keep their spelling.
PROBES = ['logistic regression', 'linear SVM', 'network']
METHODS = PROBES + ['JLens', 'ordinary lens', 'model opening label']


def fit_probe(training_states, training_labels, scoring_states, kind):
    """Train one classifier, then score states without changing its weights."""
    torch.manual_seed(SEED)

    # Each row is an email; each column is one value in its hidden state.
    # Learn the mean and scale from training only. Validation and testing cannot change them.
    training_mean = training_states.mean(dim=0, keepdim=True)
    training_scale = training_states.std(dim=0, keepdim=True).clamp_min(1e-6)
    training_states = (training_states - training_mean) / training_scale
    scoring_states = (scoring_states - training_mean) / training_scale

    # The neural network has two learned layers, including its output layer.
    if kind == 'network':
        probe = torch.nn.Sequential(torch.nn.Linear(training_states.shape[1], HIDDEN_UNITS), torch.nn.ReLU(), torch.nn.Linear(HIDDEN_UNITS, 1))
    elif kind in ['logistic regression', 'linear SVM']:
        probe = torch.nn.Linear(training_states.shape[1], 1)
    else:
        raise ValueError('Unknown probe type.')
    probe = probe.to(training_states.device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # Match the output shape: one score per email, stored as a column.
    targets = training_labels.unsqueeze(1)

    for epoch in range(EPOCHS):
        # Clear the previous update before measuring this epoch's errors.
        optimizer.zero_grad()
        output = probe(training_states)
        if kind == 'linear SVM':
            # SVM hinge loss penalizes the wrong sign and scores too close to zero.
            loss = torch.clamp(1 - (2 * targets - 1) * output, min=0).mean()
        else:
            # Logistic Regression and the neural network use binary cross entropy.
            loss = torch.nn.functional.binary_cross_entropy_with_logits(output, targets)
        # Calculate and apply the weight update; the language model stays frozen.
        loss.backward()
        optimizer.step()

    # Score all requested states without learning from their labels.
    probe.eval()
    with torch.no_grad():
        scores = probe(scoring_states).squeeze(1).cpu()
    # Preserve checkpoint field names so existing saved probes remain compatible.
    checkpoint = {
        'kind': kind, 'width': training_states.shape[1], 'centre': training_mean.cpu(), 'scale': training_scale.cpu(),
        'weights': {name: value.detach().cpu().clone() for name, value in probe.state_dict().items()},
        'values': scores
    }
    return checkpoint


def choose_rule(values, truth):
    """Choose a score threshold using equal weight for injection and safe accuracy."""
    # Float64 keeps a threshold between scores from rounding onto one of the scores.
    values = np.asarray(values, dtype=np.float64)
    truth = np.asarray(truth)
    if not np.isfinite(values).all() or set(np.unique(truth)) != {0, 1}:
        raise ValueError('Threshold selection needs finite scores and both validation labels.')
    # Every gap is a possible decision boundary. Also test always choosing one label.
    unique_scores = np.unique(values)
    midpoints = unique_scores[:-1] / 2 + unique_scores[1:] / 2
    below_all_scores = np.nextafter(unique_scores[0], -np.inf)
    above_all_scores = np.nextafter(unique_scores[-1], np.inf)
    thresholds = np.unique(np.concatenate(([0.0, below_all_scores], midpoints, [above_all_scores])))

    # Each column contains predictions from one candidate threshold.
    predictions = values[:, None] >= thresholds
    injection_accuracy = predictions[truth == 1].mean(axis=0)
    safe_accuracy = (~predictions[truth == 0]).mean(axis=0)
    average_accuracy = (injection_accuracy + safe_accuracy) / 2
    best_indices = np.flatnonzero(average_accuracy == average_accuracy.max())
    # Break equal-accuracy ties with the threshold closest to zero.
    closest_to_zero = np.argmin(np.abs(thresholds[best_indices]))
    return float(thresholds[best_indices[closest_to_zero]])


def choose_readers(scores):
    """Choose layers for historical results, keeping their saved column names."""
    # 'development' is the old CSV key for validation, not a different data split.
    validation_scores = scores.loc[scores['split'].eq('development')]
    rows = []
    for (detector, layer), layer_scores in validation_scores.groupby(['reader', 'layer']):
        values = layer_scores['score'].to_numpy(dtype=np.float64)
        truth = layer_scores['label'].to_numpy()
        threshold = choose_rule(values, truth)
        predicted = values >= threshold
        injection_accuracy = predicted[truth == 1].mean()
        safe_accuracy = (~predicted[truth == 0]).mean()
        rows.append({'reader': detector, 'layer': int(layer), 'rule': threshold, 'injection accuracy': injection_accuracy, 'safe accuracy': safe_accuracy, 'development accuracy': (injection_accuracy + safe_accuracy) / 2})
    # Keep the best validation accuracy, then the earliest layer if scores tie.
    results = pd.DataFrame(rows).sort_values(['reader', 'development accuracy', 'layer'], ascending=[True, False, True])
    return results, results.drop_duplicates('reader').reset_index(drop=True)
