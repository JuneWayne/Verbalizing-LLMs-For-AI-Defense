"""Run FIPI preparation, probe fitting, candidate scoring and data selection

Stage fit fits the three probe types and chooses layers for the two existing lenses
Stage score calculates candidate probabilities using those detectors
Stage select forms the eight training groups from eligible pairs
"""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time

import pandas as pd
import torch

from scripts.curation.common import ROOT, settings, write_json
from scripts.curation.selection.data_curation_selection import (
    PROBES,
    fit_lens,
    fit_probe,
    prepare_review,
    select_diversity,
    select_failures,
    select_lens,
    select_probe,
    select_random,
)


# Run the complete data preparation and selection pipeline in its required order
# First the pool script assigns related source prompts to separate dataset groups
# Next the recording script collects probe training and validation hidden state vectors
# The fit stage trains the three probes and chooses layers for those probes and both existing lenses
# Candidate vectors are then recorded and scored using those chosen detectors
# The model generates both responses for each candidate pair before eligibility and final selection are calculated
# The select stage forms all eight training groups and compares their overlap
# A process lock prevents two pipelines from writing to the same experiment at once
# This runner does not fit new Jacobian matrices or start LoRA fine-tuning
def run(stage="all"):
    """Run the complete preparation and selection sequence with progress tracking"""
    config, output = settings()
    # a process lock prevents two runs from writing the same recording batches
    with (output / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # keep source splitting, classifier fitting, response generation and selection in this order
        commands = [
            ("prepare_pool", ["-m", "scripts.curation.preparation.pool"]),
            ("probe_hidden_state_vectors", ["-m", "scripts.curation.preparation.Record_hidden_state_vectors_and_model_response", "--stage", "probe_states"]),
            ("fit_probes_and_choose_lens_layers", ["-m", "scripts.curation.preparation.preparation", "--stage", "fit"]),
            ("candidate_hidden_state_vectors", ["-m", "scripts.curation.preparation.Record_hidden_state_vectors_and_model_response", "--stage", "candidate_states"]),
            ("score_candidates", ["-m", "scripts.curation.preparation.preparation", "--stage", "score"]),
            ("generate_candidate_responses", ["-m", "scripts.curation.preparation.Record_hidden_state_vectors_and_model_response", "--stage", "responses"]),
            ("select_all_methods", ["-m", "scripts.curation.preparation.preparation", "--stage", "select"]),
        ]
        try:
            for name, command in commands:
                write_json(output / "run_status.json", {"status": "running", "stage": name, "pid": os.getpid(), "updated_unix": time.time()})
                print(f"Starting {name}", flush=True)
                subprocess.run([sys.executable, "-u", *command], cwd=ROOT, check=True)
            write_json(output / "run_status.json", {"status": "complete", "pid": os.getpid(), "updated_unix": time.time()})
        except BaseException as error:
            write_json(output / "run_status.json", {"status": "failed", "stage": name, "error": str(error), "updated_unix": time.time()})
            raise


# Choose whether to run every stage or one stage of the existing pipeline
# The all option follows the full order listed in run
# The fit option trains probes and chooses detector layers
# The score option calculates candidate probabilities, and select builds the training groups
# Existing compatible results are reused, while missing fitting results can trigger probe fitting
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", "fit", "score", "select"], default="all")
    parser.add_argument("--background", action="store_true", help="continue locally after the terminal closes")
    args = parser.parse_args()
    # Background mode starts this same command in a separate local process
    # Progress is written to the run log so closing the terminal does not stop it
    if args.background:
        _, output = settings()
        with (output / "run.log").open("a") as log:
            process = subprocess.Popen([sys.executable, "-u", "-m", "scripts.curation.preparation.preparation", "--stage", args.stage], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        print(json.dumps({"pid": process.pid, "log": str(output / "run.log")}), flush=True)
    elif args.stage == "all":
        run()
    else:

        torch.set_num_threads(4)
        config, output = settings()
        # Both stages use the same detectors, with candidate scoring enabled only for score
        # The lens branch chooses among existing Jacobian matrices instead of fitting a new lens
        if args.stage in ("fit", "score"):
            # the fit stage chooses layers, while the score stage also evaluates candidate prompts
            score_candidates = args.stage == "score"
            for method in PROBES:
                _, _, details = fit_probe(method, score_candidates=score_candidates)
                print(json.dumps(details), flush=True)
            for method in config["lenses"]:
                _, _, details = fit_lens(method, score_candidates=score_candidates)
                print(json.dumps(details), flush=True)
        else:
            # these three methods select pairs directly, without classifier label probabilities
            selections = {
                "random": select_random(),
                "text_diversity": select_diversity(),
                "model_failures": select_failures(),
            }
            # probes and both lenses rank the same candidate pool by pairwise selection score
            for method in PROBES:
                _, selections[method] = select_probe(method)
            for method in config["lenses"]:
                _, selections[method] = select_lens(method)
            # Summarize how many pairs and injected-answer cases each method selected
            # Classifier scores are pairwise probabilities, while the other methods use their own ordering rules
            summary = pd.DataFrame([{"method": method, "selected_pairs": len(rows), "mean_selection_score": rows.selection_score.mean(), "witness_redirection_pairs": int(rows.witness_redirection.sum())} for method, rows in selections.items()])
            summary.to_csv(output / "selection_summary.csv", index=False)
            # count shared pairs for every two methods, including each method with itself
            overlap = pd.DataFrame(index=selections, columns=selections, dtype=int)
            for first, left in selections.items():
                for second, right in selections.items():
                    overlap.loc[first, second] = len(set(left.pair_id) & set(right.pair_id))
            overlap = overlap.astype(int)
            overlap.to_csv(output / "selection_overlap.csv")
            prepare_review()
            print(summary.to_string(index=False), flush=True)
            # refresh the reader-facing tables only after the entire candidate pool has finished
            notebooks = ["collect_fipi_responses", "select_random", "select_text_diversity", "select_model_failures", "select_logistic_regression", "select_linear_svm", "select_neural_network", "select_jlens", "compare_selections", "train_lora"]
            if os.environ.get("FIPI_SKIP_NOTEBOOKS") == "1":
                notebooks = []
            for name in notebooks:
                subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--execute", "--to", "notebook", "--inplace", "--ExecutePreprocessor.timeout=3600", str(ROOT / "notebooks/curation" / f"{name}.ipynb")], cwd=ROOT, check=True)
