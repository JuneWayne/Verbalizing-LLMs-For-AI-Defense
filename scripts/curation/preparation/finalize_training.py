"""Export reviewed safe and injection-following completions before LoRA training"""

import argparse
import json

from transformers import AutoTokenizer
import pandas as pd

from scripts.curation.common import (
    ROOT,
    digest,
    experiment_fingerprint,
    messages,
    read_jsonl,
    settings,
    write_json,
    write_jsonl,
)


def finalize(method, write=False, allow_unreviewed=False):
    config, output = settings()
    selection_path = output / "selections" / method / "pairs.jsonl"
    selection = read_jsonl(selection_path)
    manifest = json.loads(selection_path.with_name("manifest.json").read_text())
    if manifest["fingerprint"] != experiment_fingerprint() or manifest["pairs_sha256"] != digest(selection_path):
        raise ValueError("The saved selection changed after it was recorded")
    review_path = output / "training_review.csv"
    review = pd.read_csv(review_path, keep_default_na=False).set_index("pair_id", verify_integrity=True)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / config["model"], local_files_only=True)
    rows, blocked = [], []
    # preserve the historical intermediate export label expected by the recorded token metadata
    # the final adapter uses reference-response loss after these paired tokens are prepared
    objective = "odile_triplet"
    for row in selection:
        key = row["pair_id"]
        if key not in review.index:
            raise ValueError(f"Missing selected pair in response table: {key}")
        if not allow_unreviewed and review.loc[key, "safe_task_correct"].strip().casefold() != "yes":
            blocked.append(key)
            continue
        checked = review.loc[key]
        # an inserted instruction can be ignored, so the prompt label alone cannot approve a harmful completion
        if not allow_unreviewed and checked["injection_followed"].strip().casefold() != "yes":
            blocked.append(key)
            continue
        # the approval must belong to this exact task and prompt pair
        if any(checked[field] != row[field] for field in ("user_instruction", "clean_data", "injected_data")):
            raise ValueError(f"Review text differs from the saved pair: {key}")
        completions = {"safe": checked.get("approved_safe_response", "").strip() or checked["safe_response"].strip(), "injection": checked["injection_response"].strip()}
        if any(not text for text in completions.values()):
            raise ValueError(f"Reviewed completion is empty: {key}")
        # keep each response with its own prompt so the two roles have different completion states
        completion_ids = {condition: tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id] for condition, text in completions.items()}
        prefixes = {condition: tokenizer.apply_chat_template(messages(row, condition, config), tokenize=True, add_generation_prompt=True, return_dict=False) for condition in ("safe", "injection")}
        if any(len(ids) > config["max_new_tokens"] for ids in completion_ids.values()) or any(len(ids) > config["max_input_tokens"] for ids in prefixes.values()):
            raise ValueError(f"Reviewed target or prompt exceeds the fixed token budget: {key}")
        rows.append({"pair_id": key, "prefix_ids": prefixes, "completion_ids": completion_ids, "completion_text": completions})

    # authorization to use unreviewed responses does not certify their correctness
    report = {"method": method, "objective": objective, "selected_pairs": len(selection), "exported_pairs": len(rows), "approved_pairs": 0 if allow_unreviewed else len(rows), "unapproved_pairs": len(selection) if allow_unreviewed else len(blocked), "review_policy": "user-authorized unreviewed generated responses" if allow_unreviewed else "human-reviewed completion pairs", "completion_quality_verified": not allow_unreviewed, "completion_tokens": {condition: sum(len(row["completion_ids"][condition]) for row in rows) for condition in ("safe", "injection")}, "prompt_tokens_both_conditions": sum(len(ids) for row in rows for ids in row["prefix_ids"].values()), "training_started": False}
    if write:
        if blocked or len(rows) != config["selected_pairs"]:
            raise ValueError("Approve every selected safe response and confirm each injection response follows the injection before exporting; keep the same pair budget")
        path = output / "lora_training_odile" / f"{method}.jsonl"
        if path.exists():
            raise FileExistsError("Final training data already exists; use a new experiment for changed targets")
        write_jsonl(path, rows)
        write_json(path.with_suffix(".json"), {**report, "fingerprint": experiment_fingerprint(), "selection_sha256": digest(selection_path), "review_sha256": digest(review_path), "training_sha256": digest(path), "tokenizer_config_sha256": digest(ROOT / config["model"] / "tokenizer_config.json")})
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True)
    parser.add_argument("--write", action="store_true", help="export reviewed training pairs; does not train")
    parser.add_argument("--allow-unreviewed", action="store_true", help="use generated responses with explicit authorization, without claiming verified correctness")
    args = parser.parse_args()
    print(json.dumps(finalize(args.method, args.write, args.allow_unreviewed), indent=2))
