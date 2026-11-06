"""Prepare the fixed validation responses without changing any model weights"""

import json

from transformers import AutoTokenizer

from scripts.curation.common import (
    ROOT,
    digest,
    experiment_fingerprint,
    load_pairs,
    messages,
    read_jsonl,
    settings,
    write_json,
    write_jsonl,
)


def export_validation():
    config, output = settings()
    pairs = load_pairs("validation")[:config["lora_validation_pairs"]]
    records = [row for path in sorted((output / "responses/validation").glob("*.jsonl")) for row in read_jsonl(path)]
    responses = {(row["pair_id"], row["condition"]): row for row in records}
    expected = {(row["pair_id"], condition) for row in pairs for condition in ("safe", "injection")}
    if set(responses) != expected or len(records) != len(expected):
        raise ValueError("Validation recording needs both responses for all fixed validation pairs")
    tokenizer = AutoTokenizer.from_pretrained(ROOT / config["model"], local_files_only=True)
    rows = []
    for row in pairs:
        # retain the original response tokens, including any natural end token
        prefixes = {condition: tokenizer.apply_chat_template(messages(row, condition, config), tokenize=True, add_generation_prompt=True, return_dict=False) for condition in ("safe", "injection")}
        completions = {condition: responses[row["pair_id"], condition]["response_ids"] for condition in prefixes}
        if any(not ids for ids in completions.values()):
            raise ValueError("Validation cannot use an empty response")
        rows.append({"pair_id": row["pair_id"], "prefix_ids": prefixes, "completion_ids": completions})
    path = output / "lora_validation.jsonl"
    if path.exists() and read_jsonl(path) != rows:
        raise ValueError("Existing validation responses differ")
    write_jsonl(path, rows)
    # validation measures the same fifty-token loss, including prefixes of unfinished responses
    report = {"fingerprint": experiment_fingerprint(), "pairs": len(rows), "sha256": digest(path), "unfinished_responses": sum(not row["finished"] for row in records), "completion_quality_verified": False, "selection_rule": "first 100 pairs in the fixed seeded validation split, before generation"}
    write_json(path.with_suffix(".json"), report)
    return report


if __name__ == "__main__":
    print(json.dumps(export_validation(), indent=2))
