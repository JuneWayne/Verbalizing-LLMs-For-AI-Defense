"""Choose fixed source groups without rewriting FIPI's prepared prompts"""

from collections import Counter, defaultdict
import hashlib
import json

import numpy as np

from scripts.curation.common import CONFIG_PATH, ROOT, digest, has_witness, settings, write_json, write_jsonl


def source_groups(rows):
    # join source IDs and identical clean text so neither can cross a split
    parent = {}

    def root(key):
        parent.setdefault(key, key)
        if parent[key] != key:
            parent[key] = root(parent[key])
        return parent[key]

    for row in rows:
        source = ("id", str(row["user_prompt_id"]))
        text = ("text", row["clean_data"])
        parent[root(source)] = root(text)

    groups = defaultdict(list)
    for row in rows:
        groups[root(("id", str(row["user_prompt_id"])))].append(row)
    return list(groups.values())


def build_pool():
    config, output = settings()
    files = {name: ROOT / config[name + "_data"] for name in ("training", "testing")}
    hashes = {name: digest(path) for name, path in files.items()}
    fingerprint = hashlib.sha256(json.dumps([digest(CONFIG_PATH), hashes], sort_keys=True).encode()).hexdigest()
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["fingerprint"] != fingerprint:
            raise ValueError("Existing experiment differs: choose a new output folder")
        return manifest

    raw = {name: json.loads(path.read_text()) for name, path in files.items()}
    if config["candidate_pairs"] == "all_remaining" or config.get("pool_mode") == "sample_remaining_variants":
        return build_full_pool(config, output, raw, hashes, fingerprint)
    shared_text = {row["clean_data"] for row in raw["training"]} & {row["clean_data"] for row in raw["testing"]}
    rng = np.random.default_rng(config["seed"])
    selected, exclusions, available = {}, {}, {}
    for name, rows in raw.items():
        valid = []
        counts = Counter()
        for row in rows:
            # leave the source files untouched and record each exclusion from this experiment
            if not row["user_instruction"].strip():
                counts["missing_task"] += 1
            elif row["clean_data"] in shared_text:
                counts["clean_text_shared_between_training_and_testing"] += 1
            elif not row["witness"].strip() or has_witness(row["user_instruction"] + " " + row["clean_data"], row["witness"]):
                counts["ambiguous_witness"] += 1
            else:
                valid.append(row)

        groups = source_groups(valid)
        rng.shuffle(groups)
        # one injection variant per group prevents repeated clean prompts from dominating selection
        representatives = [dict(group[int(rng.integers(len(group)))]) for group in groups]
        for row in representatives:
            row["pair_id"] = f"{name}:{row['data_id']}"
            row["source_split"] = name
        available[name] = len(representatives)
        exclusions[name] = dict(counts)
        sizes = {"selector_training": config["selector_training_pairs"], "validation": config["validation_pairs"], "candidates": config["candidate_pairs"]} if name == "training" else {"testing": config["testing_pairs"]}
        if sum(sizes.values()) > len(representatives):
            raise ValueError(f"Too few distinct source groups in {name}")
        start = 0
        for split, count in sizes.items():
            selected[split] = representatives[start:start + count]
            write_jsonl(output / "splits" / f"{split}.jsonl", selected[split])
            start += count

    manifest = {
        "fingerprint": fingerprint, "config_sha256": digest(CONFIG_PATH), "config": config,
        "source_sha256": hashes, "excluded_records": exclusions, "available_source_groups": available,
        "split_pairs": {name: len(rows) for name, rows in selected.items()},
        "split_sha256": {name: digest(output / "splits" / f"{name}.jsonl") for name in selected},
        "split_rule": "source ID and exact clean text stay together; one variant per source group",
        "testing_rule": "reserved from the official testing file; not read by selectors",
        "selection_scope": "pilot pool, not the full released FIPI dataset",
    }
    write_json(manifest_path, manifest)
    return manifest


def build_full_pool(config, output, raw, hashes, fingerprint):
    # inspect every training row, but keep test sources and probe-fitting sources out of selection
    test_text = {row["clean_data"] for row in raw["testing"]}
    test_ids = {str(row["user_prompt_id"]) for row in raw["testing"]}
    rng = np.random.default_rng(config["seed"])
    groups = source_groups(raw["training"])
    rng.shuffle(groups)
    selected = {name: [] for name in ("selector_training", "validation", "candidates")}
    audit, usable_groups = [], []
    for group in groups:
        test_overlap = any(row["clean_data"] in test_text or str(row["user_prompt_id"]) in test_ids for row in group)
        valid = []
        for row in group:
            reason = ""
            if test_overlap:
                reason = "source_shared_with_testing"
            elif not row["user_instruction"].strip():
                reason = "missing_task"
            elif not row["witness"].strip() or has_witness(row["user_instruction"] + " " + row["clean_data"], row["witness"]):
                reason = "ambiguous_witness"
            if reason:
                audit.append({"pair_id": f"training:{row['data_id']}", "use": reason})
            else:
                valid.append(row)
        if valid:
            usable_groups.append(valid)

    required = config["selector_training_pairs"] + config["validation_pairs"]
    if len(usable_groups) <= required:
        raise ValueError("Too few separate source groups for the requested probe split")
    for index, group in enumerate(usable_groups):
        if index < config["selector_training_pairs"]:
            split = "selector_training"
        elif index < required:
            split = "validation"
        else:
            split = "candidates"
        # one variant per fitting source gives the probes 1,500 distinct training sources
        chosen = group if split == "candidates" else [group[int(rng.integers(len(group)))]]
        chosen_ids = {row["data_id"] for row in chosen}
        for row in group:
            use = split if row["data_id"] in chosen_ids else "related_to_" + split
            audit.append({"pair_id": f"training:{row['data_id']}", "use": use})
        selected[split].extend({**row, "pair_id": f"training:{row['data_id']}", "source_split": "training"} for row in chosen)

    # sample pairs only after removing every source group used for fitting or validation
    if config.get("pool_mode") == "sample_remaining_variants":
        count = config["candidate_pairs"]
        if not 0 < count <= len(selected["candidates"]):
            raise ValueError("Candidate count must fit the remaining source groups")
        indices = sorted(rng.choice(len(selected["candidates"]), count, replace=False))
        selected["candidates"] = [selected["candidates"][index] for index in indices]
        chosen_ids = {row["pair_id"] for row in selected["candidates"]}
        for row in audit:
            if row["use"] == "candidates" and row["pair_id"] not in chosen_ids:
                row["use"] = "outside_candidate_sample"

    # reserve the complete official test file without generating or scoring its prompts
    selected["testing"] = [{**row, "pair_id": f"testing:{row['data_id']}", "source_split": "testing"} for row in raw["testing"]]
    for split, rows in selected.items():
        write_jsonl(output / "splits" / f"{split}.jsonl", rows)
    write_jsonl(output / "pool_audit.jsonl", audit)
    manifest = {
        "fingerprint": fingerprint, "config_sha256": digest(CONFIG_PATH), "config": config,
        "source_sha256": hashes, "training_rows_audited": len(audit),
        "training_row_use": dict(Counter(row["use"] for row in audit)),
        "split_pairs": {name: len(rows) for name, rows in selected.items()},
        "split_sha256": {name: digest(output / "splits" / f"{name}.jsonl") for name in selected},
        "split_rule": "source ID and exact clean text stay together; seeded pair sample from remaining variants" if config.get("pool_mode") == "sample_remaining_variants" else "source ID and exact clean text stay together; all candidate injection variants retained",
        "testing_rule": "complete official testing file reserved; no testing generation or selector fitting",
        "selection_scope": "seeded sample of usable training pairs outside probe training, validation and related source groups" if config.get("pool_mode") == "sample_remaining_variants" else "all usable training rows outside probe training, validation and related source groups",
    }
    write_json(output / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    print(json.dumps(build_pool(), indent=2))
