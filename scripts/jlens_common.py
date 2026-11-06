"""Shared, model-agnostic helpers for fitting and applying J-lenses."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any
import csv
import hashlib
import json
import random

from transformers import AutoModelForCausalLM, AutoTokenizer
import jlens
import torch
import yaml


# Preserve historical fitting text exactly; inference prompts live in reasoning_detectors.py.
PROMPT_TEMPLATES = {
    "raw": "{text}",

    # The input column already contains the complete prompt. This is an
    # explicit identity mode for materialized, pre-inspected datasets.
    "preformatted": "{text}",
    "classify": (
        "Classify this email as either Injection or Safe.\n"
        "Reply with exactly one label.\n\n"
        "Email:\n{text}\n\n"
        "Classification Label:"
    ),
}


def resolve_device(requested: str) -> str:
    """Use the requested device, or choose an available GPU before CPU."""

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(name: str):
    """Translate the saved precision name to a PyTorch numeric type."""

    try:
        return {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[name]
    except KeyError as error:
        raise ValueError("dtype must be float32, bfloat16, or float16") from error


def load_model(
    model_path: str | Path,
    *,
    device: str = "auto",
    dtype: str = "float32",
    trust_remote_code: bool = False,
):
    """Load a Hugging Face causal LM and wrap it for J-lens."""

    source = str(model_path)
    local_only = Path(source).exists()
    selected_device = resolve_device(device)
    selected_dtype = resolve_dtype(dtype)

    # the tokenizer converts the prepared text into the token IDs expected by this model
    tokenizer = AutoTokenizer.from_pretrained(
        source,
        local_files_only=local_only,
        trust_remote_code=trust_remote_code,
    )

    # load the transformer weights at the requested precision before constructing the Jlens wrapper
    hf_model = AutoModelForCausalLM.from_pretrained(
        source,
        dtype=selected_dtype,
        local_files_only=local_only,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    ).to(selected_device)
    hf_model.eval()

    # Return the language model, its tokenizer, and the Jlens wrapper as separate objects.
    return hf_model, tokenizer, jlens.from_hf(hf_model, tokenizer)


def iter_records(path: str | Path) -> Iterator[dict[str, Any]]:
    """Read dataset rows without loading the full file when its format allows it."""
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".csv":
        with source.open(newline="", encoding="utf-8") as handle:
            yield from csv.DictReader(handle)
        return
    if suffix in {".jsonl", ".ndjson"}:
        with source.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
        return
    if suffix == ".json":
        with source.open(encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, list):
            raise ValueError(f"{source} must contain a JSON list")
        yield from value
        return
    if suffix in {".yaml", ".yml"}:

        with source.open(encoding="utf-8") as handle:
            value = yaml.safe_load(handle)
        if not isinstance(value, list):
            raise ValueError(f"{source} must contain a YAML list")
        yield from value
        return
    if suffix == ".txt":
        with source.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield {"text": line.strip()}
        return
    raise ValueError(f"unsupported data format: {source.suffix}")


def read_records(path: str | Path) -> list[dict[str, Any]]:
    """Load all dataset rows when the caller needs random access or sampling."""
    return list(iter_records(path))


def select_prompts(
    data_path: str | Path,
    *,
    text_column: str,
    split_column: str | None,
    split_value: str | None,
    num_prompts: int,
    seed: int,
    template: str,
) -> list[str]:
    """Choose a repeatable prompt sample from a dataset prepared before training."""
    if num_prompts < 1:
        raise ValueError("num_prompts must be at least 1")
    if template not in PROMPT_TEMPLATES:
        raise ValueError(f"unknown prompt template: {template}")
    if bool(split_column) != bool(split_value):
        raise ValueError("split_column and split_value must be supplied together")


    random_generator = random.Random(seed)
    text_columns = [
        column.strip() for column in text_column.split(",") if column.strip()
    ]
    if not text_columns:
        raise ValueError("text_column must name at least one column")
    texts: list[str] = []
    usable_count = 0

    # filter the requested split before sampling so the fit uses only the chosen data group
    for row in iter_records(data_path):
        if split_column and str(row.get(split_column)) != str(split_value):
            continue

        # Keep only text fields that are present and nonempty.
        parts = []
        for column in text_columns:
            if row.get(column) is not None and str(row[column]).strip():
                parts.append((column, str(row[column]).strip()))
        if len(text_columns) == 1:
            text = parts[0][1] if parts else ""
        else:
            text = "\n\n".join(f"{column.title()}:\n{value}" for column, value in parts)
        if not text:
            continue

        # count usable prompts in their original order before applying seeded reservoir sampling
        usable_count += 1
        if len(texts) < num_prompts:
            texts.append(text)
        else:
            # Reservoir sampling gives every usable row the same chance without loading all rows.
            replacement = random_generator.randrange(usable_count)
            if replacement < num_prompts:
                texts[replacement] = text

    if usable_count < num_prompts:
        raise ValueError(
            f"requested {num_prompts} prompts, but {usable_count} usable rows remain"
        )

    return [PROMPT_TEMPLATES[template].format(text=text) for text in texts]


def fingerprint_prompts(prompts: list[str]) -> str:
    """Identify the exact prompt text and order used in a lens fit."""
    digest = hashlib.sha256()
    for prompt in prompts:
        digest.update(prompt.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    """Finish writing a temporary file before replacing an existing result."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(destination)


def as_bool(value: Any) -> bool:
    """Normalize common prompt-injection labels to booleans."""
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "injection", "malicious"}:
        return True
    if normalized in {"false", "0", "no", "safe", "benign"}:
        return False
    raise ValueError(f"cannot interpret label as boolean: {value!r}")


def verbalizer_ids(tokenizer, label: str) -> list[int]:
    """Read historical label variants that each occupy one complete token."""
    spellings = (label.upper(), label.lower(), label.title())
    token_ids = set()

    # The current comparison uses the actual generated lowercase label, not this historical helper.
    for spelling in spellings:
        for variant in [spelling, f" {spelling}"]:
            encoded = tokenizer.encode(variant, add_special_tokens=False)
            if len(encoded) == 1:
                token_ids.add(encoded[0])
    if not token_ids:
        raise ValueError(f"{label!r} has no single-token verbalization")
    return sorted(token_ids)


def verbalizer_start_ids(tokenizer, label: str) -> list[int]:
    """Reproduce historical first-token scoring, which may use only part of a word."""
    spellings = (label.upper(), label.lower(), label.title())
    token_ids = set()

    # Keep this behavior only for reading earlier experiments, not full-label scoring.
    for spelling in spellings:
        for variant in [spelling, f" {spelling}"]:
            encoded = tokenizer.encode(variant, add_special_tokens=False)
            if encoded:
                token_ids.add(encoded[0])
    if not token_ids:
        raise ValueError(f"{label!r} has no tokenization")
    return sorted(token_ids)


def label_margin(logits, positive_ids: list[int], negative_ids: list[int]) -> float:
    """Calculate the logit difference; the function name is kept for old callers."""

    positive = torch.logsumexp(logits[positive_ids].float(), dim=0)
    negative = torch.logsumexp(logits[negative_ids].float(), dim=0)
    return float((positive - negative).item())


def roc_auc(labels: list[bool], scores: list[float]) -> float | None:
    """Compute binary ROC-AUC with average ranks for tied scores."""
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have equal length")
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    if not positive_count or not negative_count:
        return None

    ordered = sorted(zip(scores, labels, strict=True))
    positive_rank_sum = 0.0
    start = 0
    while start < len(ordered):
        stop = start + 1
        while stop < len(ordered) and ordered[stop][0] == ordered[start][0]:
            stop += 1
        average_rank = ((start + 1) + stop) / 2

        # Tied scores receive the average rank, not an arbitrary ordering.
        tied_labels = [label for score, label in ordered[start:stop]]
        positive_rank_sum += average_rank * sum(tied_labels)
        start = stop
    mann_whitney = positive_rank_sum - positive_count * (positive_count + 1) / 2
    return mann_whitney / (positive_count * negative_count)


def metrics(
    labels: list[bool], predictions: list[bool]
) -> dict[str, float | int | None]:
    """Compute balanced accuracy and standard binary classification metrics."""
    if not labels or len(labels) != len(predictions):
        raise ValueError("labels and predictions must be non-empty and equal length")
    true_positives = sum(
        label and prediction
        for label, prediction in zip(labels, predictions, strict=True)
    )
    true_negatives = sum(
        not label and not prediction
        for label, prediction in zip(labels, predictions, strict=True)
    )
    false_positives = sum(
        not label and prediction
        for label, prediction in zip(labels, predictions, strict=True)
    )
    false_negatives = sum(
        label and not prediction
        for label, prediction in zip(labels, predictions, strict=True)
    )
    positive_recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives else None
    negative_recall = true_negatives / (true_negatives + false_positives) if true_negatives + false_positives else None
    balanced = (
        (positive_recall + negative_recall) / 2
        if positive_recall is not None and negative_recall is not None
        else None
    )
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.0
    recall = positive_recall
    f1 = (
        2 * precision * recall / (precision + recall)
        if recall is not None and precision + recall
        else 0.0
    )
    return {
        "balanced_accuracy": balanced,
        "pint_balanced_accuracy": balanced,
        "accuracy": (true_positives + true_negatives) / len(labels),
        "precision": precision,
        "recall": recall,
        "specificity": negative_recall,
        "false_positive_rate": (
            1 - negative_recall if negative_recall is not None else None
        ),
        "f1": f1,
        "tp": true_positives,
        "tn": true_negatives,
        "fp": false_positives,
        "fn": false_negatives,
    }
