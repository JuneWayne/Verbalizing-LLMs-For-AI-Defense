"""Materialize the exact classifier-formatted prompts used for old J-lens fits.

This is a preprocessing command, not part of fitting. The output contains no
complete raw-email column: it stores only the model-specific token window, the
final prompt, provenance fields, and measured token counts.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import sys

from transformers import AutoTokenizer
import pandas as pd

from scripts.jlens_common import PROMPT_TEMPLATES, fingerprint_prompts, write_json
from scripts.jlens_config import MODELS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_SOURCE = PROJECT_ROOT / "data/source/Jlens-windowed-pair-manifest.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/Jlens-windowed-training.csv"
REQUIRED_COLUMNS = {"pair_id", "label", "text", "focus_start", "focus_end"}


def build_email_window(tokenizer, text: str, start: int, end: int, budget: int) -> str:
    """Reproduce the historical token window centered on an inserted span."""
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = encoded["offset_mapping"]
    if not offsets:
        return text.strip()

    focus_tokens = [
        index
        for index, (token_start, token_end) in enumerate(offsets)
        if token_end > start and token_start < max(end, start + 1)
    ]
    if focus_tokens:
        first_token, last_token = focus_tokens[0], focus_tokens[-1]
    else:
        boundary_token = next(
            (
                index
                for index, (token_start, token_end) in enumerate(offsets)
                if token_end > start
            ),
            len(offsets) - 1,
        )
        first_token = last_token = boundary_token

    # Keep the inserted span when it fits; divide spare tokens around it.
    focus_width = last_token - first_token + 1
    if focus_width >= budget:
        selected_first = first_token
        selected_last = first_token + budget - 1
    else:
        left_padding = (budget - focus_width) // 2
        selected_first = max(0, first_token - left_padding)
        selected_last = min(len(offsets) - 1, selected_first + budget - 1)
        selected_first = max(0, selected_last - budget + 1)

    char_start = offsets[selected_first][0]
    char_end = offsets[selected_last][1]
    return text[char_start:char_end].strip()


def validate_selected_rows(rows: pd.DataFrame, num_prompts: int) -> None:
    """Check that the chosen rows contain complete, ordered safe/injection pairs."""
    if num_prompts < 2 or num_prompts % 2:
        raise ValueError("num_prompts must be a positive even number")
    if len(rows) < num_prompts:
        raise ValueError(f"requested {num_prompts} rows, but only {len(rows)} exist")

    selected = rows.iloc[:num_prompts]
    for offset in range(0, num_prompts, 2):
        pair = selected.iloc[offset : offset + 2]
        if pair["pair_id"].nunique() != 1 or pair["label"].astype(int).tolist() != [0, 1]:
            raise ValueError(
                f"manifest rows {offset} and {offset + 1} are not an ordered "
                "Safe/Injection pair"
            )


def curate_model(
    source: pd.DataFrame,
    *,
    model_key: str,
    num_prompts: int,
    max_seq_len: int,
    token_margin: int,
) -> tuple[list[dict], dict]:
    """Prepare one model's exact prompt windows and record their token lengths."""
    model_settings = MODELS[model_key]
    model_path = model_settings.model_path(PROJECT_ROOT)
    if not (model_path / "tokenizer_config.json").is_file():
        raise FileNotFoundError(f"tokenizer is missing from {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    empty_prompt = PROMPT_TEMPLATES["classify"].format(text="")
    template_tokens = len(tokenizer.encode(empty_prompt, add_special_tokens=True))
    # Reserve space for the classification instruction and token-boundary differences.
    email_budget = max_seq_len - template_tokens - token_margin
    if email_budget < 16:
        raise ValueError(
            f"{model_key}: template uses {template_tokens} tokens, leaving only "
            f"{email_budget} for the email"
        )

    curated: list[dict] = []
    selected = source.iloc[:num_prompts]
    for prompt_index, (source_index, row) in enumerate(selected.iterrows()):
        text = str(row["text"])
        focus_start = int(row["focus_start"])
        focus_end = int(row["focus_end"])
        window = build_email_window(
            tokenizer,
            text,
            focus_start,
            focus_end,
            email_budget,
        )
        prompt = PROMPT_TEMPLATES["classify"].format(text=window)
        email_tokens = len(tokenizer.encode(window, add_special_tokens=False))
        prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=True))
        if prompt_tokens > max_seq_len:
            raise ValueError(
                f"{model_key} prompt {prompt_index} has {prompt_tokens} tokens; "
                f"maximum is {max_seq_len}"
            )

        label = int(row["label"])
        # Preserve the historical prompt and label capitalization used by these fits.
        curated.append(
            {
                "model": model_key,
                "prompt_index": prompt_index,
                "source_manifest_row": int(source_index),
                "pair_id": int(row["pair_id"]),
                "label": label,
                "label_name": "Injection" if label else "Safe",
                "focus_kind": "inserted_span" if label else "clean_twin_boundary",
                "focus_text": text[focus_start:focus_end].strip(),
                "email_window": window,
                "prompt": prompt,
                "email_token_count": email_tokens,
                "prompt_token_count": prompt_tokens,
                "max_seq_len": max_seq_len,
            }
        )

    prompts = [row["prompt"] for row in curated]
    metadata = {
        "model": model_key,
        "model_path": str(model_path.relative_to(PROJECT_ROOT)),
        "num_prompts": len(curated),
        "num_pairs": len(curated) // 2,
        "label_counts": {"Safe": len(curated) // 2, "Injection": len(curated) // 2},
        "max_seq_len": max_seq_len,
        "template_token_count": template_tokens,
        "email_token_budget": email_budget,
        "token_margin": token_margin,
        "maximum_observed_prompt_tokens": max(row["prompt_token_count"] for row in curated),
        "prompt_sha256": fingerprint_prompts(prompts),
    }
    return curated, metadata


def materialize(
    source_path: Path,
    output_path: Path,
    model_keys: list[str],
    num_prompts: int,
    max_seq_len: int,
    token_margin: int,
) -> None:
    """Save prepared training prompts and metadata without fitting a lens."""
    source = pd.read_csv(source_path)
    missing = REQUIRED_COLUMNS.difference(source.columns)
    if missing:
        raise ValueError(f"source manifest is missing columns: {sorted(missing)}")
    validate_selected_rows(source, num_prompts)

    all_rows: list[dict] = []
    model_metadata: dict[str, dict] = {}
    for model_key in model_keys:
        rows, metadata = curate_model(
            source,
            model_key=model_key,
            num_prompts=num_prompts,
            max_seq_len=max_seq_len,
            token_margin=token_margin,
        )
        all_rows.extend(rows)
        model_metadata[model_key] = metadata
        print(
            f"{model_key}: {len(rows)} prompts, "
            f"max tokens={metadata['maximum_observed_prompt_tokens']}",
            flush=True,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(all_rows)
    frame.to_csv(output_path, index=False)
    write_json(
        output_path.with_suffix(output_path.suffix + ".json"),
        {
            "source_manifest": str(source_path.relative_to(PROJECT_ROOT)),
            "output": str(output_path.relative_to(PROJECT_ROOT)),
            "row_count": len(frame),
            "model": model_metadata,
        },
    )
    print(f"saved {len(frame)} final prompts to {output_path}", flush=True)


def parse_args() -> argparse.Namespace:
    """Read the source, model choices, and prompt-length settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--models", nargs="+", choices=tuple(MODELS), default=list(MODELS))
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--token-margin", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    materialize(
        arguments.source.resolve(),
        arguments.output.resolve(),
        arguments.models,
        arguments.num_prompts,
        arguments.max_seq_len,
        arguments.token_margin,
    )
