"""Fit Jacobian matrices that estimate later hidden state vectors from earlier layers

For each prompt, automatic differentiation measures how final vector values change
when earlier hidden state vector values change, then Jlens averages these derivatives
The prepared prompts determine the fitting examples while the model weights stay fixed
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
import argparse
import json
import logging

import jlens

from scripts.jlens_common import fingerprint_prompts, load_model, select_prompts, write_json


# Estimate how much time remains while Jlens processes fitting prompts
# This logging filter uses the average time of completed prompts to estimate the remaining duration
# It only adjusts Jlens fitting messages, leaving other logs unchanged
# The estimate supports monitoring and has no effect on the Jacobian calculation
class TrainingEtaFilter(logging.Filter):
    """Add an ETA to each per-prompt J-lens log line"""

    # Start the elapsed-time counters used for fitting progress estimates
    # Each completed prompt contributes its measured processing time
    # Those values determine the average duration used by the logging filter
    def __init__(self) -> None:
        """Track completed prompts and their total processing time"""
        super().__init__()
        self.prompt_seconds = 0.0
        self.prompts_seen = 0

    # Add the remaining-time estimate to a Jlens fitting progress message
    # The prompt number and elapsed time are read from the existing logging arguments
    # Only messages matching that progress format are changed
    # The fitting computation continues with the same prompts and parameters
    def filter(self, record: logging.LogRecord) -> bool:
        """Add a time estimate to fitting progress messages and keep other logs intact"""
        if record.name != "jlens.fitting":
            return True
        if not isinstance(record.args, tuple) or len(record.args) < 5:
            return True
        if not str(record.msg).lstrip().startswith("prompt %d/%d"):
            return True

        prompt_number = int(record.args[0])
        total_prompts = int(record.args[1])
        self.prompt_seconds += float(record.args[4])
        self.prompts_seen += 1

        average_seconds = self.prompt_seconds / self.prompts_seen
        remaining_seconds = (total_prompts - prompt_number) * average_seconds
        finish_time = datetime.now().astimezone() + timedelta(seconds=remaining_seconds)

        if remaining_seconds < 60:
            remaining_text = f"{remaining_seconds:.0f}s"
        elif remaining_seconds < 3600:
            remaining_text = f"{remaining_seconds / 60:.0f}m"
        else:
            remaining_text = f"{remaining_seconds / 3600:.1f}h"

        message = record.getMessage()
        record.msg = (
            f"{message}  ETA={remaining_text}  "
            f"finish~{finish_time:%Y-%m-%d %H:%M %Z}"
        )
        record.args = ()
        return True


# Keep the parameters for one approved lens fit together in a configuration object
# The data and template parameters identify the exact prepared fitting prompts
# num_prompts and max_seq_len control how many prompts and tokens enter the fit
# The device, precision and derivative batch size determine how the computation runs
# Checkpoint settings allow an interrupted fit to resume with the same prompt set
# These defaults describe the implementation and do not approve a new training run
@dataclass(frozen=True)
class FitConfig:
    """Record one lens fit without changing the language model's weights"""
    model_path: str
    data_path: str
    output_path: str
    checkpoint_path: str
    text_column: str = "text"
    split_column: str | None = None
    split_value: str | None = None
    num_prompts: int = 100
    seed: int = 0
    template: str = "raw"
    max_seq_len: int = 128
    dim_batch: int = 8
    checkpoint_every: int = 1
    device: str = "auto"
    dtype: str = "float32"
    trust_remote_code: bool = False
    resume: bool = True


# Choose the metadata filename associated with a lens checkpoint
# This record contains the fitting configuration, prompt hash and completion status
# Keeping it beside the checkpoint allows a resumed fit to check that it still uses the same prompts
def metadata_path(output_path: str | Path) -> Path:
    """Place the run description beside its lens checkpoint"""
    output = Path(output_path)
    return output.with_suffix(output.suffix + ".json")


# Compute the Jacobian matrices from an already prepared set of fitting prompts
# The selected raw or preformatted template determines how each prompt reaches the model
# The Hugging Face model supplies the transformer, while the Jlens wrapper exposes its layers for differentiation
# Automatic differentiation measures how changes in earlier hidden state vector values affect final-layer values
# The pinned Jlens fitter combines those derivatives across selected token positions and fitting prompts
# One matrix is produced for each inspected layer while the transformer weights remain fixed
# The metadata records the exact prompt set and settings, allowing the fitted matrices to be traced back to their inputs
def fit(config: FitConfig) -> dict:
    """Fit a lens from already-curated prompts and keep a record of its inputs"""

    output_path = Path(config.output_path)
    checkpoint_path = Path(config.checkpoint_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # use the prepared fitting prompts and the selected raw or classification format
    prompts = select_prompts(
        config.data_path,
        text_column=config.text_column,
        split_column=config.split_column,
        split_value=config.split_value,
        num_prompts=config.num_prompts,
        seed=config.seed,
        template=config.template,
    )

    # identify the exact fitting prompts and their order before allowing a resumed fit
    prompt_hash = fingerprint_prompts(prompts)
    metadata_file = metadata_path(output_path)

    # resume only when the fitting prompts and their order match the earlier run
    if config.resume and checkpoint_path.exists() and metadata_file.exists():
        previous = json.loads(metadata_file.read_text(encoding="utf-8"))
        if previous.get("prompt_sha256") != prompt_hash:
            raise ValueError(
                "checkpoint prompt set differs from this run; use a new run name "
                "or pass --no-resume"
            )

    run_metadata = {
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "prompt_sha256": prompt_hash,
        "config": asdict(config),
    }
    write_json(metadata_file, run_metadata)

    try:
        jlens.configure_logging()
        for handler in logging.getLogger("jlens").handlers:
            handler.addFilter(TrainingEtaFilter())
        logging.getLogger("jlens.fitting").setLevel(logging.INFO)

        # hf_model contains the transformer, while lens_model exposes its layers to Jlens
        hf_model, tokenizer, lens_model = load_model(
            config.model_path,
            device=config.device,
            dtype=config.dtype,
            trust_remote_code=config.trust_remote_code,
        )
        print(f"loaded {config.model_path}: {lens_model}", flush=True)

        # fit one Jacobian matrix per inspected layer while keeping transformer weights fixed
        # max_seq_len limits fitting prompt length, and dim_batch controls derivative batching
        # automatic differentiation estimates how final vector values change with earlier values
        # Jlens averages the selected derivatives into one Jacobian matrix per inspected layer
        lens = jlens.fit(
            lens_model,
            prompts=prompts,
            checkpoint_path=str(checkpoint_path),
            checkpoint_every=config.checkpoint_every,
            dim_batch=config.dim_batch,
            max_seq_len=config.max_seq_len,
            resume=config.resume,
        )

        # retain the fitted matrices together so later scoring can choose the matching layer
        lens.save(str(output_path))
    except BaseException as error:
        run_metadata.update(
            {
                "status": "failed",
                "completed_at": datetime.now(UTC).isoformat(),
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        write_json(metadata_file, run_metadata)
        raise

    run_metadata.update(
        {
            "status": "complete",
            "completed_at": datetime.now(UTC).isoformat(),
            "n_prompts_fitted": lens.n_prompts,
            "d_model": lens.d_model,
            "source_layers": lens.source_layers,
        }
    )
    write_json(metadata_file, run_metadata)
    del lens_model, hf_model, tokenizer
    print(f"saved lens to {output_path}", flush=True)
    return run_metadata


# Read the command-line choices into the same FitConfig object used by the fitter
# Required arguments identify the model, prepared prompt data and lens destinations
# Optional arguments set the prompt template, token limit, precision and checkpoint behavior
# The returned configuration describes a fit, with its exact settings requiring review before execution
def parse_args() -> FitConfig:
    """Read command-line settings into the configuration used by the fitter"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data", dest="data_path", required=True)
    parser.add_argument("--output", dest="output_path", required=True)
    parser.add_argument("--checkpoint", dest="checkpoint_path", required=True)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--split-column")
    parser.add_argument("--split-value")
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--template",
        choices=("raw", "preformatted", "classify"),
        default="raw",
    )
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--dim-batch", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype", choices=("float32", "bfloat16", "float16"), default="float32"
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    return FitConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    fit(parse_args())
