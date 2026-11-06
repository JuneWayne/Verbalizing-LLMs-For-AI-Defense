"""Single source of truth for the models in the J-lens comparison."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    """Keep the result name, local weight folder, and Jacobian fitting batch size together."""
    key: str
    model_dir: str
    dim_batch: int

    def model_path(self, root: Path) -> Path:
        """Find this model's downloaded files."""
        # Model folder names follow the downloaded checkpoint, not always the result name.
        return root / "model" / self.model_dir

    def run_dir(self, root: Path, run_name: str) -> Path:
        """Keep this model's lens experiments in separate folders."""
        # Separate runs so different prompt formats do not overwrite each other's lenses.
        return root / "model/artifacts" / self.key / run_name


# dim_batch controls how many Jacobian directions are computed together during fitting.
MODELS: dict[str, ModelSpec] = {
    "antares-350m": ModelSpec("antares-350m", "antares-350m", 16),
    "antares-1b": ModelSpec("antares-1b", "antares-1b", 16),
    "foundation-sec-8b": ModelSpec(
        "foundation-sec-8b", "Foundation-Sec-8B-Reasoning", 8
    ),
    "qwen2.5-0.5b": ModelSpec("qwen2.5-0.5b", "Qwen2.5-0.5B-Instruct", 16),
    "qwen2.5-1.5b": ModelSpec("qwen2.5-1.5b", "Qwen2.5-1.5B-Instruct", 16),
    "qwen2.5-7b": ModelSpec("qwen2.5-7b", "Qwen2.5-7B-Instruct", 8),
    "qwen3.5-4b": ModelSpec("qwen3.5-4b", "Qwen3.5-4B", 8),
}


def get_model(key: str) -> ModelSpec:
    """Look up a model and explain the available names if the key is unknown."""
    try:
        return MODELS[key]
    except KeyError as error:
        choices = ", ".join(MODELS)
        raise ValueError(f"unknown model {key!r}; choose one of: {choices}") from error
