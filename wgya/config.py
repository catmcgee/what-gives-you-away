"""Paths and immutable identifiers for the released experiment."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct"
MODEL_REVISION = "006f5dcd1393c3add266de40994ba96225e9689d"
MODEL_REVISIONS = {
    MODEL_NAME: MODEL_REVISION,
    "unsloth/Meta-Llama-3.1-8B-Instruct": ("a2856192dd7c25b842431f39c179a6c2c2f627d1"),
    "allenai/OLMo-2-0325-32B-Instruct": ("b96024342a77a69aa0dda815c3454a671f477463"),
}
SEED = 0


def model_revision(model_name: str) -> str:
    """Return the pinned revision for the study model."""

    if model_name not in MODEL_REVISIONS:
        raise ValueError(f"unsupported model: {model_name!r}")
    return MODEL_REVISIONS[model_name]
