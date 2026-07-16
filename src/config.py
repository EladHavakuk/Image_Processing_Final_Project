"""
Central path configuration. Every path in this project is derived from the
repository's own location on disk (via __file__), NOT hardcoded to any specific
machine or username. This is what makes the repo portable: clone/extract it
anywhere and every script finds data/models/results relative to itself.

(An earlier version of this codebase had absolute paths like
"/home/claude/project/..." hardcoded into ~9 files -- that was specific to the
sandbox this project was originally built in and would not exist on any other
computer. If you're reading this after cloning the repo fresh: that's exactly
the bug this file fixes. See README troubleshooting log.)
"""
from pathlib import Path

# src/config.py -> repo root is one level up
PROJECT_ROOT = Path(__file__).resolve().parent.parent

SRC_DIR = PROJECT_ROOT / "src"

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "bdd_subset"
IMAGES_DIR = DATA_DIR / "images"
LABELS_PATH = DATA_DIR / "labels_subset.json"

FINETUNE_DIR = PROJECT_ROOT / "data" / "finetune"
FINETUNE_V2_DIR = PROJECT_ROOT / "data" / "finetune_v2"

MODELS_DIR = PROJECT_ROOT / "models"
BASELINE_WEIGHTS = MODELS_DIR / "yolov8n.pt"
FINETUNED_WEIGHTS = MODELS_DIR / "yolov8n_finetuned.pt"

RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

FULL_RESULTS_CSV = TABLES_DIR / "full_results.csv"
PROGRESS_FILE = TABLES_DIR / "progress.txt"

PRESENTATIONS_DIR = PROJECT_ROOT / "presentations"
PRESENTATION_ASSETS_DIR = PRESENTATIONS_DIR / "assets"


def ensure_dirs():
    """Create every output directory this project writes to, if missing."""
    for d in [DATA_DIR, MODELS_DIR, TABLES_DIR, FIGURES_DIR, PRESENTATION_ASSETS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Quick self-check: run `python config.py` after cloning to confirm paths
    # resolve to real locations before running anything else.
    print(f"PROJECT_ROOT   = {PROJECT_ROOT}")
    print(f"IMAGES_DIR     = {IMAGES_DIR}  (exists: {IMAGES_DIR.exists()})")
    print(f"LABELS_PATH    = {LABELS_PATH}  (exists: {LABELS_PATH.exists()})")
    print(f"BASELINE_WEIGHTS = {BASELINE_WEIGHTS}  (exists: {BASELINE_WEIGHTS.exists()})")
    print(f"TABLES_DIR     = {TABLES_DIR}  (exists: {TABLES_DIR.exists()})")
    print(f"FIGURES_DIR    = {FIGURES_DIR}  (exists: {FIGURES_DIR.exists()})")
