"""Project configuration: paths, class definitions, training defaults."""

import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

# ── Paths ──────────────────────────────────────────────────────────────────────
# Auto-detect base directory (works on both Windows and macOS)
_config_file = Path(__file__).resolve()
BASE_DIR = _config_file.parent.parent  # plants/ directory
DATA_DIR = BASE_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"
PREVIEW_DIR = BASE_DIR / "preview"
OUTPUT_DIR = BASE_DIR / "output"  # training runs, exports, checkpoints


# ── Species & Microscopes ─────────────────────────────────────────────────────
SPECIES = ["Millet", "Rice", "Sorghum", "Tomato"]
MICROSCOPES = ["C10", "Olympus", "Zeiss"]
CHANNELS = ["DAPI", "FITC", "TRITC"]  # Blue, Green, Red

# ── Annotated classes (as stored in .txt files) ───────────────────────────────
ANNOTATED_CLASSES = {
    0: "Whole Root",
    1: "Aerenchyma",
    2: "Outer Endodermis",
    3: "Inner Endodermis",
    4: "Outer Exodermis",
    5: "Inner Exodermis",
}

# ── Target classes (semantic meaning for model training) ──────────────────────
TARGET_CLASSES = {
    0: "Whole Root",
    1: "Aerenchyma",
    2: "Endodermis",   # ring: outer - inner
    3: "Vascular",     # area inside inner endodermis
}

TARGET_CLASSES_5 = {
    0: "Whole Root",
    1: "Aerenchyma",
    2: "Endodermis",   # ring: outer - inner
    3: "Vascular",     # area inside inner endodermis
    4: "Exodermis",    # ring: outer - inner (tomato only)
}

NUM_CLASSES = len(TARGET_CLASSES_5)

def get_target_classes(num_classes: int = 5) -> dict:
    """Return class dict by count. Prefer get_model_classes() for model-specific lookup."""
    if num_classes == 6:
        return ANNOTATED_CLASSES
    return TARGET_CLASSES_5 if num_classes >= 5 else TARGET_CLASSES


def get_model_classes(model: str) -> dict:
    """Return the class dict a model is trained and evaluated on.

    yolo:            6 raw annotation classes (Whole Root … Inner Exodermis).
    unet_multilabel: 6 raw annotation classes.
    unet_semantic:   TODO — 7 semantic classes (bg + 6 regions), define later.
    sam:             TODO — define later.
    cellpose:        TODO — define later.
    """
    if model in ("yolo", "unet_multilabel"):
        return ANNOTATED_CLASSES
    # Placeholder — other models will be added later
    return TARGET_CLASSES_5


def get_model_colors(model: str) -> dict:
    """Return the RGB color dict matching a model's class space.

    yolo / unet_multilabel: CLASS_COLORS_RGB (6 annotated class colors).
    Others: TARGET_CLASS_COLORS_RGB (5 target class colors) — to be refined.
    """
    if model in ("yolo", "unet_multilabel"):
        return CLASS_COLORS_RGB
    return TARGET_CLASS_COLORS_RGB


# Which target classes are valid (annotated) per species.
# All species now have complete annotations for all 5 target classes.
SPECIES_VALID_CLASSES = {
    "Millet":  {0, 1, 2, 3, 4},
    "Rice":    {0, 1, 2, 3, 4},
    "Sorghum": {0, 1, 2, 3, 4},
    "Tomato":  {0, 1, 2, 3, 4},
}

# ── Visualization colors (BGR for OpenCV, RGB for matplotlib) ─────────────────
# Colors for *annotated* classes (as stored in .txt files)
CLASS_COLORS_RGB = {
    0: (0, 0, 255),      # Whole Root — Blue
    1: (255, 255, 0),     # Aerenchyma — Yellow
    2: (0, 255, 0),       # Outer Endodermis — Green
    3: (255, 0, 0),       # Inner Endodermis — Red
    4: (255, 128, 0),     # Outer Exodermis — Orange
    5: (128, 0, 255),     # Inner Exodermis — Purple
}

# Colors for *target* classes (derived semantic regions for model training)
TARGET_CLASS_COLORS_RGB = {
    0: (0, 0, 255),      # Whole Root — Blue
    1: (255, 255, 0),     # Aerenchyma — Yellow
    2: (0, 255, 0),       # Endodermis — Green
    3: (255, 0, 0),       # Vascular — Red
    4: (0, 255, 255),     # Exodermis — Cyan
}

# ── Training defaults ─────────────────────────────────────────────────────────
DEFAULT_IMG_SIZE = 1024
DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 200
DEFAULT_PATIENCE = 15
DEFAULT_LR = 1e-4
DEFAULT_BACKBONE_LR = 1e-5


@dataclass
class SampleRecord:
    """Metadata for a single annotated sample."""
    species: str
    microscope: str
    experiment: str
    sample_name: str
    image_dir: Path
    annotation_path: Path

    @property
    def group_key(self) -> str:
        """Key for experiment-level grouping: Species/Microscope/Experiment."""
        return f"{self.species}/{self.microscope}/{self.experiment}"

    @property
    def uid(self) -> str:
        """Unique sample identifier."""
        return f"{self.species}_{self.microscope}_{self.experiment}_{self.sample_name}"

    def channel_path(self, channel: str) -> Path:
        """Path to a specific channel TIF file."""
        return self.image_dir / f"{self.sample_name}_{channel}.tif"


def make_run_subfolder(parent_dir: Path) -> Path:
    """Create a dated subfolder like 2026-03-16_001 inside parent_dir.

    Auto-increments the sequence number if the same date already has
    existing subfolders.
    """
    today = datetime.date.today().isoformat()  # "2026-03-16"
    parent_dir = Path(parent_dir)
    parent_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(parent_dir.glob(f"{today}_*"))
    seq = 0
    for d in existing:
        suffix = d.name.split("_")[-1]
        try:
            seq = max(seq, int(suffix))
        except ValueError:
            continue
    seq += 1

    subfolder = parent_dir / f"{today}_{seq:03d}"
    subfolder.mkdir(parents=True, exist_ok=True)
    return subfolder


def save_hparams(run_dir: Path, args) -> Path:
    """Save argparse Namespace (or dict) as hparams.yaml in run_dir."""
    hparams = vars(args) if not isinstance(args, dict) else args

    serializable = {}
    for k, v in hparams.items():
        if isinstance(v, Path):
            serializable[k] = str(v)
        else:
            serializable[k] = v

    out_path = Path(run_dir) / "hparams.yaml"
    with open(out_path, "w") as f:
        yaml.dump(serializable, f, default_flow_style=False, sort_keys=True)

    print(f"Hyperparameters saved to {out_path}")
    return out_path
