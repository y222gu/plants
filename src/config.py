"""Project configuration: paths, class definitions, training defaults."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# ── Paths ──────────────────────────────────────────────────────────────────────
# Auto-detect base directory (works on both Windows and macOS)
_config_file = Path(__file__).resolve()
BASE_DIR = _config_file.parent.parent  # plants/ directory
DATA_DIR = BASE_DIR / "data"
IMAGE_DIR = DATA_DIR / "image"
ANNOTATION_DIR = DATA_DIR / "annotation"
PREVIEW_DIR = BASE_DIR / "preview"
OUTPUT_DIR = BASE_DIR / "output"  # training runs, exports, checkpoints


# ── Species & Microscopes ─────────────────────────────────────────────────────
SPECIES = ["Millet", "Rice", "Sorghum"]
MICROSCOPES = ["C10", "Olympus", "Zeiss"]
CHANNELS = ["DAPI", "FITC", "TRITC"]  # Blue, Green, Red

# ── Annotated classes (as stored in .txt files) ───────────────────────────────
ANNOTATED_CLASSES = {
    0: "Whole Root",
    1: "Aerenchyma",
    2: "Outer Endodermis",
    3: "Inner Endodermis",
}

# ── Target classes (semantic meaning for model training) ──────────────────────
TARGET_CLASSES = {
    0: "Whole Root",
    1: "Aerenchyma",
    2: "Endodermis",   # ring: outer - inner
    3: "Vascular",     # area inside inner endodermis
}

NUM_CLASSES = len(TARGET_CLASSES)

# ── Visualization colors (BGR for OpenCV, RGB for matplotlib) ─────────────────
CLASS_COLORS_RGB = {
    0: (0, 0, 255),      # Whole Root — Blue
    1: (255, 255, 0),     # Aerenchyma — Yellow
    2: (0, 255, 0),       # Endodermis — Green
    3: (255, 0, 0),       # Vascular — Red
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
