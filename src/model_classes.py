"""Per-model class registry: native class definitions, GT loaders, bio-7 converters.

Each model declares its own native class space. All models convert to the same
7 biological classes for cross-model comparison.

Usage:
    from src.model_classes import MODEL_REGISTRY, BIO_7_CLASSES
    cfg = MODEL_REGISTRY["yolo_overlap_false"]
    gt = cfg.load_gt(sample, h, w)
    bio = cfg.to_bio7(gt, h, w)
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, Tuple

import cv2
import numpy as np

from .annotation_utils import (
    load_sample_annotations,
    parse_yolo_annotations,
    polygons_to_raw_semantic_mask,
)
from .config import SampleRecord


# ═══════════════════════════════════════════════════════════════════════════════
# Shared: 7 biological classes (the target all models convert to)
# ═══════════════════════════════════════════════════════════════════════════════

BIO_7_CLASSES = {
    0: "Whole Root",
    1: "Epidermis",
    2: "Exodermis",
    3: "Cortex",          # includes aerenchyma area
    4: "Aerenchyma",
    5: "Endodermis",
    6: "Vascular",
}

BIO_7_COLORS_RGB = {
    0: (0, 0, 255),       # Whole Root — Blue
    1: (255, 165, 0),     # Epidermis — Orange
    2: (0, 255, 255),     # Exodermis — Cyan
    3: (0, 200, 0),       # Cortex — Dark Green
    4: (255, 255, 0),     # Aerenchyma — Yellow
    5: (0, 255, 0),       # Endodermis — Lime
    6: (255, 0, 0),       # Vascular — Red
}

BIO_7_PUB_COLORS = {
    "Whole Root":  "#0072B2",
    "Epidermis":   "#E69F00",
    "Exodermis":   "#56B4E9",
    "Cortex":      "#009E73",
    "Aerenchyma":  "#F0E442",
    "Endodermis":  "#D55E00",
    "Vascular":    "#CC79A7",
}

BIO_7_NAMES = [BIO_7_CLASSES[i] for i in range(7)]


# ═══════════════════════════════════════════════════════════════════════════════
# Shared utilities
# ═══════════════════════════════════════════════════════════════════════════════

def fill_contours(mask: np.ndarray) -> np.ndarray:
    """Fill external contours to reconstruct filled polygon from ring-like mask.
    No-op on already-filled masks."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    return filled


def _sub(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pixel-wise a - b, clipped to [0, 1]."""
    return np.clip(a.astype(np.int8) - b.astype(np.int8), 0, 1).astype(np.uint8)


def merge_classes(masks: np.ndarray, labels: np.ndarray,
                  h: int, w: int, num_classes: int = 6) -> Dict[int, np.ndarray]:
    """Merge per-instance masks into one binary mask per class. No post-processing."""
    merged = {}
    for cls_id in range(num_classes):
        idx = np.where(labels == cls_id)[0]
        if len(idx) > 0:
            merged[cls_id] = np.clip(masks[idx].sum(axis=0), 0, 1).astype(np.uint8)
        else:
            merged[cls_id] = np.zeros((h, w), dtype=np.uint8)
    return merged


def get_filled_classes(masks: np.ndarray, labels: np.ndarray,
                       h: int, w: int, num_classes: int = 6) -> Dict[int, np.ndarray]:
    """Merge per-instance masks into one filled binary mask per class.
    Applies fill_contours to each merged mask. Used for saving predictions
    (polygon editor, downstream), NOT for IoU computation."""
    filled = merge_classes(masks, labels, h, w, num_classes)
    for cls_id in filled:
        filled[cls_id] = fill_contours(filled[cls_id])
    return filled


def get_raw_classes(masks: np.ndarray, labels: np.ndarray,
                    h: int, w: int, num_classes: int = 6) -> Dict[int, np.ndarray]:
    """Merge per-instance masks into one binary mask per class.
    No contour-fill — keeps ring predictions as-is."""
    raw = {}
    for cls_id in range(num_classes):
        idx = np.where(labels == cls_id)[0]
        if len(idx) > 0:
            raw[cls_id] = np.clip(masks[idx].sum(axis=0), 0, 1).astype(np.uint8)
        else:
            raw[cls_id] = np.zeros((h, w), dtype=np.uint8)
    return raw


# ═══════════════════════════════════════════════════════════════════════════════
# ModelClassConfig dataclass
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ModelClassConfig:
    """Per-model class configuration."""
    name: str
    native_classes: Dict[int, str]
    native_colors_rgb: Dict[int, Tuple[int, int, int]]
    load_gt: Callable      # (sample: SampleRecord, h: int, w: int) -> native GT
    to_bio7: Callable      # (native_output, h: int, w: int) -> Dict[str, np.ndarray]

    @property
    def num_native_classes(self) -> int:
        return len(self.native_classes)


# ═══════════════════════════════════════════════════════════════════════════════
# YOLO overlap=False (6 raw filled polygon classes)
# ═══════════════════════════════════════════════════════════════════════════════

YOLO_OVERLAP_FALSE_CLASSES = {
    0: "Whole Root",
    1: "Aerenchyma",
    2: "Outer Endodermis",
    3: "Inner Endodermis",
    4: "Outer Exodermis",
    5: "Inner Exodermis",
}

YOLO_OVERLAP_FALSE_COLORS = {
    0: (0, 0, 255),       # Blue
    1: (255, 255, 0),     # Yellow
    2: (0, 255, 0),       # Green
    3: (255, 0, 0),       # Red
    4: (255, 128, 0),     # Orange
    5: (128, 0, 255),     # Purple
}


def yolo_overlap_false_load_gt(sample: SampleRecord, h: int, w: int) -> Dict[int, np.ndarray]:
    """Load GT as 6 filled polygon masks for YOLO overlap=False.
    Returns dict {0..5: (H,W) uint8 binary mask}, each a filled polygon."""
    gt = load_sample_annotations(sample, h, w, raw_classes=True)
    return get_filled_classes(gt["masks"], gt["labels"], h, w)


def yolo_overlap_false_to_bio7(filled: Dict[int, np.ndarray],
                                h: int, w: int) -> Dict[str, np.ndarray]:
    """Convert 6 filled polygon masks → 7 bio classes via subtraction.

    filled: {0: Whole Root, 1: Aerenchyma, 2: Outer Endo (filled),
             3: Inner Endo (filled), 4: Outer Exo (filled), 5: Inner Exo (filled)}
    """
    return {
        "Whole Root": filled[0],
        "Epidermis":  _sub(filled[0], filled[4]),        # WR - OuterExo
        "Exodermis":  _sub(filled[4], filled[5]),        # OuterExo - InnerExo
        "Cortex":     _sub(filled[5], filled[2]),        # InnerExo - OuterEndo (includes aer area)
        "Aerenchyma": filled[1],
        "Endodermis": _sub(filled[2], filled[3]),        # OuterEndo - InnerEndo
        "Vascular":   filled[3],                         # InnerEndo
    }


# ═══════════════════════════════════════════════════════════════════════════════
# YOLO overlap=True (6 ring-like classes, learned from overlap_mask painting)
# ═══════════════════════════════════════════════════════════════════════════════

YOLO_OVERLAP_TRUE_CLASSES = {
    0: "Epidermis (WR - OuterExo)",
    1: "Aerenchyma",
    2: "Endodermis ring (OuterEndo - InnerEndo)",
    3: "Vascular (InnerEndo)",
    4: "Exodermis ring (OuterExo - InnerExo)",
    5: "Cortex (excl. aerenchyma)",
}

YOLO_OVERLAP_TRUE_COLORS = {
    0: (255, 165, 0),     # Epidermis — Orange
    1: (255, 255, 0),     # Aerenchyma — Yellow
    2: (0, 255, 0),       # Endodermis — Green
    3: (255, 0, 0),       # Vascular — Red
    4: (0, 255, 255),     # Exodermis — Cyan
    5: (0, 200, 0),       # Cortex — Dark Green
}


def yolo_overlap_true_load_gt(sample: SampleRecord, h: int, w: int) -> Dict[int, np.ndarray]:
    """Load GT as 6 ring masks matching what Ultralytics overlap_mask=True trains on.

    Derives rings from raw filled annotations using the same subtraction
    that polygons2masks_overlap() effectively applies:
        class 0 = WR - OuterExo (epidermis)
        class 1 = Aerenchyma (unchanged)
        class 2 = OuterEndo - InnerEndo (endodermis ring)
        class 3 = InnerEndo (vascular, unchanged)
        class 4 = OuterExo - InnerExo (exodermis ring)
        class 5 = InnerExo - OuterEndo - Aerenchyma (cortex excl. aerenchyma)
    """
    gt = load_sample_annotations(sample, h, w, raw_classes=True)
    filled = get_filled_classes(gt["masks"], gt["labels"], h, w)
    return {
        0: _sub(filled[0], filled[4]),                          # Epidermis
        1: filled[1],                                           # Aerenchyma
        2: _sub(filled[2], filled[3]),                          # Endodermis ring
        3: filled[3],                                           # Vascular
        4: _sub(filled[4], filled[5]),                          # Exodermis ring
        5: _sub(_sub(filled[5], filled[2]), filled[1]),         # Cortex (excl. aer)
    }


def yolo_overlap_true_to_bio7(rings: Dict[int, np.ndarray],
                               h: int, w: int) -> Dict[str, np.ndarray]:
    """Convert 6 ring predictions → 7 bio classes.

    rings: {0: Epidermis, 1: Aerenchyma, 2: Endodermis ring,
            3: Vascular, 4: Exodermis ring, 5: Cortex (excl. aer)}
    """
    # Cortex = cortex_excl_aer + aerenchyma (add aerenchyma back)
    cortex = np.clip(rings[5].astype(np.uint8) + rings[1].astype(np.uint8),
                     0, 1).astype(np.uint8)

    # Whole Root = union of all regions
    wr = np.zeros_like(rings[0])
    for cls_id in rings:
        wr = np.clip(wr + rings[cls_id], 0, 1)

    return {
        "Whole Root": wr.astype(np.uint8),
        "Epidermis":  rings[0],
        "Exodermis":  rings[4],
        "Cortex":     cortex,
        "Aerenchyma": rings[1],
        "Endodermis": rings[2],
        "Vascular":   rings[3],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# U-Net++ semantic (7 mutually exclusive classes via softmax)
# ═══════════════════════════════════════════════════════════════════════════════

UNET_SEMANTIC_CLASSES = {
    0: "Background",
    1: "Epidermis",
    2: "Aerenchyma",
    3: "Endodermis ring",
    4: "Vascular",
    5: "Exodermis ring",
    6: "Cortex (excl. aerenchyma)",
}

UNET_SEMANTIC_COLORS = {
    0: (0, 0, 0),         # Background — Black
    1: (255, 165, 0),     # Epidermis — Orange
    2: (255, 255, 0),     # Aerenchyma — Yellow
    3: (0, 255, 0),       # Endodermis — Green
    4: (255, 0, 0),       # Vascular — Red
    5: (0, 255, 255),     # Exodermis — Cyan
    6: (0, 200, 0),       # Cortex — Dark Green
}


def unet_semantic_load_gt(sample: SampleRecord, h: int, w: int) -> np.ndarray:
    """Load GT as (H,W) semantic mask with 7 labels (0=bg, 1-6=regions).
    Uses paint-order derivation from raw annotations."""
    anns = parse_yolo_annotations(sample.annotation_path, w, h)
    return polygons_to_raw_semantic_mask(anns, h, w)


def unet_semantic_to_bio7(sem_mask: np.ndarray,
                           h: int, w: int) -> Dict[str, np.ndarray]:
    """Convert 7-class semantic mask → 7 bio classes.

    Semantic labels: 0=bg, 1=epidermis, 2=aerenchyma, 3=endodermis,
                     4=vascular, 5=exodermis, 6=cortex (excl. aer)

    Note: Bio-7 Cortex INCLUDES aerenchyma area (labels 6 OR 2).
    """
    return {
        "Whole Root": (sem_mask >= 1).astype(np.uint8),
        "Epidermis":  (sem_mask == 1).astype(np.uint8),
        "Exodermis":  (sem_mask == 5).astype(np.uint8),
        "Cortex":     np.clip((sem_mask == 6).astype(np.uint8) +
                              (sem_mask == 2).astype(np.uint8), 0, 1),
        "Aerenchyma": (sem_mask == 2).astype(np.uint8),
        "Endodermis": (sem_mask == 3).astype(np.uint8),
        "Vascular":   (sem_mask == 4).astype(np.uint8),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# U-Net++ multilabel (6 overlapping sigmoid channels = filled polygons)
# ═══════════════════════════════════════════════════════════════════════════════

UNET_MULTILABEL_CLASSES = {
    0: "Whole Root",
    1: "Aerenchyma",
    2: "Outer Endodermis",
    3: "Inner Endodermis",
    4: "Outer Exodermis",
    5: "Inner Exodermis",
}

UNET_MULTILABEL_COLORS = {
    0: (0, 0, 255),       # Blue
    1: (255, 255, 0),     # Yellow
    2: (0, 255, 0),       # Green
    3: (255, 0, 0),       # Red
    4: (255, 128, 0),     # Orange
    5: (128, 0, 255),     # Purple
}


def unet_multilabel_load_gt(sample: SampleRecord, h: int, w: int) -> Dict[int, np.ndarray]:
    """Load GT as 6 filled polygon masks for U-Net++ multilabel.

    Each channel is a filled binary mask (channels can overlap).
    Returns dict {0..5: (H,W) uint8 binary mask}.
    """
    gt = load_sample_annotations(sample, h, w, raw_classes=True)
    filled = {}
    for cls_id in range(6):
        idx = np.where(gt["labels"] == cls_id)[0]
        if len(idx) > 0:
            filled[cls_id] = np.clip(gt["masks"][idx].sum(axis=0), 0, 1).astype(np.uint8)
        else:
            filled[cls_id] = np.zeros((h, w), dtype=np.uint8)
    return filled


def unet_multilabel_to_bio7(filled: Dict[int, np.ndarray],
                             h: int, w: int) -> Dict[str, np.ndarray]:
    """Convert 6 filled sigmoid channel masks → 7 bio classes via subtraction.

    filled: {0: Whole Root, 1: Aerenchyma, 2: Outer Endo (filled),
             3: Inner Endo (filled), 4: Outer Exo (filled), 5: Inner Exo (filled)}

    Each channel is a filled polygon from sigmoid thresholding. Channels overlap
    (e.g., Outer Endo contains Inner Endo pixels). Subtract to derive rings.
    """
    return {
        "Whole Root": filled[0],
        "Epidermis":  _sub(filled[0], filled[4]),        # WR - OuterExo
        "Exodermis":  _sub(filled[4], filled[5]),        # OuterExo - InnerExo
        "Cortex":     _sub(filled[5], filled[2]),        # InnerExo - OuterEndo (includes aer area)
        "Aerenchyma": filled[1],
        "Endodermis": _sub(filled[2], filled[3]),        # OuterEndo - InnerEndo
        "Vascular":   filled[3],                         # InnerEndo
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Placeholders for future models (add load_gt and to_bio7 when trained)
# ═══════════════════════════════════════════════════════════════════════════════

# micro-SAM: 6 per-class models predict filled polygon instances (same as YOLO overlap=False)
# Reuses yolo_overlap_false GT loader and Bio-7 converter since output format is identical.
# Cellpose: expected to predict 6 raw filled classes (per-class instance models)


# ═══════════════════════════════════════════════════════════════════════════════
# Model Registry
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_REGISTRY: Dict[str, ModelClassConfig] = {
    "yolo_overlap_false": ModelClassConfig(
        name="YOLO (overlap=False)",
        native_classes=YOLO_OVERLAP_FALSE_CLASSES,
        native_colors_rgb=YOLO_OVERLAP_FALSE_COLORS,
        load_gt=yolo_overlap_false_load_gt,
        to_bio7=yolo_overlap_false_to_bio7,
    ),
    "yolo_overlap_true": ModelClassConfig(
        name="YOLO (overlap=True)",
        native_classes=YOLO_OVERLAP_TRUE_CLASSES,
        native_colors_rgb=YOLO_OVERLAP_TRUE_COLORS,
        load_gt=yolo_overlap_true_load_gt,
        to_bio7=yolo_overlap_true_to_bio7,
    ),
    "unet_multilabel": ModelClassConfig(
        name="U-Net++ Multilabel",
        native_classes=UNET_MULTILABEL_CLASSES,
        native_colors_rgb=UNET_MULTILABEL_COLORS,
        load_gt=unet_multilabel_load_gt,
        to_bio7=unet_multilabel_to_bio7,
    ),
    "unet_semantic": ModelClassConfig(
        name="U-Net++ Semantic",
        native_classes=UNET_SEMANTIC_CLASSES,
        native_colors_rgb=UNET_SEMANTIC_COLORS,
        load_gt=unet_semantic_load_gt,
        to_bio7=unet_semantic_to_bio7,
    ),
    "microsam": ModelClassConfig(
        name="micro-SAM",
        native_classes=YOLO_OVERLAP_FALSE_CLASSES,
        native_colors_rgb=YOLO_OVERLAP_FALSE_COLORS,
        load_gt=yolo_overlap_false_load_gt,
        to_bio7=yolo_overlap_false_to_bio7,
    ),
    "sam_semantic": ModelClassConfig(
        name="SAM UNETR Semantic",
        native_classes=UNET_SEMANTIC_CLASSES,
        native_colors_rgb=UNET_SEMANTIC_COLORS,
        load_gt=unet_semantic_load_gt,
        to_bio7=unet_semantic_to_bio7,
    ),
    "sam_unetpp": ModelClassConfig(
        name="SAM UNet++ Semantic",
        native_classes=UNET_SEMANTIC_CLASSES,
        native_colors_rgb=UNET_SEMANTIC_COLORS,
        load_gt=unet_semantic_load_gt,
        to_bio7=unet_semantic_to_bio7,
    ),
    "timm_semantic": ModelClassConfig(
        name="Timm Semantic",
        native_classes=UNET_SEMANTIC_CLASSES,
        native_colors_rgb=UNET_SEMANTIC_COLORS,
        load_gt=unet_semantic_load_gt,
        to_bio7=unet_semantic_to_bio7,
    ),
}
