"""Prediction mask post-processing pipeline.

Each step is a standalone function that takes and returns a PredictionResult.
The PostProcessor class chains steps together with configurable on/off toggles.

Pipeline order:
    1. fill_holes        — Fill internal holes in each mask
    2. cleanup_whole_root — Morphological close + keep largest component (class 0)
    3. clip_aerenchyma   — Clip aerenchyma to inside whole root boundary
    4. yolo_to_target    — Endodermis ring = outer endo - inner endo (YOLO only)
"""

from dataclasses import dataclass, field
from typing import Dict, List

import cv2
import numpy as np

from .evaluation import PredictionResult


# ── Individual post-processing steps ──────────────────────────────────────────

def fill_holes(pred: PredictionResult) -> PredictionResult:
    """Fill internal holes in every instance mask.

    For each mask, extracts outer contours and re-fills them, removing any
    internal holes caused by self-intersecting polygons or noisy predictions.
    """
    if len(pred.masks) == 0:
        return pred
    filled = np.zeros_like(pred.masks)
    for i in range(len(pred.masks)):
        mask = pred.masks[i]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(filled[i], contours, -1, 1, thickness=cv2.FILLED)
        else:
            filled[i] = mask
    return PredictionResult(masks=filled, labels=pred.labels, scores=pred.scores)


def cleanup_whole_root(pred: PredictionResult) -> PredictionResult:
    """Clean whole root (class 0) masks.

    1. Merge all class-0 masks into one.
    2. Morphological closing to bridge thin line artifacts.
    3. Keep only the largest connected component (one root per image).
    """
    if len(pred.masks) == 0:
        return pred

    wr_idx = np.where(pred.labels == 0)[0]
    if len(wr_idx) == 0:
        return pred

    h, w = pred.masks.shape[1], pred.masks.shape[2]

    # Merge all whole-root masks
    merged = np.clip(pred.masks[wr_idx].sum(axis=0), 0, 1).astype(np.uint8)

    # Morphological closing to bridge thin gaps/lines
    kern_size = max(11, int(min(h, w) * 0.01) | 1)  # ~1% of image, odd
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kern_size, kern_size))
    merged = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Keep only the largest connected component
    n_comp, comp_map = cv2.connectedComponents(merged)
    if n_comp > 2:  # background + multiple components
        largest_id = 0
        largest_area = 0
        for cid in range(1, n_comp):
            area = int((comp_map == cid).sum())
            if area > largest_area:
                largest_area = area
                largest_id = cid
        merged = (comp_map == largest_id).astype(np.uint8)

    # Rebuild: drop old class-0 masks, add single cleaned one
    new_masks = []
    new_labels = []
    new_scores = []

    wr_score = float(pred.scores[wr_idx].max())

    for i in range(len(pred.masks)):
        if pred.labels[i] == 0:
            continue
        new_masks.append(pred.masks[i])
        new_labels.append(pred.labels[i])
        new_scores.append(pred.scores[i])

    if merged.sum() > 0:
        new_masks.insert(0, merged)
        new_labels.insert(0, 0)
        new_scores.insert(0, wr_score)

    if not new_masks:
        return PredictionResult(
            masks=np.zeros((0, h, w), dtype=np.uint8),
            labels=np.zeros(0, dtype=np.int32),
            scores=np.zeros(0, dtype=np.float32),
        )

    return PredictionResult(
        masks=np.stack(new_masks),
        labels=np.array(new_labels, dtype=np.int32),
        scores=np.array(new_scores, dtype=np.float32),
    )


def clip_aerenchyma(pred: PredictionResult) -> PredictionResult:
    """Clip aerenchyma (class 1) masks to inside the whole root (class 0).

    Removes any aerenchyma pixels outside the whole root boundary and drops
    instances that become empty after clipping.
    """
    if len(pred.masks) == 0:
        return pred

    wr_idx = np.where(pred.labels == 0)[0]
    aer_idx = np.where(pred.labels == 1)[0]
    if len(wr_idx) == 0 or len(aer_idx) == 0:
        return pred

    wr_mask = np.clip(pred.masks[wr_idx].sum(axis=0), 0, 1).astype(np.uint8)

    new_masks = list(pred.masks)
    keep = np.ones(len(pred.masks), dtype=bool)
    for i in aer_idx:
        clipped = pred.masks[i] & wr_mask
        if clipped.sum() == 0:
            keep[i] = False
        else:
            new_masks[i] = clipped

    if keep.all():
        return PredictionResult(
            masks=np.stack(new_masks),
            labels=pred.labels.copy(),
            scores=pred.scores.copy(),
        )

    return PredictionResult(
        masks=np.stack([new_masks[i] for i in range(len(new_masks)) if keep[i]]),
        labels=pred.labels[keep],
        scores=pred.scores[keep],
    )


def yolo_to_target(pred: PredictionResult) -> PredictionResult:
    """Convert YOLO annotated classes to target classes.

    YOLO is trained on the original 4 annotation classes:
        class 2 = filled outer endodermis polygon
        class 3 = filled inner endodermis polygon (= vascular area)

    Converts to target classes:
        class 2 = endodermis ring (outer - inner)
        class 3 = vascular (kept as-is)
    """
    if len(pred.masks) == 0:
        return pred

    outer_idx = np.where(pred.labels == 2)[0]
    inner_idx = np.where(pred.labels == 3)[0]

    if len(outer_idx) == 0 or len(inner_idx) == 0:
        return pred

    outer_mask = np.clip(pred.masks[outer_idx].sum(axis=0), 0, 1).astype(np.uint8)
    inner_mask = np.clip(pred.masks[inner_idx].sum(axis=0), 0, 1).astype(np.uint8)

    ring_mask = np.clip(
        outer_mask.astype(np.int8) - inner_mask.astype(np.int8), 0, 1
    ).astype(np.uint8)

    new_masks = []
    new_labels = []
    new_scores = []

    for i in range(len(pred.masks)):
        if pred.labels[i] == 2:
            continue
        new_masks.append(pred.masks[i])
        new_labels.append(pred.labels[i])
        new_scores.append(pred.scores[i])

    if ring_mask.sum() > 0:
        new_masks.append(ring_mask)
        new_labels.append(2)
        new_scores.append(float(pred.scores[outer_idx[0]]))

    if not new_masks:
        h, w = pred.masks.shape[1], pred.masks.shape[2]
        return PredictionResult(
            masks=np.zeros((0, h, w), dtype=np.uint8),
            labels=np.zeros(0, dtype=np.int32),
            scores=np.zeros(0, dtype=np.float32),
        )

    return PredictionResult(
        masks=np.stack(new_masks),
        labels=np.array(new_labels, dtype=np.int32),
        scores=np.array(new_scores, dtype=np.float32),
    )


# ── PostProcessor: configurable pipeline ─────────────────────────────────────

# Registry of all available steps: (name, function, description)
STEPS = [
    ("fill_holes",        fill_holes,        "Fill internal holes in each mask"),
    ("cleanup_whole_root", cleanup_whole_root, "Close gaps + keep largest whole root"),
    ("clip_aerenchyma",   clip_aerenchyma,   "Clip aerenchyma to inside whole root"),
    ("yolo_to_target",    yolo_to_target,    "Endodermis ring = outer - inner (YOLO)"),
]

# Default steps per model type
DEFAULT_STEPS: Dict[str, List[str]] = {
    "yolo":    ["fill_holes", "cleanup_whole_root", "clip_aerenchyma", "yolo_to_target"],
    "maskrcnn": ["fill_holes", "cleanup_whole_root", "clip_aerenchyma"],
    "unet":    ["fill_holes", "cleanup_whole_root", "clip_aerenchyma"],
}

_STEP_MAP = {name: fn for name, fn, _ in STEPS}


class PostProcessor:
    """Configurable post-processing pipeline for prediction masks.

    Args:
        model: Model type ("yolo", "maskrcnn", "unet") — determines defaults.
        enable: List of step names to force-enable (on top of defaults).
        disable: List of step names to force-disable.

    The pipeline order is always fixed (as defined in STEPS). Enabling/disabling
    only controls which steps are active, not their order.
    """

    def __init__(
        self,
        model: str = "yolo",
        enable: List[str] = None,
        disable: List[str] = None,
    ):
        active = set(DEFAULT_STEPS.get(model, []))
        if enable:
            active |= set(enable)
        if disable:
            active -= set(disable)

        # Build ordered pipeline from the global STEPS registry
        self.pipeline = []
        for name, fn, desc in STEPS:
            if name in active:
                self.pipeline.append((name, fn, desc))

        self.model = model

    def run(self, pred: PredictionResult) -> PredictionResult:
        """Apply all active steps to a single prediction."""
        for _, fn, _ in self.pipeline:
            pred = fn(pred)
        return pred

    def run_all(self, predictions: dict, verbose: bool = True) -> dict:
        """Apply pipeline to all predictions in a dict, with progress logging."""
        if verbose:
            print(f"Post-processing pipeline ({self.model}):")
            for name, _, desc in self.pipeline:
                print(f"  + {name:25s} {desc}")
            if not self.pipeline:
                print("  (no steps enabled)")

        for uid in predictions:
            predictions[uid] = self.run(predictions[uid])

        return predictions

    def summary(self) -> str:
        """Return a human-readable summary of active steps."""
        lines = [f"PostProcessor ({self.model}):"]
        for name, _, desc in self.pipeline:
            lines.append(f"  + {name}: {desc}")
        return "\n".join(lines)
