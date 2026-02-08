"""YOLO annotation parsing, polygon→mask conversion, endodermis ring derivation."""

from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .config import SampleRecord


def parse_yolo_annotations(
    path: Path,
    img_w: int,
    img_h: int,
) -> List[Dict]:
    """Parse a YOLO polygon annotation file.

    Args:
        path: Path to annotation .txt file.
        img_w: Image width in pixels.
        img_h: Image height in pixels.

    Returns:
        List of dicts with keys: class_id (int), polygon (Nx2 int32 array).
    """
    annotations = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:  # class + at least 3 points
                continue
            class_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]
            points = []
            for i in range(0, len(coords) - 1, 2):
                x = coords[i] * img_w
                y = coords[i + 1] * img_h
                points.append([x, y])
            if len(points) >= 3:
                annotations.append({
                    "class_id": class_id,
                    "polygon": np.array(points, dtype=np.float32),
                })
    return annotations


def polygon_to_mask(polygon: np.ndarray, h: int, w: int) -> np.ndarray:
    """Rasterize a single polygon to a binary mask.

    Args:
        polygon: (N, 2) float32 array of (x, y) coordinates.
        h, w: Mask dimensions.

    Returns:
        (H, W) uint8 binary mask (0 or 1).
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.round(polygon).astype(np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 1)
    return mask


def polygons_to_instance_masks(
    annotations: List[Dict],
    h: int,
    w: int,
) -> Dict[str, np.ndarray]:
    """Convert YOLO annotations to instance masks with target class mapping.

    Handles endodermis ring derivation:
    - Annotated class 2 (Outer Endo) + class 3 (Inner Endo) → target class 2 (ring)
    - Annotated class 3 (Inner Endo) → target class 3 (Vascular)

    Returns:
        Dict with:
            masks: (N, H, W) uint8 binary masks
            labels: (N,) int32 target class IDs
            boxes: (N, 4) float32 [x1, y1, x2, y2] bounding boxes
    """
    masks_list = []
    labels_list = []

    # Separate annotations by class
    outer_endo = None
    inner_endo = None
    others = []

    for ann in annotations:
        cid = ann["class_id"]
        if cid == 2:
            outer_endo = ann["polygon"]
        elif cid == 3:
            inner_endo = ann["polygon"]
        else:
            others.append(ann)

    # Class 0 (Whole Root) and Class 1 (Aerenchyma) — direct mapping
    for ann in others:
        mask = polygon_to_mask(ann["polygon"], h, w)
        if mask.sum() > 0:
            masks_list.append(mask)
            labels_list.append(ann["class_id"])

    # Endodermis ring: outer - inner (target class 2)
    if outer_endo is not None and inner_endo is not None:
        outer_mask = polygon_to_mask(outer_endo, h, w)
        inner_mask = polygon_to_mask(inner_endo, h, w)
        ring_mask = np.clip(outer_mask.astype(np.int8) - inner_mask.astype(np.int8), 0, 1).astype(np.uint8)
        if ring_mask.sum() > 0:
            masks_list.append(ring_mask)
            labels_list.append(2)  # Endodermis

    # Vascular: inner endodermis polygon (target class 3)
    if inner_endo is not None:
        vasc_mask = polygon_to_mask(inner_endo, h, w)
        if vasc_mask.sum() > 0:
            masks_list.append(vasc_mask)
            labels_list.append(3)  # Vascular

    if not masks_list:
        return {
            "masks": np.zeros((0, h, w), dtype=np.uint8),
            "labels": np.zeros(0, dtype=np.int32),
            "boxes": np.zeros((0, 4), dtype=np.float32),
        }

    masks = np.stack(masks_list)
    labels = np.array(labels_list, dtype=np.int32)
    boxes = masks_to_boxes(masks)

    return {"masks": masks, "labels": labels, "boxes": boxes}


def masks_to_boxes(masks: np.ndarray) -> np.ndarray:
    """Compute bounding boxes from binary masks.

    Args:
        masks: (N, H, W) uint8 binary masks.

    Returns:
        (N, 4) float32 boxes as [x1, y1, x2, y2].
    """
    n = masks.shape[0]
    boxes = np.zeros((n, 4), dtype=np.float32)
    for i in range(n):
        ys, xs = np.where(masks[i] > 0)
        if len(xs) == 0:
            continue
        boxes[i] = [xs.min(), ys.min(), xs.max(), ys.max()]
    return boxes


def polygons_to_semantic_mask(
    annotations: List[Dict],
    h: int,
    w: int,
) -> np.ndarray:
    """Convert YOLO annotations to a semantic segmentation mask.

    Priority (painted in order, later overwrites): background(0) < whole root(1)
    < aerenchyma(2) < endodermis(3) < vascular(4).

    Note: semantic mask uses 0=background, classes shifted by +1.

    Returns:
        (H, W) int32 semantic mask. 0=background, 1-4=classes 0-3.
    """
    sem = np.zeros((h, w), dtype=np.int32)

    outer_endo = None
    inner_endo = None

    # Paint in priority order
    # 1. Whole Root (class 0 → label 1)
    for ann in annotations:
        if ann["class_id"] == 0:
            mask = polygon_to_mask(ann["polygon"], h, w)
            sem[mask > 0] = 1

    # 2. Aerenchyma (class 1 → label 2)
    for ann in annotations:
        if ann["class_id"] == 1:
            mask = polygon_to_mask(ann["polygon"], h, w)
            sem[mask > 0] = 2
        elif ann["class_id"] == 2:
            outer_endo = ann["polygon"]
        elif ann["class_id"] == 3:
            inner_endo = ann["polygon"]

    # 3. Endodermis ring (→ label 3)
    if outer_endo is not None and inner_endo is not None:
        outer_m = polygon_to_mask(outer_endo, h, w)
        inner_m = polygon_to_mask(inner_endo, h, w)
        ring = np.clip(outer_m.astype(np.int8) - inner_m.astype(np.int8), 0, 1)
        sem[ring > 0] = 3

    # 4. Vascular (→ label 4)
    if inner_endo is not None:
        vasc_m = polygon_to_mask(inner_endo, h, w)
        sem[vasc_m > 0] = 4

    return sem


def load_sample_annotations(
    sample: SampleRecord,
    img_h: int,
    img_w: int,
) -> Dict[str, np.ndarray]:
    """Load and convert annotations for a sample to instance masks.

    Args:
        sample: SampleRecord with annotation_path.
        img_h, img_w: Image dimensions.

    Returns:
        Dict with masks, labels, boxes arrays.
    """
    anns = parse_yolo_annotations(sample.annotation_path, img_w, img_h)
    return polygons_to_instance_masks(anns, img_h, img_w)
