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
    # Fix holes from self-intersecting polygons: cv2.fillPoly uses the
    # even-odd rule, so self-crossings create unfilled interior regions.
    # Re-fill from outer contour to produce a solid mask.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(mask, contours, -1, 1, thickness=cv2.FILLED)
    return mask


def polygons_to_instance_masks(
    annotations: List[Dict],
    h: int,
    w: int,
    num_classes: int = 5,
) -> Dict[str, np.ndarray]:
    """Convert YOLO annotations to instance masks with target class mapping.

    Handles endodermis ring derivation:
    - Annotated class 2 (Outer Endo) + class 3 (Inner Endo) → target class 2 (ring)
    - Annotated class 3 (Inner Endo) → target class 3 (Vascular)

    When num_classes >= 5, also derives exodermis ring:
    - Annotated class 4 (Outer Exo) + class 5 (Inner Exo) → target class 4 (ring)

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
    outer_exo = None
    inner_exo = None
    others = []

    for ann in annotations:
        cid = ann["class_id"]
        if cid == 2:
            outer_endo = ann["polygon"]
        elif cid == 3:
            inner_endo = ann["polygon"]
        elif cid == 4:
            outer_exo = ann["polygon"]
        elif cid == 5:
            inner_exo = ann["polygon"]
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

    # Exodermis ring: outer - inner (target class 4), only when 5-class mode
    if num_classes >= 5 and outer_exo is not None and inner_exo is not None:
        outer_mask = polygon_to_mask(outer_exo, h, w)
        inner_mask = polygon_to_mask(inner_exo, h, w)
        ring_mask = np.clip(outer_mask.astype(np.int8) - inner_mask.astype(np.int8), 0, 1).astype(np.uint8)
        if ring_mask.sum() > 0:
            masks_list.append(ring_mask)
            labels_list.append(4)  # Exodermis

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
    num_classes: int = 5,
) -> np.ndarray:
    """Convert YOLO annotations to a semantic segmentation mask.

    Priority (painted in order, later overwrites): background(0) < whole root(1)
    < aerenchyma(2) < endodermis(3) < vascular(4) [< exodermis(5) if num_classes>=5].

    Note: semantic mask uses 0=background, classes shifted by +1.

    Returns:
        (H, W) int32 semantic mask. 0=background, 1-4 (or 1-5) = target classes.
    """
    sem = np.zeros((h, w), dtype=np.int32)

    outer_endo = None
    inner_endo = None
    outer_exo = None
    inner_exo = None

    # Paint in priority order
    # 1. Whole Root (class 0 → label 1)
    for ann in annotations:
        if ann["class_id"] == 0:
            mask = polygon_to_mask(ann["polygon"], h, w)
            sem[mask > 0] = 1

    # 2. Aerenchyma (class 1 → label 2) + collect ring annotations
    for ann in annotations:
        if ann["class_id"] == 1:
            mask = polygon_to_mask(ann["polygon"], h, w)
            sem[mask > 0] = 2
        elif ann["class_id"] == 2:
            outer_endo = ann["polygon"]
        elif ann["class_id"] == 3:
            inner_endo = ann["polygon"]
        elif ann["class_id"] == 4:
            outer_exo = ann["polygon"]
        elif ann["class_id"] == 5:
            inner_exo = ann["polygon"]

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

    # 5. Exodermis ring (→ label 5), only when 5-class mode
    if num_classes >= 5 and outer_exo is not None and inner_exo is not None:
        outer_m = polygon_to_mask(outer_exo, h, w)
        inner_m = polygon_to_mask(inner_exo, h, w)
        ring = np.clip(outer_m.astype(np.int8) - inner_m.astype(np.int8), 0, 1)
        sem[ring > 0] = 5

    return sem


def polygons_to_multilabel_mask(
    annotations: List[Dict],
    h: int,
    w: int,
    num_classes: int = 5,
) -> np.ndarray:
    """Convert YOLO annotations to multi-label mask with independent channels.

    Each pixel can belong to multiple classes (e.g., aerenchyma is also inside
    the whole root). Returns binary channels with sigmoid-compatible targets.

    Channel mapping:
        0: Whole Root (entire root area)
        1: Aerenchyma (air spaces — also marked in ch0)
        2: Endodermis ring (outer - inner endo — also marked in ch0)
        3: Vascular (inner endo area — also marked in ch0)
        4: Exodermis ring (outer - inner exo — only when num_classes >= 5)

    Returns:
        (num_classes, H, W) float32 mask with values 0.0 or 1.0.
    """
    multilabel = np.zeros((num_classes, h, w), dtype=np.float32)

    outer_endo = None
    inner_endo = None
    outer_exo = None
    inner_exo = None

    for ann in annotations:
        cid = ann["class_id"]
        if cid == 0:
            mask = polygon_to_mask(ann["polygon"], h, w)
            multilabel[0][mask > 0] = 1.0
        elif cid == 1:
            mask = polygon_to_mask(ann["polygon"], h, w)
            multilabel[1][mask > 0] = 1.0
            # Aerenchyma is inside whole root
            multilabel[0][mask > 0] = 1.0
        elif cid == 2:
            outer_endo = ann["polygon"]
        elif cid == 3:
            inner_endo = ann["polygon"]
        elif cid == 4:
            outer_exo = ann["polygon"]
        elif cid == 5:
            inner_exo = ann["polygon"]

    # Endodermis ring: outer - inner (channel 2)
    if outer_endo is not None and inner_endo is not None:
        outer_mask = polygon_to_mask(outer_endo, h, w)
        inner_mask = polygon_to_mask(inner_endo, h, w)
        ring = np.clip(outer_mask.astype(np.int8) - inner_mask.astype(np.int8), 0, 1)
        multilabel[2][ring > 0] = 1.0
        # Endodermis is inside whole root
        multilabel[0][ring > 0] = 1.0

    # Vascular: inner endodermis area (channel 3)
    if inner_endo is not None:
        vasc_mask = polygon_to_mask(inner_endo, h, w)
        multilabel[3][vasc_mask > 0] = 1.0
        # Vascular is inside whole root
        multilabel[0][vasc_mask > 0] = 1.0

    # Exodermis ring: outer - inner (channel 4), only when 5-class mode
    if num_classes >= 5 and outer_exo is not None and inner_exo is not None:
        outer_mask = polygon_to_mask(outer_exo, h, w)
        inner_mask = polygon_to_mask(inner_exo, h, w)
        ring = np.clip(outer_mask.astype(np.int8) - inner_mask.astype(np.int8), 0, 1)
        multilabel[4][ring > 0] = 1.0
        # Exodermis is inside whole root
        multilabel[0][ring > 0] = 1.0

    return multilabel


def polygons_to_raw_semantic_mask(
    annotations: List[Dict],
    h: int,
    w: int,
) -> np.ndarray:
    """Convert YOLO annotations to a 7-class semantic mask using raw annotation classes.

    Paints filled polygons from largest to smallest so that inner regions
    overwrite outer regions, producing 7 mutually exclusive anatomical regions:

        0 = background
        1 = epidermis (whole root area not covered by other structures)
        2 = aerenchyma (holes in cortex)
        3 = endodermis ring (outer endo minus inner endo, via paint order)
        4 = vascular (inner endo area)
        5 = exodermis ring (outer exo minus inner exo, via paint order)
        6 = cortex (inner exo minus outer endo minus aerenchyma, via paint order)

    Paint order: whole root → outer exo → inner exo → outer endo → inner endo → aerenchyma.
    Each later paint overwrites earlier labels at overlapping pixels.

    Returns:
        (H, W) int32 semantic mask with labels 0-6.
    """
    sem = np.zeros((h, w), dtype=np.int32)

    # Collect polygons by annotation class
    class_polys = {i: [] for i in range(6)}
    for ann in annotations:
        cid = ann["class_id"]
        if cid in class_polys:
            class_polys[cid].append(ann["polygon"])

    # Paint order: largest region first, smaller regions overwrite
    # 1. Whole root (class 0) → label 1
    for poly in class_polys[0]:
        mask = polygon_to_mask(poly, h, w)
        sem[mask > 0] = 1

    # 2. Outer exo (class 4) → label 5
    for poly in class_polys[4]:
        mask = polygon_to_mask(poly, h, w)
        sem[mask > 0] = 5

    # 3. Inner exo (class 5) → label 6 (cortex region)
    for poly in class_polys[5]:
        mask = polygon_to_mask(poly, h, w)
        sem[mask > 0] = 6

    # 4. Outer endo (class 2) → label 3
    for poly in class_polys[2]:
        mask = polygon_to_mask(poly, h, w)
        sem[mask > 0] = 3

    # 5. Inner endo (class 3) → label 4 (vascular)
    for poly in class_polys[3]:
        mask = polygon_to_mask(poly, h, w)
        sem[mask > 0] = 4

    # 6. Aerenchyma (class 1) → label 2 (painted last, overwrites cortex)
    for poly in class_polys[1]:
        mask = polygon_to_mask(poly, h, w)
        sem[mask > 0] = 2

    return sem


def polygons_to_raw_binary_masks(
    annotations: List[Dict],
    h: int,
    w: int,
) -> Dict[int, np.ndarray]:
    """Convert YOLO annotations to per-class binary masks (one per raw annotation class).

    Returns dict mapping raw annotation class ID (0-5) to (H, W) uint8 binary mask.
    Each mask is the union of all polygons for that class.
    """
    masks = {}
    class_polys = {i: [] for i in range(6)}
    for ann in annotations:
        cid = ann["class_id"]
        if cid in class_polys:
            class_polys[cid].append(ann["polygon"])

    for cid in range(6):
        combined = np.zeros((h, w), dtype=np.uint8)
        for poly in class_polys[cid]:
            mask = polygon_to_mask(poly, h, w)
            combined = np.clip(combined + mask, 0, 1)
        masks[cid] = combined

    return masks


def polygons_to_raw_instance_masks(
    annotations: List[Dict],
    h: int,
    w: int,
) -> Dict[str, np.ndarray]:
    """Convert YOLO annotations to instance masks using raw annotation classes.

    No ring subtraction — each polygon becomes one instance with its raw class ID (0-5).
    This is the same representation used by YOLO and U-Net binary.

    Returns:
        Dict with:
            masks: (N, H, W) uint8 binary masks
            labels: (N,) int32 raw class IDs (0-5)
            boxes: (N, 4) float32 [x1, y1, x2, y2] bounding boxes
    """
    masks_list = []
    labels_list = []

    for ann in annotations:
        mask = polygon_to_mask(ann["polygon"], h, w)
        if mask.sum() > 0:
            masks_list.append(mask)
            labels_list.append(ann["class_id"])

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


def load_sample_annotations(
    sample: SampleRecord,
    img_h: int,
    img_w: int,
    num_classes: int = 5,
    raw_classes: bool = False,
) -> Dict[str, np.ndarray]:
    """Load and convert annotations for a sample to instance masks.

    Args:
        sample: SampleRecord with annotation_path.
        img_h, img_w: Image dimensions.
        num_classes: Number of target classes (4 or 5). Ignored when raw_classes=True.
        raw_classes: If True, return raw annotation classes (0-5) without
            ring subtraction. Each polygon becomes one instance.

    Returns:
        Dict with masks, labels, boxes arrays.
    """
    anns = parse_yolo_annotations(sample.annotation_path, img_w, img_h)
    if raw_classes:
        return polygons_to_raw_instance_masks(anns, img_h, img_w)
    return polygons_to_instance_masks(anns, img_h, img_w, num_classes=num_classes)
