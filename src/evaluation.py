"""Unified evaluation: convert any model output to standard format and compute metrics."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from .annotation_utils import (
    parse_yolo_annotations,
    polygon_to_mask,
    polygons_to_instance_masks,
)
from .config import SampleRecord


@dataclass
class PredictionResult:
    """Standard prediction format for all models."""
    masks: np.ndarray        # (N, H, W) uint8 binary
    labels: np.ndarray       # (N,) int32 target class IDs
    scores: np.ndarray       # (N,) float32 confidence scores


def convert_yolo_predictions(
    pred_path: Path,
    img_h: int,
    img_w: int,
) -> PredictionResult:
    """Convert YOLO polygon predictions to standard format.

    YOLO predictions keep 4 original classes. We apply the same
    endodermis subtraction as for GT annotations.
    """
    anns = parse_yolo_annotations(pred_path, img_w, img_h)
    # Add default score=1.0 if not present, YOLO stores confidence in separate file
    result = polygons_to_instance_masks(anns, img_h, img_w)
    n = len(result["labels"])
    return PredictionResult(
        masks=result["masks"],
        labels=result["labels"],
        scores=np.ones(n, dtype=np.float32),
    )


def convert_detectron2_instances(instances, img_h: int, img_w: int) -> PredictionResult:
    """Convert Detectron2 Instances to standard format."""
    if len(instances) == 0:
        return PredictionResult(
            masks=np.zeros((0, img_h, img_w), dtype=np.uint8),
            labels=np.zeros(0, dtype=np.int32),
            scores=np.zeros(0, dtype=np.float32),
        )
    masks = instances.pred_masks.cpu().numpy().astype(np.uint8)
    labels = instances.pred_classes.cpu().numpy().astype(np.int32)
    scores = instances.scores.cpu().numpy().astype(np.float32)
    return PredictionResult(masks=masks, labels=labels, scores=scores)


def convert_semantic_to_instances(
    sem_mask: np.ndarray,
    score: float = 1.0,
) -> PredictionResult:
    """Convert semantic mask to instance predictions.

    Semantic mask: 0=bg, 1=whole root, 2=aerenchyma, 3=endodermis, 4=vascular.
    Connected components used for aerenchyma (class 1) which has multiple instances.
    """
    masks_list = []
    labels_list = []
    scores_list = []

    # Class 0 (Whole Root) — union of all non-background pixels
    # In semantic segmentation, sem_mask==1 is just the cortex portion;
    # the full root boundary includes aerenchyma, endo, and vascular too.
    wr = (sem_mask >= 1).astype(np.uint8)
    if wr.sum() > 0:
        masks_list.append(wr)
        labels_list.append(0)
        scores_list.append(score)

    # Extract endodermis and vascular masks
    endo = (sem_mask == 3).astype(np.uint8)
    vasc = (sem_mask == 4).astype(np.uint8)

    # Class 1 (Aerenchyma) — multiple instances (semantic label 2)
    # Post-processing: morphological cleanup + anatomical constraints
    aer = (sem_mask == 2).astype(np.uint8)
    if aer.sum() > 0:
        # Morphological closing (fill holes) + opening (remove fragments)
        kern_size = max(5, int(min(sem_mask.shape) * 0.005) | 1)  # ensure odd
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kern_size, kern_size))
        aer = cv2.morphologyEx(aer, cv2.MORPH_CLOSE, kernel, iterations=2)
        aer = cv2.morphologyEx(aer, cv2.MORPH_OPEN, kernel, iterations=1)

        # Clip: aerenchyma must be inside whole root, outside endodermis/vascular
        aer = aer & wr & (~endo.astype(bool)).astype(np.uint8) & (~vasc.astype(bool)).astype(np.uint8)

        # Minimum area: filter components smaller than 0.01% of image area
        min_area = max(50, int(sem_mask.shape[0] * sem_mask.shape[1] * 0.0001))
        n_components, component_map = cv2.connectedComponents(aer)
        for comp_id in range(1, n_components):
            mask = (component_map == comp_id).astype(np.uint8)
            if mask.sum() >= min_area:
                masks_list.append(mask)
                labels_list.append(1)
                scores_list.append(score)

    # Class 2 (Endodermis) — single ring instance (semantic label 3)
    # Morphological closing connects fragmented endodermis pixels into a ring,
    # then subtract the vascular interior to preserve the ring shape.
    if endo.sum() > 0:
        kern_size = max(7, int(min(sem_mask.shape) * 0.01) | 1)  # larger kernel for ring
        endo_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kern_size, kern_size))
        endo = cv2.morphologyEx(endo, cv2.MORPH_CLOSE, endo_kernel, iterations=3)
        # Remove small isolated fragments
        endo = cv2.morphologyEx(endo, cv2.MORPH_OPEN, endo_kernel, iterations=1)
        # Clip to inside root, outside vascular
        endo = endo & wr & (~vasc.astype(bool)).astype(np.uint8)
        if endo.sum() > 0:
            masks_list.append(endo)
            labels_list.append(2)
            scores_list.append(score)

    # Class 3 (Vascular) — single instance (semantic label 4)
    if vasc.sum() > 0:
        masks_list.append(vasc)
        labels_list.append(3)
        scores_list.append(score)

    if not masks_list:
        h, w = sem_mask.shape
        return PredictionResult(
            masks=np.zeros((0, h, w), dtype=np.uint8),
            labels=np.zeros(0, dtype=np.int32),
            scores=np.zeros(0, dtype=np.float32),
        )

    return PredictionResult(
        masks=np.stack(masks_list),
        labels=np.array(labels_list, dtype=np.int32),
        scores=np.array(scores_list, dtype=np.float32),
    )


def convert_multilabel_to_instances(
    ml_mask: np.ndarray,
    score: float = 1.0,
) -> PredictionResult:
    """Convert multilabel mask (4, H, W) to instance predictions.

    Channels: 0=whole_root, 1=aerenchyma, 2=endodermis, 3=vascular.
    Each channel is thresholded at 0.5 and converted to instances.
    Aerenchyma uses connected components; others are single instances.
    """
    masks_list = []
    labels_list = []
    scores_list = []

    h, w = ml_mask.shape[1], ml_mask.shape[2]

    # Channel 0: Whole Root (single instance)
    wr = (ml_mask[0] > 0.5).astype(np.uint8)
    if wr.sum() > 0:
        masks_list.append(wr)
        labels_list.append(0)
        scores_list.append(score)

    # Channel 2: Endodermis (single ring instance)
    endo = (ml_mask[2] > 0.5).astype(np.uint8)

    # Channel 3: Vascular (single instance)
    vasc = (ml_mask[3] > 0.5).astype(np.uint8)

    # Channel 1: Aerenchyma (multiple instances via connected components)
    aer = (ml_mask[1] > 0.5).astype(np.uint8)
    if aer.sum() > 0:
        # Morphological cleanup
        kern_size = max(5, int(min(h, w) * 0.005) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kern_size, kern_size))
        aer = cv2.morphologyEx(aer, cv2.MORPH_CLOSE, kernel, iterations=2)
        aer = cv2.morphologyEx(aer, cv2.MORPH_OPEN, kernel, iterations=1)

        # Clip: aerenchyma must be inside whole root, outside endo/vascular
        if wr.sum() > 0:
            aer = aer & wr & (~endo.astype(bool)).astype(np.uint8) & (~vasc.astype(bool)).astype(np.uint8)

        # Connected components with minimum area filter
        min_area = max(50, int(h * w * 0.0001))
        n_components, component_map = cv2.connectedComponents(aer)
        for comp_id in range(1, n_components):
            mask = (component_map == comp_id).astype(np.uint8)
            if mask.sum() >= min_area:
                masks_list.append(mask)
                labels_list.append(1)
                scores_list.append(score)

    # Endodermis instance
    if endo.sum() > 0:
        # Morphological cleanup for ring
        kern_size = max(7, int(min(h, w) * 0.01) | 1)
        endo_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kern_size, kern_size))
        endo = cv2.morphologyEx(endo, cv2.MORPH_CLOSE, endo_kernel, iterations=3)
        endo = cv2.morphologyEx(endo, cv2.MORPH_OPEN, endo_kernel, iterations=1)
        # Clip to inside root, outside vascular
        if wr.sum() > 0:
            endo = endo & wr & (~vasc.astype(bool)).astype(np.uint8)
        if endo.sum() > 0:
            masks_list.append(endo)
            labels_list.append(2)
            scores_list.append(score)

    # Vascular instance
    if vasc.sum() > 0:
        masks_list.append(vasc)
        labels_list.append(3)
        scores_list.append(score)

    if not masks_list:
        return PredictionResult(
            masks=np.zeros((0, h, w), dtype=np.uint8),
            labels=np.zeros(0, dtype=np.int32),
            scores=np.zeros(0, dtype=np.float32),
        )

    return PredictionResult(
        masks=np.stack(masks_list),
        labels=np.array(labels_list, dtype=np.int32),
        scores=np.array(scores_list, dtype=np.float32),
    )


