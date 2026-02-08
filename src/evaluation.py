"""Unified evaluation: convert any model output to standard format and compute metrics."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from .annotation_utils import (
    load_sample_annotations,
    parse_yolo_annotations,
    polygon_to_mask,
    polygons_to_instance_masks,
)
from .config import NUM_CLASSES, TARGET_CLASSES, SampleRecord
from .metrics import SegmentationMetrics
from .preprocessing import load_sample_normalized


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

    # Class 0 (Whole Root) — single instance (semantic label 1)
    wr = (sem_mask == 1).astype(np.uint8)
    if wr.sum() > 0:
        masks_list.append(wr)
        labels_list.append(0)
        scores_list.append(score)

    # Class 1 (Aerenchyma) — multiple instances (semantic label 2)
    aer = (sem_mask == 2).astype(np.uint8)
    if aer.sum() > 0:
        n_components, component_map = cv2.connectedComponents(aer)
        for comp_id in range(1, n_components):
            mask = (component_map == comp_id).astype(np.uint8)
            if mask.sum() > 10:  # filter tiny noise
                masks_list.append(mask)
                labels_list.append(1)
                scores_list.append(score)

    # Class 2 (Endodermis) — single instance (semantic label 3)
    endo = (sem_mask == 3).astype(np.uint8)
    if endo.sum() > 0:
        masks_list.append(endo)
        labels_list.append(2)
        scores_list.append(score)

    # Class 3 (Vascular) — single instance (semantic label 4)
    vasc = (sem_mask == 4).astype(np.uint8)
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


def evaluate_samples(
    predictions: Dict[str, PredictionResult],
    samples: List[SampleRecord],
) -> SegmentationMetrics:
    """Evaluate predictions against GT for a list of samples.

    Args:
        predictions: Dict mapping sample.uid → PredictionResult.
        samples: List of SampleRecord to evaluate.

    Returns:
        SegmentationMetrics with accumulated results.
    """
    metrics = SegmentationMetrics(
        num_classes=NUM_CLASSES,
        class_names=TARGET_CLASSES,
    )

    for sample in samples:
        if sample.uid not in predictions:
            continue

        pred = predictions[sample.uid]
        img = load_sample_normalized(sample)
        h, w = img.shape[:2]
        gt = load_sample_annotations(sample, h, w)

        metrics.add_sample(
            pred_masks=pred.masks,
            pred_labels=pred.labels,
            pred_scores=pred.scores,
            gt_masks=gt["masks"],
            gt_labels=gt["labels"],
            species=sample.species,
            microscope=sample.microscope,
            sample_id=sample.uid,
        )

    return metrics
