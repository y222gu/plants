"""Segmentation evaluation metrics: mAP, IoU, Dice, pixel accuracy."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute IoU between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return float(intersection / union) if union > 0 else 0.0


def compute_dice(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute Dice coefficient between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    total = mask1.sum() + mask2.sum()
    return float(2 * intersection / total) if total > 0 else 0.0


def match_predictions_to_gt(
    pred_masks: np.ndarray,
    pred_labels: np.ndarray,
    pred_scores: np.ndarray,
    gt_masks: np.ndarray,
    gt_labels: np.ndarray,
    iou_threshold: float = 0.5,
) -> List[Dict]:
    """Match predicted instances to ground truth by IoU.

    Returns per-prediction match info: {pred_idx, gt_idx, iou, class_id, score, tp}.
    """
    matches = []
    gt_matched = set()

    # Sort predictions by score (highest first)
    order = np.argsort(-pred_scores)

    for pred_idx in order:
        pred_cls = pred_labels[pred_idx]
        best_iou = 0.0
        best_gt = -1

        for gt_idx in range(len(gt_masks)):
            if gt_idx in gt_matched:
                continue
            if gt_labels[gt_idx] != pred_cls:
                continue
            iou = compute_iou(pred_masks[pred_idx], gt_masks[gt_idx])
            if iou > best_iou:
                best_iou = iou
                best_gt = gt_idx

        tp = best_iou >= iou_threshold and best_gt >= 0
        if tp:
            gt_matched.add(best_gt)

        matches.append({
            "pred_idx": int(pred_idx),
            "gt_idx": int(best_gt),
            "iou": best_iou,
            "class_id": int(pred_cls),
            "score": float(pred_scores[pred_idx]),
            "tp": tp,
        })

    return matches


def compute_ap(
    matches: List[Dict],
    n_gt: int,
    iou_threshold: float = 0.5,
) -> float:
    """Compute Average Precision for a single class at a given IoU threshold."""
    if n_gt == 0:
        return 0.0

    # Sort by score descending
    sorted_matches = sorted(matches, key=lambda m: -m["score"])

    tp_cumsum = 0
    fp_cumsum = 0
    precisions = []
    recalls = []

    for m in sorted_matches:
        if m["tp"]:
            tp_cumsum += 1
        else:
            fp_cumsum += 1
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        recall = tp_cumsum / n_gt
        precisions.append(precision)
        recalls.append(recall)

    # COCO-style AP: 101-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        p_at_r = 0.0
        for p, r in zip(precisions, recalls):
            if r >= t:
                p_at_r = max(p_at_r, p)
        ap += p_at_r
    ap /= 101

    return ap


class SegmentationMetrics:
    """Accumulate predictions across a dataset and compute metrics.

    Supports grouping by species and microscope for breakdown reporting.
    """

    def __init__(self, num_classes: int = 4, class_names: Optional[Dict[int, str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or {i: f"class_{i}" for i in range(num_classes)}
        self._records: List[Dict] = []

    def add_sample(
        self,
        pred_masks: np.ndarray,
        pred_labels: np.ndarray,
        pred_scores: np.ndarray,
        gt_masks: np.ndarray,
        gt_labels: np.ndarray,
        species: str = "",
        microscope: str = "",
        sample_id: str = "",
    ):
        """Add a single sample's predictions and ground truth."""
        self._records.append({
            "pred_masks": pred_masks,
            "pred_labels": pred_labels,
            "pred_scores": pred_scores,
            "gt_masks": gt_masks,
            "gt_labels": gt_labels,
            "species": species,
            "microscope": microscope,
            "sample_id": sample_id,
        })

    def compute(self) -> Dict:
        """Compute all metrics overall and by group.

        Returns dict with keys: overall, per_species, per_microscope.
        Each contains: mAP_50, mAP_50_95, per_class_ap, per_class_iou,
        per_class_dice, pixel_accuracy.
        """
        results = {}

        # Overall
        results["overall"] = self._compute_group(self._records)

        # Per species
        species_groups = defaultdict(list)
        for r in self._records:
            if r["species"]:
                species_groups[r["species"]].append(r)
        results["per_species"] = {
            sp: self._compute_group(recs) for sp, recs in sorted(species_groups.items())
        }

        # Per microscope
        micro_groups = defaultdict(list)
        for r in self._records:
            if r["microscope"]:
                micro_groups[r["microscope"]].append(r)
        results["per_microscope"] = {
            m: self._compute_group(recs) for m, recs in sorted(micro_groups.items())
        }

        return results

    def _compute_group(self, records: List[Dict]) -> Dict:
        """Compute metrics for a group of samples."""
        # Instance-level metrics: mAP
        iou_thresholds = np.arange(0.5, 1.0, 0.05)
        per_class_ap = defaultdict(lambda: defaultdict(float))

        # Collect all matches per class per IoU threshold
        per_class_gt_counts = defaultdict(int)
        per_class_matches = defaultdict(list)

        for rec in records:
            gt_masks = rec["gt_masks"]
            gt_labels = rec["gt_labels"]
            pred_masks = rec["pred_masks"]
            pred_labels = rec["pred_labels"]
            pred_scores = rec["pred_scores"]

            # Count GT per class
            for lbl in gt_labels:
                per_class_gt_counts[int(lbl)] += 1

            if len(pred_masks) == 0:
                continue

            # Match at each IoU threshold
            for iou_t in iou_thresholds:
                matches = match_predictions_to_gt(
                    pred_masks, pred_labels, pred_scores,
                    gt_masks, gt_labels, iou_t,
                )
                for m in matches:
                    per_class_matches[(int(m["class_id"]), float(iou_t))].append(m)

        # Compute AP per class per threshold
        aps_50 = {}
        aps_50_95 = {}
        for cls_id in range(self.num_classes):
            aps_at_thresholds = []
            for iou_t in iou_thresholds:
                key = (cls_id, float(round(iou_t, 2)))
                class_matches = per_class_matches.get(key, [])
                n_gt = per_class_gt_counts.get(cls_id, 0)
                ap = compute_ap(class_matches, n_gt, iou_t)
                aps_at_thresholds.append(ap)
                if abs(iou_t - 0.5) < 1e-6:
                    aps_50[cls_id] = ap

            aps_50_95[cls_id] = float(np.mean(aps_at_thresholds)) if aps_at_thresholds else 0.0

        mAP_50 = float(np.mean(list(aps_50.values()))) if aps_50 else 0.0
        mAP_50_95 = float(np.mean(list(aps_50_95.values()))) if aps_50_95 else 0.0

        # Pixel-level metrics per class (aggregate over samples)
        per_class_iou = {}
        per_class_dice = {}
        total_correct = 0
        total_pixels = 0

        for cls_id in range(self.num_classes):
            cls_intersection = 0
            cls_union = 0
            cls_pred_sum = 0
            cls_gt_sum = 0

            for rec in records:
                # Aggregate all masks of this class into single binary mask
                gt_cls = self._class_mask(rec["gt_masks"], rec["gt_labels"], cls_id)
                pred_cls = self._class_mask(rec["pred_masks"], rec["pred_labels"], cls_id)

                cls_intersection += np.logical_and(gt_cls, pred_cls).sum()
                cls_union += np.logical_or(gt_cls, pred_cls).sum()
                cls_pred_sum += pred_cls.sum()
                cls_gt_sum += gt_cls.sum()

            per_class_iou[cls_id] = float(cls_intersection / cls_union) if cls_union > 0 else 0.0
            denom = cls_pred_sum + cls_gt_sum
            per_class_dice[cls_id] = float(2 * cls_intersection / denom) if denom > 0 else 0.0

        # Pixel accuracy (using semantic-like aggregation)
        for rec in records:
            h = rec["gt_masks"].shape[1] if len(rec["gt_masks"]) > 0 else 0
            w = rec["gt_masks"].shape[2] if len(rec["gt_masks"]) > 0 else 0
            if h == 0 or w == 0:
                continue
            gt_sem = np.zeros((h, w), dtype=np.int32)
            pred_sem = np.zeros((h, w), dtype=np.int32)
            # Priority: higher class overwrites lower
            for cls_id in range(self.num_classes):
                gt_cls = self._class_mask(rec["gt_masks"], rec["gt_labels"], cls_id)
                pred_cls = self._class_mask(rec["pred_masks"], rec["pred_labels"], cls_id)
                gt_sem[gt_cls > 0] = cls_id + 1
                pred_sem[pred_cls > 0] = cls_id + 1
            total_correct += (gt_sem == pred_sem).sum()
            total_pixels += h * w

        pixel_accuracy = float(total_correct / total_pixels) if total_pixels > 0 else 0.0

        return {
            "n_samples": len(records),
            "mAP_50": mAP_50,
            "mAP_50_95": mAP_50_95,
            "per_class_AP_50": {self.class_names.get(k, str(k)): v for k, v in aps_50.items()},
            "per_class_AP_50_95": {self.class_names.get(k, str(k)): v for k, v in aps_50_95.items()},
            "per_class_IoU": {self.class_names.get(k, str(k)): v for k, v in per_class_iou.items()},
            "per_class_Dice": {self.class_names.get(k, str(k)): v for k, v in per_class_dice.items()},
            "mean_IoU": float(np.mean(list(per_class_iou.values()))),
            "mean_Dice": float(np.mean(list(per_class_dice.values()))),
            "pixel_accuracy": pixel_accuracy,
        }

    @staticmethod
    def _class_mask(masks: np.ndarray, labels: np.ndarray, cls_id: int) -> np.ndarray:
        """Merge all instance masks of a given class into a single binary mask."""
        if len(masks) == 0:
            return np.zeros((1, 1), dtype=np.uint8)
        idx = np.where(labels == cls_id)[0]
        if len(idx) == 0:
            return np.zeros(masks.shape[1:], dtype=np.uint8)
        return np.clip(masks[idx].sum(axis=0), 0, 1).astype(np.uint8)

    def save(self, path: Path) -> None:
        """Save computed metrics to JSON."""
        results = self.compute()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(results, f, indent=2)

    def print_summary(self) -> None:
        """Print formatted metrics summary."""
        results = self.compute()
        overall = results["overall"]
        print(f"\n{'='*60}")
        print(f"OVERALL ({overall['n_samples']} samples)")
        print(f"  mAP@0.5:      {overall['mAP_50']:.4f}")
        print(f"  mAP@0.5:0.95: {overall['mAP_50_95']:.4f}")
        print(f"  Mean IoU:     {overall['mean_IoU']:.4f}")
        print(f"  Mean Dice:    {overall['mean_Dice']:.4f}")
        print(f"  Pixel Acc:    {overall['pixel_accuracy']:.4f}")
        print(f"\n  Per-class AP@0.5:")
        for cls, ap in overall["per_class_AP_50"].items():
            print(f"    {cls}: {ap:.4f}")
        print(f"\n  Per-class IoU:")
        for cls, iou in overall["per_class_IoU"].items():
            print(f"    {cls}: {iou:.4f}")

        for group_name in ["per_species", "per_microscope"]:
            if group_name in results:
                print(f"\n{'─'*60}")
                print(f"BY {group_name.upper().replace('PER_', '')}:")
                for key, metrics in results[group_name].items():
                    print(f"  {key} ({metrics['n_samples']} samples): "
                          f"mAP50={metrics['mAP_50']:.4f}, "
                          f"IoU={metrics['mean_IoU']:.4f}, "
                          f"Dice={metrics['mean_Dice']:.4f}")
