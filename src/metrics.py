"""Segmentation evaluation metrics: mAP, IoU, Dice, pixel accuracy."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np


def _downscale_masks(masks: np.ndarray, max_dim: int = 256) -> np.ndarray:
    """Downscale masks for faster IoU computation."""
    if len(masks) == 0:
        return masks
    h, w = masks.shape[1], masks.shape[2]
    if max(h, w) <= max_dim:
        return masks
    scale = max_dim / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    out = np.zeros((len(masks), new_h, new_w), dtype=np.uint8)
    for i in range(len(masks)):
        out[i] = cv2.resize(masks[i], (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return out


def _compute_iou_matrix(
    pred_masks: np.ndarray,
    gt_masks: np.ndarray,
) -> np.ndarray:
    """Compute pairwise IoU matrix between predictions and GT.

    Args:
        pred_masks: (N_pred, H, W) uint8
        gt_masks: (N_gt, H, W) uint8

    Returns:
        (N_pred, N_gt) float32 IoU matrix.
    """
    n_pred = len(pred_masks)
    n_gt = len(gt_masks)
    if n_pred == 0 or n_gt == 0:
        return np.zeros((n_pred, n_gt), dtype=np.float32)

    # Flatten for vectorized computation
    pred_flat = pred_masks.reshape(n_pred, -1).astype(bool)
    gt_flat = gt_masks.reshape(n_gt, -1).astype(bool)

    # intersection[i,j] = sum(pred_i & gt_j)
    intersection = pred_flat.astype(np.float32) @ gt_flat.astype(np.float32).T
    pred_areas = pred_flat.sum(axis=1, keepdims=True).astype(np.float32)  # (N_pred, 1)
    gt_areas = gt_flat.sum(axis=1, keepdims=True).astype(np.float32).T    # (1, N_gt)
    union = pred_areas + gt_areas - intersection
    iou = np.where(union > 0, intersection / union, 0.0)
    return iou.astype(np.float32)


def _match_at_threshold(
    iou_matrix: np.ndarray,
    pred_labels: np.ndarray,
    pred_scores: np.ndarray,
    gt_labels: np.ndarray,
    iou_threshold: float,
) -> List[Dict]:
    """Match predictions to GT using a precomputed IoU matrix at a given threshold."""
    matches = []
    gt_matched = set()
    order = np.argsort(-pred_scores)

    for pred_idx in order:
        pred_cls = pred_labels[pred_idx]
        best_iou = 0.0
        best_gt = -1

        for gt_idx in range(len(gt_labels)):
            if gt_idx in gt_matched:
                continue
            if gt_labels[gt_idx] != pred_cls:
                continue
            iou_val = iou_matrix[pred_idx, gt_idx]
            if iou_val > best_iou:
                best_iou = iou_val
                best_gt = gt_idx

        tp = best_iou >= iou_threshold and best_gt >= 0
        if tp:
            gt_matched.add(best_gt)

        matches.append({
            "class_id": int(pred_cls),
            "score": float(pred_scores[pred_idx]),
            "tp": tp,
        })

    return matches


def compute_ap(matches: List[Dict], n_gt: int) -> float:
    """Compute Average Precision for a single class (101-point interpolation)."""
    if n_gt == 0:
        return 0.0

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
        precisions.append(tp_cumsum / (tp_cumsum + fp_cumsum))
        recalls.append(tp_cumsum / n_gt)

    if not recalls:
        return 0.0

    # Vectorized 101-point interpolation
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        mask = recalls >= t
        ap += precisions[mask].max() if mask.any() else 0.0
    return float(ap / 101)


class SegmentationMetrics:
    """Accumulate predictions across a dataset and compute metrics.

    Supports grouping by species and microscope for breakdown reporting.
    Precomputes per-sample match results at add_sample time to avoid
    redundant computation during grouped reporting.
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
        """Add a single sample's predictions and ground truth.

        Precomputes IoU matrix and per-threshold matches immediately.
        """
        iou_thresholds = np.round(np.arange(0.5, 1.0, 0.05), 2)

        # Precompute per-threshold matches using downscaled masks
        per_threshold_matches = {}
        if len(pred_masks) > 0 and len(gt_masks) > 0:
            small_pred = _downscale_masks(pred_masks)
            small_gt = _downscale_masks(gt_masks)
            iou_matrix = _compute_iou_matrix(small_pred, small_gt)

            for iou_t in iou_thresholds:
                per_threshold_matches[round(float(iou_t), 2)] = _match_at_threshold(
                    iou_matrix, pred_labels, pred_scores, gt_labels, iou_t,
                )

        # Determine spatial dimensions from whichever mask set is non-empty
        if len(gt_masks) > 0:
            spatial_shape = gt_masks.shape[1:]
        elif len(pred_masks) > 0:
            spatial_shape = pred_masks.shape[1:]
        else:
            spatial_shape = (1, 1)

        # Precompute per-class pixel stats (at original resolution)
        pixel_stats = {}
        for cls_id in range(self.num_classes):
            gt_cls = self._class_mask(gt_masks, gt_labels, cls_id, shape=spatial_shape)
            pred_cls = self._class_mask(pred_masks, pred_labels, cls_id, shape=spatial_shape)
            pixel_stats[cls_id] = {
                "intersection": int(np.logical_and(gt_cls, pred_cls).sum()),
                "union": int(np.logical_or(gt_cls, pred_cls).sum()),
                "pred_sum": int(pred_cls.sum()),
                "gt_sum": int(gt_cls.sum()),
            }

        # Precompute pixel accuracy
        if len(gt_masks) > 0:
            h, w = gt_masks.shape[1], gt_masks.shape[2]
            gt_sem = np.zeros((h, w), dtype=np.int32)
            pred_sem = np.zeros((h, w), dtype=np.int32)
            for cls_id in range(self.num_classes):
                gt_cls = self._class_mask(gt_masks, gt_labels, cls_id, shape=(h, w))
                pred_cls = self._class_mask(pred_masks, pred_labels, cls_id, shape=(h, w))
                gt_sem[gt_cls > 0] = cls_id + 1
                pred_sem[pred_cls > 0] = cls_id + 1
            pa_correct = int((gt_sem == pred_sem).sum())
            pa_total = h * w
        else:
            pa_correct = 0
            pa_total = 0

        # GT class counts
        gt_class_counts = defaultdict(int)
        for lbl in gt_labels:
            gt_class_counts[int(lbl)] += 1

        self._records.append({
            "per_threshold_matches": per_threshold_matches,
            "pixel_stats": pixel_stats,
            "pa_correct": pa_correct,
            "pa_total": pa_total,
            "gt_class_counts": dict(gt_class_counts),
            "species": species,
            "microscope": microscope,
            "sample_id": sample_id,
        })

    def compute(self) -> Dict:
        """Compute all metrics overall and by group."""
        results = {}
        results["overall"] = self._compute_group(self._records)

        species_groups = defaultdict(list)
        micro_groups = defaultdict(list)
        combo_groups = defaultdict(list)
        for r in self._records:
            if r["species"]:
                species_groups[r["species"]].append(r)
            if r["microscope"]:
                micro_groups[r["microscope"]].append(r)
            if r["species"] and r["microscope"]:
                combo_groups[f"{r['species']}/{r['microscope']}"].append(r)

        results["per_species"] = {
            sp: self._compute_group(recs) for sp, recs in sorted(species_groups.items())
        }
        results["per_microscope"] = {
            m: self._compute_group(recs) for m, recs in sorted(micro_groups.items())
        }
        results["per_species_microscope"] = {
            k: self._compute_group(recs) for k, recs in sorted(combo_groups.items())
        }
        return results

    def _compute_group(self, records: List[Dict]) -> Dict:
        """Compute metrics from precomputed per-sample results."""
        iou_thresholds = np.round(np.arange(0.5, 1.0, 0.05), 2)

        # Aggregate GT counts and matches
        per_class_gt_counts = defaultdict(int)
        per_class_matches = defaultdict(list)  # (cls_id, iou_t) -> matches

        for rec in records:
            for cls_id_str, cnt in rec["gt_class_counts"].items():
                per_class_gt_counts[int(cls_id_str)] += cnt

            for iou_t_key, matches in rec["per_threshold_matches"].items():
                for m in matches:
                    per_class_matches[(m["class_id"], iou_t_key)].append(m)

        # Compute AP
        aps_50 = {}
        aps_50_95 = {}
        for cls_id in range(self.num_classes):
            aps_at_thresholds = []
            n_gt = per_class_gt_counts.get(cls_id, 0)
            for iou_t in iou_thresholds:
                key = (cls_id, round(float(iou_t), 2))
                class_matches = per_class_matches.get(key, [])
                ap = compute_ap(class_matches, n_gt)
                aps_at_thresholds.append(ap)
                if abs(iou_t - 0.5) < 1e-6:
                    aps_50[cls_id] = ap
            aps_50_95[cls_id] = float(np.mean(aps_at_thresholds)) if aps_at_thresholds else 0.0

        mAP_50 = float(np.mean(list(aps_50.values()))) if aps_50 else 0.0
        mAP_50_95 = float(np.mean(list(aps_50_95.values()))) if aps_50_95 else 0.0

        # Aggregate pixel stats
        per_class_iou = {}
        per_class_dice = {}
        for cls_id in range(self.num_classes):
            total_inter = sum(r["pixel_stats"][cls_id]["intersection"] for r in records)
            total_union = sum(r["pixel_stats"][cls_id]["union"] for r in records)
            total_pred = sum(r["pixel_stats"][cls_id]["pred_sum"] for r in records)
            total_gt = sum(r["pixel_stats"][cls_id]["gt_sum"] for r in records)
            per_class_iou[cls_id] = float(total_inter / total_union) if total_union > 0 else 0.0
            denom = total_pred + total_gt
            per_class_dice[cls_id] = float(2 * total_inter / denom) if denom > 0 else 0.0

        total_correct = sum(r["pa_correct"] for r in records)
        total_pixels = sum(r["pa_total"] for r in records)
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
    @staticmethod
    def _class_mask(masks: np.ndarray, labels: np.ndarray, cls_id: int,
                    shape: tuple = None) -> np.ndarray:
        """Merge all instance masks of a given class into a single binary mask."""
        if len(masks) == 0:
            if shape is not None:
                return np.zeros(shape, dtype=np.uint8)
            return np.zeros((1, 1), dtype=np.uint8)
        idx = np.where(labels == cls_id)[0]
        if len(idx) == 0:
            return np.zeros(masks.shape[1:], dtype=np.uint8)
        return np.clip(masks[idx].sum(axis=0), 0, 1).astype(np.uint8)

    def save(self, path: Path, _cached_results: Optional[Dict] = None) -> None:
        """Save computed metrics to JSON."""
        results = _cached_results or self.compute()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(results, f, indent=2)

    def print_summary(self) -> Dict:
        """Print formatted metrics summary. Returns computed results to avoid recomputation."""
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

        for group_name in ["per_species", "per_microscope", "per_species_microscope"]:
            if group_name in results:
                print(f"\n{'─'*60}")
                print(f"BY {group_name.upper().replace('PER_', '')}:")
                for key, m in results[group_name].items():
                    print(f"  {key} ({m['n_samples']} samples): "
                          f"mAP50={m['mAP_50']:.4f}, "
                          f"IoU={m['mean_IoU']:.4f}, "
                          f"Dice={m['mean_Dice']:.4f}")
        return results
