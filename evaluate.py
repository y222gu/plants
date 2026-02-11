"""Unified evaluation entry point.

Run any trained model on a test set and produce metrics CSV/JSON.

Usage:
    python evaluate.py --model yolo --checkpoint output/runs/yolo/run/weights/best.pt
    python evaluate.py --model unet --checkpoint output/runs/unet/run/checkpoints/best.ckpt
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.annotation_utils import load_sample_annotations
from src.config import (
    CLASS_COLORS_RGB,
    DEFAULT_IMG_SIZE,
    NUM_CLASSES,
    OUTPUT_DIR,
    TARGET_CLASSES,
)
from src.dataset import SampleRegistry
from src.evaluation import (
    PredictionResult,
    convert_multilabel_to_instances,
    convert_semantic_to_instances,
)
from src.postprocessing import PostProcessor, STEPS
from src.metrics import SegmentationMetrics
from src.preprocessing import load_sample_normalized, to_uint8
from src.splits import get_split, print_split_summary


def draw_masks_overlay(
    img_uint8: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Draw instance masks on an image with semi-transparent fill + contours.

    Draws largest masks first so smaller ones appear on top.
    """
    result = img_uint8.copy()
    if len(masks) == 0:
        return result

    # Sort by mask area (largest first)
    areas = [m.sum() for m in masks]
    order = np.argsort(areas)[::-1]

    overlay = result.copy()
    for idx in order:
        mask = masks[idx]
        cls_id = int(labels[idx])
        color = CLASS_COLORS_RGB.get(cls_id, (128, 128, 128))
        overlay[mask > 0] = color

    result = cv2.addWeighted(result, 1 - alpha, overlay, alpha, 0)

    # Draw contours on top
    for idx in order:
        mask = masks[idx]
        cls_id = int(labels[idx])
        color = CLASS_COLORS_RGB.get(cls_id, (128, 128, 128))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, color, 2)

    return result


def _compute_sample_metrics(
    pred_masks: np.ndarray,
    pred_labels: np.ndarray,
    gt_masks: np.ndarray,
    gt_labels: np.ndarray,
) -> dict:
    """Compute per-class IoU and Dice for a single sample."""
    results = {}
    for cls_id, cls_name in TARGET_CLASSES.items():
        # Merge all instance masks for this class into binary mask
        gt_idx = np.where(gt_labels == cls_id)[0]
        pred_idx = np.where(pred_labels == cls_id)[0]
        if len(gt_masks) > 0 and len(gt_idx) > 0:
            gt_cls = np.clip(gt_masks[gt_idx].sum(axis=0), 0, 1).astype(bool)
        else:
            gt_cls = np.zeros(gt_masks.shape[1:] if len(gt_masks) > 0
                              else pred_masks.shape[1:], dtype=bool)
        if len(pred_masks) > 0 and len(pred_idx) > 0:
            pred_cls = np.clip(pred_masks[pred_idx].sum(axis=0), 0, 1).astype(bool)
        else:
            pred_cls = np.zeros_like(gt_cls, dtype=bool)

        inter = np.logical_and(gt_cls, pred_cls).sum()
        union = np.logical_or(gt_cls, pred_cls).sum()
        iou = float(inter / union) if union > 0 else float('nan')
        denom = gt_cls.sum() + pred_cls.sum()
        dice = float(2 * inter / denom) if denom > 0 else float('nan')

        results[cls_id] = {"name": cls_name, "iou": iou, "dice": dice}
    return results


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    """Load a standard TrueType font, falling back gracefully."""
    for name in ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _pil_text(
    img: np.ndarray,
    text: str,
    xy: tuple,
    font: ImageFont.FreeTypeFont,
    fill: tuple,
    outline: tuple = None,
) -> np.ndarray:
    """Draw text on a numpy RGB image using PIL for clean font rendering."""
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    if outline:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx != 0 or dy != 0:
                    draw.text((xy[0] + dx, xy[1] + dy), text, font=font, fill=outline)
    draw.text(xy, text, font=font, fill=fill)
    return np.array(pil_img)


def save_visualizations(
    samples,
    predictions: dict,
    vis_dir: Path,
    max_dim: int = 800,
):
    """Save side-by-side GT vs prediction overlay images with per-class metrics."""
    vis_dir.mkdir(parents=True, exist_ok=True)

    title_font = _load_font(22)
    metric_font = _load_font(15)
    legend_font = _load_font(14)

    for sample in tqdm(samples, desc="Saving visualizations"):
        if sample.uid not in predictions:
            continue
        pred = predictions[sample.uid]

        # Load image
        img = load_sample_normalized(sample)
        h, w = img.shape[:2]
        img_uint8 = to_uint8(img)

        # Load GT
        gt = load_sample_annotations(sample, h, w)

        # Compute per-class metrics at original resolution
        sample_metrics = _compute_sample_metrics(
            pred.masks, pred.labels, gt["masks"], gt["labels"],
        )

        # Downscale for reasonable file sizes
        scale = min(1.0, max_dim / max(h, w))
        if scale < 1.0:
            new_h, new_w = int(h * scale), int(w * scale)
            img_small = cv2.resize(img_uint8, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            def resize_masks(masks):
                if len(masks) == 0:
                    return masks
                out = np.zeros((len(masks), new_h, new_w), dtype=np.uint8)
                for i in range(len(masks)):
                    out[i] = cv2.resize(masks[i], (new_w, new_h),
                                        interpolation=cv2.INTER_NEAREST)
                return out

            gt_masks = resize_masks(gt["masks"])
            pred_masks = resize_masks(pred.masks)
        else:
            img_small = img_uint8
            new_h, new_w = h, w
            gt_masks = gt["masks"]
            pred_masks = pred.masks

        # Draw overlays
        orig_vis = img_small.copy()
        gt_vis = draw_masks_overlay(img_small, gt_masks, gt["labels"])
        pred_vis = draw_masks_overlay(img_small, pred_masks, pred.labels)

        # Add titles using PIL
        orig_vis = _pil_text(orig_vis, "Original", (10, 8), title_font,
                             fill=(255, 255, 255), outline=(0, 0, 0))
        gt_vis = _pil_text(gt_vis, "Ground Truth", (10, 8), title_font,
                           fill=(255, 255, 255), outline=(0, 0, 0))
        pred_vis = _pil_text(pred_vis, "Prediction", (10, 8), title_font,
                             fill=(255, 255, 255), outline=(0, 0, 0))

        # Add per-class metrics on the prediction panel
        y_offset = 40
        for cls_id in sorted(sample_metrics.keys()):
            m = sample_metrics[cls_id]
            color = CLASS_COLORS_RGB.get(cls_id, (255, 255, 255))
            iou_str = f"{m['iou']:.2f}" if not np.isnan(m['iou']) else "N/A"
            dice_str = f"{m['dice']:.2f}" if not np.isnan(m['dice']) else "N/A"
            text = f"{m['name']}: IoU={iou_str}  Dice={dice_str}"
            pred_vis = _pil_text(pred_vis, text, (10, y_offset), metric_font,
                                 fill=color, outline=(0, 0, 0))
            y_offset += 20

        # Side-by-side: Original | Ground Truth | Prediction
        divider = np.full((new_h, 3, 3), 255, dtype=np.uint8)
        combined = np.concatenate(
            [orig_vis, divider, gt_vis, divider, pred_vis], axis=1
        )

        # Add legend bar at bottom using PIL
        legend_h = 32
        legend = np.zeros((legend_h, combined.shape[1], 3), dtype=np.uint8)
        pil_legend = Image.fromarray(legend)
        draw = ImageDraw.Draw(pil_legend)
        x_offset = 10
        for cls_id, cls_name in TARGET_CLASSES.items():
            color = CLASS_COLORS_RGB.get(cls_id, (128, 128, 128))
            draw.rectangle([x_offset, 6, x_offset + 20, 26], fill=color)
            draw.text((x_offset + 25, 8), cls_name, font=legend_font, fill=(255, 255, 255))
            bbox = legend_font.getbbox(cls_name)
            text_w = bbox[2] - bbox[0]
            x_offset += 25 + text_w + 20
        legend = np.array(pil_legend)

        combined = np.concatenate([combined, legend], axis=0)

        # Save
        out_path = vis_dir / f"{sample.uid}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

    print(f"Saved {len(samples)} visualizations to {vis_dir}")


def _setup_pub_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,  # editable text in PDF
        "ps.fonttype": 42,
    })


# Color-blind-friendly palette (adapted from Wong 2011, Nature Methods)
_PUB_CLASS_COLORS = {
    "Whole Root":   "#0072B2",  # blue
    "Aerenchyma":   "#E69F00",  # orange
    "Endodermis":   "#009E73",  # green
    "Vascular":     "#CC79A7",  # pink
}
_PUB_HATCHES = {
    "Whole Root":   "",
    "Aerenchyma":   "//",
    "Endodermis":   "\\\\",
    "Vascular":     "xx",
}


def save_metric_comparison_plots(results: dict, out_dir: Path, model_tag: str,
                                  per_sample_rows: list = None):
    """Save publication-quality box plots comparing per-sample metrics.

    Box plots use 5th/95th percentile whiskers and white diagonal hatching.
    Falls back to loading per-sample CSV if per_sample_rows not provided.
    """
    _setup_pub_style()
    plt.rcParams['hatch.color'] = 'white'
    plt.rcParams['hatch.linewidth'] = 1.0

    class_names = list(results["overall"]["per_class_IoU"].keys())

    # Load per-sample CSV if not provided directly (e.g. --plot-only mode)
    if per_sample_rows is None:
        csv_path = out_dir / f"{model_tag}_per_sample.csv"
        if csv_path.exists():
            import csv as csv_mod
            with open(csv_path) as f:
                reader = csv_mod.DictReader(f)
                per_sample_rows = []
                for row in reader:
                    parsed = {}
                    for k, v in row.items():
                        if k in ("sample_id", "species", "microscope", "experiment"):
                            parsed[k] = v
                        else:
                            try:
                                parsed[k] = float(v) if v != "" else float('nan')
                            except ValueError:
                                parsed[k] = float('nan')
                    per_sample_rows.append(parsed)

    if not per_sample_rows:
        print("Warning: no per-sample data available for box plots, skipping")
        return

    # ── helpers ──

    def _get_vals(rows, key):
        """Extract finite float values for a column from sample rows."""
        out = []
        for r in rows:
            v = r.get(key)
            if v is None or v == "":
                continue
            try:
                fv = float(v)
                if not np.isnan(fv):
                    out.append(fv)
            except (ValueError, TypeError):
                pass
        return out

    def _build_groups(rows, group_key):
        """Return ordered dict: 'Overall' followed by sorted sub-groups."""
        groups = {"Overall": rows}
        if group_key:
            sub = {}
            for r in rows:
                g = r.get(group_key, "")
                if g:
                    sub.setdefault(g, []).append(r)
            for k in sorted(sub):
                groups[k] = sub[k]
        return groups

    def _draw_boxes(ax, groups, metric_keys, colors, ylabel):
        """Draw side-by-side box plots (5th/95th whiskers, white diagonal hatch)."""
        group_names = list(groups.keys())
        n_g = len(group_names)
        n_m = len(metric_keys)
        w = 0.72 / n_m

        data, pos, cols = [], [], []
        for gi, gn in enumerate(group_names):
            rows = groups[gn]
            for mi, mk in enumerate(metric_keys):
                vals = _get_vals(rows, mk)
                data.append(vals if vals else [0.0])
                pos.append(gi + (mi - n_m / 2 + 0.5) * w)
                cols.append(colors[mi])

        bp = ax.boxplot(
            data, positions=pos, widths=w * 0.80,
            patch_artist=True, whis=[5, 95], showfliers=False,
            medianprops=dict(color="black", linewidth=1.0),
            whiskerprops=dict(color="#444444", linewidth=0.6),
            capprops=dict(color="#444444", linewidth=0.6),
        )
        for patch, c in zip(bp["boxes"], cols):
            patch.set_facecolor(c)
            patch.set_edgecolor("#555555")
            patch.set_hatch("///")
            patch.set_linewidth(0.5)

        ax.set_xticks(range(n_g))
        ax.set_xticklabels([f"{g}\n($n$={len(groups[g])})" for g in group_names])
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.08)
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="y", which="major", linewidth=0.3, alpha=0.5)

    # ── prepare data ──

    # Compute per-sample mean_Dice (mean_IoU already present in rows)
    for row in per_sample_rows:
        dice_vals = []
        for cls in class_names:
            v = row.get(f"{cls}_Dice")
            if v is not None and v != "":
                try:
                    fv = float(v)
                    if not np.isnan(fv):
                        dice_vals.append(fv)
                except (ValueError, TypeError):
                    pass
        row["mean_Dice"] = round(np.mean(dice_vals), 4) if dice_vals else 0.0

    species_groups = _build_groups(per_sample_rows, "species")
    micro_groups = _build_groups(per_sample_rows, "microscope")

    cls_iou_keys = [f"{c}_IoU" for c in class_names]
    cls_dice_keys = [f"{c}_Dice" for c in class_names]
    cls_colors = [_PUB_CLASS_COLORS.get(c, "#999999") for c in class_names]

    from matplotlib.patches import Patch

    # ── Figure 1: Per-class IoU & Dice by Species and Microscope (2x2) ──
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))
    _draw_boxes(axes[0, 0], species_groups, cls_iou_keys, cls_colors, "IoU")
    _draw_boxes(axes[0, 1], micro_groups, cls_iou_keys, cls_colors, "IoU")
    _draw_boxes(axes[1, 0], species_groups, cls_dice_keys, cls_colors, "Dice")
    _draw_boxes(axes[1, 1], micro_groups, cls_dice_keys, cls_colors, "Dice")

    axes[0, 0].set_title("By Species")
    axes[0, 1].set_title("By Microscope")
    axes[0, 0].annotate("IoU", xy=(-0.22, 0.5), xycoords="axes fraction",
                         fontsize=10, fontweight="bold", ha="center", va="center",
                         rotation=90)
    axes[1, 0].annotate("Dice", xy=(-0.22, 0.5), xycoords="axes fraction",
                         fontsize=10, fontweight="bold", ha="center", va="center",
                         rotation=90)
    handles = [Patch(facecolor=_PUB_CLASS_COLORS.get(c, "#999999"),
                     edgecolor="#555555", hatch="///", label=c) for c in class_names]
    fig.legend(handles=handles, loc="lower center", ncol=len(class_names),
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0.03, 0.05, 1, 1])
    fig.subplots_adjust(hspace=0.30, wspace=0.28)

    for fmt in ("png", "pdf"):
        fig.savefig(out_dir / f"{model_tag}_per_class_comparison.{fmt}")
    plt.close(fig)
    print(f"Per-class comparison saved to {out_dir / model_tag}_per_class_comparison.[png|pdf]")

    # ── Figure 2: Summary (mean IoU, mean Dice) by Species and Microscope (1x2) ──
    sum_keys = ["mean_IoU", "mean_Dice"]
    sum_labels = ["Mean IoU", "Mean Dice"]
    sum_colors = ["#009E73", "#CC79A7"]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    _draw_boxes(axes[0], species_groups, sum_keys, sum_colors, "Score")
    _draw_boxes(axes[1], micro_groups, sum_keys, sum_colors, "Score")
    axes[0].set_title("By Species")
    axes[1].set_title("By Microscope")
    handles = [Patch(facecolor=c, edgecolor="#555555", hatch="///", label=l)
               for c, l in zip(sum_colors, sum_labels)]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.subplots_adjust(wspace=0.28)

    for fmt in ("png", "pdf"):
        fig.savefig(out_dir / f"{model_tag}_summary_comparison.{fmt}")
    plt.close(fig)
    print(f"Summary comparison saved to {out_dir / model_tag}_summary_comparison.[png|pdf]")

    # ── Figure 3: Per-class IoU & Dice by Species+Microscope combo (2x1) ──
    combo_sub = {}
    for r in per_sample_rows:
        sp, mic = r.get("species", ""), r.get("microscope", "")
        if sp and mic:
            combo_sub.setdefault(f"{sp}/{mic}", []).append(r)
    if combo_sub:
        combo_groups = {"Overall": per_sample_rows}
        for k in sorted(combo_sub):
            combo_groups[k] = combo_sub[k]

        fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.4))
        _draw_boxes(axes[0], combo_groups, cls_iou_keys, cls_colors, "IoU")
        _draw_boxes(axes[1], combo_groups, cls_dice_keys, cls_colors, "Dice")
        axes[0].set_title("Per-class IoU by Species / Microscope")
        axes[1].set_title("Per-class Dice by Species / Microscope")
        for ax in axes:
            ax.tick_params(axis="x", rotation=25)
        handles = [Patch(facecolor=_PUB_CLASS_COLORS.get(c, "#999999"),
                         edgecolor="#555555", hatch="///", label=c) for c in class_names]
        fig.legend(handles=handles, loc="lower center", ncol=len(class_names),
                   frameon=False, bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout(rect=[0, 0.04, 1, 1])
        fig.subplots_adjust(hspace=0.45)

        for fmt in ("png", "pdf"):
            fig.savefig(out_dir / f"{model_tag}_species_microscope_comparison.{fmt}")
        plt.close(fig)
        print(f"Species+Microscope comparison saved to "
              f"{out_dir / model_tag}_species_microscope_comparison.[png|pdf]")


def predict_yolo(checkpoint: str, samples, img_size: int) -> dict:
    """Run YOLO inference and return predictions."""
    from ultralytics import YOLO

    model = YOLO(checkpoint)
    predictions = {}

    for sample in tqdm(samples, desc="YOLO inference"):
        img = load_sample_normalized(sample)
        img_uint8 = to_uint8(img)
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)

        results = model(img_bgr, imgsz=img_size, verbose=False)[0]

        if results.masks is not None:
            masks = results.masks.data.cpu().numpy().astype(np.uint8)
            labels = results.boxes.cls.cpu().numpy().astype(np.int32)
            scores = results.boxes.conf.cpu().numpy().astype(np.float32)

            # Resize masks to original image size with bilinear + threshold
            # for smooth edges (INTER_NEAREST creates blocky artifacts)
            h, w = img.shape[:2]
            resized = np.zeros((len(masks), h, w), dtype=np.uint8)
            for i in range(len(masks)):
                smooth = cv2.resize(masks[i].astype(np.float32), (w, h),
                                    interpolation=cv2.INTER_LINEAR)
                resized[i] = (smooth > 0.5).astype(np.uint8)
            masks = resized
        else:
            h, w = img.shape[:2]
            masks = np.zeros((0, h, w), dtype=np.uint8)
            labels = np.zeros(0, dtype=np.int32)
            scores = np.zeros(0, dtype=np.float32)

        predictions[sample.uid] = PredictionResult(
            masks=masks, labels=labels, scores=scores,
        )

    return predictions


def predict_unet(checkpoint: str, samples, img_size: int,
                  unet_mode: str = "semantic") -> dict:
    """Run U-Net inference and return predictions.

    Args:
        unet_mode: "semantic" (softmax/argmax) or "multilabel" (sigmoid/threshold).
    """
    if unet_mode == "multilabel":
        from train_unet import MultiLabelSegmentationModule
        model = MultiLabelSegmentationModule.load_from_checkpoint(checkpoint)
    else:
        from train_unet import SegmentationModule
        model = SegmentationModule.load_from_checkpoint(checkpoint)
    model.eval()
    model.cuda()

    predictions = {}
    for sample in tqdm(samples, desc="U-Net inference"):
        img = load_sample_normalized(sample)
        h, w = img.shape[:2]
        img_resized = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float().cuda()

        with torch.no_grad():
            logits = model(tensor)

        if unet_mode == "multilabel":
            # Sigmoid → (4, H, W) probabilities
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()  # (4, img_size, img_size)
            # Resize each channel back to original size
            ml_mask = np.zeros((4, h, w), dtype=np.float32)
            for c in range(4):
                ml_mask[c] = cv2.resize(probs[c], (w, h), interpolation=cv2.INTER_LINEAR)
            pred = convert_multilabel_to_instances(ml_mask)
        else:
            sem_mask = logits.argmax(dim=1).squeeze().cpu().numpy()
            sem_mask = cv2.resize(sem_mask.astype(np.uint8), (w, h),
                                  interpolation=cv2.INTER_NEAREST).astype(np.int32)
            pred = convert_semantic_to_instances(sem_mask)

        predictions[sample.uid] = pred

    return predictions


def predict_maskrcnn(checkpoint: str, samples, img_size: int) -> dict:
    """Run Mask R-CNN inference."""
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from detectron2 import model_zoo
    from src.evaluation import convert_detectron2_instances

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = checkpoint
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.INPUT.MIN_SIZE_TEST = img_size
    cfg.INPUT.MAX_SIZE_TEST = img_size

    predictor = DefaultPredictor(cfg)
    predictions = {}

    for sample in tqdm(samples, desc="Mask R-CNN inference"):
        img = load_sample_normalized(sample)
        h, w = img.shape[:2]
        img_uint8 = to_uint8(img)
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)

        outputs = predictor(img_bgr)
        instances = outputs["instances"].to("cpu")
        pred = convert_detectron2_instances(instances, h, w)
        predictions[sample.uid] = pred

    return predictions


def main():
    parser = argparse.ArgumentParser(description="Unified model evaluation")
    parser.add_argument("--model", default=None,
                        choices=["yolo", "maskrcnn", "unet"],
                        help="Model type (required unless --plot-only)")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to model checkpoint (required unless --plot-only)")
    parser.add_argument("--strategy", default=None,
                        choices=["strategy1", "strategy2", "strategy3"],
                        help="Splitting strategy. Auto-detected from checkpoint path if not set.")
    parser.add_argument("--species", default=None)
    parser.add_argument("--subset", default="test",
                        help="Which split to evaluate on")
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path for metrics")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unet-mode", default="semantic",
                        choices=["semantic", "multilabel"],
                        help="U-Net mode: semantic (softmax) or multilabel (sigmoid)")
    parser.add_argument("--no-vis", action="store_true",
                        help="Skip saving visualization overlay images")
    parser.add_argument("--vis-dir", type=str, default=None,
                        help="Directory for visualization PNGs (default: output/evaluation/vis_*)")
    parser.add_argument("--plot-only", type=str, default=None,
                        help="Skip inference; regenerate plots from an existing metrics JSON file")

    # Post-processing toggles
    step_names = [name for name, _, _ in STEPS]
    parser.add_argument("--enable-pp", nargs="*", default=[], metavar="STEP",
                        choices=step_names,
                        help=f"Force-enable post-processing steps: {step_names}")
    parser.add_argument("--disable-pp", nargs="*", default=[], metavar="STEP",
                        choices=step_names,
                        help=f"Force-disable post-processing steps: {step_names}")
    parser.add_argument("--no-postprocess", action="store_true",
                        help="Disable ALL post-processing steps")
    args = parser.parse_args()

    # ── Plot-only mode: load existing JSON, regenerate plots, exit ──
    if args.plot_only:
        json_path = Path(args.plot_only)
        if not json_path.exists():
            print(f"Error: {json_path} not found")
            return
        with open(json_path) as f:
            results = json.load(f)
        out_dir = json_path.parent
        plot_tag = json_path.stem  # e.g. "yolo_strategy1_test"
        save_metric_comparison_plots(results, out_dir, plot_tag)
        return

    # Validate required args for full evaluation
    if not args.model or not args.checkpoint:
        parser.error("--model and --checkpoint are required unless --plot-only is used")

    # Auto-detect strategy from checkpoint path if not explicitly set
    if args.strategy is None:
        ckpt_lower = args.checkpoint.lower()
        if "strategy2" in ckpt_lower:
            args.strategy = "strategy2"
        elif "strategy3" in ckpt_lower:
            args.strategy = "strategy3"
        else:
            args.strategy = "strategy1"
        print(f"Auto-detected strategy: {args.strategy}")

    # Setup
    registry = SampleRegistry()
    split = get_split(args.strategy, registry, seed=args.seed, species=args.species)
    samples = split[args.subset]
    print(f"Evaluating {args.model} on {len(samples)} {args.subset} samples")

    # Run inference
    if args.model == "unet":
        predictions = predict_unet(args.checkpoint, samples, args.img_size,
                                   unet_mode=args.unet_mode)
    else:
        predict_fn = {
            "yolo": predict_yolo,
            "maskrcnn": predict_maskrcnn,
        }[args.model]
        predictions = predict_fn(args.checkpoint, samples, args.img_size)

    # Post-processing pipeline
    if args.no_postprocess:
        disable = [name for name, _, _ in STEPS]
    else:
        disable = args.disable_pp
    pp = PostProcessor(
        model=args.model,
        enable=args.enable_pp,
        disable=disable,
    )
    predictions = pp.run_all(predictions)

    # Evaluate — load GT and compute metrics with progress
    print("Computing metrics...")
    metrics = SegmentationMetrics(
        num_classes=NUM_CLASSES,
        class_names=TARGET_CLASSES,
    )

    per_sample_rows = []
    for sample in tqdm(samples, desc="Loading GT & scoring"):
        if sample.uid not in predictions:
            continue
        pred = predictions[sample.uid]

        # Get image dimensions from prediction masks (avoid reloading TIFs)
        if len(pred.masks) > 0:
            h, w = pred.masks.shape[1], pred.masks.shape[2]
        else:
            # Fall back to loading image for dimension (rare: no predictions)
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

        # Collect per-sample metrics for CSV
        sm = _compute_sample_metrics(pred.masks, pred.labels, gt["masks"], gt["labels"])
        row = {
            "sample_id": sample.uid,
            "species": sample.species,
            "microscope": sample.microscope,
            "experiment": sample.experiment,
        }
        iou_vals = []
        for cls_id in sorted(sm.keys()):
            m = sm[cls_id]
            row[f"{m['name']}_IoU"] = round(m["iou"], 4) if not np.isnan(m["iou"]) else ""
            row[f"{m['name']}_Dice"] = round(m["dice"], 4) if not np.isnan(m["dice"]) else ""
            if not np.isnan(m["iou"]):
                iou_vals.append(m["iou"])
        row["mean_IoU"] = round(np.mean(iou_vals), 4) if iou_vals else 0.0
        per_sample_rows.append(row)

    # Print and save (compute only once)
    results = metrics.print_summary()

    out_dir = OUTPUT_DIR / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.output:
        out_path = Path(args.output)
    else:
        model_tag = args.model
        if args.model == "unet":
            model_tag = f"unet_{args.unet_mode}"
        out_path = out_dir / f"{model_tag}_{args.strategy}_{args.subset}.json"

    metrics.save(out_path, _cached_results=results)
    print(f"\nMetrics saved to {out_path}")

    # Save per-sample CSV sorted by mean_IoU (worst first)
    per_sample_rows.sort(key=lambda r: r["mean_IoU"])
    csv_path = out_dir / f"{model_tag}_{args.strategy}_{args.subset}_per_sample.csv"
    if per_sample_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=per_sample_rows[0].keys())
            writer.writeheader()
            writer.writerows(per_sample_rows)
        print(f"Per-sample metrics saved to {csv_path} (sorted worst-first)")
        # Print worst 10
        print(f"\nWorst 10 samples by mean IoU:")
        for row in per_sample_rows[:10]:
            print(f"  {row['sample_id']:50s}  mean_IoU={row['mean_IoU']:.4f}")

    # Save comparison plots by species and microscope
    plot_tag = f"{model_tag}_{args.strategy}_{args.subset}"
    save_metric_comparison_plots(results, out_dir, plot_tag,
                                 per_sample_rows=per_sample_rows)

    # Save visualizations (default on, use --no-vis to skip)
    if not args.no_vis:
        if args.vis_dir:
            vis_dir = Path(args.vis_dir)
        else:
            vis_dir = OUTPUT_DIR / "evaluation" / f"vis_{model_tag}_{args.strategy}_{args.subset}"
        save_visualizations(samples, predictions, vis_dir)


if __name__ == "__main__":
    main()
