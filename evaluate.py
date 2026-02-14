"""Unified evaluation entry point.

Run any trained model on a test set and produce metrics CSV/JSON,
visualizations, and comparison plots.

Usage:
    python evaluate.py --model yolo --checkpoint output/runs/yolo/run/weights/best.pt
    python evaluate.py --model unet --checkpoint output/runs/unet/run/checkpoints/best.ckpt
    python evaluate.py --model yolo --from-predictions data/prediction/ --strategy strategy1
    python evaluate.py --plot-only output/evaluation/yolo_strategy1_test.json
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
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
    convert_yolo_predictions,
)
from src.postprocessing import PostProcessor, STEPS
from src.metrics import SegmentationMetrics
from src.preprocessing import load_sample_normalized, to_uint8
from src.visualization import (
    draw_masks_overlay,
    downscale_for_vis,
    load_font,
    make_legend_bar,
    pil_text,
    setup_pub_style,
    PUB_CLASS_COLORS,
)


def _compute_sample_metrics(
    pred_masks: np.ndarray,
    pred_labels: np.ndarray,
    gt_masks: np.ndarray,
    gt_labels: np.ndarray,
) -> dict:
    """Compute per-class IoU and Dice for a single sample."""
    results = {}
    for cls_id, cls_name in TARGET_CLASSES.items():
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


def save_visualizations(
    samples,
    predictions: dict,
    vis_dir: Path,
    max_dim: int = 800,
):
    """Save side-by-side GT vs prediction overlay images with per-class metrics."""
    vis_dir.mkdir(parents=True, exist_ok=True)

    title_font = load_font(22)
    metric_font = load_font(15)

    for sample in tqdm(samples, desc="Saving visualizations"):
        if sample.uid not in predictions:
            continue
        pred = predictions[sample.uid]

        img = load_sample_normalized(sample)
        h, w = img.shape[:2]
        img_uint8 = to_uint8(img)

        gt = load_sample_annotations(sample, h, w)

        sample_metrics = _compute_sample_metrics(
            pred.masks, pred.labels, gt["masks"], gt["labels"],
        )

        img_small, gt_masks, new_h, new_w = downscale_for_vis(img_uint8, gt["masks"], max_dim)
        _, pred_masks, _, _ = downscale_for_vis(img_uint8, pred.masks, max_dim)

        # Draw overlays
        orig_vis = pil_text(img_small.copy(), "Original", (10, 8), title_font,
                            fill=(255, 255, 255), outline=(0, 0, 0))
        gt_vis = draw_masks_overlay(img_small, gt_masks, gt["labels"])
        gt_vis = pil_text(gt_vis, "Ground Truth", (10, 8), title_font,
                          fill=(255, 255, 255), outline=(0, 0, 0))
        pred_vis = draw_masks_overlay(img_small, pred_masks, pred.labels)
        pred_vis = pil_text(pred_vis, "Prediction", (10, 8), title_font,
                            fill=(255, 255, 255), outline=(0, 0, 0))

        # Per-class metrics on prediction panel
        y_offset = 40
        for cls_id in sorted(sample_metrics.keys()):
            m = sample_metrics[cls_id]
            color = CLASS_COLORS_RGB.get(cls_id, (255, 255, 255))
            iou_str = f"{m['iou']:.2f}" if not np.isnan(m['iou']) else "N/A"
            dice_str = f"{m['dice']:.2f}" if not np.isnan(m['dice']) else "N/A"
            text = f"{m['name']}: IoU={iou_str}  Dice={dice_str}"
            pred_vis = pil_text(pred_vis, text, (10, y_offset), metric_font,
                                fill=color, outline=(0, 0, 0))
            y_offset += 20

        # Side-by-side: Original | GT | Prediction + legend
        divider = np.full((new_h, 3, 3), 255, dtype=np.uint8)
        combined = np.concatenate(
            [orig_vis, divider, gt_vis, divider, pred_vis], axis=1
        )
        legend = make_legend_bar(combined.shape[1])
        combined = np.concatenate([combined, legend], axis=0)

        out_path = vis_dir / f"{sample.uid}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

    print(f"Saved {len(samples)} visualizations to {vis_dir}")


def save_metric_comparison_plots(results: dict, out_dir: Path, model_tag: str,
                                  per_sample_rows: list = None):
    """Save publication-quality box plots comparing per-sample metrics."""
    setup_pub_style()
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
    cls_colors = [PUB_CLASS_COLORS.get(c, "#999999") for c in class_names]

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
    handles = [Patch(facecolor=PUB_CLASS_COLORS.get(c, "#999999"),
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
        handles = [Patch(facecolor=PUB_CLASS_COLORS.get(c, "#999999"),
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


# ── Inference functions ───────────────────────────────────────────────────────

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
    """Run U-Net inference and return predictions."""
    if unet_mode == "multilabel":
        from train.train_unet import MultiLabelSegmentationModule
        model = MultiLabelSegmentationModule.load_from_checkpoint(checkpoint)
    else:
        from train.train_unet import SegmentationModule
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
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
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


def load_predictions_from_dir(pred_dir: Path, samples) -> dict:
    """Load saved YOLO .txt prediction files into PredictionResult dict."""
    predictions = {}
    missing = 0
    for sample in samples:
        txt_path = pred_dir / f"{sample.uid}.txt"
        if not txt_path.exists():
            missing += 1
            continue
        img = load_sample_normalized(sample)
        h, w = img.shape[:2]
        pred = convert_yolo_predictions(txt_path, h, w)
        predictions[sample.uid] = pred

    if missing:
        print(f"Warning: {missing} samples missing prediction files in {pred_dir}")
    print(f"Loaded {len(predictions)} predictions from {pred_dir}")
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Unified model evaluation")
    parser.add_argument("--data-dir", default="data/",
                        help="Data directory with image/ and annotation/ subfolders")
    parser.add_argument("--model", default=None,
                        choices=["yolo", "unet"],
                        help="Model type (required unless --plot-only or --from-predictions)")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--from-predictions", default=None,
                        help="Load saved YOLO .txt predictions from this directory (skip inference)")
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path for metrics")
    parser.add_argument("--unet-mode", default="semantic",
                        choices=["semantic", "multilabel"],
                        help="U-Net mode: semantic (softmax) or multilabel (sigmoid)")
    parser.add_argument("--no-vis", action="store_true",
                        help="Skip saving visualization overlay images")
    parser.add_argument("--vis-dir", type=str, default=None,
                        help="Directory for visualization PNGs")
    parser.add_argument("--no-metrics", action="store_true",
                        help="Skip segmentation metric computation")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip metric plot generation")
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

    # ── Plot-only mode ──
    if args.plot_only:
        json_path = Path(args.plot_only)
        if not json_path.exists():
            print(f"Error: {json_path} not found")
            return
        with open(json_path) as f:
            results = json.load(f)
        out_dir = json_path.parent
        plot_tag = json_path.stem
        save_metric_comparison_plots(results, out_dir, plot_tag)
        return

    # ── Determine model tag ──
    if args.from_predictions:
        model_tag = "yolo"
        if not args.model:
            args.model = "yolo"
    else:
        if not args.model or not args.checkpoint:
            parser.error("--model and --checkpoint are required unless --plot-only or --from-predictions is used")
        model_tag = args.model
        if args.model == "unet":
            model_tag = f"unet_{args.unet_mode}"

    # Discover all samples with annotations in data-dir
    data_dir = Path(args.data_dir)
    registry = SampleRegistry(data_dir=data_dir, require_annotations=True)
    samples = registry.samples
    if not samples:
        print(f"No annotated samples found in {data_dir}")
        print("Expected: {data-dir}/image/{Species}/{Microscope}/{Exp}/{Sample}/ "
              "with matching annotation .txt files in {data-dir}/annotation/")
        return
    print(f"Evaluating {args.model} on {len(samples)} samples")

    # ── Get predictions ──
    if args.from_predictions:
        pred_dir = Path(args.from_predictions)
        predictions = load_predictions_from_dir(pred_dir, samples)
    elif args.model == "unet":
        predictions = predict_unet(args.checkpoint, samples, args.img_size,
                                   unet_mode=args.unet_mode)
    else:
        predict_fn = {
            "yolo": predict_yolo,
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

    # ── Metrics ──
    out_dir = OUTPUT_DIR / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_sample_rows = []
    results = None

    if not args.no_metrics:
        print("Computing metrics...")
        metrics = SegmentationMetrics(
            num_classes=NUM_CLASSES,
            class_names=TARGET_CLASSES,
        )

        for sample in tqdm(samples, desc="Loading GT & scoring"):
            if sample.uid not in predictions:
                continue
            pred = predictions[sample.uid]

            if len(pred.masks) > 0:
                h, w = pred.masks.shape[1], pred.masks.shape[2]
            else:
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

        results = metrics.print_summary()

        if args.output:
            out_path = Path(args.output)
        else:
            out_path = out_dir / f"{model_tag}_metrics.json"
        metrics.save(out_path, _cached_results=results)
        print(f"\nMetrics saved to {out_path}")

        # Per-sample CSV
        per_sample_rows.sort(key=lambda r: r["mean_IoU"])
        csv_path = out_dir / f"{model_tag}_per_sample.csv"
        if per_sample_rows:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=per_sample_rows[0].keys())
                writer.writeheader()
                writer.writerows(per_sample_rows)
            print(f"Per-sample metrics saved to {csv_path} (sorted worst-first)")
            print(f"\nWorst 10 samples by mean IoU:")
            for row in per_sample_rows[:10]:
                print(f"  {row['sample_id']:50s}  mean_IoU={row['mean_IoU']:.4f}")

    # ── Plots ──
    if not args.no_plots and results is not None:
        save_metric_comparison_plots(results, out_dir, model_tag,
                                     per_sample_rows=per_sample_rows)

    # ── Visualizations ──
    if not args.no_vis:
        if args.vis_dir:
            vis_dir = Path(args.vis_dir)
        else:
            vis_dir = OUTPUT_DIR / "evaluation" / f"vis_{model_tag}"
        save_visualizations(samples, predictions, vis_dir)


if __name__ == "__main__":
    main()
