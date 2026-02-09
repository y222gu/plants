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
    fill_prediction_holes,
)
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
        gt_vis = draw_masks_overlay(img_small, gt_masks, gt["labels"])
        pred_vis = draw_masks_overlay(img_small, pred_masks, pred.labels)

        # Add titles using PIL
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

        # Side-by-side
        divider = np.full((new_h, 3, 3), 255, dtype=np.uint8)
        combined = np.concatenate([gt_vis, divider, pred_vis], axis=1)

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
    parser.add_argument("--model", required=True,
                        choices=["yolo", "maskrcnn", "unet"],
                        help="Model type")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--strategy", default="strategy1",
                        choices=["strategy1", "strategy2", "strategy3"])
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
    args = parser.parse_args()

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

    # Fill holes in all predicted masks (mask → outer contour → fill)
    print("Filling mask holes...")
    for uid in predictions:
        predictions[uid] = fill_prediction_holes(predictions[uid])

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

    # Save visualizations (default on, use --no-vis to skip)
    if not args.no_vis:
        if args.vis_dir:
            vis_dir = Path(args.vis_dir)
        else:
            vis_dir = OUTPUT_DIR / "evaluation" / f"vis_{model_tag}_{args.strategy}_{args.subset}"
        save_visualizations(samples, predictions, vis_dir)


if __name__ == "__main__":
    main()
