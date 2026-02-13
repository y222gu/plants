"""Run YOLO prediction on images without ground truth and visualize results.

Accepts either:
  - A single sample directory containing {Sample}_DAPI.tif, _FITC.tif, _TRITC.tif
  - A parent directory to walk (finds all leaf dirs with 3 channel TIFs)

Usage:
    # Single sample folder
    python predict.py --input data/image/Rice/Olympus/Exp1/Sample1

    # Walk a directory tree
    python predict.py --input data/image/Rice/Olympus/Exp1

    # Specify output directory and checkpoint
    python predict.py --input path/to/images --output output/predictions \
        --checkpoint output/runs/yolo/yolo11m-seg_strategy1/weights/best.pt
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from src.config import CLASS_COLORS_RGB, DEFAULT_IMG_SIZE, OUTPUT_DIR, TARGET_CLASSES
from src.preprocessing import normalize_percentile, to_uint8


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for name in ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _pil_text(img, text, xy, font, fill, outline=None):
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    if outline:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx != 0 or dy != 0:
                    draw.text((xy[0] + dx, xy[1] + dy), text, font=font, fill=outline)
    draw.text(xy, text, font=font, fill=fill)
    return np.array(pil_img)


def discover_samples(input_dir: Path) -> list:
    """Find all leaf directories containing 3-channel TIF files.

    Returns list of dicts with 'name', 'dir', and channel paths.
    """
    import os
    samples = []

    # Check if input_dir itself is a sample directory
    tifs = {f.stem.split("_")[-1].upper(): f
            for f in input_dir.iterdir()
            if f.suffix.lower() in (".tif", ".tiff")}
    if all(ch in tifs for ch in ["DAPI", "FITC", "TRITC"]):
        samples.append({
            "name": input_dir.name,
            "dir": input_dir,
            "DAPI": tifs["DAPI"],
            "FITC": tifs["FITC"],
            "TRITC": tifs["TRITC"],
        })
        return samples

    # Walk subdirectories
    for root, dirs, files in os.walk(input_dir):
        if dirs:
            continue  # only leaf directories
        root_path = Path(root)
        tifs = {}
        for f in files:
            if f.lower().endswith((".tif", ".tiff")):
                ch = f.rsplit("_", 1)[-1].split(".")[0].upper()
                if ch in ("DAPI", "FITC", "TRITC"):
                    tifs[ch] = root_path / f
        if all(ch in tifs for ch in ["DAPI", "FITC", "TRITC"]):
            samples.append({
                "name": root_path.name,
                "dir": root_path,
                "DAPI": tifs["DAPI"],
                "FITC": tifs["FITC"],
                "TRITC": tifs["TRITC"],
            })

    samples.sort(key=lambda s: s["name"])
    return samples


def load_image(sample: dict) -> np.ndarray:
    """Load 3-channel image as (H, W, 3) float32 normalized to [0,1]."""
    channels = []
    for ch_name in ["TRITC", "FITC", "DAPI"]:  # RGB order
        img = tifffile.imread(str(sample[ch_name]))
        if img.ndim > 2:
            img = img[0] if img.shape[0] < img.shape[-1] else img[..., 0]
        channels.append(img.astype(np.float32))
    raw = np.stack(channels, axis=-1)
    return normalize_percentile(raw)


def draw_masks_overlay(img_uint8, masks, labels, scores=None, alpha=0.45):
    """Draw instance masks with semi-transparent fill + contours."""
    result = img_uint8.copy()
    if len(masks) == 0:
        return result

    areas = [m.sum() for m in masks]
    order = np.argsort(areas)[::-1]

    overlay = result.copy()
    for idx in order:
        mask = masks[idx]
        cls_id = int(labels[idx])
        color = CLASS_COLORS_RGB.get(cls_id, (128, 128, 128))
        overlay[mask > 0] = color

    result = cv2.addWeighted(result, 1 - alpha, overlay, alpha, 0)

    for idx in order:
        mask = masks[idx]
        cls_id = int(labels[idx])
        color = CLASS_COLORS_RGB.get(cls_id, (128, 128, 128))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, color, 2)

    return result


def run_yolo(checkpoint, samples, img_size):
    """Run YOLO inference on discovered samples."""
    from ultralytics import YOLO
    model = YOLO(checkpoint)
    results = {}

    for sample in tqdm(samples, desc="YOLO inference"):
        img = load_image(sample)
        h, w = img.shape[:2]
        img_uint8 = to_uint8(img)
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)

        preds = model(img_bgr, imgsz=img_size, verbose=False)[0]

        if preds.masks is not None:
            masks = preds.masks.data.cpu().numpy().astype(np.uint8)
            labels = preds.boxes.cls.cpu().numpy().astype(np.int32)
            scores = preds.boxes.conf.cpu().numpy().astype(np.float32)
            resized = np.zeros((len(masks), h, w), dtype=np.uint8)
            for i in range(len(masks)):
                smooth = cv2.resize(masks[i].astype(np.float32), (w, h),
                                    interpolation=cv2.INTER_LINEAR)
                resized[i] = (smooth > 0.5).astype(np.uint8)
            masks = resized
        else:
            masks = np.zeros((0, h, w), dtype=np.uint8)
            labels = np.zeros(0, dtype=np.int32)
            scores = np.zeros(0, dtype=np.float32)

        results[sample["name"]] = {
            "img": img,
            "masks": masks,
            "labels": labels,
            "scores": scores,
        }

    return results


def save_visualizations(samples, predictions, out_dir, max_dim=800):
    """Save side-by-side Original | Prediction overlay images."""
    out_dir.mkdir(parents=True, exist_ok=True)

    title_font = _load_font(22)
    metric_font = _load_font(15)
    legend_font = _load_font(14)

    for sample in tqdm(samples, desc="Saving visualizations"):
        name = sample["name"]
        if name not in predictions:
            continue
        pred = predictions[name]
        img = pred["img"]
        masks = pred["masks"]
        labels = pred["labels"]
        scores = pred["scores"]
        h, w = img.shape[:2]
        img_uint8 = to_uint8(img)

        # Downscale for reasonable file sizes
        scale = min(1.0, max_dim / max(h, w))
        if scale < 1.0:
            new_h, new_w = int(h * scale), int(w * scale)
            img_small = cv2.resize(img_uint8, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            if len(masks) > 0:
                masks_small = np.zeros((len(masks), new_h, new_w), dtype=np.uint8)
                for i in range(len(masks)):
                    masks_small[i] = cv2.resize(masks[i], (new_w, new_h),
                                                interpolation=cv2.INTER_NEAREST)
            else:
                masks_small = masks
        else:
            img_small = img_uint8
            new_h, new_w = h, w
            masks_small = masks

        # Draw overlays
        orig_vis = img_small.copy()
        pred_vis = draw_masks_overlay(img_small, masks_small, labels, scores)

        # Add titles
        orig_vis = _pil_text(orig_vis, "Original", (10, 8), title_font,
                             fill=(255, 255, 255), outline=(0, 0, 0))
        pred_vis = _pil_text(pred_vis, "Prediction", (10, 8), title_font,
                             fill=(255, 255, 255), outline=(0, 0, 0))

        # Add per-class instance counts + confidence on prediction panel
        y_offset = 40
        for cls_id, cls_name in TARGET_CLASSES.items():
            cls_mask = labels == cls_id
            n_inst = int(cls_mask.sum())
            if n_inst == 0:
                continue
            color = CLASS_COLORS_RGB.get(cls_id, (255, 255, 255))
            cls_scores = scores[cls_mask]
            conf_str = f"conf={cls_scores.mean():.2f}" if len(cls_scores) > 0 else ""
            text = f"{cls_name}: {n_inst} inst  {conf_str}"
            pred_vis = _pil_text(pred_vis, text, (10, y_offset), metric_font,
                                 fill=color, outline=(0, 0, 0))
            y_offset += 20

        # Side-by-side: Original | Prediction
        divider = np.full((new_h, 3, 3), 255, dtype=np.uint8)
        combined = np.concatenate([orig_vis, divider, pred_vis], axis=1)

        # Legend bar at bottom
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

        out_path = out_dir / f"{name}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

    print(f"Saved {len(samples)} visualizations to {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Run YOLO prediction on unlabeled images and visualize results")
    parser.add_argument("--input", required=True,
                        help="Directory containing sample folders with DAPI/FITC/TRITC TIFs")
    parser.add_argument("--checkpoint",
                        default="output/runs/yolo/yolo11m-seg_strategy1/weights/best.pt",
                        help="YOLO checkpoint path")
    parser.add_argument("--output", default=None,
                        help="Output directory for visualizations (default: output/predictions/)")
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--max-dim", type=int, default=800,
                        help="Max dimension for visualization images")
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"Error: {input_dir} does not exist")
        return

    # Discover samples
    samples = discover_samples(input_dir)
    if not samples:
        print(f"No samples found in {input_dir}")
        print("Expected directories containing *_DAPI.tif, *_FITC.tif, *_TRITC.tif")
        return
    print(f"Found {len(samples)} samples in {input_dir}")

    # Run inference
    predictions = run_yolo(args.checkpoint, samples, args.img_size)

    # Save visualizations
    if args.output:
        out_dir = Path(args.output)
    else:
        out_dir = OUTPUT_DIR / "predictions"
    save_visualizations(samples, predictions, out_dir, max_dim=args.max_dim)


if __name__ == "__main__":
    main()
