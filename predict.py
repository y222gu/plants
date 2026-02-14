"""Run YOLO inference, save predictions as YOLO .txt files, and optionally visualize.

Merges the functionality of the old predict.py (generic directory discovery,
visualization) and generate_predictions.py (batched inference, YOLO txt output).

Usage:
    # Run on the main data directory (uses SampleRegistry)
    python predict.py --data-dir data/ --checkpoint best.pt

    # Run on arbitrary directory of TIF triplets
    python predict.py --data-dir path/to/images --checkpoint best.pt

    # Skip visualization
    python predict.py --data-dir data/ --checkpoint best.pt --no-vis
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import tifffile
from tqdm import tqdm

from src.config import CLASS_COLORS_RGB, DEFAULT_IMG_SIZE, TARGET_CLASSES
from src.preprocessing import normalize_percentile, to_uint8
from src.visualization import (
    draw_masks_overlay,
    downscale_for_vis,
    load_font,
    make_legend_bar,
    pil_text,
)


# ── Sample discovery ──────────────────────────────────────────────────────────

def discover_samples_generic(input_dir: Path) -> list:
    """Find all leaf directories containing 3-channel TIF files.

    Returns list of dicts with 'name', 'dir', and channel paths.
    Works on any directory structure (no SampleRegistry needed).
    """
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


def _load_image_generic(sample: dict) -> np.ndarray:
    """Load 3-channel image from a generic sample dict as (H, W, 3) float32 [0,1]."""
    channels = []
    for ch_name in ["TRITC", "FITC", "DAPI"]:  # RGB order
        img = tifffile.imread(str(sample[ch_name]))
        if img.ndim > 2:
            img = img[0] if img.shape[0] < img.shape[-1] else img[..., 0]
        channels.append(img.astype(np.float32))
    raw = np.stack(channels, axis=-1)
    return normalize_percentile(raw)


# ── Mask-to-polygon conversion ───────────────────────────────────────────────

def mask_to_polygon(mask: np.ndarray, simplify_epsilon: float = 2.0) -> np.ndarray:
    """Convert binary mask to polygon points.

    Returns (N, 2) array of polygon points, or empty array if no contour found.
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return np.array([])
    contour = max(contours, key=cv2.contourArea)
    simplified = cv2.approxPolyDP(contour, simplify_epsilon, True)
    return simplified.reshape(-1, 2).astype(np.float32)


# ── Main logic ────────────────────────────────────────────────────────────────

def run_inference(checkpoint: str, samples, img_size: int, conf_thresh: float,
                  batch_size: int, use_registry: bool):
    """Run batched YOLO inference on samples.

    Returns dict: sample_name -> {img, masks, labels, scores, h, w}.
    """
    from ultralytics import YOLO
    from src.preprocessing import load_sample_normalized

    model = YOLO(checkpoint)
    results = {}

    for batch_start in tqdm(range(0, len(samples), batch_size), desc="YOLO inference"):
        batch = samples[batch_start:batch_start + batch_size]

        # Preprocess batch
        batch_imgs_bgr = []
        batch_imgs = []
        batch_sizes = []
        batch_names = []

        for sample in batch:
            if use_registry:
                img = load_sample_normalized(sample)
                name = sample.uid
            else:
                img = _load_image_generic(sample)
                name = sample["name"]

            img_uint8 = to_uint8(img)
            h, w = img.shape[:2]
            batch_imgs.append(img)
            batch_sizes.append((h, w))
            batch_names.append(name)
            batch_imgs_bgr.append(cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))

        # Batched GPU inference
        batch_results = model(batch_imgs_bgr, imgsz=img_size, conf=conf_thresh, verbose=False)

        for idx, result in enumerate(batch_results):
            h, w = batch_sizes[idx]
            name = batch_names[idx]

            if result.masks is not None:
                masks = result.masks.data.cpu().numpy().astype(np.uint8)
                labels = result.boxes.cls.cpu().numpy().astype(np.int32)
                scores = result.boxes.conf.cpu().numpy().astype(np.float32)
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

            results[name] = {
                "img": batch_imgs[idx],
                "masks": masks,
                "labels": labels,
                "scores": scores,
            }

    return results


def save_predictions_txt(predictions: dict, out_dir: Path):
    """Save predictions as YOLO-format .txt files (one per sample)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, pred in predictions.items():
        masks = pred["masks"]
        labels = pred["labels"]
        h, w = masks.shape[1], masks.shape[2] if len(masks) > 0 else (0, 0)

        txt_path = out_dir / f"{name}.txt"
        with open(txt_path, "w") as f:
            for i in range(len(masks)):
                polygon = mask_to_polygon(masks[i])
                if len(polygon) < 3:
                    continue
                cls_id = int(labels[i])
                coords = []
                for pt in polygon:
                    coords.append(f"{pt[0] / w:.6f}")
                    coords.append(f"{pt[1] / h:.6f}")
                f.write(f"{cls_id} " + " ".join(coords) + "\n")

    print(f"Saved {len(predictions)} prediction .txt files to {out_dir}")


def save_visualizations(predictions: dict, out_dir: Path, max_dim: int = 800):
    """Save 2-panel (Original | Prediction) overlay images."""
    out_dir.mkdir(parents=True, exist_ok=True)

    title_font = load_font(22)
    metric_font = load_font(15)

    for name, pred in tqdm(predictions.items(), desc="Saving visualizations"):
        img = pred["img"]
        masks = pred["masks"]
        labels = pred["labels"]
        scores = pred["scores"]
        img_uint8 = to_uint8(img)

        img_small, masks_small, new_h, new_w = downscale_for_vis(img_uint8, masks, max_dim)

        # Draw panels
        orig_vis = pil_text(img_small.copy(), "Original", (10, 8), title_font,
                            fill=(255, 255, 255), outline=(0, 0, 0))
        pred_vis = draw_masks_overlay(img_small, masks_small, labels, scores)
        pred_vis = pil_text(pred_vis, "Prediction", (10, 8), title_font,
                            fill=(255, 255, 255), outline=(0, 0, 0))

        # Per-class instance counts + confidence
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
            pred_vis = pil_text(pred_vis, text, (10, y_offset), metric_font,
                                fill=color, outline=(0, 0, 0))
            y_offset += 20

        # Combine: Original | Prediction + legend
        divider = np.full((new_h, 3, 3), 255, dtype=np.uint8)
        combined = np.concatenate([orig_vis, divider, pred_vis], axis=1)
        legend = make_legend_bar(combined.shape[1])
        combined = np.concatenate([combined, legend], axis=0)

        out_path = out_dir / f"{name}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

    print(f"Saved {len(predictions)} visualizations to {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Run YOLO inference, save prediction .txt files and visualizations")
    parser.add_argument("--data-dir", required=True,
                        help="Data directory (supports SampleRegistry structure or generic TIF dirs)")
    parser.add_argument("--checkpoint", required=True, help="YOLO checkpoint path")
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--conf-thresh", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--no-vis", action="store_true", help="Skip visualization")
    parser.add_argument("--max-dim", type=int, default=800, help="Max dimension for vis images")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: {data_dir} does not exist")
        return

    image_dir = data_dir / "image"
    if not image_dir.exists():
        print(f"Error: {image_dir} does not exist")
        print("Images must be placed in a subfolder called 'image/' inside the data directory.")
        return

    # Try SampleRegistry first (structured: image/{Sp}/{Mic}/{Exp}/{Sample}/)
    use_registry = False
    try:
        from src.dataset import SampleRegistry
        registry = SampleRegistry(data_dir=data_dir, require_annotations=False)
        samples = registry.samples
        if samples:
            use_registry = True
            print(f"Found {len(samples)} samples via SampleRegistry")
    except Exception:
        pass

    if not use_registry:
        # Fall back to generic directory discovery inside image/
        samples = discover_samples_generic(image_dir)
        if not samples:
            print(f"No samples found in {image_dir}")
            print("Expected directories containing *_DAPI.tif, *_FITC.tif, *_TRITC.tif")
            return
        print(f"Found {len(samples)} samples via directory scan")

    # Run inference
    predictions = run_inference(
        args.checkpoint, samples, args.img_size, args.conf_thresh,
        args.batch_size, use_registry,
    )

    # Save YOLO .txt predictions
    pred_dir = data_dir / "prediction"
    save_predictions_txt(predictions, pred_dir)

    # Save visualizations
    if not args.no_vis:
        vis_dir = pred_dir / "vis"
        save_visualizations(predictions, vis_dir, max_dim=args.max_dim)


if __name__ == "__main__":
    main()
