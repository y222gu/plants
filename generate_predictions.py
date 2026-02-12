"""Generate YOLO-format prediction files and overlay images for annotation review.

Usage:
    python generate_predictions.py --checkpoint path/to/best.pt --output-dir predictions/

This script:
1. Runs YOLO inference on all samples
2. Saves predictions as YOLO .txt files (same format as annotations)
3. Optionally saves overlay images for quick visual review
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.config import CLASS_COLORS_RGB, DEFAULT_IMG_SIZE
from src.dataset import SampleRegistry
from src.preprocessing import load_sample_normalized, to_uint8


def mask_to_polygon(mask: np.ndarray, simplify_epsilon: float = 2.0) -> np.ndarray:
    """Convert binary mask to polygon points.

    Args:
        mask: (H, W) binary mask
        simplify_epsilon: Douglas-Peucker simplification threshold (higher = fewer points)

    Returns:
        (N, 2) array of polygon points, or empty array if no contour found
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return np.array([])

    # Take the largest contour
    contour = max(contours, key=cv2.contourArea)

    # Simplify the contour
    epsilon = simplify_epsilon
    simplified = cv2.approxPolyDP(contour, epsilon, True)

    # Convert to (N, 2) array
    points = simplified.reshape(-1, 2).astype(np.float32)
    return points


def draw_overlay(img_uint8: np.ndarray, polygons: list, alpha: float = 0.4) -> np.ndarray:
    """Draw polygons on image with semi-transparent overlay."""
    result = img_uint8.copy()
    overlay = result.copy()

    for poly_data in polygons:
        class_id = poly_data["class_id"]
        polygon = poly_data["polygon"]
        color = CLASS_COLORS_RGB.get(class_id, (128, 128, 128))

        pts = np.round(polygon).astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(result, [pts], True, color, 2)

    result = cv2.addWeighted(result, 1 - alpha, overlay, alpha, 0)
    return result


def main():
    parser = argparse.ArgumentParser(description="Generate YOLO predictions for annotation review")
    parser.add_argument("--checkpoint", required=True, help="Path to YOLO model checkpoint (.pt)")
    parser.add_argument("--output-dir", required=True, help="Output directory for predictions")
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE, help="Inference image size")
    parser.add_argument("--save-overlays", action="store_true", help="Also save overlay images")
    parser.add_argument("--conf-thresh", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--batch-size", type=int, default=86, help="Inference batch size (higher = more GPU usage)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    overlay_dir = output_dir / "overlays"
    if args.save_overlays:
        overlay_dir.mkdir(exist_ok=True)

    # Load model
    from ultralytics import YOLO
    model = YOLO(args.checkpoint)

    # Load samples
    registry = SampleRegistry()
    samples = registry.samples
    print(f"Processing {len(samples)} samples...")

    # Process in batches for maximum GPU utilization
    batch_size = args.batch_size
    for batch_start in tqdm(range(0, len(samples), batch_size)):
        batch_samples = samples[batch_start : batch_start + batch_size]

        # Preprocess entire batch on CPU
        batch_imgs_bgr = []
        batch_imgs_uint8 = []
        batch_sizes = []
        for sample in batch_samples:
            img = load_sample_normalized(sample)
            img_uint8 = to_uint8(img)
            h, w = img.shape[:2]
            batch_sizes.append((h, w))
            batch_imgs_uint8.append(img_uint8)
            batch_imgs_bgr.append(cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))

        # Run batched inference — single GPU call for the whole batch
        batch_results = model(batch_imgs_bgr, imgsz=args.img_size, conf=args.conf_thresh, verbose=False)

        # Post-process each result
        for idx, result in enumerate(batch_results):
            sample = batch_samples[idx]
            h, w = batch_sizes[idx]

            polygons = []
            if result.masks is not None:
                masks = result.masks.data.cpu().numpy().astype(np.uint8)
                labels = result.boxes.cls.cpu().numpy().astype(np.int32)

                for i in range(len(masks)):
                    mask_resized = cv2.resize(masks[i], (w, h), interpolation=cv2.INTER_NEAREST)
                    polygon = mask_to_polygon(mask_resized)
                    if len(polygon) >= 3:
                        polygons.append({
                            "class_id": int(labels[i]),
                            "polygon": polygon
                        })

            # Save YOLO format txt file
            txt_path = output_dir / f"{sample.uid}.txt"
            with open(txt_path, "w") as f:
                for poly_data in polygons:
                    class_id = poly_data["class_id"]
                    polygon = poly_data["polygon"]

                    coords = []
                    for pt in polygon:
                        coords.append(f"{pt[0] / w:.6f}")
                        coords.append(f"{pt[1] / h:.6f}")

                    line = f"{class_id} " + " ".join(coords)
                    f.write(line + "\n")

            # Save overlay image
            if args.save_overlays:
                overlay = draw_overlay(batch_imgs_uint8[idx], polygons)
                overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(overlay_dir / f"{sample.uid}.png"), overlay_bgr)

    print(f"\nPredictions saved to: {output_dir}")
    print(f"Total samples processed: {len(registry.samples)}")

    if args.save_overlays:
        print(f"Overlays saved to: {overlay_dir}")

    print("\nTo review and correct annotations, run:")
    print(f"  python annotation_editor.py --pred-dir {output_dir}")


if __name__ == "__main__":
    main()
