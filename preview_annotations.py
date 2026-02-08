"""
Script to preview annotations overlaid on composite images.

Combines 3 channel images (DAPI, FITC, TRITC) into RGB and overlays polygon masks
from annotation files with class-specific colors.

Applies contrast enhancement to make dark images more visible.
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import cv2
import tifffile
from skimage import exposure
from tqdm import tqdm

# Paths
DATA_DIR = Path(r"C:\Users\Yifei\Documents\plants\data")
IMAGE_DIR = DATA_DIR / "image"
ANNOTATION_DIR = DATA_DIR / "annotation"
OUTPUT_DIR = Path(r"C:\Users\Yifei\Documents\plants\preview")


LABEL_CLASSES = {
    0: ("Whole Root", (0, 0, 255)),          # Blue
    1: ("Aerenchyma", (255, 255, 0)),        # Yellow
    2: ("Outer Endodermis", (0, 255, 0)),    # Green
    3: ("Inner Endodermis", (255, 0, 0)),    # Red
}

def enhance_channel(img, clip_limit=0.03, percentile_low=1, percentile_high=99.5):
    """
    Enhance a single channel image to improve visibility of dark images.

    Uses percentile-based rescaling followed by CLAHE for local contrast.
    This avoids over-exposure while revealing details in dark regions.
    """
    # Convert to float
    img_float = img.astype(np.float32)

    # Percentile-based rescaling (robust to outliers)
    p_low = np.percentile(img_float, percentile_low)
    p_high = np.percentile(img_float, percentile_high)

    if p_high > p_low:
        img_float = (img_float - p_low) / (p_high - p_low)
        img_float = np.clip(img_float, 0, 1)
    else:
        # Fallback for very uniform images
        if img_float.max() > 0:
            img_float = img_float / img_float.max()

    # Apply CLAHE for local contrast enhancement
    # Use a moderate clip limit to avoid amplifying noise too much
    img_enhanced = exposure.equalize_adapthist(img_float, clip_limit=clip_limit)

    # Convert to 8-bit
    img_8bit = (img_enhanced * 255).astype(np.uint8)

    return img_8bit


def load_and_combine_channels(sample_dir):
    """
    Load 3 channel images and combine into RGB.

    Channel mapping (standardized filenames):
    - {sample}_DAPI.tif -> Blue
    - {sample}_FITC.tif -> Green
    - {sample}_TRITC.tif -> Red
    """
    sample_name = sample_dir.name
    dapi_file = sample_dir / f"{sample_name}_DAPI.tif"
    fitc_file = sample_dir / f"{sample_name}_FITC.tif"
    tritc_file = sample_dir / f"{sample_name}_TRITC.tif"

    if not all(f.exists() for f in [dapi_file, fitc_file, tritc_file]):
        return None

    # Load channels
    try:
        ch1 = tifffile.imread(dapi_file)   # DAPI -> Blue
        ch2 = tifffile.imread(fitc_file)   # FITC -> Green
        ch3 = tifffile.imread(tritc_file)  # TRITC -> Red
    except Exception as e:
        print(f"Error loading images from {sample_dir}: {e}")
        return None

    # Handle multi-dimensional images (take first frame if stack)
    if ch1.ndim > 2:
        ch1 = ch1[0] if ch1.shape[0] < ch1.shape[-1] else ch1[..., 0]
    if ch2.ndim > 2:
        ch2 = ch2[0] if ch2.shape[0] < ch2.shape[-1] else ch2[..., 0]
    if ch3.ndim > 2:
        ch3 = ch3[0] if ch3.shape[0] < ch3.shape[-1] else ch3[..., 0]

    # Enhance each channel
    blue = enhance_channel(ch1)   # DAPI
    green = enhance_channel(ch2)  # FITC
    red = enhance_channel(ch3)    # TRITC

    # Combine into RGB
    rgb = np.stack([red, green, blue], axis=-1)

    return rgb


def parse_annotation(annotation_path, img_width, img_height):
    """
    Parse YOLO-style polygon annotation file.

    Each line: class_id x1 y1 x2 y2 ... xn yn
    Coordinates are normalized (0-1).

    Returns list of (class_id, polygon_points)
    """
    annotations = []

    with open(annotation_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:  # Need at least class + 3 points
                continue

            class_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]

            # Convert to pixel coordinates
            points = []
            for i in range(0, len(coords), 2):
                if i + 1 < len(coords):
                    x = int(coords[i] * img_width)
                    y = int(coords[i + 1] * img_height)
                    points.append((x, y))

            if len(points) >= 3:
                annotations.append((class_id, points))

    return annotations


def polygon_area(points):
    """Calculate polygon area using shoelace formula."""
    n = len(points)
    if n < 3:
        return 0
    area = 0
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    return abs(area) / 2


def draw_annotations(img, annotations, alpha=0.5):
    """
    Draw polygon annotations on image with semi-transparent solid fill.
    Uses OpenCV fillPoly for robust handling of complex polygons.

    Draws polygons in order of decreasing area so smaller regions
    appear on top of larger regions.

    Returns image with overlays.
    """
    # Ensure we're working with uint8
    result = img.copy()
    if result.dtype != np.uint8:
        result = np.clip(result, 0, 255).astype(np.uint8)

    # Sort annotations by area (largest first) so smaller regions draw on top
    annotations_with_area = [(class_id, points, polygon_area(points))
                             for class_id, points in annotations]
    annotations_sorted = sorted(annotations_with_area, key=lambda x: -x[2])

    # Create a copy for the overlay (we'll draw filled polygons here)
    overlay = result.copy()

    for class_id, points, area in annotations_sorted:
        if class_id in LABEL_CLASSES:
            name, color = LABEL_CLASSES[class_id]
        else:
            name, color = f"Class {class_id}", (128, 128, 128)

        # Convert points to numpy array for OpenCV
        # OpenCV expects points as array of shape (n, 1, 2) with int32
        pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))

        # Draw filled polygon on overlay
        cv2.fillPoly(overlay, [pts], color)

    # Blend overlay with original image: result = (1-alpha)*original + alpha*overlay
    result = cv2.addWeighted(result, 1 - alpha, overlay, alpha, 0)

    # Draw outlines on top for clarity (same order - largest first)
    for class_id, points, area in annotations_sorted:
        if class_id in LABEL_CLASSES:
            name, color = LABEL_CLASSES[class_id]
        else:
            name, color = f"Class {class_id}", (128, 128, 128)

        pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(result, [pts], isClosed=True, color=color, thickness=2)

    return result


def get_sample_folders():
    """Find all sample folders with images."""
    sample_folders = []

    for root, dirs, files in os.walk(IMAGE_DIR):
        if not dirs and files:  # Leaf directory with files
            tif_files = [f for f in files if f.lower().endswith(('.tif', '.tiff'))]
            if tif_files:
                full_path = Path(root)
                rel_path = full_path.relative_to(IMAGE_DIR)
                sample_folders.append((full_path, rel_path))

    return sample_folders


def path_to_annotation_key(rel_path):
    """Convert sample path to annotation filename."""
    key = "_".join(rel_path.parts)
    return f"{key}.txt"


def main():
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Get all sample folders
    sample_folders = get_sample_folders()
    print(f"Found {len(sample_folders)} sample folders")

    # Get available annotations
    annotation_files = {f.name for f in ANNOTATION_DIR.iterdir() if f.is_file()}
    print(f"Found {len(annotation_files)} annotation files")

    # Process samples with annotations
    processed = 0
    skipped = 0

    for sample_path, rel_path in tqdm(sample_folders, desc="Processing samples"):
        annotation_name = path_to_annotation_key(rel_path)

        if annotation_name not in annotation_files:
            skipped += 1
            continue

        # Load and combine channels
        rgb_img = load_and_combine_channels(sample_path)
        if rgb_img is None:
            print(f"Could not load images from {sample_path}")
            skipped += 1
            continue

        # Parse annotations
        annotation_path = ANNOTATION_DIR / annotation_name
        h, w = rgb_img.shape[:2]
        annotations = parse_annotation(annotation_path, w, h)

        if not annotations:
            print(f"No valid annotations in {annotation_name}")

        # Draw annotations
        result = draw_annotations(rgb_img, annotations)

        # Save result
        # Put all previews in one flat folder with descriptive filename
        # Use the annotation key (path joined by _) as filename
        flat_name = "_".join(rel_path.parts) + "_preview.png"
        output_file = OUTPUT_DIR / flat_name

        Image.fromarray(result).save(output_file)
        processed += 1

    print(f"\nDone!")
    print(f"Processed: {processed}")
    print(f"Skipped (no annotation or load error): {skipped}")
    print(f"Output directory: {OUTPUT_DIR}")


def create_legend():
    """Create a legend image showing class colors."""
    from PIL import ImageFont

    legend_height = 30 * len(LABEL_CLASSES) + 20
    legend_width = 250
    legend = Image.new('RGB', (legend_width, legend_height), (255, 255, 255))
    draw = ImageDraw.Draw(legend)

    y = 10
    for class_id, (name, color) in sorted(LABEL_CLASSES.items()):
        # Draw color box
        draw.rectangle([10, y, 40, y + 20], fill=color, outline=(0, 0, 0))
        # Draw label
        draw.text((50, y + 2), f"{class_id}: {name}", fill=(0, 0, 0))
        y += 30

    legend.save(OUTPUT_DIR / "legend.png")
    print(f"Legend saved to {OUTPUT_DIR / 'legend.png'}")


if __name__ == "__main__":
    main()
    create_legend()
