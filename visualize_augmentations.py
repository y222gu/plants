"""Visualize training augmentations with mask overlays.

Picks 10 random samples, augments each 10 times, and saves side-by-side
(original + 10 augmented) images with mask overlays to verify sync.
"""

import random
from pathlib import Path

import cv2
import numpy as np

from src.annotation_utils import parse_yolo_annotations, polygons_to_semantic_mask
from src.augmentation import get_train_transform
from src.config import CLASS_COLORS_RGB, DEFAULT_IMG_SIZE, OUTPUT_DIR, TARGET_CLASSES
from src.dataset import SampleRegistry
from src.preprocessing import load_sample_normalized


def overlay_semantic_mask(img_uint8, sem_mask, alpha=0.45):
    """Draw semantic mask overlay on RGB uint8 image."""
    overlay = img_uint8.copy()
    for label_val, cls_id in [(1, 0), (2, 1), (3, 2), (4, 3)]:
        color = CLASS_COLORS_RGB.get(cls_id, (128, 128, 128))
        overlay[sem_mask == label_val] = color

    result = cv2.addWeighted(img_uint8, 1 - alpha, overlay, alpha, 0)

    # Draw contours
    for label_val, cls_id in [(1, 0), (2, 1), (3, 2), (4, 3)]:
        color = CLASS_COLORS_RGB.get(cls_id, (128, 128, 128))
        binary = (sem_mask == label_val).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, color, 1)

    return result


def to_uint8(img):
    """Convert [0,1] float32 to uint8."""
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def main():
    rng = random.Random(42)
    img_size = DEFAULT_IMG_SIZE
    n_samples = 10
    n_augments = 10

    registry = SampleRegistry()
    samples = rng.sample(registry.samples, n_samples)

    transform = get_train_transform(img_size)

    out_dir = OUTPUT_DIR / "augmentation_check"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Also build a legend strip
    legend_h = 30
    legend_w = (n_augments + 1) * img_size + n_augments * 2  # images + gaps

    for si, sample in enumerate(samples):
        print(f"[{si+1}/{n_samples}] {sample.uid}")

        # Load original
        img = load_sample_normalized(sample)  # (H, W, 3) float32
        h, w = img.shape[:2]
        anns = parse_yolo_annotations(sample.annotation_path, w, h)
        sem_mask = polygons_to_semantic_mask(anns, h, w)

        # Resize original to target size for display
        img_resized = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(
            sem_mask.astype(np.uint8), (img_size, img_size),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.int32)

        panels = []

        # Original panel
        orig_vis = overlay_semantic_mask(to_uint8(img_resized), mask_resized)
        # Add "Original" label
        cv2.putText(orig_vis, "Original", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(orig_vis, "Original", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        panels.append(orig_vis)

        # Augmented panels
        for ai in range(n_augments):
            transformed = transform(image=img_resized.copy(), mask=mask_resized.copy())
            aug_img = transformed["image"]
            aug_mask = transformed["mask"]

            aug_vis = overlay_semantic_mask(to_uint8(aug_img), aug_mask)
            cv2.putText(aug_vis, f"Aug {ai+1}", (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(aug_vis, f"Aug {ai+1}", (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            panels.append(aug_vis)

        # Arrange: top row = original + aug1-5, bottom row = aug6-10
        gap = 2
        row1 = panels[:6]
        row2 = panels[6:]

        def make_row(panel_list):
            parts = []
            for i, p in enumerate(panel_list):
                if i > 0:
                    parts.append(np.zeros((img_size, gap, 3), dtype=np.uint8))
                parts.append(p)
            return np.concatenate(parts, axis=1)

        top = make_row(row1)
        bot = make_row(row2)

        # Pad bottom row to match top width if needed
        if bot.shape[1] < top.shape[1]:
            pad = np.zeros((img_size, top.shape[1] - bot.shape[1], 3), dtype=np.uint8)
            bot = np.concatenate([bot, pad], axis=1)

        h_gap = np.zeros((gap, top.shape[1], 3), dtype=np.uint8)
        grid = np.concatenate([top, h_gap, bot], axis=0)

        # Add legend at bottom
        legend = np.zeros((legend_h, grid.shape[1], 3), dtype=np.uint8)
        x_off = 10
        for cls_id, cls_name in TARGET_CLASSES.items():
            color = CLASS_COLORS_RGB.get(cls_id, (128, 128, 128))
            cv2.rectangle(legend, (x_off, 5), (x_off + 18, 25), color, -1)
            cv2.putText(legend, cls_name, (x_off + 24, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            x_off += 24 + len(cls_name) * 10 + 15
        grid = np.concatenate([grid, legend], axis=0)

        # Save
        out_path = out_dir / f"{si+1:02d}_{sample.uid}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

    print(f"\nSaved {n_samples} augmentation grids to {out_dir}")


if __name__ == "__main__":
    main()
