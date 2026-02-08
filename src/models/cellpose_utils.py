"""Cellpose data preparation utilities."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..annotation_utils import load_sample_annotations
from ..config import OUTPUT_DIR, TARGET_CLASSES, SampleRecord
from ..preprocessing import load_sample_normalized


def prepare_cellpose_data(
    samples: List[SampleRecord],
    output_dir: Path = None,
    img_size: int = 1024,
    target_class: Optional[int] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Prepare images and label masks for Cellpose training.

    Cellpose expects integer-labeled instance masks where each instance
    has a unique integer ID. Background = 0.

    Args:
        samples: List of SampleRecord.
        output_dir: If provided, save images and masks to disk.
        img_size: Target image size.
        target_class: If specified, only include instances of this class.
            If None, include all classes (each instance gets unique ID).

    Returns:
        (images, labels) where images are (H,W,C) uint8 and labels are (H,W) int32.
    """
    images = []
    labels = []

    for sample in samples:
        img = load_sample_normalized(sample)
        h, w = img.shape[:2]
        ann = load_sample_annotations(sample, h, w)
        masks = ann["masks"]
        cls_labels = ann["labels"]

        # Resize
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)

        # Build integer-labeled mask
        label_mask = np.zeros((img_size, img_size), dtype=np.int32)
        instance_id = 1

        for i in range(len(masks)):
            if target_class is not None and cls_labels[i] != target_class:
                continue
            m = cv2.resize(masks[i], (img_size, img_size),
                           interpolation=cv2.INTER_NEAREST)
            label_mask[m > 0] = instance_id
            instance_id += 1

        images.append(img_uint8)
        labels.append(label_mask)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for i, sample in enumerate(samples):
            np.save(output_dir / f"{sample.uid}_img.npy", images[i])
            np.save(output_dir / f"{sample.uid}_masks.npy", labels[i])
        print(f"Saved {len(images)} samples to {output_dir}")

    return images, labels


def export_cellpose_dataset(
    splits: Dict[str, List[SampleRecord]],
    output_dir: Path = None,
    img_size: int = 1024,
    target_class: Optional[int] = None,
) -> Path:
    """Export full dataset for Cellpose training.

    Args:
        splits: Dict with train/val/test sample lists.
        output_dir: Output directory.
        img_size: Target image size.
        target_class: If specified, train per-class model.

    Returns:
        Output directory path.
    """
    if output_dir is None:
        class_suffix = f"_class{target_class}" if target_class is not None else ""
        output_dir = OUTPUT_DIR / f"cellpose_dataset{class_suffix}"

    for subset, samples in splits.items():
        print(f"Exporting {subset}: {len(samples)} samples")
        subset_dir = output_dir / subset
        prepare_cellpose_data(samples, subset_dir, img_size, target_class)

    print(f"Cellpose dataset exported to {output_dir}")
    return output_dir
