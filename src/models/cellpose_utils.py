"""Cellpose data preparation utilities."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from ..annotation_utils import load_sample_annotations
from ..config import OUTPUT_DIR, SampleRecord
from ..preprocessing import load_sample_normalized


def _load_one_sample(sample: 'SampleRecord', img_size: int, target_class: Optional[int],
                     num_classes: int = 4):
    """Load and process a single sample (for parallel loading)."""
    img = load_sample_normalized(sample)
    h, w = img.shape[:2]
    ann = load_sample_annotations(sample, h, w, num_classes=num_classes)
    masks = ann["masks"]
    cls_labels = ann["labels"]

    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)

    label_mask = np.zeros((img_size, img_size), dtype=np.int32)
    instance_id = 1
    for i in range(len(masks)):
        if target_class is not None and cls_labels[i] != target_class:
            continue
        m = cv2.resize(masks[i], (img_size, img_size),
                       interpolation=cv2.INTER_NEAREST)
        label_mask[m > 0] = instance_id
        instance_id += 1

    return img_uint8, label_mask


def prepare_cellpose_data(
    samples: List[SampleRecord],
    output_dir: Path = None,
    img_size: int = 1024,
    target_class: Optional[int] = None,
    num_workers: int = 8,
    num_classes: int = 4,
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
        num_workers: Number of threads for parallel loading.
        num_classes: Number of target classes (4 or 5).

    Returns:
        (images, labels) where images are (H,W,C) uint8 and labels are (H,W) int32.
    """
    images = []
    labels = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_load_one_sample, s, img_size, target_class, num_classes)
                   for s in samples]
        for f in tqdm(futures, desc="Loading data", total=len(futures)):
            img_uint8, label_mask = f.result()
            images.append(img_uint8)
            labels.append(label_mask)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for i, sample in enumerate(samples):
            np.save(output_dir / f"{sample.uid}_img.npy", images[i])
            np.save(output_dir / f"{sample.uid}_masks.npy", labels[i])
        print(f"Saved {len(images)} samples to {output_dir}")

    return images, labels


def preload_cellpose_data(
    samples: List[SampleRecord],
    img_size: int = 1024,
    num_classes: int = 4,
    num_workers: int = 8,
) -> List[Tuple[np.ndarray, List[np.ndarray], List[int]]]:
    """Load images and annotations once, returning per-sample cached data.

    Returns list of (img_uint8, masks_list, labels_list) tuples where
    masks_list contains all instance masks and labels_list their class IDs.
    """
    def _load_one(sample):
        img = load_sample_normalized(sample)
        h, w = img.shape[:2]
        ann = load_sample_annotations(sample, h, w, num_classes=num_classes)
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        resized_masks = []
        for m in ann["masks"]:
            resized_masks.append(
                cv2.resize(m, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
            )
        return img_uint8, resized_masks, list(ann["labels"])

    cache = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_load_one, s) for s in samples]
        for f in tqdm(futures, desc="Loading data", total=len(futures)):
            cache.append(f.result())
    return cache


def build_class_labels(
    cache: List[Tuple[np.ndarray, List[np.ndarray], List[int]]],
    target_class: Optional[int],
    img_size: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Build per-class instance label masks from pre-loaded cache (no disk I/O)."""
    images = []
    labels = []
    for img_uint8, masks_list, cls_labels in cache:
        label_mask = np.zeros((img_size, img_size), dtype=np.int32)
        instance_id = 1
        for m, cls_id in zip(masks_list, cls_labels):
            if target_class is not None and cls_id != target_class:
                continue
            label_mask[m > 0] = instance_id
            instance_id += 1
        images.append(img_uint8)
        labels.append(label_mask)
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
