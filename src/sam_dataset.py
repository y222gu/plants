"""PyTorch Dataset for micro-SAM per-class instance segmentation training.

Creates integer instance masks for a target raw annotation class from YOLO
polygon annotations. Applies albumentations augmentation BEFORE distance
transform computation. Returns (image, label) tuples compatible with
micro_sam.training.train_sam().

Raw annotation classes (one model per class):
    0 = Whole Root       (1 instance per sample, all species)
    1 = Aerenchyma       (many instances, cereals only)
    2 = Outer Endodermis (1 instance, all species)
    3 = Inner Endodermis (1 instance, all species)
    4 = Outer Exodermis  (1 instance, all species)
    5 = Inner Exodermis  (1 instance, all species)

Usage:
    from src.sam_dataset import MicroSAMDataset
    ds = MicroSAMDataset(samples, class_id=1, transform=train_transform)
    img, label = ds[0]  # (3,H,W) float32 [0,255], (4,H,W) float32
    # label channels: [instances, foreground, center_dist, boundary_dist]
"""

from pathlib import Path
from typing import List, Optional

import albumentations as A
import cv2
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import find_boundaries
from torch.utils.data import Dataset

from .annotation_utils import parse_yolo_annotations, polygon_to_mask
from .config import ANNOTATED_CLASSES, SampleRecord
from .preprocessing import load_sample_normalized


def _compute_distance_labels(
    instance_mask: np.ndarray,
    min_size: int = 25,
) -> np.ndarray:
    """Compute 4-channel UNETR target from integer instance mask using scipy.

    Replaces torch_em's PerObjectDistanceTransform which requires vigra
    (no ARM builds). Produces equivalent output using scipy.

    Returns:
        (4, H, W) float32: [instances, foreground, center_dist, boundary_dist]
        Channel order for channels 1-3 matches micro-SAM's expected UNETR
        label format (JointSamTrainer takes y[:,1:] → loss/inference expect
        [foreground, center_dist, boundary_dist]).
    """
    h, w = instance_mask.shape
    out = np.zeros((4, h, w), dtype=np.float32)

    # Relabel consecutive
    ids = np.unique(instance_mask)
    ids = ids[ids > 0]
    relabeled = np.zeros_like(instance_mask)
    for new_id, old_id in enumerate(ids, 1):
        mask = instance_mask == old_id
        if mask.sum() < min_size:
            continue
        relabeled[mask] = new_id

    out[0] = relabeled.astype(np.float32)  # instances (for SAM prompt path)
    out[1] = (relabeled > 0).astype(np.float32)  # foreground

    # Per-object center and boundary distances
    obj_ids = np.unique(relabeled)
    obj_ids = obj_ids[obj_ids > 0]

    for obj_id in obj_ids:
        mask = (relabeled == obj_id).astype(np.uint8)

        # Center distance: EDT from object boundary (peaks at center)
        center_dist = distance_transform_edt(mask)
        if center_dist.max() > 0:
            center_dist = center_dist / center_dist.max()
        out[2][mask > 0] = np.maximum(out[2][mask > 0], center_dist[mask > 0])

        # Boundary distance: EDT from non-boundary pixels
        boundaries = find_boundaries(mask, mode="inner").astype(np.uint8)
        interior = mask & (~boundaries.astype(bool))
        boundary_dist = distance_transform_edt(interior.astype(np.uint8))
        if boundary_dist.max() > 0:
            boundary_dist = boundary_dist / boundary_dist.max()
        out[3][mask > 0] = np.maximum(out[3][mask > 0], boundary_dist[mask > 0])

    return out


def _has_class(annotation_path: Path, class_id: int) -> bool:
    """Check if annotation file contains at least one polygon of class_id."""
    prefix = f"{class_id} "
    with open(annotation_path) as f:
        for line in f:
            if line.startswith(prefix):
                return True
    return False


class MicroSAMDataset(Dataset):
    """Per-class instance segmentation dataset for micro-SAM.

    For a given raw annotation class, creates integer instance masks where
    each polygon gets a unique ID (1, 2, 3, ...). Background = 0.

    Returns (image, label) tuples:
        image: (3, H, W) float32 in [0, 255] (SAM's expected range)
        label: (4, H, W) float32 [instances, foreground, center_dist, boundary_dist]
               when with_segmentation_decoder=True, else (1, H, W) instance IDs.
               Channels 1-3 match micro-SAM's expected UNETR label order.

    Augmentation is applied to both image and instance mask jointly before
    the distance transform is computed, so the UNETR decoder trains on
    augmented geometry.
    """

    def __init__(
        self,
        samples: List[SampleRecord],
        class_id: int,
        transform: Optional[A.Compose] = None,
        img_size: int = 1024,
        with_segmentation_decoder: bool = True,
        min_instance_size: int = 25,
    ):
        self.class_id = class_id
        self.transform = transform
        self.img_size = img_size
        self.with_segmentation_decoder = with_segmentation_decoder

        # Filter to samples that have at least one instance of this class
        self.samples = [s for s in samples
                        if _has_class(s.annotation_path, class_id)]
        print(f"  Class {class_id} ({ANNOTATED_CLASSES[class_id]}): "
              f"{len(self.samples)}/{len(samples)} samples have instances")

        self.min_instance_size = min_instance_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        # Load 3-channel fluorescence image: (H, W, 3) float32 [0, 1]
        img = load_sample_normalized(sample)
        h, w = img.shape[:2]

        # Parse annotations → integer instance mask for target class
        anns = parse_yolo_annotations(sample.annotation_path, w, h)
        instance_mask = self._make_instance_mask(anns, h, w)

        # Resize to model input size
        img = cv2.resize(img, (self.img_size, self.img_size),
                         interpolation=cv2.INTER_LINEAR)
        # cv2.resize needs float for INTER_NEAREST on int32
        instance_mask = cv2.resize(
            instance_mask.astype(np.float32),
            (self.img_size, self.img_size),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.int32)

        # Augmentation (applied BEFORE distance transform so UNETR trains
        # on augmented geometry). albumentations uses INTER_NEAREST for
        # mask targets, preserving integer instance IDs.
        if self.transform is not None:
            augmented = self.transform(image=img, mask=instance_mask)
            img = augmented["image"]
            instance_mask = augmented["mask"]

        # Scale image to [0, 255] for SAM (SAM normalizes internally)
        img_255 = np.clip(img * 255, 0, 255).astype(np.float32)
        img_tensor = torch.from_numpy(img_255.copy()).permute(2, 0, 1)  # (3, H, W)

        # Compute label: distance maps for UNETR decoder, or raw instances
        if self.with_segmentation_decoder:
            label_np = _compute_distance_labels(
                instance_mask, min_size=self.min_instance_size)  # (4, H, W)
            label_tensor = torch.from_numpy(label_np)
        else:
            label_tensor = torch.from_numpy(
                instance_mask[None].astype(np.float32))  # (1, H, W)

        return img_tensor, label_tensor

    def _make_instance_mask(
        self, annotations: list, h: int, w: int,
    ) -> np.ndarray:
        """Create integer instance mask for target class.

        Each polygon of class_id gets a unique ID (1, 2, 3, ...).
        Background = 0. For overlapping polygons of the same class
        (e.g., multiple aerenchyma), later polygons overwrite earlier ones
        at overlapping pixels.
        """
        instance_mask = np.zeros((h, w), dtype=np.int32)
        inst_id = 1
        for ann in annotations:
            if ann["class_id"] == self.class_id:
                mask = polygon_to_mask(ann["polygon"], h, w)
                instance_mask[mask > 0] = inst_id
                inst_id += 1
        return instance_mask
