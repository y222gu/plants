"""PyTorch Dataset for U-Net semantic/multilabel segmentation training."""

from pathlib import Path
from typing import Callable, Dict, List, Optional

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..annotation_utils import (
    parse_yolo_annotations,
    polygons_to_multilabel_mask,
    polygons_to_semantic_mask,
)
from ..config import SPECIES_VALID_CLASSES, SampleRecord
from ..preprocessing import load_sample_normalized


class UNetDataset(Dataset):
    """Dataset for segmentation with U-Net / U-Net++.

    Supports two modes:
        "semantic": Returns (H, W) int64 mask (0=bg, 1-N=classes). Mutually exclusive.
        "multilabel": Returns (C, H, W) float32 mask. Independent binary channels (sigmoid).

    Args:
        samples: List of SampleRecord.
        transform: Albumentations Compose pipeline.
        img_size: Target image size (square).
        mode: "semantic" or "multilabel".
        num_classes: Number of target classes (4 or 5).
        mask_missing: If True, return a per-channel validity mask for masked loss.
    """

    def __init__(
        self,
        samples: List[SampleRecord],
        transform: Optional[A.Compose] = None,
        img_size: int = 1024,
        mode: str = "semantic",
        num_classes: int = 4,
        mask_missing: bool = False,
    ):
        assert mode in ("semantic", "multilabel"), f"Unknown mode: {mode}"
        self.samples = samples
        self.transform = transform
        self.img_size = img_size
        self.mode = mode
        self.num_classes = num_classes
        self.mask_missing = mask_missing

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load image
        img = load_sample_normalized(sample)  # (H, W, 3) float32 [0,1]
        h, w = img.shape[:2]

        # Load annotations
        anns = parse_yolo_annotations(sample.annotation_path, w, h)

        if self.mode == "semantic":
            result = self._get_semantic(img, anns, h, w, sample.uid)
        else:
            result = self._get_multilabel(img, anns, h, w, sample.uid)

        # Add per-channel validity mask when mask_missing is enabled
        if self.mask_missing:
            valid_classes = SPECIES_VALID_CLASSES.get(sample.species, set(range(self.num_classes)))
            valid_mask = torch.zeros(self.num_classes, dtype=torch.float32)
            for c in range(self.num_classes):
                if c in valid_classes:
                    valid_mask[c] = 1.0
            result["valid_mask"] = valid_mask

        return result

    def _get_semantic(self, img, anns, h, w, uid):
        sem_mask = polygons_to_semantic_mask(anns, h, w, num_classes=self.num_classes)  # (H, W) int32

        # Resize to target
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        sem_mask = cv2.resize(
            sem_mask.astype(np.uint8), (self.img_size, self.img_size),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.int32)

        # Apply augmentation
        if self.transform is not None:
            transformed = self.transform(image=img, mask=sem_mask)
            img = transformed["image"]
            sem_mask = transformed["mask"]

        # Convert to tensors
        img_tensor = torch.from_numpy(img.copy()).permute(2, 0, 1).float()  # (3, H, W)
        mask_tensor = torch.from_numpy(sem_mask.copy()).long()               # (H, W)

        return {"image": img_tensor, "mask": mask_tensor, "uid": uid}

    def _get_multilabel(self, img, anns, h, w, uid):
        nc = self.num_classes
        ml_mask = polygons_to_multilabel_mask(anns, h, w, num_classes=nc)  # (C, H, W) float32

        # Resize to target
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        # Resize each channel of multilabel mask with nearest neighbor
        resized_mask = np.zeros((nc, self.img_size, self.img_size), dtype=np.float32)
        for c in range(nc):
            resized_mask[c] = cv2.resize(
                ml_mask[c], (self.img_size, self.img_size),
                interpolation=cv2.INTER_NEAREST,
            )

        # Apply augmentation: albumentations handles (H, W, C) masks
        if self.transform is not None:
            # Transpose mask to (H, W, C) for albumentations
            mask_hwc = resized_mask.transpose(1, 2, 0)  # (H, W, C)
            transformed = self.transform(image=img, mask=mask_hwc)
            img = transformed["image"]
            mask_hwc = transformed["mask"]
            resized_mask = mask_hwc.transpose(2, 0, 1)  # (C, H, W)

        # Convert to tensors
        img_tensor = torch.from_numpy(img.copy()).permute(2, 0, 1).float()    # (3, H, W)
        mask_tensor = torch.from_numpy(resized_mask.copy()).float()            # (C, H, W)

        return {"image": img_tensor, "mask": mask_tensor, "uid": uid}
