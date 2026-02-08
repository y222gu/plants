"""PyTorch Dataset for U-Net semantic segmentation training."""

from pathlib import Path
from typing import Callable, Dict, List, Optional

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..annotation_utils import parse_yolo_annotations, polygons_to_semantic_mask
from ..config import SampleRecord
from ..preprocessing import load_sample_normalized


class UNetDataset(Dataset):
    """Dataset for semantic segmentation with U-Net / U-Net++.

    Returns (image, mask) pairs where:
        image: (3, H, W) float32 tensor
        mask: (H, W) int64 tensor (0=bg, 1-4=classes)
    """

    def __init__(
        self,
        samples: List[SampleRecord],
        transform: Optional[A.Compose] = None,
        img_size: int = 1024,
    ):
        self.samples = samples
        self.transform = transform
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load image
        img = load_sample_normalized(sample)  # (H, W, 3) float32 [0,1]
        h, w = img.shape[:2]

        # Load semantic mask
        anns = parse_yolo_annotations(sample.annotation_path, w, h)
        sem_mask = polygons_to_semantic_mask(anns, h, w)  # (H, W) int32

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

        return {"image": img_tensor, "mask": mask_tensor, "uid": sample.uid}
