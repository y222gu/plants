"""Dataset for SAM (Segment Anything Model) fine-tuning with prompt generation."""

import random
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..annotation_utils import load_sample_annotations, masks_to_boxes, parse_yolo_annotations
from ..config import SampleRecord
from ..preprocessing import load_sample_normalized


class _LRUCache:
    """Simple LRU cache that is picklable (unlike functools.lru_cache)."""

    def __init__(self, maxsize: int = 64):
        self.maxsize = maxsize
        self._cache: OrderedDict = OrderedDict()

    def get(self, key):
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key, value):
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self.maxsize:
                self._cache.popitem(last=False)
            self._cache[key] = value


class SAMDataset(Dataset):
    """Dataset for SAM fine-tuning.

    Each item returns a random instance from a random sample,
    with point and/or box prompts generated from the GT mask.

    Uses num_workers=0 in DataLoader for in-process caching.
    """

    def __init__(
        self,
        samples: List[SampleRecord],
        img_size: int = 1024,
        points_per_mask: int = 3,
        use_box_prompt: bool = True,
        use_point_prompt: bool = True,
        cache_size: int = 64,
    ):
        self.samples = samples
        self.img_size = img_size
        self.points_per_mask = points_per_mask
        self.use_box_prompt = use_box_prompt
        self.use_point_prompt = use_point_prompt
        self._cache = _LRUCache(maxsize=cache_size)

        # Build index by parsing annotation text files only (no image loading)
        self._index: List[Tuple[int, int]] = []
        for si, sample in enumerate(samples):
            anns = parse_yolo_annotations(sample.annotation_path, 1, 1)
            # Count instances: class 0 and 1 directly, plus endodermis ring + vascular
            n_direct = sum(1 for a in anns if a["class_id"] in (0, 1))
            has_outer = any(a["class_id"] == 2 for a in anns)
            has_inner = any(a["class_id"] == 3 for a in anns)
            n_derived = (1 if (has_outer and has_inner) else 0) + (1 if has_inner else 0)
            n_instances = n_direct + n_derived
            for ii in range(n_instances):
                self._index.append((si, ii))

    def _load_sample(self, sample_idx: int):
        cached = self._cache.get(sample_idx)
        if cached is not None:
            return cached
        sample = self.samples[sample_idx]
        img = load_sample_normalized(sample)
        h, w = img.shape[:2]
        ann = load_sample_annotations(sample, h, w)
        self._cache.put(sample_idx, (img, ann))
        return img, ann

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_idx, instance_idx = self._index[idx]

        # Load image and annotations (cached)
        img, ann = self._load_sample(sample_idx)
        h, w = img.shape[:2]

        gt_mask = ann["masks"][instance_idx]  # (H, W)
        class_id = ann["labels"][instance_idx]

        # Resize to SAM input size
        scale_x = self.img_size / w
        scale_y = self.img_size / h
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        gt_mask = cv2.resize(gt_mask, (self.img_size, self.img_size),
                             interpolation=cv2.INTER_NEAREST)

        result = {
            "image": torch.from_numpy(img.copy()).permute(2, 0, 1).float(),
            "gt_mask": torch.from_numpy(gt_mask.copy()).float(),
            "class_id": torch.tensor(class_id, dtype=torch.long),
        }

        # Generate point prompts: sample points from mask interior
        if self.use_point_prompt:
            ys, xs = np.where(gt_mask > 0)
            if len(xs) >= self.points_per_mask:
                indices = np.random.choice(len(xs), self.points_per_mask, replace=False)
                points = np.stack([xs[indices], ys[indices]], axis=-1)  # (K, 2) xy format
            else:
                points = np.stack([xs, ys], axis=-1)
                # Pad if needed
                while len(points) < self.points_per_mask:
                    points = np.concatenate([points, points[:1]])
            point_labels = np.ones(len(points), dtype=np.float32)  # 1 = foreground
            result["point_coords"] = torch.from_numpy(points).float()
            result["point_labels"] = torch.from_numpy(point_labels)

        # Generate box prompt: bounding box with small random jitter
        if self.use_box_prompt:
            ys, xs = np.where(gt_mask > 0)
            if len(xs) > 0:
                x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
                # Add small random jitter (up to 5% of box size)
                bw, bh = x2 - x1, y2 - y1
                jitter = 0.05
                x1 = max(0, x1 - random.uniform(0, jitter * bw))
                y1 = max(0, y1 - random.uniform(0, jitter * bh))
                x2 = min(self.img_size, x2 + random.uniform(0, jitter * bw))
                y2 = min(self.img_size, y2 + random.uniform(0, jitter * bh))
                box = np.array([x1, y1, x2, y2], dtype=np.float32)
            else:
                box = np.zeros(4, dtype=np.float32)
            result["box"] = torch.from_numpy(box)

        return result
