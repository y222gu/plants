"""Export dataset as mask arrays (NPZ) for U-Net and other mask-based models."""

from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from ..annotation_utils import (
    load_sample_annotations,
    parse_yolo_annotations,
    polygons_to_semantic_mask,
)
from ..config import OUTPUT_DIR, SampleRecord
from ..preprocessing import load_sample_normalized, to_uint8


def export_mask_dataset(
    splits: Dict[str, List[SampleRecord]],
    output_dir: Path = None,
    img_size: int = 1024,
    include_semantic: bool = True,
    include_instance: bool = True,
) -> Path:
    """Export samples as images + NPZ mask files.

    Each NPZ contains:
        - semantic_mask: (H, W) int32, 0=bg, 1-4=classes
        - instance_masks: (N, H, W) uint8 binary
        - instance_labels: (N,) int32 target class IDs

    Returns:
        Output directory path.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR / "mask_dataset"

    for subset, samples in splits.items():
        print(f"Exporting {subset}: {len(samples)} samples")
        img_dir = output_dir / subset / "images"
        mask_dir = output_dir / subset / "masks"
        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        for sample in samples:
            uid = sample.uid

            # Load and save image
            img = load_sample_normalized(sample)
            orig_h, orig_w = img.shape[:2]

            if img_size is not None:
                img_resized = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
            else:
                img_resized = img
            img_uint8 = to_uint8(img_resized)
            cv2.imwrite(
                str(img_dir / f"{uid}.png"),
                cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR),
            )

            # Build masks
            npz_data = {}

            if include_semantic:
                anns = parse_yolo_annotations(sample.annotation_path, orig_w, orig_h)
                sem = polygons_to_semantic_mask(anns, orig_h, orig_w)
                if img_size is not None:
                    sem = cv2.resize(sem.astype(np.uint8), (img_size, img_size),
                                     interpolation=cv2.INTER_NEAREST).astype(np.int32)
                npz_data["semantic_mask"] = sem

            if include_instance:
                ann_data = load_sample_annotations(sample, orig_h, orig_w)
                masks = ann_data["masks"]
                labels = ann_data["labels"]
                if img_size is not None and len(masks) > 0:
                    resized_masks = np.zeros((len(masks), img_size, img_size), dtype=np.uint8)
                    for i in range(len(masks)):
                        resized_masks[i] = cv2.resize(
                            masks[i], (img_size, img_size),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    masks = resized_masks
                npz_data["instance_masks"] = masks
                npz_data["instance_labels"] = labels

            np.savez_compressed(mask_dir / f"{uid}.npz", **npz_data)

    print(f"Mask dataset exported to {output_dir}")
    return output_dir
