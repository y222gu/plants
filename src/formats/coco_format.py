"""Export dataset to COCO JSON format for Detectron2 training."""

import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from pycocotools import mask as mask_util

from ..annotation_utils import load_sample_annotations
from ..config import OUTPUT_DIR, TARGET_CLASSES, SampleRecord
from ..preprocessing import load_sample_normalized, to_uint8


def _mask_to_rle(binary_mask: np.ndarray) -> dict:
    """Convert binary mask to COCO RLE format."""
    # pycocotools expects Fortran-order uint8
    rle = mask_util.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def export_coco_dataset(
    splits: Dict[str, List[SampleRecord]],
    output_dir: Path = None,
    img_size: int = 1024,
) -> Dict[str, Path]:
    """Export samples to COCO format with images + annotation JSON.

    Uses target classes (with endodermis ring derived from subtraction).
    Masks stored as RLE in annotations.

    Returns:
        Dict mapping subset name → JSON annotation path.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR / "coco_dataset"

    # COCO categories (1-indexed)
    categories = [
        {"id": cls_id + 1, "name": name}
        for cls_id, name in TARGET_CLASSES.items()
    ]

    json_paths = {}

    for subset, samples in splits.items():
        print(f"Exporting {subset}: {len(samples)} samples")
        img_dir = output_dir / subset
        img_dir.mkdir(parents=True, exist_ok=True)

        images_list = []
        annotations_list = []
        ann_id = 1

        for img_id, sample in enumerate(samples, start=1):
            uid = sample.uid

            # Save composite image
            img = load_sample_normalized(sample)
            orig_h, orig_w = img.shape[:2]
            if img_size is not None:
                img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
            img_uint8 = to_uint8(img)
            img_filename = f"{uid}.png"
            cv2.imwrite(
                str(img_dir / img_filename),
                cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR),
            )

            h, w = img_uint8.shape[:2]
            images_list.append({
                "id": img_id,
                "file_name": img_filename,
                "width": w,
                "height": h,
            })

            # Get instance masks (with endodermis subtraction)
            ann_data = load_sample_annotations(sample, orig_h, orig_w)
            masks = ann_data["masks"]
            labels = ann_data["labels"]
            boxes = ann_data["boxes"]

            # Resize masks to target size
            for i in range(len(masks)):
                if img_size is not None:
                    m = cv2.resize(
                        masks[i], (img_size, img_size),
                        interpolation=cv2.INTER_NEAREST,
                    )
                else:
                    m = masks[i]

                rle = _mask_to_rle(m)
                area = int(m.sum())
                # Recompute bbox after resize
                ys, xs = np.where(m > 0)
                if len(xs) == 0:
                    continue
                x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
                bbox = [x1, y1, x2 - x1, y2 - y1]  # COCO format: [x, y, w, h]

                annotations_list.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": int(labels[i]) + 1,  # 1-indexed
                    "segmentation": rle,
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0,
                })
                ann_id += 1

        coco_json = {
            "images": images_list,
            "annotations": annotations_list,
            "categories": categories,
        }

        json_path = output_dir / f"{subset}.json"
        with open(json_path, "w") as f:
            json.dump(coco_json, f)

        json_paths[subset] = json_path
        print(f"  {len(images_list)} images, {len(annotations_list)} annotations")

    print(f"COCO dataset exported to {output_dir}")
    return json_paths
