"""Export dataset to YOLO directory structure for Ultralytics training."""

import shutil
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import yaml

from ..config import ANNOTATED_CLASSES, OUTPUT_DIR, SampleRecord
from ..preprocessing import load_sample_normalized, to_uint8


def export_yolo_dataset(
    splits: Dict[str, List[SampleRecord]],
    output_dir: Path = None,
    img_size: int = 1024,
    num_classes: int = 6,
) -> Path:
    """Export samples to YOLO directory structure.

    Creates:
        output_dir/
            images/train/  images/val/  images/test/
            labels/train/  labels/val/  labels/test/
            data.yaml

    YOLO trains on the original 6 raw annotation classes (endodermis and
    exodermis ring subtraction is done post-inference via yolo_to_target).
    Images are composited to uint8 RGB PNG.

    Returns:
        Path to data.yaml.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR / "yolo_dataset"

    # YOLO uses the original annotated classes (0-5 for all 6 raw classes)
    # Filter out exodermis classes (4, 5) only when explicitly requesting 4 classes
    filter_exo = (num_classes <= 4)

    # Create directory structure
    for subset in ["train", "val", "test"]:
        (output_dir / "images" / subset).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / subset).mkdir(parents=True, exist_ok=True)

    # Export samples
    for subset, samples in splits.items():
        print(f"Exporting {subset}: {len(samples)} samples")
        for sample in samples:
            uid = sample.uid

            # Composite image → uint8 PNG
            img = load_sample_normalized(sample)
            if img_size is not None:
                img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
            img_uint8 = to_uint8(img)
            img_path = output_dir / "images" / subset / f"{uid}.png"
            cv2.imwrite(str(img_path), cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))

            # Copy annotation file, filtering exodermis classes if needed
            label_path = output_dir / "labels" / subset / f"{uid}.txt"
            if filter_exo:
                with open(sample.annotation_path) as fin, open(label_path, "w") as fout:
                    for line in fin:
                        parts = line.strip().split()
                        if len(parts) >= 7 and int(parts[0]) not in (4, 5):
                            fout.write(line)
            else:
                shutil.copy2(sample.annotation_path, label_path)

    # Create data.yaml — YOLO uses the original annotated classes
    if num_classes <= 4:
        yolo_classes = {k: v for k, v in ANNOTATED_CLASSES.items() if k <= 3}
    else:
        yolo_classes = dict(ANNOTATED_CLASSES)  # all 6 raw classes (0-5)
    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(yolo_classes),
        "names": yolo_classes,
    }

    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

    print(f"YOLO dataset exported to {output_dir}")
    print(f"data.yaml: {yaml_path}")
    return yaml_path
