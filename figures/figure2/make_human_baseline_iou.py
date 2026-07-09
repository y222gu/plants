# Figure 2 human baseline

import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
sys.path.insert(0, str(PROJECT))

from src.annotation_utils import parse_yolo_annotations, polygons_to_raw_binary_masks
from src.model_classes import fill_contours, yolo_overlap_false_to_bio7

HH_DIR    = HERE / "hh_variance"
GT_DIR    = HH_DIR / "gt_annotation"
ANN2_DIR  = HH_DIR / "annotator2"
META_CSV  = HH_DIR / "sample_metadata.csv"
OUT_CSV   = HERE / "human_baseline_iou.csv"

BIO_CLASSES_NO_ROOT = ["Epidermis", "Exodermis", "Cortex",
                       "Aerenchyma", "Endodermis", "Vascular"]

SPECIES_RENAME = {"Tomato": "Solanum"}


def load_polygons_to_bio7(txt_path, h, w):
    anns = parse_yolo_annotations(txt_path, w, h)
    raw  = polygons_to_raw_binary_masks(anns, h, w)
    raw  = {k: fill_contours(v) for k, v in raw.items()}
    return yolo_overlap_false_to_bio7(raw, h, w)


def iou(a, b):
    a, b = a.astype(bool), b.astype(bool)
    union = (a | b).sum()
    if union == 0:
        return None
    return float((a & b).sum()) / float(union)


def main():
    meta = {}
    with open(META_CSV) as f:
        for r in csv.DictReader(f):
            meta[r["uid"]] = {
                "species":    r["species"],
                "microscope": r["microscope"],
                "experiment": r["experiment"],
                "height":     int(r["height"]),
                "width":      int(r["width"]),
            }

    gt_uids   = {p.stem for p in GT_DIR.glob("*.txt")}
    ann2_uids = {p.stem for p in ANN2_DIR.glob("*.txt")}
    shared = sorted(gt_uids & ann2_uids)
    print(f"GT: {len(gt_uids)}, annotator2: {len(ann2_uids)}, shared: {len(shared)}")

    rows = []
    for uid in shared:
        m = meta.get(uid)
        if m is None:
            print(f"  [skip] {uid}: no metadata row")
            continue
        h, w = m["height"], m["width"]

        gt   = load_polygons_to_bio7(GT_DIR   / f"{uid}.txt", h, w)
        ann2 = load_polygons_to_bio7(ANN2_DIR / f"{uid}.txt", h, w)

        ious = {cls: iou(gt[cls], ann2[cls]) for cls in BIO_CLASSES_NO_ROOT}
        valid = [v for v in ious.values() if v is not None]
        mean_iou = float(np.mean(valid)) if valid else None

        rows.append({
            "sample_id":  uid,
            "species":    SPECIES_RENAME.get(m["species"], m["species"]),
            "microscope": m["microscope"],
            "experiment": m["experiment"],
            **{f"{cls}_IoU": (f"{v:.4f}" if v is not None else "")
               for cls, v in ious.items()},
            "mean_IoU":   f"{mean_iou:.4f}" if mean_iou is not None else "",
        })
        print(f"  {uid}: mean={mean_iou:.4f}")

    if not rows:
        print("No rows written.")
        return

    fieldnames = ["sample_id", "species", "microscope", "experiment"] + \
                 [f"{c}_IoU" for c in BIO_CLASSES_NO_ROOT] + ["mean_IoU"]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n→ {OUT_CSV} ({len(rows)} samples)")


if __name__ == "__main__":
    main()
