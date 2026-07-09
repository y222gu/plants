# Figure 2 b/c/d/e/f data prep

import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
METRICS_DIR = HERE / "eval_metrics"


MODELS = {
    "DINOv3 + DPT-meta":      "dinov3_dpt",
    "DINOv2 + DPT-meta":      "dinov2_dpt",
    "ResNet34 + UNet++ (IN)": "resnet34_unetpp",
    "DINOv2 + MS-Linear":     "dinov2_mslinear",
    "DINOv3 + SegDINO-MLP":   "dinov3_segdino",
    "ResNet50 + UNet++ (IN)": "resnet50_unetpp",
    "MicroSAM + UNETR":       "microsam_unetr",
    "YOLO26m-seg (COCO)":     "yolo26m",
}

BIO_CLASSES_NO_ROOT = [
    "Epidermis", "Exodermis", "Cortex",
    "Aerenchyma", "Endodermis", "Vascular",
]

SPLITS = ["test", "zero-shot"]

SPECIES_RENAME = {"Tomato": "Solanum"}


def _floats(r, classes, metric):
    vals = []
    for c in classes:
        s = r.get(f"{c}_{metric}", "")
        if s == "" or s.lower() == "nan":
            continue
        vals.append(float(s))
    return vals


def load_per_sample(metrics_csv):
    out = []
    with open(metrics_csv) as f:
        for r in csv.DictReader(f):
            row = {
                "sample_id":  r["sample_id"],
                "species":    SPECIES_RENAME.get(r["species"], r["species"]),
                "microscope": r["microscope"],
                "experiment": r.get("experiment", ""),
            }
            for c in BIO_CLASSES_NO_ROOT:
                row[f"{c}_IoU"]  = r.get(f"{c}_IoU",  "")
                row[f"{c}_Dice"] = r.get(f"{c}_Dice", "")
            ious  = _floats(r, BIO_CLASSES_NO_ROOT, "IoU")
            dices = _floats(r, BIO_CLASSES_NO_ROOT, "Dice")
            if not ious:
                continue
            row["mean_IoU"]  = sum(ious)  / len(ious)
            row["mean_Dice"] = sum(dices) / len(dices) if dices else None
            out.append(row)
    return out


def main():
    by_model_split = {}
    for label, key in MODELS.items():
        for split in SPLITS:
            metrics = METRICS_DIR / f"{key}_{split}.csv"
            if not metrics.exists():
                print(f"  [skip] {label} / {split}: missing {metrics}")
                continue
            rows = load_per_sample(metrics)
            by_model_split[(label, split)] = rows
            print(f"  {label:25s} {split:10s}  n={len(rows)}")

    headline = "DINOv3 + DPT-meta"

    iou_path = HERE / "per_sample_iou.csv"
    iou_fields = ["sample_id", "species", "microscope", "experiment"]
    for c in BIO_CLASSES_NO_ROOT:
        iou_fields += [f"{c}_IoU", f"{c}_Dice"]
    iou_fields += ["mean_IoU", "mean_Dice", "split"]
    with open(iou_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=iou_fields)
        w.writeheader()
        for split in SPLITS:
            for r in by_model_split.get((headline, split), []):
                row = {k: r.get(k, "") for k in iou_fields if k not in ("mean_IoU", "mean_Dice", "split")}
                row["mean_IoU"]  = f"{r['mean_IoU']:.4f}"
                row["mean_Dice"] = f"{r['mean_Dice']:.4f}" if r["mean_Dice"] is not None else ""
                row["split"]     = split
                w.writerow(row)
    print(f"\n→ {iou_path}")

    all_path = HERE / "per_sample_miou_all_models.csv"
    all_fields = ["model", "split", "sample_id", "species"] \
        + [f"{c}_IoU" for c in BIO_CLASSES_NO_ROOT] + ["mean_IoU"]
    with open(all_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_fields)
        w.writeheader()
        for label in MODELS:
            for split in SPLITS:
                for r in by_model_split.get((label, split), []):
                    row = {
                        "model":     label,
                        "split":     split,
                        "sample_id": r["sample_id"],
                        "species":   r["species"],
                        "mean_IoU":  f"{r['mean_IoU']:.4f}",
                    }
                    for c in BIO_CLASSES_NO_ROOT:
                        row[f"{c}_IoU"] = r.get(f"{c}_IoU", "")
                    w.writerow(row)
    print(f"→ {all_path}")

    by_mic_path = HERE / "per_sample_miou_by_microscope.csv"
    with open(by_mic_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["microscope", "split", "sample_id", "species", "mean_IoU"])
        w.writeheader()
        for split in SPLITS:
            for r in by_model_split.get((headline, split), []):
                w.writerow({
                    "microscope": r["microscope"],
                    "split":      split,
                    "sample_id":  r["sample_id"],
                    "species":    r["species"],
                    "mean_IoU":   f"{r['mean_IoU']:.4f}",
                })
    print(f"→ {by_mic_path}")


if __name__ == "__main__":
    main()
