"""Extract downstream measurements from saved YOLO polygon predictions.

Loads prediction .txt files (YOLO polygon format) instead of running
model inference, so no GPU is required. Works for all model types since
eval_bio7.py saves all predictions in the same 6-class YOLO format.

Usage:
    # Single run
    python downstream_measure_from_predictions.py --predictions-dir output/runs/unet/.../eval_test/predictions --strategy A --out-dir output/runs/unet/.../downstream

    # Batch mode: process all runs missing test-set downstream
    python downstream_measure_from_predictions.py --batch --strategy A

    # With intensity thresholds (raw uint16 scale): for each intensity
    # measurement, restrict the mask to pixels above the threshold before
    # averaging. Default is no thresholding.
    #
    # Global per-channel: applies the same threshold to Exodermis, Endodermis, Vascular.
    python downstream_measure_from_predictions.py --batch --strategy A --tritc-threshold 5000 --fitc-threshold 1000
    #
    # Per-structure: REGION in {Exodermis,Endodermis,Vascular}, CHANNEL in {TRITC,FITC}.
    # Repeatable; overrides the global flags for the specific (region, channel).
    python downstream_measure_from_predictions.py --batch --strategy A --threshold Exodermis:TRITC=5000 --threshold Endodermis:TRITC=4000 --threshold Vascular:FITC=800

GT measurements are recomputed from the annotation polygons every run and saved
to <out_dir>/gt_measurements.csv alongside pred_measurements.csv (no shared
cache file), so a different threshold choice always produces a fresh paired GT.
"""
import csv, cv2, sys, argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.splits import get_split
from src.preprocessing import load_sample_raw
from src.annotation_utils import parse_yolo_annotations, polygon_to_mask
from src.model_classes import get_filled_classes, yolo_overlap_false_to_bio7, BIO_7_NAMES
from downstream_measure_from_model import (
    extract_measurements, MEASUREMENT_COLS,
    parse_threshold_args, get_gt_bio7,
)
from src.model_classes import MODEL_REGISTRY


def load_pred_bio7(pred_path, h, w):
    """Load saved YOLO polygon predictions and convert to bio-7 masks."""
    anns = parse_yolo_annotations(pred_path, w, h)
    if not anns:
        filled = {i: np.zeros((h, w), dtype=np.uint8) for i in range(6)}
    else:
        masks = []
        labels = []
        for ann in anns:
            mask = polygon_to_mask(ann["polygon"], h, w)
            masks.append(mask)
            labels.append(ann["class_id"])
        masks = np.array(masks)
        labels = np.array(labels, dtype=np.int32)
        filled = get_filled_classes(masks, labels, h, w)
    return yolo_overlap_false_to_bio7(filled, h, w)


def process_run(predictions_dir, samples, out_dir, intensity_thresholds=None):
    """Process one run: extract measurements from saved predictions and GT.

    Always recomputes GT measurements from the annotation polygons (no shared
    cache), so a different threshold choice always produces a fresh paired GT.

    Args:
        predictions_dir: Path to eval_test/predictions/ with .txt files
        samples: list of SampleRecord for the test set
        out_dir: output directory for downstream CSVs (gt + pred)
        intensity_thresholds: optional nested dict {region: {channel: float}} —
            restrict mask to pixels above the threshold before computing mean
            intensity for that (region, channel). Default None = no thresholding.

    Returns:
        (pred_csv, gt_csv) — paths to the written CSVs.
    """
    predictions_dir = Path(predictions_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_fields = ["sample_id", "species", "microscope", "experiment"] + MEASUREMENT_COLS

    pred_files = {f.stem: f for f in predictions_dir.glob("*.txt")}
    cfg = MODEL_REGISTRY["yolo_overlap_false"]

    gt_rows = []
    pred_rows = []
    missing = []

    for sample in tqdm(samples, desc=f"Processing {predictions_dir.parent.parent.name}"):
        pred_file = pred_files.get(sample.uid)
        if pred_file is None:
            missing.append(sample.uid)
            continue

        raw_image = load_sample_raw(sample)
        h, w = raw_image.shape[:2]

        base_row = {
            "sample_id": sample.uid,
            "species": sample.species,
            "microscope": sample.microscope,
            "experiment": sample.experiment,
        }

        # Prediction measurements
        pred_bio7 = load_pred_bio7(pred_file, h, w)
        pred_m = extract_measurements(pred_bio7, raw_image, intensity_thresholds)
        pred_row = dict(base_row)
        for k, v in pred_m.items():
            pred_row[k] = round(v, 6)
        pred_rows.append(pred_row)

        # GT measurements (recomputed every run)
        gt_bio7 = get_gt_bio7(sample, cfg)
        gt_m = extract_measurements(gt_bio7, raw_image, intensity_thresholds)
        gt_row = dict(base_row)
        for k, v in gt_m.items():
            gt_row[k] = round(v, 6)
        gt_rows.append(gt_row)

    if missing:
        print(f"  Warning: {len(missing)} samples missing predictions")

    pred_csv = out_dir / "pred_measurements.csv"
    with open(pred_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(pred_rows)
    print(f"  Saved {pred_csv} ({len(pred_rows)} samples)")

    gt_csv = out_dir / "gt_measurements.csv"
    with open(gt_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(gt_rows)
    print(f"  Saved {gt_csv} ({len(gt_rows)} samples)")

    return pred_csv, gt_csv


def find_all_runs(base_dir):
    """Find all runs and check if they need test-set downstream."""
    base = Path(base_dir)
    runs = []
    for pred_dir in sorted(base.rglob("eval_test/predictions")):
        run_dir = pred_dir.parent.parent  # go up from eval_test/predictions
        downstream_dir = run_dir / "downstream"
        pred_csv = downstream_dir / "pred_measurements.csv"

        # Check if downstream exists and has test-set data (>36 lines = not just Zeiss)
        needs_downstream = True
        if pred_csv.exists():
            with open(pred_csv) as f:
                line_count = sum(1 for _ in f)
            if line_count > 36:
                needs_downstream = False

        txt_count = len(list(pred_dir.glob("*.txt")))
        runs.append({
            "run_dir": run_dir,
            "pred_dir": pred_dir,
            "downstream_dir": downstream_dir,
            "txt_count": txt_count,
            "needs_downstream": needs_downstream,
        })
    return runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-dir", help="Path to eval_test/predictions/")
    parser.add_argument("--strategy", default="A")
    parser.add_argument("--out-dir", help="Output directory")
    parser.add_argument("--batch", action="store_true",
                        help="Process all runs in lambda_output missing test-set downstream")
    parser.add_argument("--base-dir",
                        default="lambda_output/runs",
                        help="Base directory for batch mode")
    parser.add_argument("--plot", action="store_true",
                        help="Generate comparison plots after extracting measurements")
    parser.add_argument("--tritc-threshold", type=str, default=None,
                        metavar="RANGE",
                        help="Global TRITC keep-range applied to Exodermis, Endodermis, "
                             "and Vascular intensity measurements (raw uint16 scale). "
                             "Format: LOW-HIGH where each side is a number or 'min'/'max'. "
                             "Examples: 4000-5000, min-5000, 5000-max. "
                             "Inclusive on both ends. Override per structure with --threshold.")
    parser.add_argument("--fitc-threshold", type=str, default=None,
                        metavar="RANGE",
                        help="Global FITC keep-range; same format as --tritc-threshold.")
    parser.add_argument("--threshold", action="append", default=[],
                        metavar="REGION:CHANNEL=RANGE",
                        help="Per-structure keep-range, repeatable. REGION in "
                             "{Exodermis,Endodermis,Vascular}, CHANNEL in {TRITC,FITC}. "
                             "RANGE same syntax as --tritc-threshold. Overrides the "
                             "global flag for that (region, channel). "
                             "Example: --threshold Exodermis:TRITC=4000-5000")
    args = parser.parse_args()

    intensity_thresholds = parse_threshold_args(
        args.tritc_threshold, args.fitc_threshold, args.threshold)
    if intensity_thresholds:
        print(f"Intensity thresholds: {intensity_thresholds}")

    split = get_split(strategy=args.strategy)
    samples = split["test"]
    print(f"Test set: strategy {args.strategy}, {len(samples)} samples")

    if args.batch:
        runs = find_all_runs(args.base_dir)
        need_runs = [r for r in runs if r["needs_downstream"]]
        print(f"Found {len(runs)} total runs, {len(need_runs)} need test-set downstream")

        for i, run in enumerate(need_runs):
            print(f"\n[{i+1}/{len(need_runs)}] {run['run_dir']}")
            pred_csv, gt_csv = process_run(
                run["pred_dir"], samples, run["downstream_dir"],
                intensity_thresholds=intensity_thresholds,
            )
            if args.plot:
                generate_plot(str(gt_csv), str(pred_csv), str(run["downstream_dir"]))

    elif args.predictions_dir:
        out_dir = args.out_dir or str(Path(args.predictions_dir).parent.parent / "downstream")
        pred_csv, gt_csv = process_run(
            args.predictions_dir, samples, out_dir,
            intensity_thresholds=intensity_thresholds,
        )
        if args.plot:
            generate_plot(str(gt_csv), str(pred_csv), out_dir)
    else:
        parser.error("Specify --predictions-dir or --batch")


def generate_plot(gt_csv, pred_csv, out_dir):
    """Run downstream_plot_correlations.py to generate comparison plots."""
    import subprocess
    cmd = [sys.executable, "downstream_plot_correlations.py",
           "--gt", gt_csv, "--pred", pred_csv, "--out-dir", out_dir]
    print(f"  Generating plots...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Plot generation failed: {result.stderr[:200]}")
    else:
        print(f"  Plots saved to {out_dir}")


if __name__ == "__main__":
    main()
