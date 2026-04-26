"""Extract downstream biological measurements from any mask directory.

Source-agnostic: --mask-dir can point at GT annotations OR at a model's
predictions (or any other directory of YOLO polygon .txt files in the project's
6-class raw format). The script walks --image-dir for samples laid out as
{Species}/{Microscope}/{Exp}/{Sample}/{Sample}_{DAPI|FITC|TRITC}.tif and pairs
each with a mask file named {Species}_{Microscope}_{Exp}_{Sample}.txt inside
--mask-dir. There is no GT vs prediction comparison — the output is a single
CSV of measurements.

Usage:
    # GT
    python downstream_measure_from_masks.py --image-dir data/image --mask-dir data/annotation --out-csv output/downstream/gt_all.csv

    # Predictions from a trained model
    python downstream_measure_from_masks.py --image-dir data/image --mask-dir output/runs/dinov3_dpt/.../eval/test/predictions --out-csv output/downstream/pred_all.csv

    # With per-structure keep-ranges (raw uint16 scale): pixels in [low, high] are
    # kept (inclusive). Required form: LOW-HIGH (e.g. 4000-5000, min-5000, 5000-max).
    python downstream_measure_from_masks.py --image-dir data/image --mask-dir data/annotation --out-csv output/downstream/gt_thr.csv --threshold Exodermis:TRITC=4000-5000 --threshold Vascular:FITC=800-max --save-vis-dir output/downstream/vis
"""
import csv, argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.config import SampleRecord
from src.preprocessing import load_sample_raw
from src.annotation_utils import parse_yolo_annotations, polygon_to_mask
from src.model_classes import get_filled_classes, yolo_overlap_false_to_bio7
from downstream_measure_from_model import (
    extract_measurements, MEASUREMENT_COLS, parse_threshold_args,
    INTENSITY_REGIONS, INTENSITY_CHANNELS, CHANNEL_IDX, format_range,
)


CHANNELS = ("TRITC", "FITC", "DAPI")

# Per-region colours for outline overlays (RGB)
REGION_COLORS_RGB = {
    "Exodermis":  (0, 255, 255),   # Cyan
    "Endodermis": (0, 255, 0),     # Lime
    "Vascular":   (255, 0, 0),     # Red
}


def discover_samples(image_dir: Path):
    """Walk image_dir for {Species}/{Microscope}/{Exp}/{Sample}/{Sample}_TRITC.tif.

    Yields SampleRecord(annotation_path=None) for every leaf folder that has
    all three channel TIFs. The script's pairing with masks is done separately
    by matching on UID.
    """
    image_dir = image_dir.resolve()
    samples = []
    for tritc in image_dir.rglob("*_TRITC.tif"):
        leaf = tritc.parent
        try:
            rel = leaf.relative_to(image_dir).parts
        except ValueError:
            continue
        if len(rel) != 4:
            continue
        species, microscope, experiment, sample_name = rel
        if not all((leaf / f"{sample_name}_{ch}.tif").exists() for ch in CHANNELS):
            continue
        samples.append(SampleRecord(
            species=species,
            microscope=microscope,
            experiment=experiment,
            sample_name=sample_name,
            image_dir=leaf,
            annotation_path=None,
        ))
    samples.sort(key=lambda s: s.uid)
    return samples


def _normalize_to_uint8(channel_2d):
    """Percentile-normalize a 2D channel to uint8 for display."""
    lo, hi = np.percentile(channel_2d, [1, 99.5])
    if hi <= lo:
        return np.zeros(channel_2d.shape, dtype=np.uint8)
    img = np.clip((channel_2d - lo) / (hi - lo), 0, 1)
    return (img * 255).astype(np.uint8)


def _blend_fill(img_rgb, mask, color, alpha=0.55):
    """Blend a flat-coloured fill into img_rgb where mask is True."""
    if not mask.any():
        return
    overlay = img_rgb.copy()
    overlay[mask > 0] = color
    np.copyto(img_rgb, cv2.addWeighted(overlay, alpha, img_rgb, 1 - alpha, 0))


def _label(img, text):
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (255, 255, 255), 2, cv2.LINE_AA)


def save_threshold_diagnostic(out_dir: Path, uid: str, raw_image, bio7,
                               intensity_thresholds: dict):
    """Save per-channel diagnostic PNG for a single sample.

    For each channel with at least one region keep-range, writes one composite
    PNG `{uid}_{CHANNEL}.png` with four panels side by side:
        1) Mask before threshold overlay — original masks blended over the channel image
        2) Mask before threshold         — original masks alone (on black)
        3) Mask after threshold          — post-threshold masks alone (on black)
        4) Mask after threshold overlay  — post-threshold masks blended over the channel image
    Channels without any range are skipped.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    for channel in INTENSITY_CHANNELS:
        regions_thr = {
            r: intensity_thresholds.get(r, {}).get(channel)
            for r in INTENSITY_REGIONS
            if intensity_thresholds.get(r, {}).get(channel) is not None
        }
        if not regions_thr:
            continue

        ch_idx = CHANNEL_IDX[channel]
        ch_full = raw_image[..., ch_idx]
        ch_disp = _normalize_to_uint8(ch_full)
        base = cv2.cvtColor(ch_disp, cv2.COLOR_GRAY2RGB)
        h, w = ch_full.shape
        black = np.zeros((h, w, 3), dtype=np.uint8)

        p1 = base.copy()      # masks over image
        p2 = black.copy()     # masks alone
        p3 = black.copy()     # thresholded masks alone
        p4 = base.copy()      # thresholded masks over image

        for region, rng in regions_thr.items():
            mask = (bio7[region] > 0)
            if mask.shape != ch_full.shape:
                mask = cv2.resize(mask.astype(np.uint8),
                                  (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            low, high = rng
            kept = mask.copy()
            if low is not None:
                kept &= (ch_full >= low)
            if high is not None:
                kept &= (ch_full <= high)
            color = REGION_COLORS_RGB[region]
            _blend_fill(p1, mask, color)
            p2[mask] = color
            p3[kept] = color
            _blend_fill(p4, kept, color)

        _label(p1, "Mask before threshold overlay")
        _label(p2, "Mask before threshold")
        _label(p3, "Mask after threshold")
        _label(p4, "Mask after threshold overlay")

        gap = np.zeros((h, 8, 3), dtype=np.uint8)
        panels = np.hstack([p1, gap, p2, gap, p3, gap, p4])
        thr_str = ", ".join(f"{r} in [{format_range(rng)}]"
                             for r, rng in regions_thr.items())
        title_h = 40
        title_bar = np.zeros((title_h, panels.shape[1], 3), dtype=np.uint8)
        cv2.putText(title_bar, f"{uid}  |  {channel}: {thr_str}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
        composite = np.vstack([title_bar, panels])
        cv2.imwrite(str(out_dir / f"{uid}_{channel}.png"),
                    cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))


def load_mask_bio7(mask_path: Path, h: int, w: int):
    """Load a YOLO polygon .txt file and convert to bio-7 masks."""
    anns = parse_yolo_annotations(mask_path, w, h)
    if not anns:
        filled = {i: np.zeros((h, w), dtype=np.uint8) for i in range(6)}
    else:
        masks, labels = [], []
        for ann in anns:
            masks.append(polygon_to_mask(ann["polygon"], h, w))
            labels.append(ann["class_id"])
        filled = get_filled_classes(
            np.array(masks), np.array(labels, dtype=np.int32), h, w,
        )
    return yolo_overlap_false_to_bio7(filled, h, w)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image-dir", required=True, type=Path,
                        help="Root image directory laid out as "
                             "{Species}/{Microscope}/{Exp}/{Sample}/*_{TRITC,FITC,DAPI}.tif "
                             "(typically data/image).")
    parser.add_argument("--mask-dir", required=True, type=Path,
                        help="Directory of YOLO polygon .txt files named "
                             "{Species}_{Microscope}_{Exp}_{Sample}.txt. "
                             "May be GT (data/annotation) or model predictions.")
    parser.add_argument("--out-csv", required=True, type=Path,
                        help="Output measurements CSV path.")
    parser.add_argument("--save-vis-dir", type=Path, default=None,
                        help="When thresholds are set, save per-sample diagnostic PNGs "
                             "into this directory: {uid}_{channel}.png — one composite "
                             "per channel with any region threshold. Each PNG has four "
                             "side-by-side panels: Mask before threshold overlay, "
                             "Mask before threshold, Mask after threshold, "
                             "Mask after threshold overlay.")
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

    if not args.image_dir.exists():
        parser.error(f"--image-dir does not exist: {args.image_dir}")
    if not args.mask_dir.exists():
        parser.error(f"--mask-dir does not exist: {args.mask_dir}")

    intensity_thresholds = parse_threshold_args(
        args.tritc_threshold, args.fitc_threshold, args.threshold)
    if intensity_thresholds:
        print(f"Intensity thresholds: {intensity_thresholds}")

    save_vis = args.save_vis_dir is not None and bool(intensity_thresholds)
    if args.save_vis_dir is not None and not intensity_thresholds:
        print("Warning: --save-vis-dir set but no thresholds given; nothing to visualize.")
    if save_vis:
        args.save_vis_dir.mkdir(parents=True, exist_ok=True)
        print(f"Diagnostic PNGs → {args.save_vis_dir}")

    samples = discover_samples(args.image_dir)
    print(f"Discovered {len(samples)} samples under {args.image_dir}")

    mask_files = {f.stem: f for f in args.mask_dir.glob("*.txt")}
    print(f"Found {len(mask_files)} mask files under {args.mask_dir}")

    csv_fields = ["sample_id", "species", "microscope", "experiment"] + MEASUREMENT_COLS
    rows = []
    missing = []

    for sample in tqdm(samples, desc="Extracting measurements"):
        mask_file = mask_files.get(sample.uid)
        if mask_file is None:
            missing.append(sample.uid)
            continue

        raw_image = load_sample_raw(sample)
        h, w = raw_image.shape[:2]
        bio7 = load_mask_bio7(mask_file, h, w)
        m = extract_measurements(bio7, raw_image, intensity_thresholds)

        row = {
            "sample_id": sample.uid,
            "species": sample.species,
            "microscope": sample.microscope,
            "experiment": sample.experiment,
        }
        for k, v in m.items():
            row[k] = round(v, 6)
        rows.append(row)

        if save_vis:
            save_threshold_diagnostic(
                args.save_vis_dir, sample.uid, raw_image, bio7,
                intensity_thresholds,
            )

    if missing:
        print(f"Warning: {len(missing)} samples had no matching mask "
              f"(first few: {missing[:3]})")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} samples to {args.out_csv}")


if __name__ == "__main__":
    main()
