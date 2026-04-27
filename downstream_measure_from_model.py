"""Extract biological measurements from bio-7 masks on raw images.

Measurements:
    - Aerenchyma ratio: aerenchyma area / whole root area
    - Exodermis TRITC/FITC intensity (on raw image)
    - Endodermis TRITC/FITC intensity (on raw image)
    - Vascular TRITC/FITC intensity (on raw image)

Usage:
    # Both GT and predictions (for comparison)
    python downstream_measure_from_model.py --model-key yolo_overlap_false --checkpoint path/to/best.pt --source both

    # GT only (no model needed)
    python downstream_measure_from_model.py --source gt

    # Predictions only
    python downstream_measure_from_model.py --model-key yolo_overlap_false --checkpoint path/to/best.pt --source prediction
"""
import csv, cv2, numpy as np, argparse
from pathlib import Path
from tqdm import tqdm

from src.splits import get_split
from src.preprocessing import load_sample_raw, load_sample_normalized, to_uint8
from src.model_classes import MODEL_REGISTRY, BIO_7_NAMES, get_filled_classes, get_raw_classes


# ── Measurement functions ────────────────────────────────────────────────────

def measure_aerenchyma_ratio(bio7):
    """Aerenchyma area / whole root area. Returns 0.0 if no root."""
    wr_area = bio7["Whole Root"].sum()
    if wr_area == 0:
        return 0.0
    return float(bio7["Aerenchyma"].sum() / wr_area)


def measure_region_intensity(raw_image, mask, channel_idx, threshold_range=None):
    """Mean intensity of a channel under a binary mask on the raw image.

    If `threshold_range = (low, high)` is given, restrict the mask to pixels
    where `low <= raw_image[..., channel_idx] <= high`. Either bound may be
    None for unbounded (e.g. (None, 5000) → keep <= 5000;
    (5000, None) → keep >= 5000). Returns 0.0 if the kept mask is empty.
    """
    chan = raw_image[:, :, channel_idx]
    region = mask > 0
    if threshold_range is not None:
        low, high = threshold_range
        if low is not None:
            region = region & (chan >= low)
        if high is not None:
            region = region & (chan <= high)
    if not region.any():
        return 0.0
    return float(chan[region].mean())


# Channel name → index in raw_image (R=TRITC, G=FITC, B=DAPI)
CHANNEL_IDX = {"TRITC": 0, "FITC": 1, "DAPI": 2}

# Region → bio-7 mask name. Order matters for the measurement column order.
INTENSITY_REGIONS = ["Exodermis", "Endodermis", "Vascular"]
INTENSITY_CHANNELS = ["TRITC", "FITC"]


def parse_range(spec):
    """Parse a keep-range spec into a (low, high) tuple of floats-or-None.

    Required form: 'LOW-HIGH' (case-insensitive on min/max).
        "4000-5000"  → (4000.0, 5000.0)
        "min-5000"   → (None,   5000.0)
        "5000-max"   → (5000.0, None)
    Bounds are inclusive on both ends. Bare numbers are rejected to avoid
    ambiguity (write 'min-N' or 'N-max' explicitly).
    """
    import argparse as _ap
    s = str(spec).strip().lower()
    if "-" not in s:
        raise _ap.ArgumentTypeError(
            f"Invalid range {spec!r}; expected LOW-HIGH form (e.g. 4000-5000, "
            f"min-5000, 5000-max). Bare numbers are not allowed.")
    lo_s, hi_s = s.split("-", 1)
    if not lo_s or not hi_s:
        raise _ap.ArgumentTypeError(
            f"Invalid range {spec!r}; both LOW and HIGH are required "
            f"(use 'min'/'max' for unbounded ends).")
    try:
        low  = None if lo_s == "min" else float(lo_s)
        high = None if hi_s == "max" else float(hi_s)
    except ValueError:
        raise _ap.ArgumentTypeError(
            f"Invalid range {spec!r}; LOW and HIGH must be numbers, 'min', or 'max'.")
    if low is not None and high is not None and low > high:
        raise _ap.ArgumentTypeError(f"Range {spec!r}: low ({low}) > high ({high})")
    return (low, high)


def format_range(rng):
    """Render a (low, high) tuple as 'min-5000', '4000-5000', etc."""
    low, high = rng
    lo_s = "min" if low is None else f"{low:g}"
    hi_s = "max" if high is None else f"{high:g}"
    return f"{lo_s}-{hi_s}"


def extract_measurements(bio7, raw_image, intensity_thresholds=None):
    """Extract all downstream measurements from bio-7 masks and raw image.

    Args:
        bio7: dict {name: (H,W) uint8 mask} — 7 biological class masks
        raw_image: (H,W,3) float32 raw image — R=TRITC(0), G=FITC(1), B=DAPI(2)
        intensity_thresholds: optional nested dict
            {region_name: {channel_name: (low, high)}} — keep-range per
            (region, channel). low/high may each be None for unbounded.
            Pixels with raw value in [low, high] are kept; the rest are
            removed before averaging. Missing entries → no thresholding for
            that (region, channel). Default None = no thresholding anywhere.

    Returns:
        dict with measurement values.
    """
    # Resize masks to raw image dimensions if needed
    raw_h, raw_w = raw_image.shape[:2]
    mask_h, mask_w = bio7["Whole Root"].shape[:2]

    if (raw_h, raw_w) != (mask_h, mask_w):
        bio7_resized = {}
        for name, mask in bio7.items():
            bio7_resized[name] = cv2.resize(mask, (raw_w, raw_h),
                                             interpolation=cv2.INTER_NEAREST)
        bio7 = bio7_resized

    thr = intensity_thresholds or {}

    out = {"aerenchyma_ratio": measure_aerenchyma_ratio(bio7)}
    for region in INTENSITY_REGIONS:
        for channel in INTENSITY_CHANNELS:
            rng = thr.get(region, {}).get(channel)
            out[f"{region.lower()}_{channel}"] = measure_region_intensity(
                raw_image, bio7[region], CHANNEL_IDX[channel], rng,
            )
    return out


def parse_threshold_args(global_tritc, global_fitc, threshold_specs):
    """Build nested {region: {channel: (low, high)}} dict from CLI args.

    Args:
        global_tritc: optional range string applied to all regions on TRITC.
        global_fitc:  optional range string applied to all regions on FITC.
        threshold_specs: list of "REGION:CHANNEL=RANGE" strings; per-(region,channel)
            entries override the globals. RANGE follows `parse_range` syntax.
    """
    import argparse as _ap
    thresholds = {r: {} for r in INTENSITY_REGIONS}
    if global_tritc is not None:
        rng = parse_range(global_tritc)
        for r in INTENSITY_REGIONS:
            thresholds[r]["TRITC"] = rng
    if global_fitc is not None:
        rng = parse_range(global_fitc)
        for r in INTENSITY_REGIONS:
            thresholds[r]["FITC"] = rng
    for spec in threshold_specs or []:
        try:
            rc, val = spec.split("=", 1)
            region, channel = rc.split(":")
        except ValueError:
            raise _ap.ArgumentTypeError(
                f"Invalid --threshold spec {spec!r}; expected REGION:CHANNEL=RANGE")
        if region not in INTENSITY_REGIONS:
            raise _ap.ArgumentTypeError(
                f"Unknown region {region!r}; must be one of {INTENSITY_REGIONS}")
        if channel not in INTENSITY_CHANNELS:
            raise _ap.ArgumentTypeError(
                f"Unknown channel {channel!r}; must be one of {INTENSITY_CHANNELS}")
        thresholds[region][channel] = parse_range(val)
    # Drop empty regions for cleaner display
    return {r: c for r, c in thresholds.items() if c}


MEASUREMENT_COLS = [
    "aerenchyma_ratio",
    "exodermis_TRITC", "exodermis_FITC",
    "endodermis_TRITC", "endodermis_FITC",
    "vascular_TRITC", "vascular_FITC",
]


# ── Inference helpers (reused from eval_bio7.py) ────────────────────────────

def run_yolo_inference(model, sample, img_size):
    img = load_sample_normalized(sample)
    h, w = img.shape[:2]
    img_bgr = cv2.cvtColor(to_uint8(img), cv2.COLOR_RGB2BGR)
    results = model(img_bgr, imgsz=img_size, verbose=False, retina_masks=True)[0]
    if results.masks is not None:
        masks = results.masks.data.cpu().numpy().astype(np.uint8)
        labels = results.boxes.cls.cpu().numpy().astype(np.int32)
    else:
        masks = np.zeros((0, h, w), dtype=np.uint8)
        labels = np.zeros(0, dtype=np.int32)
    return masks, labels, h, w


def run_unet_multilabel_inference(model, sample, img_size):
    import torch
    img = load_sample_normalized(sample)
    h, w = img.shape[:2]
    img_resized = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float()
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    with torch.no_grad():
        logits = model(tensor)
    probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
    filled = {}
    for c in range(probs.shape[0]):
        prob_full = cv2.resize(probs[c], (w, h), interpolation=cv2.INTER_LINEAR)
        filled[c] = (prob_full > 0.5).astype(np.uint8)
    return filled, h, w


def run_unet_semantic_inference(model, sample, img_size):
    """U-Net++ semantic inference → (H,W) argmax semantic mask."""
    import torch
    img = load_sample_normalized(sample)
    h, w = img.shape[:2]
    img_resized = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float()
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    with torch.no_grad():
        logits = model(tensor)
    sem_mask = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int32)
    sem_mask = cv2.resize(sem_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return sem_mask, h, w


def run_microsam_inference(models, sample, img_size):
    """micro-SAM inference with 6 per-class models → dict {0..5: (H,W) binary mask}."""
    from micro_sam.automatic_segmentation import automatic_instance_segmentation

    img = load_sample_normalized(sample)
    h, w = img.shape[:2]
    img_resized = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    img_uint8 = (np.clip(img_resized, 0, 1) * 255).astype(np.uint8)

    filled = {}
    for cls_id in range(6):
        if cls_id not in models:
            filled[cls_id] = np.zeros((h, w), dtype=np.uint8)
            continue
        predictor, segmenter = models[cls_id]

        if cls_id == 1:
            # Aerenchyma: use full AIS (watershed) for multi-instance
            pred_instances = automatic_instance_segmentation(
                predictor=predictor, segmenter=segmenter,
                input_path=img_uint8, ndim=2,
            )
            pred_binary = (pred_instances > 0).astype(np.uint8)
        else:
            # Single-instance classes (0,2-5): threshold UNETR foreground
            segmenter.initialize(img_uint8, ndim=2)
            pred_binary = (segmenter._foreground > 0.015).astype(np.uint8)

        if pred_binary.shape != (h, w):
            pred_binary = cv2.resize(pred_binary, (w, h),
                                     interpolation=cv2.INTER_NEAREST)
        filled[cls_id] = pred_binary
    return filled, h, w


def get_pred_bio7(model, sample, model_key, cfg, img_size):
    """Run inference and convert to bio-7 masks."""
    if model_key.startswith("yolo"):
        masks, labels, h, w = run_yolo_inference(model, sample, img_size)
        if model_key == "yolo_overlap_true":
            pred_native = get_raw_classes(masks, labels, h, w)
        else:
            pred_native = get_filled_classes(masks, labels, h, w)
    elif model_key == "unet_multilabel":
        pred_native, h, w = run_unet_multilabel_inference(model, sample, img_size)
    elif model_key in ("unet_semantic", "sam_semantic", "sam_unetpp", "timm_semantic"):
        pred_native, h, w = run_unet_semantic_inference(model, sample, img_size)
    elif model_key == "microsam":
        pred_native, h, w = run_microsam_inference(model, sample, img_size)
    else:
        raise ValueError(f"Inference not implemented for {model_key}")
    return cfg.to_bio7(pred_native, h, w)


def get_gt_bio7(sample, cfg):
    """Load GT and convert to bio-7 masks."""
    img = load_sample_normalized(sample)
    h, w = img.shape[:2]
    gt_native = cfg.load_gt(sample, h, w)
    return cfg.to_bio7(gt_native, h, w)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract downstream measurements from bio-7 masks")
    parser.add_argument("--model-key", choices=list(MODEL_REGISTRY.keys()),
                        help="Model key (required for --source prediction or both)")
    parser.add_argument("--checkpoint",
                        help="Path to model checkpoint (required for --source prediction or both)")
    parser.add_argument("--source", default="both", choices=["gt", "prediction", "both"],
                        help="Which masks to use: gt, prediction, or both")
    parser.add_argument("--strategy", default="A")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (default: <run_dir>/downstream/)")
    parser.add_argument("--img-size", type=int, default=1024)
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

    if args.source in ("prediction", "both") and (not args.model_key or not args.checkpoint):
        parser.error("--model-key and --checkpoint required for --source prediction or both")

    split = get_split(strategy=args.strategy)
    samples = split["test"]

    # Output directory
    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif args.checkpoint:
        ckpt = Path(args.checkpoint)
        run_dir = ckpt.parent
        while run_dir != run_dir.parent:
            if run_dir.name[:4].isdigit() and "-" in run_dir.name:
                break
            run_dir = run_dir.parent
        out_dir = run_dir / "downstream"
    else:
        out_dir = Path("output/downstream")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model if needed
    model = None
    cfg = None
    if args.source in ("prediction", "both"):
        cfg = MODEL_REGISTRY[args.model_key]
        if args.model_key.startswith("yolo"):
            from ultralytics import YOLO
            model = YOLO(args.checkpoint)
        elif args.model_key == "unet_multilabel":
            import torch
            from train.train_unet_binary import MultilabelSegModule
            model = MultilabelSegModule.load_from_checkpoint(args.checkpoint)
            model.eval()
            if torch.cuda.is_available():
                model.cuda()
        elif args.model_key == "unet_semantic":
            import torch
            from train.train_unet_semantic import SemanticSegModule
            model = SemanticSegModule.load_from_checkpoint(args.checkpoint)
            model.eval()
            if torch.cuda.is_available():
                model.cuda()
        elif args.model_key == "sam_semantic":
            import torch
            from train.train_sam_semantic import SAMSemanticModule
            model = SAMSemanticModule.load_from_checkpoint(args.checkpoint)
            model.eval()
            if torch.cuda.is_available():
                model.cuda()
        elif args.model_key == "sam_unetpp":
            import torch
            from train.train_sam_unetpp import SAMUNetPPModule
            model = SAMUNetPPModule.load_from_checkpoint(args.checkpoint)
            model.eval()
            if torch.cuda.is_available():
                model.cuda()
        elif args.model_key == "timm_semantic":
            import torch
            from train.train_timm_semantic import TimmSemanticModule
            model = TimmSemanticModule.load_from_checkpoint(args.checkpoint)
            model.eval()
            if torch.cuda.is_available():
                model.cuda()
        elif args.model_key == "microsam":
            import torch
            from micro_sam.automatic_segmentation import get_predictor_and_segmenter
            from src.config import ANNOTATED_CLASSES
            device = "cuda" if torch.cuda.is_available() else "cpu"
            run_base = Path(args.checkpoint)
            model = {}
            for cls_id in range(6):
                cls_dir = run_base / f"vit_b_lm_class{cls_id}_A"
                if not cls_dir.exists():
                    continue
                runs = sorted(cls_dir.glob("20*"))
                if not runs:
                    continue
                exported = runs[-1] / "exported_model.pt"
                if not exported.exists():
                    continue
                print(f"  Loading class {cls_id} ({ANNOTATED_CLASSES[cls_id]}): {exported}")
                predictor, segmenter = get_predictor_and_segmenter(
                    model_type="vit_b_lm", checkpoint=str(exported), device=device,
                    segmentation_mode="ais",
                )
                model[cls_id] = (predictor, segmenter)
            print(f"  Loaded {len(model)}/6 class models")

    # For GT, use any model config that produces bio-7 (yolo_overlap_false is the default)
    gt_cfg = MODEL_REGISTRY.get(args.model_key, MODEL_REGISTRY["yolo_overlap_false"])

    csv_fields = ["sample_id", "species", "microscope", "experiment"] + MEASUREMENT_COLS

    print(f"Source: {args.source}")
    print(f"Test set: {args.strategy} ({len(samples)} samples)")
    print(f"Output: {out_dir}")

    # ── Process samples ──
    gt_rows = []
    pred_rows = []

    for sample in tqdm(samples, desc="Extracting measurements"):
        raw_image = load_sample_raw(sample)
        base_row = {
            "sample_id": sample.uid,
            "species": sample.species,
            "microscope": sample.microscope,
            "experiment": sample.experiment,
        }

        # GT measurements
        if args.source in ("gt", "both"):
            gt_bio7 = get_gt_bio7(sample, gt_cfg)
            gt_measurements = extract_measurements(gt_bio7, raw_image, intensity_thresholds)
            gt_row = dict(base_row)
            for k, v in gt_measurements.items():
                gt_row[k] = round(v, 6)
            gt_rows.append(gt_row)

        # Prediction measurements
        if args.source in ("prediction", "both"):
            pred_bio7 = get_pred_bio7(model, sample, args.model_key, cfg, args.img_size)
            pred_measurements = extract_measurements(pred_bio7, raw_image, intensity_thresholds)
            pred_row = dict(base_row)
            for k, v in pred_measurements.items():
                pred_row[k] = round(v, 6)
            pred_rows.append(pred_row)

    # ── Save CSVs ──
    if gt_rows:
        gt_csv = out_dir / "gt_measurements.csv"
        with open(gt_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            writer.writerows(gt_rows)
        print(f"GT measurements saved to {gt_csv} ({len(gt_rows)} samples)")

    if pred_rows:
        pred_csv = out_dir / "pred_measurements.csv"
        with open(pred_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            writer.writerows(pred_rows)
        print(f"Prediction measurements saved to {pred_csv} ({len(pred_rows)} samples)")

    print(f"\nDone. Output: {out_dir}")


if __name__ == "__main__":
    main()
