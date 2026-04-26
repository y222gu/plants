"""Full evaluation pipeline: eval + downstream + correlation plots.

Runs segmentation evaluation (IoU/Dice), saves predictions, extracts
downstream biological measurements, and generates correlation plots.

Directory structure created:
    {run-dir}/
    ├── eval/
    │   ├── test/                   # or oneshot/
    │   │   ├── metrics_native.csv
    │   │   ├── metrics_bio7.csv
    │   │   ├── predictions/
    │   │   ├── vis_native/
    │   │   └── vis_bio7/
    │   └── ...
    └── downstream/
        ├── test_from_predictions/  # or test_from_model/
        │   ├── gt_measurements.csv
        │   ├── pred_measurements.csv
        │   ├── correlation_summary.csv
        │   └── plots/
        └── ...

Usage:
    # Full pipeline (test + oneshot, downstream from saved predictions)
    python run_eval_pipeline.py --model-key unet_semantic --checkpoint path/to/best.ckpt --run-dir path/to/run/

    # Test only
    python run_eval_pipeline.py --model-key unet_semantic --checkpoint path/to/best.ckpt --run-dir path/to/run/ --split test

    # Oneshot only
    python run_eval_pipeline.py --model-key unet_semantic --checkpoint path/to/best.ckpt --run-dir path/to/run/ --split oneshot

    # Downstream from direct model output (re-runs inference, needs GPU)
    python run_eval_pipeline.py --model-key unet_semantic --checkpoint path/to/best.ckpt --run-dir path/to/run/ --downstream-source model

    # Skip downstream
    python run_eval_pipeline.py --model-key unet_semantic --checkpoint path/to/best.ckpt --run-dir path/to/run/ --no-downstream

    # Skip eval (only run downstream on existing predictions)
    python run_eval_pipeline.py --model-key unet_semantic --checkpoint path/to/best.ckpt --run-dir path/to/run/ --no-eval

    # Force re-run even if outputs exist
    python run_eval_pipeline.py ... --force
"""
import argparse, csv, cv2, sys, subprocess
import numpy as np
from pathlib import Path
from tqdm import tqdm


def run_eval(model_key, checkpoint, run_dir, split_name, strategy, img_size,
             force=False, no_vis=False):
    """Run segmentation evaluation for one split.

    Returns path to eval output dir (e.g., run_dir/eval/test/).
    """
    eval_dir = run_dir / "eval" / split_name

    # Check if already done
    bio7_csv = eval_dir / "metrics_bio7.csv"
    if bio7_csv.exists() and not force:
        with open(bio7_csv) as f:
            n = sum(1 for _ in f) - 1
        print(f"  [skip] eval/{split_name} already exists ({n} samples). Use --force to re-run.")
        return eval_dir

    eval_dir.mkdir(parents=True, exist_ok=True)

    # Determine which strategy/split to use
    eval_strategy = "oneshot" if split_name == "oneshot" else strategy

    from src.splits import get_split
    from src.preprocessing import load_sample_normalized, to_uint8
    from src.model_classes import (
        MODEL_REGISTRY, BIO_7_NAMES, BIO_7_COLORS_RGB,
        merge_classes, get_filled_classes,
    )
    from eval_bio7 import (
        semantic_to_raw_filled, mask_to_yolo_polygons, iou_dice,
        run_yolo_inference, run_unet_multilabel_inference,
        run_unet_semantic_inference, run_microsam_inference,
        make_vis_grid,
    )

    cfg = MODEL_REGISTRY[model_key]
    split = get_split(strategy=eval_strategy)
    samples = split["test"]

    # Load model
    model = _load_model(model_key, checkpoint)

    font = cv2.FONT_HERSHEY_SIMPLEX
    native_ids = sorted(cfg.native_classes.keys())
    native_names = [cfg.native_classes[i] for i in native_ids]
    native_color_by_name = {cfg.native_classes[i]: cfg.native_colors_rgb[i] for i in native_ids}

    native_csv_fields = ["sample_id", "species", "microscope", "experiment"]
    for name in native_names:
        native_csv_fields += [f"{name}_IoU", f"{name}_Dice"]
    native_csv_fields += ["mean_IoU", "mean_Dice"]

    bio7_csv_fields = ["sample_id", "species", "microscope", "experiment"]
    for name in BIO_7_NAMES:
        bio7_csv_fields += [f"{name}_IoU", f"{name}_Dice"]
    bio7_csv_fields += ["mean_IoU", "mean_Dice"]

    # Create output dirs
    pred_dir = eval_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    if not no_vis:
        (eval_dir / "vis_native").mkdir(exist_ok=True)
        (eval_dir / "vis_bio7").mkdir(exist_ok=True)

    native_rows = []
    bio7_rows = []

    print(f"  Evaluating {split_name} ({eval_strategy}): {len(samples)} samples")

    for sample in tqdm(samples, desc=f"  eval/{split_name}"):
        img = load_sample_normalized(sample)
        h, w = img.shape[:2]
        img_uint8 = to_uint8(img)

        # Inference
        if model_key.startswith("yolo"):
            masks, labels, h, w = run_yolo_inference(model, sample, img_size)
            # Raw merge for IoU (no fill_contours)
            pred_native = merge_classes(masks, labels, h, w)
        elif model_key == "unet_multilabel":
            pred_native, h, w = run_unet_multilabel_inference(model, sample, img_size)
        elif model_key in ("unet_semantic", "sam_semantic", "sam_unetpp", "timm_semantic"):
            pred_native, h, w = run_unet_semantic_inference(model, sample, img_size)
        elif model_key == "microsam":
            pred_native, h, w = run_microsam_inference(model, sample, img_size)
        else:
            raise ValueError(f"Inference not implemented for {model_key}")

        gt_native = cfg.load_gt(sample, h, w)
        pred_bio = cfg.to_bio7(pred_native, h, w)
        gt_bio = cfg.to_bio7(gt_native, h, w)

        base_row = {
            "sample_id": sample.uid,
            "species": sample.species,
            "microscope": sample.microscope,
            "experiment": sample.experiment,
        }

        # Native metrics
        native_row = dict(base_row)
        native_ious, native_dices = [], []
        gt_native_named, pred_native_named = {}, {}
        for cls_id in native_ids:
            name = cfg.native_classes[cls_id]
            if isinstance(gt_native, np.ndarray) and gt_native.ndim == 2:
                gt_cls = (gt_native == cls_id).astype(np.uint8)
                pred_cls = (pred_native == cls_id).astype(np.uint8)
            else:
                gt_cls = gt_native[cls_id]
                pred_cls = pred_native[cls_id]
            iou, dice = iou_dice(gt_cls, pred_cls)
            native_row[f"{name}_IoU"] = round(iou, 4) if not np.isnan(iou) else ""
            native_row[f"{name}_Dice"] = round(dice, 4) if not np.isnan(dice) else ""
            if not np.isnan(iou): native_ious.append(iou)
            if not np.isnan(dice): native_dices.append(dice)
            gt_native_named[name] = gt_cls
            pred_native_named[name] = pred_cls
        native_row["mean_IoU"] = round(np.mean(native_ious), 4) if native_ious else ""
        native_row["mean_Dice"] = round(np.mean(native_dices), 4) if native_dices else ""
        native_rows.append(native_row)

        # Save predictions as YOLO polygons
        lines = []
        if isinstance(pred_native, np.ndarray) and pred_native.ndim == 2:
            raw_filled = semantic_to_raw_filled(pred_native)
            for cls_id in range(6):
                lines.extend(mask_to_yolo_polygons(raw_filled[cls_id], cls_id, h, w))
        else:
            # YOLO/multilabel/micro-SAM: save as-is
            for cls_id in sorted(pred_native.keys()):
                lines.extend(mask_to_yolo_polygons(pred_native[cls_id], cls_id, h, w))
        with open(pred_dir / f"{sample.uid}.txt", "w") as f:
            f.write("\n".join(lines))

        # Bio-7 metrics
        bio7_row = dict(base_row)
        bio_ious, bio_dices = [], []
        for name in BIO_7_NAMES:
            iou, dice = iou_dice(gt_bio[name], pred_bio[name])
            bio7_row[f"{name}_IoU"] = round(iou, 4) if not np.isnan(iou) else ""
            bio7_row[f"{name}_Dice"] = round(dice, 4) if not np.isnan(dice) else ""
            if not np.isnan(iou): bio_ious.append(iou)
            if not np.isnan(dice): bio_dices.append(dice)
        bio7_row["mean_IoU"] = round(np.mean(bio_ious), 4) if bio_ious else ""
        bio7_row["mean_Dice"] = round(np.mean(bio_dices), 4) if bio_dices else ""
        bio7_rows.append(bio7_row)

        # Visualization
        if not no_vis:
            grid_native = make_vis_grid(
                img_uint8, gt_native_named, pred_native_named, native_row,
                native_names, native_color_by_name, sample.uid, "mean_IoU", font)
            cv2.imwrite(str(eval_dir / "vis_native" / f"{sample.uid}.png"),
                        cv2.cvtColor(grid_native, cv2.COLOR_RGB2BGR))

            bio7_color_by_name = {BIO_7_NAMES[i]: BIO_7_COLORS_RGB[i] for i in range(7)}
            grid_bio7 = make_vis_grid(
                img_uint8, gt_bio, pred_bio, bio7_row,
                BIO_7_NAMES, bio7_color_by_name, sample.uid, "mean_IoU", font)
            cv2.imwrite(str(eval_dir / "vis_bio7" / f"{sample.uid}.png"),
                        cv2.cvtColor(grid_bio7, cv2.COLOR_RGB2BGR))

    # Save CSVs
    native_rows.sort(key=lambda r: float(r["mean_IoU"]) if r["mean_IoU"] != "" else 0)
    with open(eval_dir / "metrics_native.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=native_csv_fields)
        writer.writeheader()
        writer.writerows(native_rows)

    bio7_rows.sort(key=lambda r: float(r["mean_IoU"]) if r["mean_IoU"] != "" else 0)
    with open(eval_dir / "metrics_bio7.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=bio7_csv_fields)
        writer.writeheader()
        writer.writerows(bio7_rows)

    # Summary
    bio_ious_all = [float(r["mean_IoU"]) for r in bio7_rows if r["mean_IoU"] != ""]
    miou = np.mean(bio_ious_all) if bio_ious_all else float("nan")
    print(f"  Bio-7 mIoU: {miou:.4f} ({len(samples)} samples)")
    print(f"  Saved to {eval_dir}/")

    return eval_dir


def run_downstream(model_key, checkpoint, run_dir, split_name, strategy,
                   source="predictions", img_size=1024, force=False,
                   intensity_thresholds=None):
    """Run downstream biological measurements for one split.

    Args:
        source: "predictions" (load saved .txt files, no GPU) or
                "model" (re-run inference, needs GPU)
        intensity_thresholds: optional dict {"TRITC": float, "FITC": float} —
            restrict mask to pixels above the threshold for that channel before
            computing mean intensity. Default None = no thresholding.
    """
    downstream_dir = run_dir / "downstream" / f"{split_name}_from_{source}"

    pred_csv = downstream_dir / "pred_measurements.csv"
    if pred_csv.exists() and not force:
        with open(pred_csv) as f:
            n = sum(1 for _ in f) - 1
        print(f"  [skip] downstream/{split_name}_from_{source} already exists ({n} samples). Use --force to re-run.")
        return downstream_dir

    downstream_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = downstream_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    eval_strategy = "oneshot" if split_name == "oneshot" else strategy

    from src.splits import get_split
    from src.preprocessing import load_sample_raw
    from src.model_classes import MODEL_REGISTRY
    from downstream_measure_from_model import extract_measurements, MEASUREMENT_COLS

    cfg = MODEL_REGISTRY.get(model_key, MODEL_REGISTRY["yolo_overlap_false"])
    split = get_split(strategy=eval_strategy)
    samples = split["test"]

    csv_fields = ["sample_id", "species", "microscope", "experiment"] + MEASUREMENT_COLS

    gt_rows = []
    pred_rows = []

    if source == "predictions":
        # Load from saved YOLO polygon .txt files
        pred_dir = run_dir / "eval" / split_name / "predictions"
        if not pred_dir.exists():
            print(f"  [error] No predictions at {pred_dir}. Run eval first.")
            return None

        from src.annotation_utils import parse_yolo_annotations, polygon_to_mask
        from src.model_classes import get_filled_classes, yolo_overlap_false_to_bio7

        pred_files = {f.stem: f for f in pred_dir.glob("*.txt")}

        for sample in tqdm(samples, desc=f"  downstream/{split_name} (from predictions)"):
            pred_file = pred_files.get(sample.uid)
            if pred_file is None:
                continue

            raw_image = load_sample_raw(sample)
            h, w = raw_image.shape[:2]

            base_row = {
                "sample_id": sample.uid,
                "species": sample.species,
                "microscope": sample.microscope,
                "experiment": sample.experiment,
            }

            # Prediction bio-7 from saved polygons
            anns = parse_yolo_annotations(pred_file, w, h)
            if anns:
                masks, labels = [], []
                for ann in anns:
                    masks.append(polygon_to_mask(ann["polygon"], h, w))
                    labels.append(ann["class_id"])
                filled = get_filled_classes(np.array(masks),
                                            np.array(labels, dtype=np.int32), h, w)
            else:
                filled = {i: np.zeros((h, w), dtype=np.uint8) for i in range(6)}
            pred_bio7 = yolo_overlap_false_to_bio7(filled, h, w)
            pred_m = extract_measurements(pred_bio7, raw_image, intensity_thresholds)
            pred_row = dict(base_row)
            for k, v in pred_m.items():
                pred_row[k] = round(v, 6)
            pred_rows.append(pred_row)

            # GT bio-7
            from downstream_measure_from_model import get_gt_bio7
            gt_bio7 = get_gt_bio7(sample, cfg)
            gt_m = extract_measurements(gt_bio7, raw_image, intensity_thresholds)
            gt_row = dict(base_row)
            for k, v in gt_m.items():
                gt_row[k] = round(v, 6)
            gt_rows.append(gt_row)

    elif source == "model":
        # Re-run inference (needs GPU)
        from downstream_measure_from_model import get_pred_bio7, get_gt_bio7

        model = _load_model(model_key, checkpoint)

        for sample in tqdm(samples, desc=f"  downstream/{split_name} (from model)"):
            raw_image = load_sample_raw(sample)

            base_row = {
                "sample_id": sample.uid,
                "species": sample.species,
                "microscope": sample.microscope,
                "experiment": sample.experiment,
            }

            pred_bio7 = get_pred_bio7(model, sample, model_key, cfg, img_size)
            pred_m = extract_measurements(pred_bio7, raw_image, intensity_thresholds)
            pred_row = dict(base_row)
            for k, v in pred_m.items():
                pred_row[k] = round(v, 6)
            pred_rows.append(pred_row)

            gt_bio7 = get_gt_bio7(sample, cfg)
            gt_m = extract_measurements(gt_bio7, raw_image, intensity_thresholds)
            gt_row = dict(base_row)
            for k, v in gt_m.items():
                gt_row[k] = round(v, 6)
            gt_rows.append(gt_row)

    # Save CSVs
    gt_csv = downstream_dir / "gt_measurements.csv"
    with open(gt_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(gt_rows)

    with open(pred_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(pred_rows)

    print(f"  Saved {len(pred_rows)} samples to {downstream_dir}/")

    # Generate correlation plots
    print(f"  Generating correlation plots...")
    cmd = [sys.executable, "downstream_plot_correlations.py",
           "--gt", str(gt_csv), "--pred", str(pred_csv), "--out-dir", str(plot_dir)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Plot generation failed: {result.stderr[:300]}")
    else:
        # Move correlation_summary.csv to downstream_dir (next to gt/pred CSVs)
        summary_src = plot_dir / "correlation_summary.csv"
        summary_dst = downstream_dir / "correlation_summary.csv"
        if summary_src.exists():
            summary_src.rename(summary_dst)
        print(f"  Plots saved to {plot_dir}/")

    return downstream_dir


def validate_and_save_status(run_dir, splits, downstream_source):
    """Validate outputs and save eval_status.json."""
    import json
    from datetime import datetime

    status = {
        "run_dir": str(run_dir),
        "timestamp": datetime.now().isoformat(),
        "status": "complete",
        "errors": [],
    }

    # Get expected counts from actual splits
    from src.splits import get_split
    run_str = str(run_dir)
    if "B-dico" in run_str:
        strat = "B-dico"
    elif "B-mono" in run_str:
        strat = "B-mono"
    else:
        strat = "A"
    n_test = len(get_split(strategy=strat)["test"])
    n_oneshot = len(get_split(strategy="oneshot")["test"])

    for split_name in splits:
        expected_n = n_oneshot if split_name == "oneshot" else n_test
        split_status = {}

        # Check eval
        bio7_csv = run_dir / "eval" / split_name / "metrics_bio7.csv"
        native_csv = run_dir / "eval" / split_name / "metrics_native.csv"
        pred_dir = run_dir / "eval" / split_name / "predictions"

        if bio7_csv.exists():
            with open(bio7_csv) as f:
                rows = list(csv.DictReader(f))
            split_status["metrics_bio7"] = len(rows)
            if len(rows) != expected_n:
                status["errors"].append(f"{split_name}: metrics_bio7 has {len(rows)} rows, expected {expected_n}")
                status["status"] = "error"
            # Compute mIoU
            ious = [float(r["mean_IoU"]) for r in rows if r.get("mean_IoU", "") != ""]
            split_status["mIoU"] = round(np.mean(ious), 4) if ious else None
        else:
            split_status["metrics_bio7"] = 0
            status["errors"].append(f"{split_name}: metrics_bio7.csv missing")
            status["status"] = "error"

        if native_csv.exists():
            with open(native_csv) as f:
                n = sum(1 for _ in f) - 1
            split_status["metrics_native"] = n
        else:
            split_status["metrics_native"] = 0

        if pred_dir.exists():
            split_status["predictions"] = len(list(pred_dir.glob("*.txt")))
        else:
            split_status["predictions"] = 0

        vis_dir = run_dir / "eval" / split_name / "vis_bio7"
        split_status["vis"] = len(list(vis_dir.glob("*.png"))) if vis_dir.exists() else 0

        # Check downstream
        ds_dir = run_dir / "downstream" / f"{split_name}_from_predictions"
        if ds_dir.exists():
            pred_csv = ds_dir / "pred_measurements.csv"
            gt_csv = ds_dir / "gt_measurements.csv"
            corr_csv = ds_dir / "correlation_summary.csv"
            split_status["downstream_pred"] = (sum(1 for _ in open(pred_csv)) - 1) if pred_csv.exists() else 0
            split_status["downstream_gt"] = (sum(1 for _ in open(gt_csv)) - 1) if gt_csv.exists() else 0
            split_status["downstream_plots"] = corr_csv.exists()
        else:
            split_status["downstream_pred"] = 0
            split_status["downstream_gt"] = 0
            split_status["downstream_plots"] = False

        status[split_name] = split_status

    # Save
    status_path = run_dir / "eval_status.json"
    with open(status_path, "w") as f:
        json.dump(status, f, indent=2)

    return status


def _load_model(model_key, checkpoint):
    """Load model from checkpoint."""
    if model_key.startswith("yolo"):
        from ultralytics import YOLO
        return YOLO(checkpoint)
    elif model_key == "unet_multilabel":
        import torch
        from train.train_unet_binary import MultilabelSegModule
        model = MultilabelSegModule.load_from_checkpoint(checkpoint)
        model.eval()
        if torch.cuda.is_available():
            model.cuda()
        return model
    elif model_key == "unet_semantic":
        import torch
        from train.train_unet_semantic import SemanticSegModule
        model = SemanticSegModule.load_from_checkpoint(checkpoint)
        model.eval()
        if torch.cuda.is_available():
            model.cuda()
        return model
    elif model_key == "sam_semantic":
        import torch
        from train.train_sam_semantic import SAMSemanticModule
        model = SAMSemanticModule.load_from_checkpoint(checkpoint)
        model.eval()
        if torch.cuda.is_available():
            model.cuda()
        return model
    elif model_key == "sam_unetpp":
        import torch
        from train.train_sam_unetpp import SAMUNetPPModule
        model = SAMUNetPPModule.load_from_checkpoint(checkpoint)
        model.eval()
        if torch.cuda.is_available():
            model.cuda()
        return model
    elif model_key == "timm_semantic":
        import torch
        from train.train_timm_semantic import TimmSemanticModule
        model = TimmSemanticModule.load_from_checkpoint(checkpoint)
        model.eval()
        if torch.cuda.is_available():
            model.cuda()
        return model
    elif model_key == "microsam":
        import torch
        from micro_sam.automatic_segmentation import get_predictor_and_segmenter
        from src.config import ANNOTATED_CLASSES
        device = "cuda" if torch.cuda.is_available() else "cpu"
        run_base = Path(checkpoint)
        models = {}
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
            predictor, segmenter = get_predictor_and_segmenter(
                model_type="vit_b_lm", checkpoint=str(exported), device=device,
                segmentation_mode="ais",
            )
            models[cls_id] = (predictor, segmenter)
        return models
    else:
        raise ValueError(f"Unknown model key: {model_key}")


def main():
    parser = argparse.ArgumentParser(
        description="Full evaluation pipeline: eval + downstream + correlation plots")
    parser.add_argument("--model-key", required=True,
                        help="Model key (e.g., unet_semantic, yolo_overlap_false)")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--run-dir", required=True, type=Path,
                        help="Run directory (e.g., lambda_output/runs/unet/.../2026-04-14_001)")
    parser.add_argument("--split", default="all", choices=["all", "test", "oneshot"],
                        help="Which split to evaluate (default: both test and oneshot)")
    parser.add_argument("--strategy", default=None,
                        help="Data split strategy (default: auto-detect from run dir name)")
    parser.add_argument("--downstream-source", default="predictions",
                        choices=["predictions", "model"],
                        help="Downstream source: 'predictions' (no GPU) or 'model' (re-runs inference)")
    parser.add_argument("--no-downstream", action="store_true",
                        help="Skip downstream analysis")
    parser.add_argument("--no-eval", action="store_true",
                        help="Skip eval (only run downstream on existing predictions)")
    parser.add_argument("--no-vis", action="store_true",
                        help="Skip visualization PNGs")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if outputs already exist")
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

    from downstream_measure_from_model import parse_threshold_args
    intensity_thresholds = parse_threshold_args(
        args.tritc_threshold, args.fitc_threshold, args.threshold)

    # Auto-detect strategy from run dir name
    if args.strategy is None:
        run_name = str(args.run_dir)
        if "B-dico" in run_name:
            args.strategy = "B-dico"
        elif "B-mono" in run_name:
            args.strategy = "B-mono"
        else:
            args.strategy = "A"
    print(f"Strategy: {args.strategy}")

    # Determine splits to run
    if args.split == "all":
        splits = ["test", "oneshot"]
    else:
        splits = [args.split]

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: {args.model_key}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Run dir: {run_dir}")
    print(f"Splits: {splits}")
    print(f"Downstream: {'skip' if args.no_downstream else args.downstream_source}")
    if intensity_thresholds:
        print(f"Intensity thresholds: {intensity_thresholds}")
    print()

    for split_name in splits:
        print(f"{'='*60}")
        print(f"Split: {split_name}")
        print(f"{'='*60}")

        # Step 1: Eval
        if not args.no_eval:
            eval_dir = run_eval(
                args.model_key, args.checkpoint, run_dir, split_name,
                args.strategy, args.img_size, args.force, args.no_vis)
        else:
            eval_dir = run_dir / "eval" / split_name
            if not eval_dir.exists():
                print(f"  [error] eval/{split_name} not found. Run eval first.")
                continue
            print(f"  [skip] eval (--no-eval)")

        # Step 2: Downstream
        if not args.no_downstream:
            run_downstream(
                args.model_key, args.checkpoint, run_dir, split_name,
                args.strategy, args.downstream_source, args.img_size, args.force,
                intensity_thresholds=intensity_thresholds)
        else:
            print(f"  [skip] downstream (--no-downstream)")

        print()

    # Validate and save status
    print(f"{'='*60}")
    print("Validation")
    print(f"{'='*60}")
    status = validate_and_save_status(run_dir, splits, args.downstream_source)
    if status["status"] == "complete":
        print(f"  ✓ All checks passed")
    else:
        print(f"  ✗ Errors found:")
        for err in status["errors"]:
            print(f"    - {err}")

    for split_name in splits:
        s = status.get(split_name, {})
        miou = s.get("mIoU", "N/A")
        print(f"  {split_name}: bio7={s.get('metrics_bio7',0)} samples, "
              f"preds={s.get('predictions',0)}, "
              f"downstream={s.get('downstream_pred',0)}, "
              f"mIoU={miou}")

    print(f"\nStatus saved to {run_dir}/eval_status.json")
    print("Done.")


if __name__ == "__main__":
    main()
