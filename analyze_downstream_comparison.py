"""Compare downstream metrics: YOLO predictions vs ground truth.

Runs YOLO inference on test samples, computes downstream metrics from both
GT and predictions, plots predicted vs true with regression line and R².

Usage:
    python analyze_downstream_comparison.py --checkpoint output/runs/yolo/yolo11m-seg_strategy1/weights/best.pt
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.annotation_utils import load_sample_annotations
from src.config import OUTPUT_DIR, DEFAULT_IMG_SIZE
from src.dataset import SampleRegistry
from src.downstream import analyze_sample
from src.preprocessing import load_sample_normalized, to_uint8
from src.splits import get_split


def _flatten_result(result: dict) -> dict:
    """Flatten nested intensity dicts into flat columns."""
    flat = {}
    for k, v in result.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                flat[f"{k}_{sub_k}"] = sub_v
        else:
            flat[k] = v
    return flat


def run_yolo_inference(checkpoint: str, samples, img_size: int) -> dict:
    """Run YOLO inference, return {uid: (masks, labels)}."""
    from ultralytics import YOLO
    model = YOLO(checkpoint)
    predictions = {}

    for sample in tqdm(samples, desc="YOLO inference"):
        img = load_sample_normalized(sample)
        img_uint8 = to_uint8(img)
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
        results = model(img_bgr, imgsz=img_size, verbose=False)[0]

        h, w = img.shape[:2]
        if results.masks is not None:
            masks = results.masks.data.cpu().numpy().astype(np.uint8)
            labels = results.boxes.cls.cpu().numpy().astype(np.int32)
            resized = np.zeros((len(masks), h, w), dtype=np.uint8)
            for i in range(len(masks)):
                smooth = cv2.resize(masks[i].astype(np.float32), (w, h),
                                    interpolation=cv2.INTER_LINEAR)
                resized[i] = (smooth > 0.5).astype(np.uint8)
            masks = resized
        else:
            masks = np.zeros((0, h, w), dtype=np.uint8)
            labels = np.zeros(0, dtype=np.int32)

        predictions[sample.uid] = (masks, labels)

    return predictions


def make_comparison_plots(df: pd.DataFrame, out_dir: Path):
    """Plot predicted vs true for each downstream metric with regression + R²."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    })

    metrics = [
        ("aerenchyma_ratio", "Aerenchyma Ratio"),
        ("endodermis_intensity_TRITC", "Endodermis TRITC Intensity"),
        ("endodermis_intensity_FITC", "Endodermis FITC Intensity"),
        ("endodermis_intensity_DAPI", "Endodermis DAPI Intensity"),
        ("vascular_intensity_TRITC", "Vascular TRITC Intensity"),
        ("vascular_intensity_FITC", "Vascular FITC Intensity"),
        ("vascular_intensity_DAPI", "Vascular DAPI Intensity"),
    ]

    species_markers = {"Millet": "o", "Rice": "s", "Sorghum": "^"}
    species_colors = {"Millet": "#E69F00", "Rice": "#0072B2", "Sorghum": "#009E73"}

    # ── Individual metric plots ──
    for col, label in metrics:
        gt_col = f"gt_{col}"
        pred_col = f"pred_{col}"
        if gt_col not in df.columns or pred_col not in df.columns:
            continue

        valid = df[[gt_col, pred_col, "species"]].dropna()
        gt_vals = valid[gt_col].values
        pred_vals = valid[pred_col].values

        if len(gt_vals) < 3:
            continue

        # Regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(gt_vals, pred_vals)
        r2 = r_value ** 2

        fig, ax = plt.subplots(figsize=(5.5, 5.5))

        # Plot by species
        for sp in sorted(species_markers.keys()):
            mask = valid["species"] == sp
            if mask.sum() == 0:
                continue
            ax.scatter(valid.loc[mask, gt_col], valid.loc[mask, pred_col],
                       marker=species_markers[sp], color=species_colors[sp],
                       s=60, alpha=0.6, label=sp, edgecolors="white", linewidth=0.4)

        # Fit line
        x_range = np.array([gt_vals.min(), gt_vals.max()])
        ax.plot(x_range, slope * x_range + intercept, "k-", linewidth=1.2,
                label=f"$y = {slope:.2f}x + {intercept:.3f}$")

        # Identity line
        all_vals = np.concatenate([gt_vals, pred_vals])
        lo, hi = all_vals.min() * 0.95, all_vals.max() * 1.05
        if lo == hi:
            lo, hi = lo - 0.1, hi + 0.1
        ax.plot([lo, hi], [lo, hi], "--", color="#999999", linewidth=0.8, label="$y = x$")

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel(f"Ground Truth — {label}")
        ax.set_ylabel(f"Predicted — {label}")
        ax.set_title(f"{label}\n$R^2 = {r2:.4f}$,  $n = {len(gt_vals)}$")
        ax.legend(fontsize=10, loc="upper left")
        ax.grid(linewidth=0.3, alpha=0.4)

        for fmt in ("png", "pdf"):
            fig.savefig(out_dir / f"downstream_{col}.{fmt}")
        plt.close(fig)
        print(f"  {label}: R²={r2:.4f}  (slope={slope:.3f}, n={len(gt_vals)})")

    # ── Combined figure ──
    n_metrics = len(metrics)
    ncols = 4
    nrows = (n_metrics + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows))
    axes_flat = axes.flatten()
    # Hide unused axes
    for i in range(n_metrics, len(axes_flat)):
        axes_flat[i].set_visible(False)

    for idx, (col, label) in enumerate(metrics):
        if idx >= len(axes_flat):
            break
        ax = axes_flat[idx]
        gt_col = f"gt_{col}"
        pred_col = f"pred_{col}"
        if gt_col not in df.columns or pred_col not in df.columns:
            continue

        valid = df[[gt_col, pred_col, "species"]].dropna()
        gt_vals = valid[gt_col].values
        pred_vals = valid[pred_col].values

        if len(gt_vals) < 3:
            continue

        slope, intercept, r_value, _, _ = stats.linregress(gt_vals, pred_vals)
        r2 = r_value ** 2

        for sp in sorted(species_markers.keys()):
            mask = valid["species"] == sp
            if mask.sum() == 0:
                continue
            ax.scatter(valid.loc[mask, gt_col], valid.loc[mask, pred_col],
                       marker=species_markers[sp], color=species_colors[sp],
                       s=45, alpha=0.6, label=sp, edgecolors="white", linewidth=0.3)

        x_range = np.array([gt_vals.min(), gt_vals.max()])
        ax.plot(x_range, slope * x_range + intercept, "k-", linewidth=1.0)

        all_vals = np.concatenate([gt_vals, pred_vals])
        lo, hi = all_vals.min() * 0.95, all_vals.max() * 1.05
        if lo == hi:
            lo, hi = lo - 0.1, hi + 0.1
        ax.plot([lo, hi], [lo, hi], "--", color="#999999", linewidth=0.7)

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel("GT", fontsize=12)
        ax.set_ylabel("Pred", fontsize=12)
        ax.set_title(f"{label}\n$R^2 = {r2:.4f}$", fontsize=13)
        ax.tick_params(labelsize=10)
        ax.grid(linewidth=0.2, alpha=0.4)

    # Shared legend from first axes
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=12,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.subplots_adjust(hspace=0.45, wspace=0.35)

    for fmt in ("png", "pdf"):
        fig.savefig(out_dir / f"downstream_comparison_all.{fmt}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Downstream: predicted vs GT comparison")
    parser.add_argument("--checkpoint",
                        default="output/runs/yolo/yolo11m-seg_strategy1/weights/best.pt",
                        help="YOLO checkpoint path")
    parser.add_argument("--strategy", default="strategy1")
    parser.add_argument("--subset", default="test")
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot-only", type=str, default=None,
                        help="Skip inference; regenerate plots from existing CSV")
    args = parser.parse_args()

    # Plot-only mode
    if args.plot_only:
        csv_path = Path(args.plot_only)
        if not csv_path.exists():
            print(f"Error: {csv_path} not found")
            return
        df = pd.read_csv(csv_path)
        out_dir = csv_path.parent
        print(f"Regenerating plots from {csv_path} ({len(df)} samples)")
        make_comparison_plots(df, out_dir)
        print(f"Plots saved to {out_dir}/downstream_*.png|pdf")
        return

    # Include all samples (do NOT exclude bad ones)
    registry = SampleRegistry(include_excluded=True)
    split = get_split(args.strategy, registry, seed=args.seed)
    samples = split[args.subset]
    print(f"Running downstream comparison on {len(samples)} {args.subset} samples "
          f"(strategy={args.strategy}, include_excluded=True)")

    # Run YOLO inference
    predictions = run_yolo_inference(args.checkpoint, samples, args.img_size)

    # Compute GT and predicted downstream metrics for each sample
    rows = []
    for sample in tqdm(samples, desc="Computing downstream metrics"):
        img = load_sample_normalized(sample)
        h, w = img.shape[:2]

        # GT
        gt_ann = load_sample_annotations(sample, h, w)
        gt_result = _flatten_result(
            analyze_sample(img, gt_ann["masks"], gt_ann["labels"], sample_id=sample.uid)
        )

        # Prediction
        if sample.uid in predictions:
            pred_masks, pred_labels = predictions[sample.uid]
            pred_result = _flatten_result(
                analyze_sample(img, pred_masks, pred_labels, sample_id=sample.uid)
            )
        else:
            pred_result = {}

        row = {
            "sample_id": sample.uid,
            "species": sample.species,
            "microscope": sample.microscope,
            "experiment": sample.experiment,
        }
        for k, v in gt_result.items():
            if k != "sample_id":
                row[f"gt_{k}"] = v
        for k, v in pred_result.items():
            if k != "sample_id":
                row[f"pred_{k}"] = v
        rows.append(row)

    df = pd.DataFrame(rows)

    # Save CSV
    out_dir = OUTPUT_DIR / "downstream"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"comparison_yolo_{args.strategy}_{args.subset}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nCSV saved to {csv_path}")

    # Print R² summary
    print(f"\n{'='*60}")
    print("DOWNSTREAM PREDICTED vs GT — R² SUMMARY")
    print(f"{'='*60}")
    make_comparison_plots(df, out_dir)

    print(f"\nPlots saved to {out_dir}/downstream_*.png|pdf")


if __name__ == "__main__":
    main()
