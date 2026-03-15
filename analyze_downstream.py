"""Downstream analysis: aerenchyma ratios, channel intensities, instance counts.

Supports two scenarios:
  1. Prediction-only (new data without GT): compute metrics from prediction .txt files
  2. GT comparison (data with annotations): compute from both GT and predictions,
     generate scatter plots with regression lines and R²

Usage:
    # Both GT and prediction comparison (default when annotations exist)
    python analyze_downstream.py --data-dir data/ --source both

    # Prediction-only (new data without GT)
    python analyze_downstream.py --data-dir data/ --source prediction

    # GT-only analysis
    python analyze_downstream.py --data-dir data/ --source gt

    # Regenerate plots from existing CSV
    python analyze_downstream.py --plot-only output/downstream/comparison_yolo_A_test.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.annotation_utils import load_sample_annotations
from src.config import DEFAULT_IMG_SIZE
from src.dataset import SampleRegistry
from src.downstream import analyze_sample
from src.evaluation import convert_yolo_predictions
from src.preprocessing import load_sample_raw
from predict import discover_samples_generic


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

        slope, intercept, r_value, p_value, std_err = stats.linregress(gt_vals, pred_vals)
        r2 = r_value ** 2

        fig, ax = plt.subplots(figsize=(5.5, 5.5))

        for sp in sorted(species_markers.keys()):
            mask = valid["species"] == sp
            if mask.sum() == 0:
                continue
            ax.scatter(valid.loc[mask, gt_col], valid.loc[mask, pred_col],
                       marker=species_markers[sp], color=species_colors[sp],
                       s=60, alpha=0.6, label=sp, edgecolors="white", linewidth=0.4)

        x_range = np.array([gt_vals.min(), gt_vals.max()])
        ax.plot(x_range, slope * x_range + intercept, "k-", linewidth=1.2,
                label=f"$y = {slope:.2f}x + {intercept:.3f}$")

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

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=12,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.subplots_adjust(hspace=0.45, wspace=0.35)

    for fmt in ("png", "pdf"):
        fig.savefig(out_dir / f"downstream_comparison_all.{fmt}")
    plt.close(fig)


def _load_prediction_masks(pred_dir: Path, name: str, h: int, w: int):
    """Load prediction masks from YOLO .txt file."""
    txt_path = pred_dir / f"{name}.txt"
    if not txt_path.exists():
        return None, None
    pred = convert_yolo_predictions(txt_path, h, w)
    return pred.masks, pred.labels


def _load_image_raw_generic(sample: dict) -> np.ndarray:
    """Load 3-channel raw image from a generic sample dict as (H, W, 3) float32.

    No normalization is applied — values remain in the original range
    (e.g. uint16 [0, 65535] or float32 [0, 1]) so that downstream intensity
    measurements reflect the true signal.
    """
    import tifffile
    channels = []
    for ch_name in ["TRITC", "FITC", "DAPI"]:
        img = tifffile.imread(str(sample[ch_name]))
        if img.ndim > 2:
            img = img[0] if img.shape[0] < img.shape[-1] else img[..., 0]
        channels.append(img.astype(np.float32))
    return np.stack(channels, axis=-1)


def main():
    parser = argparse.ArgumentParser(description="Downstream analysis")
    parser.add_argument("--data-dir", default="data/", help="Data folder")
    parser.add_argument("--source", choices=["gt", "prediction", "both"], default=None,
                        help="Analysis source (default: 'both' if GT exists, else 'prediction')")
    parser.add_argument("--checkpoint", default=None,
                        help="YOLO checkpoint (generate predictions if needed)")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating plots")
    parser.add_argument("--plot-only", type=str, default=None,
                        help="Regenerate plots from existing CSV")
    args = parser.parse_args()

    # ── Plot-only mode ──
    if args.plot_only:
        csv_path = Path(args.plot_only)
        if not csv_path.exists():
            print(f"Error: {csv_path} not found")
            return
        df = pd.read_csv(csv_path)
        out_dir = csv_path.parent
        print(f"Regenerating plots from {csv_path} ({len(df)} samples)")
        make_comparison_plots(df, out_dir)
        print(f"Plots saved to {out_dir}")
        return

    data_dir = Path(args.data_dir)
    pred_dir = data_dir / "prediction"
    image_dir = data_dir / "image"

    if not image_dir.exists():
        print(f"Error: {image_dir} does not exist")
        print("Images must be placed in a subfolder called 'image/' inside the data directory.")
        return

    # Discover samples — try SampleRegistry first, fall back to generic discovery
    use_registry = False
    registry_samples = []
    generic_samples = []

    # Try structured discovery (image/{Sp}/{Mic}/{Exp}/{Sample}/)
    try:
        registry_with_gt = SampleRegistry(data_dir=data_dir, require_annotations=True)
        registry_all = SampleRegistry(data_dir=data_dir, require_annotations=False)
        registry_samples = registry_all.samples
        if registry_samples:
            use_registry = True
    except Exception:
        pass

    # Fall back to generic directory discovery inside image/
    if not use_registry:
        generic_samples = discover_samples_generic(image_dir)

    has_gt = use_registry and len(registry_with_gt.samples) > 0
    has_pred = pred_dir.exists() and len(list(pred_dir.glob("*.txt"))) > 0

    # Auto-detect source
    if args.source is None:
        if has_gt and has_pred:
            args.source = "both"
        elif has_pred:
            args.source = "prediction"
        elif has_gt:
            args.source = "gt"
        else:
            print(f"Error: no annotations or predictions found in {data_dir}")
            return
        print(f"Auto-detected source: {args.source}")

    # GT/both modes require registry (structured data with annotations)
    if args.source in ("gt", "both") and not has_gt:
        print(f"Error: --source {args.source} requires annotations in {data_dir}/annotation/")
        return

    # Choose sample list
    if use_registry:
        if args.source in ("gt", "both"):
            samples = registry_with_gt.samples
        else:
            samples = registry_samples
    else:
        samples = generic_samples

    if not samples:
        print(f"No samples found in {data_dir}")
        return

    # Generate predictions if needed
    if args.source in ("prediction", "both") and not has_pred:
        if args.checkpoint:
            print("Predictions not found, generating via predict.py...")
            import subprocess, sys
            subprocess.run([
                sys.executable, "predict.py",
                "--data-dir", str(data_dir),
                "--checkpoint", args.checkpoint,
                "--no-vis",
            ], check=True)
        else:
            print(f"Error: no predictions found in {pred_dir} and no --checkpoint given")
            return

    print(f"Analyzing {len(samples)} samples (source={args.source})")

    # ── Compute downstream metrics ──
    rows = []
    for sample in tqdm(samples, desc="Computing downstream metrics"):
        # Load raw image (no normalization) for accurate intensity measurements
        if use_registry:
            img = load_sample_raw(sample)
            name = sample.uid
            row = {
                "sample_id": name,
                "species": sample.species,
                "microscope": sample.microscope,
                "experiment": sample.experiment,
            }
        else:
            img = _load_image_raw_generic(sample)
            name = sample["name"]
            row = {"sample_id": name}

        h, w = img.shape[:2]

        if args.source in ("gt", "both"):
            gt_ann = load_sample_annotations(sample, h, w)
            gt_result = _flatten_result(
                analyze_sample(img, gt_ann["masks"], gt_ann["labels"], sample_id=name)
            )
            if args.source == "gt":
                for k, v in gt_result.items():
                    if k != "sample_id":
                        row[k] = v
            else:
                for k, v in gt_result.items():
                    if k != "sample_id":
                        row[f"gt_{k}"] = v

        if args.source in ("prediction", "both"):
            pred_masks, pred_labels = _load_prediction_masks(pred_dir, name, h, w)
            if pred_masks is None:
                if args.source == "prediction":
                    print(f"  Skipping {name}: no prediction file")
                    continue
                pred_result = {}
            else:
                pred_result = _flatten_result(
                    analyze_sample(img, pred_masks, pred_labels, sample_id=name)
                )

            if args.source == "prediction":
                for k, v in pred_result.items():
                    if k != "sample_id":
                        row[k] = v
            else:
                for k, v in pred_result.items():
                    if k != "sample_id":
                        row[f"pred_{k}"] = v

        rows.append(row)

    df = pd.DataFrame(rows)

    # ── Save CSV ──
    out_dir = data_dir / "downstream"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.output:
        out_path = Path(args.output)
    elif args.source == "both":
        out_path = out_dir / "comparison.csv"
    elif args.source == "gt":
        out_path = out_dir / "gt.csv"
    else:
        out_path = out_dir / "prediction.csv"

    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    # ── Print summary ──
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"  Samples analyzed: {len(df)}")

    if args.source == "both":
        print(f"\nDOWNSTREAM PREDICTED vs GT — R² SUMMARY")
        print(f"{'='*60}")
        if not args.no_plots:
            make_comparison_plots(df, out_dir)
            print(f"\nPlots saved to {out_dir}")
    else:
        ratio_col = "aerenchyma_ratio"
        count_col = "aerenchyma_count"
        if ratio_col in df.columns:
            print(f"  Aerenchyma ratio: {df[ratio_col].mean():.4f} +/- {df[ratio_col].std():.4f}")
        if count_col in df.columns:
            print(f"  Aerenchyma count: {df[count_col].mean():.1f} +/- {df[count_col].std():.1f}")

        if "species" in df.columns:
            print(f"\nPer Species:")
            for sp, group in df.groupby("species"):
                parts = []
                if ratio_col in df.columns:
                    parts.append(f"ratio={group[ratio_col].mean():.4f}")
                if count_col in df.columns:
                    parts.append(f"count={group[count_col].mean():.1f}")
                print(f"  {sp}: {', '.join(parts)}")


if __name__ == "__main__":
    main()
