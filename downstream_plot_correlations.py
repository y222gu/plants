"""Compare predicted vs GT downstream measurements.

Generates scatter plots with linear regression, R², and identity line
for each measurement. Points colored by species.

Usage:
    python downstream_plot_correlations.py --gt downstream/gt_measurements.csv --pred downstream/pred_measurements.csv
    python downstream_plot_correlations.py --gt downstream/gt_measurements.csv --pred downstream/pred_measurements.csv --out-dir downstream/
"""
import argparse, csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


MEASUREMENT_COLS = [
    "aerenchyma_ratio",
    "exodermis_TRITC", "exodermis_FITC",
    "endodermis_TRITC", "endodermis_FITC",
    "vascular_TRITC", "vascular_FITC",
]

MEASUREMENT_LABELS = {
    "aerenchyma_ratio": "Aerenchyma Ratio\n(area / whole root)",
    "exodermis_TRITC": "Exodermis\nTRITC Intensity",
    "exodermis_FITC": "Exodermis\nFITC Intensity",
    "endodermis_TRITC": "Endodermis\nTRITC Intensity",
    "endodermis_FITC": "Endodermis\nFITC Intensity",
    "vascular_TRITC": "Vascular\nTRITC Intensity",
    "vascular_FITC": "Vascular\nFITC Intensity",
}

SPECIES_COLORS = {
    "Millet": "#E69F00",
    "Rice": "#0072B2",
    "Sorghum": "#009E73",
    "Tomato": "#CC79A7",
}


def plot_correlation(gt_vals, pred_vals, species_list, metric_name, out_path, ax=None):
    """Plot scatter with linear regression, R², identity line.

    Args:
        gt_vals, pred_vals: arrays of GT and predicted values
        species_list: array of species names for coloring
        metric_name: key from MEASUREMENT_COLS
        out_path: save path (None if using ax)
        ax: matplotlib axis (None to create new figure)
    """
    save_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    # Filter valid pairs (both non-zero for intensity, or both valid for ratio)
    valid = np.isfinite(gt_vals) & np.isfinite(pred_vals)
    gt_v = gt_vals[valid]
    pred_v = pred_vals[valid]
    sp_v = species_list[valid]

    if len(gt_v) < 3:
        ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                ha="center", va="center", fontsize=10)
        if save_fig:
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
        return {"R2": float("nan"), "slope": float("nan"), "intercept": float("nan"), "n": len(gt_v)}

    # Scatter colored by species
    for sp in sorted(set(sp_v)):
        mask = sp_v == sp
        color = SPECIES_COLORS.get(sp, "#999999")
        ax.scatter(gt_v[mask], pred_v[mask], c=color, label=sp,
                   s=20, alpha=0.6, edgecolors="none")

    # Linear regression (handle constant values)
    if np.std(gt_v) < 1e-10 or np.std(pred_v) < 1e-10:
        ax.text(0.5, 0.5, f"Constant values\nn={len(gt_v)}", transform=ax.transAxes,
                ha="center", va="center", fontsize=10)
        if save_fig:
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
        return {"R2": float("nan"), "slope": float("nan"), "intercept": float("nan"), "n": len(gt_v)}

    slope, intercept, r_value, p_value, std_err = stats.linregress(gt_v, pred_v)
    r2 = r_value ** 2

    # Regression line
    x_range = np.array([gt_v.min(), gt_v.max()])
    ax.plot(x_range, slope * x_range + intercept, "r-", linewidth=1.5,
            label=f"y={slope:.3f}x+{intercept:.3f}")

    # Identity line
    all_vals = np.concatenate([gt_v, pred_v])
    lim_min = min(all_vals.min(), 0)
    lim_max = all_vals.max() * 1.05
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", alpha=0.3, linewidth=1)

    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_aspect("equal")

    label = MEASUREMENT_LABELS.get(metric_name, metric_name)
    ax.set_xlabel(f"GT {label}", fontsize=8)
    ax.set_ylabel(f"Predicted {label}", fontsize=8)
    ax.set_title(f"{metric_name}\nR²={r2:.4f}, n={len(gt_v)}", fontsize=9)
    ax.legend(fontsize=6, loc="upper left")
    ax.tick_params(labelsize=7)

    if save_fig:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return {"R2": round(r2, 4), "slope": round(slope, 4),
            "intercept": round(intercept, 4), "n": len(gt_v)}


def main():
    parser = argparse.ArgumentParser(description="Compare predicted vs GT downstream measurements")
    parser.add_argument("--gt", required=True, help="Path to gt_measurements.csv")
    parser.add_argument("--pred", required=True, help="Path to pred_measurements.csv")
    parser.add_argument("--out-dir", default=None, help="Output directory (default: same as --pred)")
    args = parser.parse_args()

    gt_df = pd.read_csv(args.gt)
    pred_df = pd.read_csv(args.pred)

    # Merge on sample_id
    merged = pd.merge(gt_df, pred_df, on="sample_id", suffixes=("_gt", "_pred"))

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path(args.pred).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"GT samples: {len(gt_df)}")
    print(f"Pred samples: {len(pred_df)}")
    print(f"Matched: {len(merged)}")
    print(f"Output: {out_dir}")

    # Publication style
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # Species column (from GT side)
    species = merged["species_gt"].values if "species_gt" in merged.columns else merged.get("species", pd.Series(["unknown"] * len(merged))).values

    # Individual plots + summary (overall)
    summary_rows = []

    print("\n=== Overall ===")
    for metric in MEASUREMENT_COLS:
        gt_col = f"{metric}_gt"
        pred_col = f"{metric}_pred"

        if gt_col not in merged.columns or pred_col not in merged.columns:
            print(f"  Skipping {metric} — column not found")
            continue

        gt_vals = merged[gt_col].values.astype(float)
        pred_vals = merged[pred_col].values.astype(float)

        # Compute stats only (combined plot generated below)
        valid = np.isfinite(gt_vals) & np.isfinite(pred_vals)
        gt_v, pred_v = gt_vals[valid], pred_vals[valid]
        if len(gt_v) >= 3 and np.std(gt_v) > 1e-10 and np.std(pred_v) > 1e-10:
            slope, intercept, r_value, _, _ = stats.linregress(gt_v, pred_v)
            result = {"R2": round(r_value**2, 4), "slope": round(slope, 4),
                      "intercept": round(intercept, 4), "n": len(gt_v)}
        else:
            result = {"R2": float("nan"), "slope": float("nan"),
                      "intercept": float("nan"), "n": len(gt_v)}
        result["metric"] = metric
        result["species"] = "Overall"
        summary_rows.append(result)
        r2_str = f"{result['R2']:.4f}" if not np.isnan(result['R2']) else "   nan"
        sl_str = f"{result['slope']:.4f}" if not np.isnan(result['slope']) else "   nan"
        print(f"  {metric:25s}  R²={r2_str}  slope={sl_str}  n={result['n']}")

    # Per-species breakdown
    unique_species = sorted(set(species))
    for sp in unique_species:
        sp_mask = species == sp
        sp_merged = merged[sp_mask]
        if len(sp_merged) < 3:
            continue

        print(f"\n=== {sp} ({len(sp_merged)} samples) ===")
        for metric in MEASUREMENT_COLS:
            gt_col = f"{metric}_gt"
            pred_col = f"{metric}_pred"
            if gt_col not in sp_merged.columns or pred_col not in sp_merged.columns:
                continue

            gt_vals = sp_merged[gt_col].values.astype(float)
            pred_vals = sp_merged[pred_col].values.astype(float)
            sp_species = np.array([sp] * len(gt_vals))

            # Compute stats only (no individual plot)
            valid = np.isfinite(gt_vals) & np.isfinite(pred_vals)
            gt_v, pred_v = gt_vals[valid], pred_vals[valid]
            if len(gt_v) >= 3 and np.std(gt_v) > 1e-10 and np.std(pred_v) > 1e-10:
                slope, intercept, r_value, _, _ = stats.linregress(gt_v, pred_v)
                result = {"R2": round(r_value**2, 4), "slope": round(slope, 4),
                          "intercept": round(intercept, 4), "n": len(gt_v)}
            else:
                result = {"R2": float("nan"), "slope": float("nan"),
                          "intercept": float("nan"), "n": len(gt_v)}
            result["metric"] = metric
            result["species"] = sp
            summary_rows.append(result)
            r2_str = f"{result['R2']:.4f}" if not np.isnan(result['R2']) else "   nan"
            sl_str = f"{result['slope']:.4f}" if not np.isnan(result['slope']) else "   nan"
            print(f"  {metric:25s}  R²={r2_str}  slope={sl_str}  n={result['n']}")

    # Combined figure — overall (2 rows x 4 cols, last cell empty)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes_flat = axes.flatten()

    for i, metric in enumerate(MEASUREMENT_COLS):
        gt_col = f"{metric}_gt"
        pred_col = f"{metric}_pred"
        if gt_col in merged.columns and pred_col in merged.columns:
            gt_vals = merged[gt_col].values.astype(float)
            pred_vals = merged[pred_col].values.astype(float)
            plot_correlation(gt_vals, pred_vals, species, metric, None, ax=axes_flat[i])

    axes_flat[7].set_visible(False)
    fig.suptitle("Overall", fontsize=12, y=1.02)
    fig.tight_layout()
    combined_path = out_dir / "correlation_combined.png"
    fig.savefig(combined_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nCombined plot saved to {combined_path}")

    # Combined figure — per species
    for sp in unique_species:
        sp_mask = species == sp
        sp_merged = merged[sp_mask]
        if len(sp_merged) < 3:
            continue

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes_flat = axes.flatten()
        sp_species = np.array([sp] * len(sp_merged))

        for i, metric in enumerate(MEASUREMENT_COLS):
            gt_col = f"{metric}_gt"
            pred_col = f"{metric}_pred"
            if gt_col in sp_merged.columns and pred_col in sp_merged.columns:
                gt_vals = sp_merged[gt_col].values.astype(float)
                pred_vals = sp_merged[pred_col].values.astype(float)
                plot_correlation(gt_vals, pred_vals, sp_species, metric, None, ax=axes_flat[i])

        axes_flat[7].set_visible(False)
        fig.suptitle(f"{sp} (n={len(sp_merged)})", fontsize=12, y=1.02)
        fig.tight_layout()
        sp_path = out_dir / f"correlation_combined_{sp}.png"
        fig.savefig(sp_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  {sp} combined plot saved to {sp_path}")

    # Summary CSV (includes per-species rows)
    summary_csv = out_dir / "correlation_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["species", "metric", "R2", "slope", "intercept", "n"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSummary saved to {summary_csv}")

    # Print summary table
    print(f"\n{'Species':10s}  {'Metric':25s}  {'R²':>8s}  {'Slope':>8s}  {'n':>5s}")
    print("-" * 65)
    for row in summary_rows:
        print(f"{row['species']:10s}  {row['metric']:25s}  {row['R2']:8.4f}  {row['slope']:8.4f}  {row['n']:5d}")


if __name__ == "__main__":
    main()
