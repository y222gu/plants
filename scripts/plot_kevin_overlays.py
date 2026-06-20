"""Per-group overlay grids for the Kevin SCZ/EXO1 mutant prediction run.

For each of the 5 sample-type folders, produces a single PNG showing every
sample in the group as two side-by-side panels:
  - left: channel composite (DAPI=cyan, FITC=yellow, TRITC=red), cropped to root bbox
  - right: same composite with the 7-class bio overlay

The saved .npy masks live at the crop resolution and the bbox was never persisted,
so we re-derive the bbox by replaying the coarse inference step from predict.py.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import tifffile
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train.train_timm_semantic import TimmSemanticModule
from src.model_classes import BIO_7_NAMES, unet_semantic_to_bio7
from predict import (
    discover_samples, load_sample_3ch, normalize_percentile,
    run_inference, find_root_bbox, find_checkpoint, CROP_MARGIN_PCT,
)

# Canonical Fig 1a anatomy palette (paper convention, per CLAUDE.md).
# Whole Root is not painted (background). Paint order: outer to inner, then aerenchyma
# last so it sits on top of cortex.
PAPER_PALETTE_HEX = {
    "Epidermis":  "#0a9396",
    "Exodermis":  "#f4a261",
    "Cortex":     "#94d2bd",
    "Endodermis": "#f6e48e",
    "Vascular":   "#e76f61",
    "Aerenchyma": "#264653",
}
PAINT_ORDER = ["Epidermis", "Exodermis", "Cortex", "Endodermis", "Vascular", "Aerenchyma"]


def hex_to_rgb01(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i+2], 16) for i in (0, 2, 4)], dtype=np.float32) / 255.0

DATA_DIR = Path("/Users/yifeigu/Downloads/Kevin_SCZ_EXO1_test_mutants")
PRED_ROOT = DATA_DIR / "predictions_dinov3_dpt"
PRED_DIR = PRED_ROOT / "predictions"
OUT_DIR = PRED_ROOT / "group_overlays"
MODEL_DIR = Path(
    "/Users/yifeigu/Documents/Siobhan_Lab/plants/output/runs/timm/"
    "dpt_meta_facebook_dinov3-vits16-pretrain-lvd1689m_equalw_drop_shuf_dfcel_semantic7c_A/"
    "2026-04-22_001"
)


def channel_composite(raw_3ch: np.ndarray) -> np.ndarray:
    """raw_3ch is (H,W,3) in (TRITC, FITC, DAPI) order, matching load_sample_3ch.
    Returns (H,W,3) float32 in [0,1] with DAPI->cyan, FITC->yellow, TRITC->red."""
    def pnorm(x):
        lo, hi = np.percentile(x, 1.0), np.percentile(x, 99.5)
        return np.clip((x - lo) / (hi - lo), 0, 1) if hi > lo else np.zeros_like(x, np.float32)
    tritc = pnorm(raw_3ch[..., 0].astype(np.float32))
    fitc = pnorm(raw_3ch[..., 1].astype(np.float32))
    dapi = pnorm(raw_3ch[..., 2].astype(np.float32))
    comp = np.zeros((*dapi.shape, 3), dtype=np.float32)
    comp[..., 1] += dapi;  comp[..., 2] += dapi    # DAPI → cyan
    comp[..., 0] += fitc;  comp[..., 1] += fitc    # FITC → yellow
    comp[..., 0] += tritc                          # TRITC → red
    return np.clip(comp, 0, 1)


def make_overlay(rgb01: np.ndarray, sem_mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    bio7 = unet_semantic_to_bio7(sem_mask, *sem_mask.shape)
    out = rgb01.copy()
    for name in PAINT_ORDER:
        m = bio7[name] > 0
        if not m.any():
            continue
        color = hex_to_rgb01(PAPER_PALETTE_HEX[name])
        out[m] = (1 - alpha) * out[m] + alpha * color
    return np.clip(out, 0, 1)


def short_name(sample_id: str) -> str:
    """Drop the long shared date/experiment prefix, keep the distinguishing tail."""
    tail = sample_id.split("/", 1)[1] if "/" in sample_id else sample_id
    return tail.replace("05072026_Tomato_Mutants_ML_", "")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    ckpt = find_checkpoint(MODEL_DIR.resolve())
    print(f"Loading checkpoint: {ckpt.name}")
    model = TimmSemanticModule.load_from_checkpoint(str(ckpt), map_location=device)
    model.eval().to(device)

    all_samples = discover_samples(DATA_DIR.resolve())
    # Skip the auto-created predictions_* candidates that have no channels (defensive).
    all_samples = [(sid, ch) for sid, ch in all_samples if "predictions_dinov3_dpt" not in sid]

    # Group by top-level folder name.
    by_group: dict[str, list] = {}
    for sid, channels in all_samples:
        group = sid.split("/", 1)[0]
        by_group.setdefault(group, []).append((sid, channels))

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=hex_to_rgb01(PAPER_PALETTE_HEX[name]))
        for name in PAINT_ORDER
    ]
    legend_labels = list(PAINT_ORDER)

    # ── Pass 1: inference + composite for every sample, keep results in memory.
    pairs_by_group: dict[str, list] = {}
    for group, items in sorted(by_group.items()):
        items.sort(key=lambda kv: kv[0])
        print(f"\n[{group}] {len(items)} samples")
        pairs_by_group[group] = []
        for sid, channels in items:
            raw = load_sample_3ch(channels)
            comp_full = channel_composite(raw)
            norm = normalize_percentile(raw)
            sem_coarse = run_inference(model, device, norm)
            bbox = find_root_bbox(sem_coarse, margin_pct=CROP_MARGIN_PCT)
            if bbox is None:
                comp_crop = comp_full
            else:
                y0, y1, x0, x1 = bbox
                comp_crop = comp_full[y0:y1, x0:x1]
            sem = np.load(PRED_DIR / f"{sid}.npy")
            if sem.shape != comp_crop.shape[:2]:
                sem = cv2.resize(sem, (comp_crop.shape[1], comp_crop.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
            overlay = make_overlay(comp_crop, sem)
            # Glue orig + overlay into one image so the pair is literally touching.
            pair_img = np.concatenate([comp_crop, overlay], axis=1)
            pairs_by_group[group].append((sid, pair_img))
            print(f"  {sid}")

    # ── Pass 2: build one combined figure with five titled subfigures.
    samples_per_row = 4
    group_order = sorted(pairs_by_group.keys())
    rows_per_group = {
        g: max(1, (len(pairs_by_group[g]) + samples_per_row - 1) // samples_per_row)
        for g in group_order
    }

    slot_w_in = 3.6              # width of one sample slot (orig+overlay glued)
    slot_h_in = 1.9              # height of one slot
    title_pad_in = 0.45          # explicit vertical space for each group title

    fig_w = slot_w_in * samples_per_row + 0.4
    section_heights = [slot_h_in * rows_per_group[g] + title_pad_in for g in group_order]
    fig_h = sum(section_heights)
    fig = plt.figure(figsize=(fig_w, fig_h))

    subfigs = fig.subfigures(
        nrows=len(group_order), ncols=1,
        height_ratios=section_heights,
        hspace=0.0,
    )
    if len(group_order) == 1:
        subfigs = [subfigs]

    for sf, group in zip(subfigs, group_order):
        # Reserve top space for the title so axes don't cover it.
        title_frac = title_pad_in / (slot_h_in * rows_per_group[group] + title_pad_in)
        sf.suptitle(group, fontsize=15, fontweight="bold", y=1.0 - title_frac * 0.25)
        n_rows = rows_per_group[group]
        axes = sf.subplots(n_rows, samples_per_row, squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for idx, (sid, pair_img) in enumerate(pairs_by_group[group]):
            r, c = idx // samples_per_row, idx % samples_per_row
            axes[r, c].imshow(pair_img)
            axes[r, c].set_title(short_name(sid), fontsize=7)
        sf.subplots_adjust(top=1 - title_frac, bottom=0.02,
                           left=0.01, right=0.99,
                           wspace=0.05, hspace=0.25)

    fig.legend(legend_handles, legend_labels, loc="lower center",
               ncol=len(legend_labels), fontsize=10, frameon=False,
               bbox_to_anchor=(0.5, -0.012))
    fig.text(0.5, -0.025, "DAPI=cyan, FITC=yellow, TRITC=red",
             ha="center", va="top", fontsize=9, transform=fig.transFigure)

    out_path = OUT_DIR / "all_groups_combined.png"
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"\n→ {out_path}")
    print(f"Done. Combined plot with {len(group_order)} groups.")


if __name__ == "__main__":
    main()
