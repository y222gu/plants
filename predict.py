"""Deployment inference: DINOv2+DPT on 3-channel microscopy samples.

Per-sample folder layout expected:
    DATA_DIR/
        {sample}_cropped_tifs/                  (or any folder name; all 3 TIFs must share a prefix)
            {sample}_DAPI.tif
            {sample}_FITC.tif
            {sample}_TRITC.tif

Outputs (in OUTPUT_DIR):
    predictions/{sample}.npy   — uint8 semantic argmax mask (0=bg, 1=epi, 2=aer, 3=endo, 4=vasc, 5=exo, 6=cortex)
    vis/{sample}.png           — input + 7-class bio overlay visualization
    measurements.csv           — one row per sample:
        sample_id, aerenchyma_ratio,
        exodermis_{DAPI,FITC,TRITC}, endodermis_{DAPI,FITC,TRITC},
        cortex_{DAPI,FITC,TRITC}, vascular_{DAPI,FITC,TRITC}

Usage:
    Edit DATA_DIR, MODEL_DIR, and (optionally) OUTPUT_DIR below, then:
        python predict.py
"""
import csv
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
import tifffile
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train.train_timm_semantic import TimmSemanticModule
from src.model_classes import (
    unet_semantic_to_bio7,
    BIO_7_NAMES, BIO_7_COLORS_RGB,
)

# ═══════════════════════════════════════════════════════════════════════════════
# ⇩  EDIT THESE TWO PATHS, THEN `python predict.py`  ⇩
# ═══════════════════════════════════════════════════════════════════════════════

# Folder of per-sample subfolders; each must contain *_DAPI.tif, *_FITC.tif, *_TRITC.tif
DATA_DIR = Path("")

# Either a run folder (e.g. .../2026-04-17_002/) containing checkpoints/best-*.ckpt,
# or a direct path to a .ckpt file.
MODEL_DIR = Path("")

# Set to None to default to <DATA_DIR>/predictions_dinov2_dpt/.
OUTPUT_DIR = Path("")

# ═══════════════════════════════════════════════════════════════════════════════

IMG_SIZE = 1024
CHANNEL_NAMES = ["TRITC", "FITC", "DAPI"]   # RGB order as in src/preprocessing.py
MEASURED_REGIONS = ["Exodermis", "Endodermis", "Cortex", "Vascular"]
CROP_MARGIN_PCT = 0.05   # 5% padding around detected root bbox


def find_checkpoint(model_dir: Path) -> Path:
    """Accept either a .ckpt file or a run folder containing checkpoints/best-*.ckpt."""
    if model_dir.is_file() and model_dir.suffix == ".ckpt":
        return model_dir
    if model_dir.is_dir():
        ckpt_dir = model_dir / "checkpoints" if (model_dir / "checkpoints").is_dir() else model_dir
        candidates = sorted(ckpt_dir.glob("best-*.ckpt"))
        if candidates:
            return candidates[-1]  # take the most recent "best-*" if multiple
        # fall back to last.ckpt or any .ckpt
        any_ckpts = sorted(ckpt_dir.glob("*.ckpt"))
        if any_ckpts:
            return any_ckpts[-1]
    raise FileNotFoundError(f"No checkpoint found under {model_dir}")


def discover_samples(data_dir: Path):
    """Return list of (sample_name, {channel: tif_path}) by looking inside each subfolder.

    Sample name = subfolder name. Channel files are matched by any of the common
    naming conventions we've seen:
        *_DAPI.tif                              (training data)
        *CH(DAPI)*                              (Zeiss/Olympus exports: "..._CH(DAPI)_CH1.ome.tif")
        *_DAPI_*.tif / *-DAPI-*.tif             (miscellaneous)
    """
    samples = []
    tif_exts = ("*.tif", "*.tiff")
    for sub in sorted(data_dir.iterdir()):
        if not sub.is_dir() or sub.name.startswith("."):
            continue

        # Collect every TIF in the subfolder
        all_tifs = []
        for ext in tif_exts:
            all_tifs.extend(sub.glob(ext))

        channels = {}
        for ch in CHANNEL_NAMES:
            # Match DAPI/FITC/TRITC as a token, not a substring of another word
            tokens = (f"_{ch}.", f"_{ch}_", f"({ch})", f"-{ch}-", f"-{ch}.")
            for f in all_tifs:
                if any(tok in f.name for tok in tokens):
                    channels[ch] = f
                    break

        if len(channels) == 3:
            samples.append((sub.name, channels))
        else:
            missing = set(CHANNEL_NAMES) - set(channels)
            print(f"  [skip] {sub.name}: missing channels {missing}")
    return samples


def load_channel(path: Path) -> np.ndarray:
    """Load a single-channel TIF as (H, W) float32."""
    img = tifffile.imread(str(path))
    if img.ndim == 3:
        img = img[..., 0]
    return img.astype(np.float32)


def load_sample_3ch(channels: dict):
    """Load 3 TIFs into (H, W, 3) float32 in R=TRITC, G=FITC, B=DAPI order."""
    stacks = [load_channel(channels[ch]) for ch in CHANNEL_NAMES]
    return np.stack(stacks, axis=-1)


def normalize_percentile(img: np.ndarray, p_low=1.0, p_high=99.5):
    out = np.empty_like(img, dtype=np.float32)
    for c in range(img.shape[-1]):
        ch = img[..., c]
        lo = np.percentile(ch, p_low); hi = np.percentile(ch, p_high)
        if hi > lo:
            out[..., c] = np.clip((ch - lo) / (hi - lo), 0, 1)
        else:
            out[..., c] = 0.0
    return out


def to_uint8(img01: np.ndarray) -> np.ndarray:
    return (np.clip(img01, 0, 1) * 255).astype(np.uint8)


def run_inference(model, device, img_norm: np.ndarray) -> np.ndarray:
    """img_norm: (H, W, 3) float32 [0,1] at full resolution. Returns (H, W) uint8 semantic mask."""
    h, w = img_norm.shape[:2]
    resized = cv2.resize(img_norm, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    x = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float().to(device)
    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)


def find_root_bbox(sem_mask: np.ndarray, margin_pct: float = CROP_MARGIN_PCT):
    """Bounding box of all foreground (class >= 1) in a semantic mask + margin.
    Returns (y0, y1, x0, x1) in the mask's own pixel coordinates, or None if empty."""
    fg = sem_mask >= 1
    if not fg.any():
        return None
    ys, xs = np.where(fg)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    bh, bw = y1 - y0, x1 - x0
    my = int(round(bh * margin_pct))
    mx = int(round(bw * margin_pct))
    H, W = sem_mask.shape
    y0 = max(0, y0 - my); y1 = min(H, y1 + my)
    x0 = max(0, x0 - mx); x1 = min(W, x1 + mx)
    return y0, y1, x0, x1


def overlay_bio7(rgb_uint8, bio7_dict, alpha=0.5):
    result = rgb_uint8.astype(np.float32)
    for cls_id, name in enumerate(BIO_7_NAMES):
        m = bio7_dict.get(name)
        if m is None:
            continue
        mask = m > 0
        if not mask.any():
            continue
        color = np.array(BIO_7_COLORS_RGB[cls_id], dtype=np.float32)
        result[mask] = (1 - alpha) * result[mask] + alpha * color
    return np.clip(result, 0, 255).astype(np.uint8)


def save_vis(rgb_u8, bio7, sample_name, out_path):
    """Input + per-region overlays + combined overlay in a 2×4 grid (matches predict_online_*)."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    axes[0, 0].imshow(rgb_u8); axes[0, 0].set_title("Input"); axes[0, 0].axis("off")
    for i, name in enumerate(BIO_7_NAMES):
        row = (i + 1) // 4
        col = (i + 1) % 4
        if row >= 2 or col >= 4:
            break
        ax = axes[row, col]
        mask = bio7[name] > 0
        frac = float(mask.sum()) / mask.size * 100
        color = np.array(BIO_7_COLORS_RGB[i], dtype=np.float32)
        overlay = rgb_u8.astype(np.float32).copy()
        overlay[mask] = 0.4 * overlay[mask] + 0.6 * color
        ax.imshow(overlay.clip(0, 255).astype(np.uint8))
        ax.set_title(f"{name} ({frac:.1f}%)")
        ax.axis("off")
    # last panel: combined
    ax = axes[1, 3]
    ax.imshow(overlay_bio7(rgb_u8, bio7))
    ax.set_title("All classes")
    ax.axis("off")

    fig.suptitle(f"{sample_name} — DINOv2+DPT", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def measure(sample_name: str, raw_img: np.ndarray, bio7: dict) -> dict:
    """Compute aerenchyma ratio + per-region per-channel mean intensity on raw image."""
    h, w = raw_img.shape[:2]
    mh, mw = bio7["Whole Root"].shape[:2]
    if (mh, mw) != (h, w):
        bio7 = {k: cv2.resize(v, (w, h), interpolation=cv2.INTER_NEAREST) for k, v in bio7.items()}

    wr = bio7["Whole Root"]
    aer = bio7["Aerenchyma"]
    aer_ratio = float(aer.sum()) / max(int(wr.sum()), 1)

    row = {"sample_id": sample_name, "aerenchyma_ratio": aer_ratio}
    # raw_img channels: 0=TRITC, 1=FITC, 2=DAPI (matches load_sample_3ch)
    ch_idx = {"TRITC": 0, "FITC": 1, "DAPI": 2}
    for region in MEASURED_REGIONS:
        mask = bio7[region] > 0
        for ch in ("DAPI", "FITC", "TRITC"):
            val = float(raw_img[..., ch_idx[ch]][mask].mean()) if mask.any() else 0.0
            row[f"{region.lower()}_{ch}"] = val
    return row


def main():
    data_dir = DATA_DIR.resolve()
    out_dir = (OUTPUT_DIR.resolve() if OUTPUT_DIR is not None
               else data_dir / "predictions_dinov2_dpt")
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (out_dir / "vis").mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    ckpt = find_checkpoint(MODEL_DIR.resolve())
    print(f"Loading checkpoint: {ckpt}")
    model = TimmSemanticModule.load_from_checkpoint(str(ckpt), map_location=device)
    model.eval().to(device)

    samples = discover_samples(data_dir)
    print(f"Discovered {len(samples)} samples in {data_dir}")
    if not samples:
        print("No samples found.")
        return

    rows = []
    for sample_name, channels in samples:
        print(f"-- {sample_name}")
        raw = load_sample_3ch(channels)          # (H, W, 3) float32 raw
        norm = normalize_percentile(raw)         # (H, W, 3) float32 [0,1]

        # Stage 1: coarse localization on the full image to find root bbox.
        sem_coarse = run_inference(model, device, norm)
        bbox = find_root_bbox(sem_coarse, margin_pct=CROP_MARGIN_PCT)

        if bbox is None:
            print("  [warn] no root detected; using full image")
            raw_crop = raw
            norm_crop = norm
        else:
            y0, y1, x0, x1 = bbox
            raw_crop = raw[y0:y1, x0:x1]
            # Re-normalize on the cropped region so percentiles reflect only tissue pixels.
            norm_crop = normalize_percentile(raw_crop)
            print(f"  root bbox: y=[{y0}:{y1}] x=[{x0}:{x1}]  "
                  f"crop {raw_crop.shape[0]}x{raw_crop.shape[1]} "
                  f"(full {raw.shape[0]}x{raw.shape[1]})")

        # Stage 2: final inference on the cropped region (resized to 1024 inside run_inference).
        sem = run_inference(model, device, norm_crop)
        bio7 = unet_semantic_to_bio7(sem, *sem.shape)

        np.save(out_dir / "predictions" / f"{sample_name}.npy", sem)
        save_vis(to_uint8(norm_crop), bio7, sample_name, out_dir / "vis" / f"{sample_name}.png")
        rows.append(measure(sample_name, raw_crop, bio7))

    # Write one CSV with all samples
    cols = ["sample_id", "aerenchyma_ratio"]
    for region in MEASURED_REGIONS:
        for ch in ("DAPI", "FITC", "TRITC"):
            cols.append(f"{region.lower()}_{ch}")
    csv_path = out_dir / "measurements.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    print(f"\nSaved {len(rows)} samples:")
    print(f"  predictions: {out_dir/'predictions'}")
    print(f"  visualizations: {out_dir/'vis'}")
    print(f"  measurements: {csv_path}")


if __name__ == "__main__":
    main()
