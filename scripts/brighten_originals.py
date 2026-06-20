"""Re-render `original.png` for each UID in v2_assets/ with a higher-contrast
channel composite, then rebuild the 5x6 montage. The mask and PCA panels
are unchanged.

Pipeline per channel (after percentile normalization to [0,1]):
    1. Gaussian smooth, sigma=0.5 px            → suppress single-pixel noise
    2. Triangle-threshold background subtract   → push dim/noise pixels to 0
       out = clip((x - t) / (1 - t), 0, 1)
    3. Sigmoid contrast curve (k=8, mid=0.35)   → bright → brighter,
                                                  mid → dimmer (S-shape)
The bg subtraction is what drives "dim → dimmer". The sigmoid is what
drives "bright → brighter". Together: clean black background outside
the root, vivid signal inside.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "figures_for_paper" / "figure3"))

from src.preprocessing import load_sample_normalized                     # noqa: E402
from src.augmentation import get_val_transform                           # noqa: E402
from generate_dense_features import (                                    # noqa: E402
    parse_uid, find_sample, save_png,
)
from scripts.render_science_art_pred_pca import (                        # noqa: E402
    INPUT_SIZE, OUT_ROOT, N_SAMPLES, build_montage, collect_uids, square_stretch,
)

NOISE_SIGMA_PX = 0.5

# Params from the interactive HTML tool — match its pipeline exactly.
PARAMS = {
    "bgFloor": 0.0,
    "sigK":    0.0,
    "sigMid":  0.35,
    "gamma":   1.0,
    "scaleD":  1.4,
    "scaleF":  1.4,
    "scaleT":  1.4,
}


def sigmoid_contrast(x: np.ndarray, k: float, mid: float) -> np.ndarray:
    """Symmetric sigmoid mapped so x∈[0,1] → out∈[0,1] with f(0)=0, f(1)=1."""
    raw = 1.0 / (1.0 + np.exp(-k * (x - mid)))
    lo = 1.0 / (1.0 + np.exp(-k * (0.0 - mid)))
    hi = 1.0 / (1.0 + np.exp(-k * (1.0 - mid)))
    return np.clip((raw - lo) / max(hi - lo, 1e-6), 0, 1)


def brighten_channel(ch: np.ndarray, scale: float) -> np.ndarray:
    """Smooth → bg-subtract → optional sigmoid → gamma → scale.
    Same order as the HTML tool's processChannel()."""
    smoothed = cv2.GaussianBlur(ch, ksize=(0, 0), sigmaX=NOISE_SIGMA_PX,
                                sigmaY=NOISE_SIGMA_PX)
    bg = PARAMS["bgFloor"]
    out = np.clip((smoothed - bg) / max(1.0 - bg, 1e-6), 0, 1)
    if PARAMS["sigK"] > 0:
        out = sigmoid_contrast(out, PARAMS["sigK"], PARAMS["sigMid"])
    if PARAMS["gamma"] != 1.0:
        out = np.power(out, PARAMS["gamma"])
    return (out * scale).astype(np.float32)


def bright_composite_paper(img_norm: np.ndarray) -> np.ndarray:
    """Channel order in img_norm is (TRITC, FITC, DAPI). Apply per-channel
    brighten, then blend per Fig 1c convention (DAPI=cyan, FITC=yellow,
    TRITC=red)."""
    tritc = brighten_channel(img_norm[..., 0], PARAMS["scaleT"])
    fitc  = brighten_channel(img_norm[..., 1], PARAMS["scaleF"])
    dapi  = brighten_channel(img_norm[..., 2], PARAMS["scaleD"])
    comp = np.zeros_like(img_norm)
    comp[..., 1] += dapi;  comp[..., 2] += dapi    # DAPI  → cyan
    comp[..., 0] += fitc;  comp[..., 1] += fitc    # FITC  → yellow
    comp[..., 0] += tritc                          # TRITC → red
    return np.clip(comp, 0, 1)


def main():
    val_transform = get_val_transform(INPUT_SIZE)
    uids = collect_uids()
    print(f"Brightening {len(uids)} originals with params={PARAMS} "
          f"(σ={NOISE_SIGMA_PX} px) …")
    for uid in uids:
        sp, sc, ex, sm = parse_uid(uid)
        rec = find_sample(sp, sc, ex, sm)
        img_norm = load_sample_normalized(rec)
        img_sq = square_stretch(img_norm, val_transform, INPUT_SIZE)
        save_png(bright_composite_paper(img_sq),
                 OUT_ROOT / uid / "original.png")
        print(f"  brightened {uid}")

    print("\nRebuilding montage …")
    build_montage(uids)


if __name__ == "__main__":
    main()
