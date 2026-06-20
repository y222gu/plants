"""Export per-channel grayscale PNGs (DAPI / FITC / TRITC) at 1024×1024 for
each UID in v2_assets/, after percentile normalization + 0.5 px Gaussian
smooth. The HTML brightening tool loads these and composites in JS.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "figures_for_paper" / "figure3"))

from src.preprocessing import load_sample_normalized                     # noqa: E402
from generate_dense_features import parse_uid, find_sample               # noqa: E402
from scripts.render_science_art_pred_pca import (                        # noqa: E402
    OUT_ROOT, collect_uids,
)

OUT_SIZE = 1024
SIGMA = 0.5


def main():
    uids = collect_uids()
    print(f"Exporting channel PNGs for {len(uids)} samples …")
    for uid in uids:
        sp, sc, ex, sm = parse_uid(uid)
        rec = find_sample(sp, sc, ex, sm)
        img = load_sample_normalized(rec)                       # (H,W,3) [0,1]
        img = cv2.resize(img, (OUT_SIZE, OUT_SIZE),
                         interpolation=cv2.INTER_LINEAR)
        ch_dir = OUT_ROOT / uid / "channels"
        ch_dir.mkdir(parents=True, exist_ok=True)
        # channel order = (TRITC, FITC, DAPI) — see preprocessing.load_sample_normalized
        for idx, name in enumerate(("tritc", "fitc", "dapi")):
            ch = img[..., idx]
            ch = cv2.GaussianBlur(ch, ksize=(0, 0),
                                  sigmaX=SIGMA, sigmaY=SIGMA)
            u8 = (np.clip(ch, 0, 1) * 255.0).astype(np.uint8)
            Image.fromarray(u8, mode="L").save(ch_dir / f"{name}.png")
        print(f"  {uid}")
    print(f"\nDone → {OUT_ROOT}/<uid>/channels/{{dapi,fitc,tritc}}.png")


if __name__ == "__main__":
    main()
