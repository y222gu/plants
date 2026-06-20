"""Export Fig 1c canonical composites (DAPI=cyan, FITC=yellow, TRITC=red)
for every sample under data/train/image/, into the science_art selection folder.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.preprocessing import load_channel, normalize_percentile  # noqa: E402

TRAIN_IMG_ROOT = PROJECT / "data" / "train" / "image"
OUT_DIR = Path("/Users/yifeigu/Documents/Siobhan_Lab/science_art/to be selected")


def make_composite(dapi: np.ndarray, fitc: np.ndarray, tritc: np.ndarray) -> np.ndarray:
    h, w = dapi.shape
    comp = np.zeros((h, w, 3), dtype=np.float32)
    comp[..., 1] += dapi; comp[..., 2] += dapi   # DAPI  → cyan
    comp[..., 0] += fitc; comp[..., 1] += fitc   # FITC  → yellow
    comp[..., 0] += tritc                        # TRITC → red
    return np.clip(comp, 0, 1)


def iter_samples():
    for species_dir in sorted(p for p in TRAIN_IMG_ROOT.iterdir() if p.is_dir()):
        for scope_dir in sorted(p for p in species_dir.iterdir() if p.is_dir()):
            for exp_dir in sorted(p for p in scope_dir.iterdir() if p.is_dir()):
                for sample_dir in sorted(p for p in exp_dir.iterdir() if p.is_dir()):
                    yield species_dir.name, scope_dir.name, exp_dir.name, sample_dir


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    samples = list(iter_samples())
    total = len(samples)
    print(f"Found {total} samples under {TRAIN_IMG_ROOT}")
    n_done = n_skip = n_fail = 0
    for i, (species, scope, exp, sample_dir) in enumerate(samples, 1):
        sample_name = sample_dir.name
        uid = f"{species}_{scope}_{exp}_{sample_name}"
        out_path = OUT_DIR / f"{uid}.png"
        if out_path.exists():
            n_skip += 1
            continue
        try:
            channels = {}
            for ch in ("DAPI", "FITC", "TRITC"):
                channels[ch] = load_channel(sample_dir / f"{sample_name}_{ch}.tif")
            stacked = np.stack([channels["TRITC"], channels["FITC"], channels["DAPI"]], axis=-1)
            stacked = normalize_percentile(stacked.astype(np.float32))
            tritc_n, fitc_n, dapi_n = stacked[..., 0], stacked[..., 1], stacked[..., 2]
            comp = make_composite(dapi_n, fitc_n, tritc_n)
            Image.fromarray((comp * 255).astype(np.uint8)).save(out_path)
            n_done += 1
        except Exception as e:
            n_fail += 1
            print(f"[FAIL] {uid}: {e}")
        if i % 50 == 0 or i == total:
            print(f"  {i}/{total}  done={n_done} skipped={n_skip} failed={n_fail}")
    print(f"\nFinished. Wrote {n_done}, skipped {n_skip}, failed {n_fail}.")


if __name__ == "__main__":
    main()
