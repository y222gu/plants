"""For each PNG in /Users/yifeigu/Documents/Siobhan_Lab/science_art/, run the
best DINOv3+DPT-Meta checkpoint to produce:
    {uid}/original.png   channel composite (DAPI=cyan, FITC=yellow, TRITC=red)
    {uid}/pred.png       Bio-7 colored prediction mask (paper palette)
    {uid}/pca.png        per-image encoder PCA → RGB

Then assemble a 6 × 6 montage where each row holds 2 samples × 3 panels:
    [img1, mask1, pca1, img2, mask2, pca2]
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "figures_for_paper" / "figure3"))

from src.preprocessing import load_sample_normalized                     # noqa: E402
from src.augmentation import get_val_transform                           # noqa: E402
from train.train_timm_semantic import TimmSemanticModule                 # noqa: E402
from generate_dense_features import (                                    # noqa: E402
    parse_uid, find_sample, composite_paper, encode_patches,
    project_with_pca, save_png, pick_device,
)
from fig3a_top10_pca_assets import (                                     # noqa: E402
    dino_pc1_fg_mask, fit_pca_per_model,
)
from figures_for_paper.figure3.fig3a_predictions_assets import (         # noqa: E402
    render_bio7,
)

CKPT = REPO / "output/runs/timm/dpt_meta_facebook_dinov3-vits16-pretrain-lvd1689m_equalw_drop_shuf_dfcel_semantic7c_A/2026-04-22_001/checkpoints/best-epoch=117-val_loss=0.2941.ckpt"
ART_DIR = Path("/Users/yifeigu/Documents/Siobhan_Lab/science_art")
OUT_ROOT = ART_DIR / "v2_assets"
def montage_out_path(rows: int, cols: int = 6) -> Path:
    return ART_DIR / f"science_art_montage_{rows}x{cols}.png"

TRAIN_IMG_SIZE = 1024              # model trained at 1024
INPUT_SIZE = 1536                  # fig3c uses 1536 → 96×96 token grid
PATCH = 16
PCA_GRID = INPUT_SIZE // PATCH

CELL = 800
GAP = 8
COLS = 6
SLOTS_PER_ROW = 2
BG = (0, 0, 0)
N_SAMPLES = 10  # legacy reference; build_montage now infers from count


_SPECIES_PREFIXES = ("Millet_", "Rice_", "Sorghum_", "Tomato_")


def collect_uids():
    uids = []
    for p in sorted(ART_DIR.iterdir()):
        if p.suffix.lower() != ".png" or not p.is_file():
            continue
        if not p.name.startswith(_SPECIES_PREFIXES):
            continue
        stem = p.stem
        for suffix in ("_image", "_gt", "_pred"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        uids.append(stem)
    # de-dup, keep order
    seen, ordered = set(), []
    for u in uids:
        if u not in seen:
            ordered.append(u); seen.add(u)
    return ordered


def square_stretch(img, val_transform, size):
    img_r = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    out = val_transform(image=img_r,
                        mask=np.zeros(img_r.shape[:2], dtype=np.int32))
    return out["image"]


def render_assets(device, dtype):
    print(f"loading checkpoint: {CKPT.name}")
    module = TimmSemanticModule.load_from_checkpoint(
        str(CKPT), map_location="cpu", strict=False,
    )
    module.eval().to(device)
    if dtype != torch.float32:
        module.to(dtype)

    val_transform = get_val_transform(INPUT_SIZE)

    uids = collect_uids()
    print(f"{len(uids)} samples: {uids}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Pass 1: encode + predict at 1536 (fig3c resolution); DINO PC1 fg mask
    per_sample = {}
    for uid in uids:
        out_dir = OUT_ROOT / uid
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"  {uid}")

        sp, sc, ex, sm = parse_uid(uid)
        rec = find_sample(sp, sc, ex, sm)
        img_norm = load_sample_normalized(rec)

        img_sq = square_stretch(img_norm, val_transform, INPUT_SIZE)
        save_png(composite_paper(img_sq), out_dir / "original.png")

        x = (torch.from_numpy(img_sq).permute(2, 0, 1).unsqueeze(0)
             .to(device).to(dtype))

        with torch.inference_mode():
            logits = module(x).float()
        pred_7 = logits.argmax(dim=1)[0].cpu().numpy().astype(np.int32)
        save_png(render_bio7(pred_7), out_dir / "pred.png")

        with torch.inference_mode():
            tokens = encode_patches(module, x).float().cpu().numpy()[0]
        fg = dino_pc1_fg_mask(tokens, PCA_GRID, PCA_GRID)
        per_sample[uid] = {"tokens": tokens, "model_fg": fg, "out_dir": out_dir}
        print(f"    fg={int(fg.sum())}/{PCA_GRID*PCA_GRID} (DINO PC1 + Otsu)")

    # Pass 2: shared PCA(n=3) across all 12 samples' foreground tokens
    # (per-sample mean-centered) — same recipe as figure 3 fig3a_top10.
    print("\nFitting shared 2-pass PCA across 12 samples …")
    pca, lo, hi, means = fit_pca_per_model(per_sample, uids, pct_clip=1.0)
    print(f"  PC ratios={[f'{v:.3f}' for v in pca.explained_variance_ratio_.tolist()]}")

    for uid in uids:
        d = per_sample[uid]
        rgb = project_with_pca(
            d["tokens"], d["model_fg"], pca, lo, hi,
            (PCA_GRID, PCA_GRID), mean=means[uid],
        )
        # Linear upsample (matches fig3c render_panel) — smooths sub-patch holes
        rgb_full = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE),
                              interpolation=cv2.INTER_LINEAR)
        save_png(rgb_full, d["out_dir"] / "pca.png")

    return uids


def build_montage(uids):
    if len(uids) % SLOTS_PER_ROW != 0:
        raise SystemExit(f"sample count {len(uids)} not divisible by "
                         f"{SLOTS_PER_ROW} samples per row")
    rows = len(uids) // SLOTS_PER_ROW
    # tomato first; rest shuffled
    tomato = [u for u in uids if u.lower().startswith("tomato")]
    others = [u for u in uids if not u.lower().startswith("tomato")]
    random.shuffle(tomato); random.shuffle(others)
    ordered = (tomato + others)[: rows * SLOTS_PER_ROW]

    W = COLS * CELL + (COLS - 1) * GAP
    H = rows * CELL + (rows - 1) * GAP
    canvas = Image.new("RGB", (W, H), BG)

    for i, uid in enumerate(ordered):
        row = i // SLOTS_PER_ROW
        col_base = (i % SLOTS_PER_ROW) * 3
        for k, name in enumerate(("original.png", "pca.png", "pred.png")):
            im = Image.open(OUT_ROOT / uid / name).convert("RGB")
            im = im.resize((CELL, CELL), Image.LANCZOS)
            x = (col_base + k) * (CELL + GAP)
            y = row * (CELL + GAP)
            canvas.paste(im, (x, y))
        print(f"  row {row}, slot {i % SLOTS_PER_ROW}: {uid}")

    out_path = montage_out_path(rows, COLS)
    canvas.save(out_path)
    print(f"\nWrote {out_path} ({W}x{H})")


def main():
    device = torch.device("cpu")
    dtype = torch.float32
    print(f"device={device} dtype={dtype}")
    uids = render_assets(device, dtype)
    build_montage(uids)


if __name__ == "__main__":
    main()
