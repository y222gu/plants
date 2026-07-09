"""Supplementary figure: each training augmentation applied individually
to one sample, arranged in a 3x4 grid.

Channel composite follows paper convention: DAPI=cyan, FITC=yellow, TRITC=red,
additively blended (same recipe as fig1c). Channel-order arithmetic is done
in the training (TRITC, FITC, DAPI) layout so ChannelDropout/ChannelShuffle
visually drop/permute fluorescence channels rather than display channels.

Output: figures_for_paper/supplementary/supp_augmentations_600dpi.png
"""

import sys
from pathlib import Path

import albumentations as A
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent.parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(PROJECT))

from src.preprocessing import load_sample_normalized
from make_supp_training_gallery import _uid_to_record
from make_supp_training_gallery import (
    scale_bar_mm, draw_scale_bar,
    load_species_genotype_map, format_species, format_genotype,
    _norm_uid,
)
import re
import matplotlib.patheffects as path_effects


SAMPLE_UID = "Sorghum_C10_Exp5_F10"
IMG_SIZE = 1024
SEED = 7
OUT_PATH = HERE / "supp_augmentations_600dpi.png"


def make_composite(img_tfd: np.ndarray) -> np.ndarray:
    """Convert a (H, W, 3) image in (TRITC, FITC, DAPI) channel order to the
    DAPI=cyan, FITC=yellow, TRITC=red composite used in Fig 1c."""
    tritc = img_tfd[..., 0]
    fitc = img_tfd[..., 1]
    dapi = img_tfd[..., 2]
    h, w = dapi.shape
    comp = np.zeros((h, w, 3), dtype=np.float32)
    comp[..., 1] += dapi; comp[..., 2] += dapi
    comp[..., 0] += fitc; comp[..., 1] += fitc
    comp[..., 0] += tritc
    return np.clip(comp, 0, 1)


def aug_pipeline(transform: A.BasicTransform, size: int) -> A.Compose:
    """Wrap one transform with a final resize so every panel has the same
    pixel dimensions as a training crop."""
    return A.Compose([transform, A.Resize(size, size)])


def main():
    uid_to_sample = _uid_to_record()
    if SAMPLE_UID not in uid_to_sample:
        raise SystemExit(f"Sample {SAMPLE_UID} not in supplementary/data/")
    img = load_sample_normalized(uid_to_sample[SAMPLE_UID]).astype(np.float32)

    # Original (just resized) for reference panel
    base_resize = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE)])
    original = base_resize(image=img)["image"]

    # Each entry: (display title, parameter string shown on panel, transform).
    # Parameters are pinned to specific values so the label matches exactly
    # what was applied - no ranges, no random sampling within the transform.
    def _drop_channel(idx):
        def _fn(image, **kw):
            out = image.copy()
            out[..., idx] = 0
            return out
        return A.Lambda(image=_fn, p=1.0)

    def _shuffle_channels(perm):
        def _fn(image, **kw):
            return image[..., list(perm)].copy()
        return A.Lambda(image=_fn, p=1.0)

    panels = [
        ("Original", "no augmentation", None),
        ("Horizontal Flip", "",
         A.HorizontalFlip(p=1.0)),
        ("Vertical Flip", "",
         A.VerticalFlip(p=1.0)),
        ("Rotate", "rotated 90°",
         A.Lambda(image=lambda image, **kw: np.rot90(image, k=1).copy(),
                  p=1.0)),
        ("Affine",
         "translate +8%/-6%, scale 0.78,\nrotate -35°, shear +8°",
         A.Affine(
            translate_percent={"x": 0.08, "y": -0.06},
            scale=0.78, rotate=-35, shear=8,
            border_mode=0, p=1.0,
        )),
        ("Elastic Transform", "α = 600, σ = 12",
         A.ElasticTransform(alpha=600, sigma=12, border_mode=0, p=1.0)),
        ("Brightness / Contrast",
         "brightness +0.25, contrast -0.25",
         A.RandomBrightnessContrast(
            brightness_limit=(0.25, 0.25),
            contrast_limit=(-0.25, -0.25), p=1.0,
        )),
        ("Gamma", "γ = 70",
         A.RandomGamma(gamma_limit=(70, 70), p=1.0)),
        ("Gaussian Blur", "kernel = 7 px",
         A.GaussianBlur(blur_limit=(7, 7), p=1.0)),
        ("Gaussian Noise", "σ = 0.20",
         A.GaussNoise(std_range=(0.20, 0.20), p=1.0)),
        ("Channel Dropout", "dropped FITC channel",
         _drop_channel(1)),
        ("Channel Shuffle", "new order: FITC, DAPI, TRITC",
         _shuffle_channels([1, 2, 0])),
    ]

    renders = []
    for i, (title, params, t) in enumerate(panels):
        if t is None:
            out = original
        else:
            pipe = aug_pipeline(t, IMG_SIZE)
            pipe.set_random_seed(SEED + i)  # deterministic per-panel sample
            out = pipe(image=img)["image"]
        renders.append((title, params, make_composite(out)))

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 7,
        "text.color": "white",
    })

    ncols, nrows = 4, 3
    cell_in = 1.8
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * cell_in, nrows * cell_in + 0.4),
        facecolor="black",
        gridspec_kw={"wspace": 0.04, "hspace": 0.32},
    )
    # Suptitle removed per user request - caption lives in the manuscript.
    fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.01)

    for idx in range(nrows * ncols):
        r, c = idx // ncols, idx % ncols
        ax = axes[r, c]
        ax.set_facecolor("black")
        if idx < len(renders):
            title, params, comp = renders[idx]
            ax.imshow(comp)
            # Title (bold) on first line, parameters (regular) below
            ax.text(0.5, 1.015, title,
                    transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=8, fontweight=900, color="white",
                    family="Helvetica")
            if params:
                ax.text(0.5, -0.02, params,
                        transform=ax.transAxes, ha="center", va="top",
                        fontsize=7, color="white")
            # 100 µm scale bar drawn on the Original tile only - every
            # augmentation panel reshapes or rescales the image, so the
            # bar would mislead there.
            if idx == 0:
                sb_mm = scale_bar_mm(SAMPLE_UID, cell_in * 25.4)
                if sb_mm is not None:
                    draw_scale_bar(ax, sb_mm, cell_in * 25.4,
                                   label_fontsize=7,
                                   label_fontweight="normal")
                # Species + genotype overlay at the top of the Original tile.
                sp_gt = load_species_genotype_map()
                sp_raw, g_raw = sp_gt.get(_norm_uid(SAMPLE_UID), ("", ""))
                sp_disp = format_species(sp_raw) \
                          or SAMPLE_UID.split("_", 1)[0]
                gt_disp = format_genotype(g_raw)
                gt_italic = bool(
                    gt_disp
                    and gt_disp == gt_disp.lower()
                    and re.fullmatch(r"[a-z0-9\-]+", gt_disp)
                    and re.search(r"[a-z]", gt_disp))
                stroke = [path_effects.withStroke(
                    linewidth=0.7, foreground="black")]
                base = dict(transform=ax.transAxes, va="top",
                            fontsize=7, color="white",
                            family="Helvetica", path_effects=stroke)
                if sp_disp and gt_disp:
                    # Combine into one centered string so the visual midpoint
                    # of the whole label sits at x=0.5. Italic-only genotypes
                    # use mathtext so italic styling is preserved.
                    if gt_italic:
                        combined = f"{sp_disp}, $\\mathit{{{gt_disp}}}$"
                    else:
                        combined = f"{sp_disp}, {gt_disp}"
                    ax.text(0.5, 0.97, combined, ha="center", **base)
                elif sp_disp:
                    ax.text(0.5, 0.97, sp_disp, ha="center", **base)
                elif gt_disp:
                    kw = dict(base)
                    if gt_italic:
                        kw["fontstyle"] = "italic"
                    ax.text(0.5, 0.97, gt_disp, ha="center", **kw)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.savefig(OUT_PATH, dpi=600, facecolor="black", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
