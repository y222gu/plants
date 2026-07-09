"""Supplementary figure: 18 test-set samples with ground-truth vs. RADIX
prediction masks, side by side.

Selection: at most one sample per (species, microscope, genotype) combination
drawn from the union of the Strategy A in-distribution test split and the
Zeiss zero-shot split, so the figure captures as much species / genotype
diversity as the test set allows. Seeded for reproducibility.

Layout: 18 sample-pairs in a 6 × 6 tile grid. Each pair sits in two
adjacent 28 mm cells: left cell = ground-truth mask, right cell = RADIX
prediction. The species + microscope and genotype labels sit above the
pair; gene/allele genotypes are italicised.

Mask palette and paint order follow the canonical Fig 1a Bio-7 palette
(make_fig1a_anatomy.py / CLAUDE.md), with the Whole Root class painted as
black background. Each tile carries a 100 µm scale bar sized from the
sample's native TIF dimensions.

Output: supp_more_test_prediction_examples_600dpi.png in this directory.
"""

import csv
import random
import re
import sys
from collections import defaultdict, OrderedDict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent.parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(PROJECT))

from src.preprocessing import load_sample_normalized
from src.annotation_utils import parse_yolo_annotations, polygons_to_raw_binary_masks
from src.model_classes import yolo_overlap_false_to_bio7
from make_supp_training_gallery import _uid_to_record

# Reuse helpers from the training-gallery module.
from make_supp_training_gallery import (
    load_species_genotype_map, format_species, format_genotype,
    microscope_for, native_px_for, scale_bar_mm, draw_scale_bar, _norm_uid,
    boost_for_microscope,
)


RADIX_EVAL    = HERE.parent / "eval" / "radix"
TEST_PRED_DIR = RADIX_EVAL / "test"     / "predictions"
ZS_PRED_DIR   = RADIX_EVAL / "oneshot"  / "predictions"
TEST_METRICS  = RADIX_EVAL / "test"     / "metrics_bio7.csv"
ZS_METRICS    = RADIX_EVAL / "oneshot"  / "metrics_bio7.csv"

# UIDs already shown in Figures 2, 3, and 4a. The supplementary figure
# must not repeat them; they get filtered out of the candidate pool.
EXCLUDED_UIDS = {
    # Figure 2 sample galleries (2a Tomato / 2b cereal / 2c Zeiss)
    "Tomato_C10_Exp3_M82_WT_C12ROI1",
    "Tomato_Olympus_Exp12_Suberin_mutants_13-24_19",
    "Tomato_Olympus_Exp5_Solanum_MajkenSlide1_14",
    "Millet_Olympus_Exp4_1-2cm__75",
    "Rice_C10_Exp10_PSY9-4-b",
    "Sorghum_C10_Exp4_E2",
    "Sorghum_Olympus_Exp74_PDB143_03",
    "Rice_Zeiss_Exp1_Image_34",
    "Rice_Zeiss_Exp2_Image_47",
    "Rice_Zeiss_Exp3_Image_38",
    # Figure 3 (Strategy A / B-mono / B-dico panels)
    "Millet_Olympus_Exp4_1-2cm__16",
    "Millet_Olympus_Exp4_1-2cm__26",
    "Rice_Olympus_Exp10_Rhizotron_Wox10-50_4_03",
    "Rice_Zeiss_Exp1_Image_36", "Rice_Zeiss_Exp1_Image_37",
    "Rice_Zeiss_Exp1_Image_39", "Rice_Zeiss_Exp1_Image_41",
    "Rice_Zeiss_Exp1_Image_42", "Rice_Zeiss_Exp1_Image_52",
    "Rice_Zeiss_Exp1_Image_59",
    "Sorghum_C10_Exp4_H1",
    "Sorghum_Olympus_Exp59_1_01",
    "Sorghum_Olympus_Exp71_PDB134_02",
    "Sorghum_Olympus_Exp75_PDB161_03",
    "Sorghum_Olympus_Exp94_LLC3456_05",
    "Tomato_Olympus_Exp11_Suberin_mutants_11-20_10",
    "Tomato_Olympus_Exp13_Suberin_mutants_21-30_07",
    "Tomato_Olympus_Exp3_Lignin_mutants_36",
    "Tomato_Olympus_Exp5_Solanum_MajkenSlide1_36",
    "Tomato_Olympus_Exp6_Solanum_WendySlide1_19",
    # Figure 4a
    "Rice_Olympus_Exp10_Rhizotron_Wox10-15_1_01",
    "Rice_Olympus_Exp11_Rhizotron_WT_1_04",
    "Sorghum_Olympus_Exp88_N3_02",
}

OUT_PATH = HERE / "supp_more_test_prediction_examples_600dpi.png"

# Paper Bio-7 palette (RGB 0-255). Matches figure1/make_fig1a_anatomy.py.
PALETTE = {
    "Epidermis":   ( 10, 147, 150),  # #0a9396
    "Exodermis":   (244, 162,  97),  # #f4a261
    "Cortex":      (148, 210, 189),  # #94d2bd
    "Aerenchyma":  ( 38,  70,  83),  # #264653
    "Endodermis":  (246, 228, 142),  # #f6e48e
    "Vascular":    (231, 111,  97),  # #e76f61
}
# Outer to inner. Aerenchyma is painted last so it overlays the Cortex.
PAINT_ORDER = ["Epidermis", "Exodermis", "Cortex", "Endodermis",
               "Vascular", "Aerenchyma"]

# ── Canvas geometry (mm) ────────────────────────────────────────────────────
CANVAS_W_MM   = 180.0
SIDE_PAD_MM   = 2.0
TOP_PAD_MM    = 6.0        # column headers above row 1
BOT_PAD_MM    = 4.0        # gap above the legend (room for last row's mIoU)
LEGEND_H_MM   = 6.0        # Bio-7 class legend strip at the bottom
SAMPLES_PER_ROW = 2
ROWS          = 5          # 10 samples / 2 per row
LABEL_W_MM    = 6.0        # narrow left-side label column (species + genotype, CCW)
INTRA_GAP_MM  = 1.5        # gap between cells within one sample
INTER_SAMPLE_GAP_MM = 4.0  # gap between the two sample-quartets in a row
ROW_GAP_MM    = 5.5        # gap between rows (fits the mIoU below pred)
COL_HEADER_PT = 7.0        # column headers (Input / GT / Pred)
LABEL_PT      = 6.0        # all other labels (species, genotype, mIoU, scale bar)
CELLS_PER_SAMPLE = 3       # Input + GT + Pred

# TILE_MM derived so the layout fills exactly the 180 mm canvas width
# (no unused right margin).
TILE_MM = ((CANVAS_W_MM - 2 * SIDE_PAD_MM
            - (SAMPLES_PER_ROW - 1) * INTER_SAMPLE_GAP_MM
            - SAMPLES_PER_ROW * LABEL_W_MM
            - SAMPLES_PER_ROW * CELLS_PER_SAMPLE * INTRA_GAP_MM)
           / (SAMPLES_PER_ROW * CELLS_PER_SAMPLE))
DPI           = 600
MM_PER_INCH   = 25.4
RANDOM_SEED   = 42
N_SAMPLES     = 10   # 9 hand-picked + 1 random Zeiss Rice replacement


# Hand-picked retained samples (paper-curated). One additional Zeiss Rice
# sample, randomly chosen with RANDOM_SEED from the unused zero-shot pool,
# is appended at runtime.
FIXED_KEEP_UIDS = [
    # Rice / Olympus
    "Rice_Olympus_Exp10_Rhizotron_Wox10-50_3_06",   # wox10-15
    # Sorghum / C10
    "Sorghum_C10_Exp3_G12",                          # SQR
    "Sorghum_C10_Exp4_G1",                           # SRN
    # Sorghum / Olympus
    "Sorghum_Olympus_Exp59_2_04",                    # SQR
    "Sorghum_Olympus_Exp16_2_02",                    # SRN39
    "Sorghum_Olympus_Exp74_PDB143_15",               # Teshale
    # Solanums
    "Tomato_Olympus_Exp14_Suberin_mutants_31-40_27", # slmyb92
    "Tomato_Olympus_Exp5_Solanum_MajkenSlide1_05",   # S. peruvianum LA0446
    "Tomato_Olympus_Exp8_Solanum_Yeonjoon_Slide3_02",# S. pimpinellifolium LA1589
]
PREVIOUS_ZEISS_UID = "Rice_Zeiss_Exp1_Image_50"


def per_sample_w_mm() -> float:
    """Width of one sample group = label + 3 tiles + 3 intra-gaps (gap
    before label is folded into INTER_SAMPLE_GAP / SIDE_PAD)."""
    return LABEL_W_MM + 3 * TILE_MM + 3 * INTRA_GAP_MM


def thumb_size_mm():
    return TILE_MM


def paint_bio7_mask(bio7_dict: dict) -> np.ndarray:
    """Render the 7-class semantic dict as an RGB image (uint8). Whole Root
    pixels are left as the (black) background."""
    sample_mask = next(iter(bio7_dict.values()))
    h, w = sample_mask.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for cls in PAINT_ORDER:
        m = bio7_dict.get(cls)
        if m is None:
            continue
        img[m > 0] = PALETTE[cls]
    return img


def make_channel_composite(record, microscope: str) -> np.ndarray:
    """Load DAPI/FITC/TRITC TIFs and return the additive composite
    (DAPI=cyan, FITC=yellow, TRITC=red) with the per-microscope boost
    applied - same recipe as Fig 1c and the supplementary training
    gallery."""
    img = load_sample_normalized(record)  # (H, W, 3) channels = (TRITC, FITC, DAPI)
    tritc, fitc, dapi = img[..., 0], img[..., 1], img[..., 2]
    h, w = dapi.shape
    comp = np.zeros((h, w, 3), dtype=np.float32)
    comp[..., 1] += dapi;  comp[..., 2] += dapi   # DAPI  → cyan
    comp[..., 0] += fitc;  comp[..., 1] += fitc   # FITC  → yellow
    comp[..., 0] += tritc                          # TRITC → red
    comp = np.clip(comp, 0, 1)
    comp_u8 = (comp * 255).astype(np.uint8)
    return boost_for_microscope(comp_u8, microscope)


def load_polygons_to_bio7(txt_path: Path, h: int, w: int) -> dict | None:
    if not txt_path.exists():
        return None
    anns = parse_yolo_annotations(txt_path, w, h)  # signature: (path, img_w, img_h)
    raw = polygons_to_raw_binary_masks(anns, h, w)
    # Ensure every raw class slot exists (an empty (H,W) mask if absent).
    for c in range(6):
        if c not in raw:
            raw[c] = np.zeros((h, w), dtype=np.uint8)
    return yolo_overlap_false_to_bio7(raw, h, w)


_PER_SAMPLE_MIOU: dict[str, float] | None = None
BIO_CLASSES_NO_ROOT = ["Vascular", "Exodermis", "Endodermis",
                       "Cortex", "Epidermis", "Aerenchyma"]


def load_per_sample_miou() -> dict[str, float]:
    """Cache uid → sample-level mIoU (mean across the six anatomy classes,
    Whole Root excluded) from the run's eval/{test,oneshot}/metrics_bio7.csv."""
    global _PER_SAMPLE_MIOU
    if _PER_SAMPLE_MIOU is not None:
        return _PER_SAMPLE_MIOU
    out: dict[str, float] = {}
    for path in [TEST_METRICS, ZS_METRICS]:
        if not path.exists():
            continue
        with open(path) as f:
            for row in csv.DictReader(f):
                vals = []
                for c in BIO_CLASSES_NO_ROOT:
                    s = row.get(f"{c}_IoU", "")
                    if s in ("", None) or s.lower() == "nan":
                        continue
                    try:
                        vals.append(float(s))
                    except ValueError:
                        pass
                if vals:
                    out[row["sample_id"]] = sum(vals) / len(vals)
    _PER_SAMPLE_MIOU = out
    return out


def get_native_hw(record) -> tuple[int, int] | None:
    tifs = list(record.image_dir.glob("*.tif"))
    if not tifs:
        return None
    w, h = Image.open(tifs[0]).size
    return h, w


def select_samples():
    """Return the curated 12-sample list for the figure:
       11 hand-picked UIDs from FIXED_KEEP_UIDS, plus one Zeiss Rice
       sample drawn at random from the zero-shot pool (excluding
       previously used UIDs and the previous figure's Zeiss pick).
    The returned list is sorted by species → microscope → genotype so
    the 6-row × 2-sample-per-row layout reads cleanly.
    """
    all_local = _uid_to_record()
    test_recs = [r for u, r in all_local.items() if "_Zeiss_" not in u]
    zs_recs   = [r for u, r in all_local.items() if "_Zeiss_" in u]
    test_ids = {r.uid for r in test_recs}
    by_uid = {r.uid: r for r in (list(test_recs) + list(zs_recs))}

    sp_gt_map = load_species_genotype_map()

    def combo_of(rec):
        sp, g = sp_gt_map.get(_norm_uid(rec.uid), ("", ""))
        if not sp:
            prefix = rec.uid.split("_", 1)[0]
            sp = prefix if prefix != "Tomato" else "Solanum_lycopersicum"
        return (sp, rec.microscope, format_genotype(g))

    rng = random.Random(RANDOM_SEED)

    # Random Zeiss Rice: any zero-shot sample not in EXCLUDED_UIDS and
    # not the prior figure's pick.
    zeiss_pool = [r for r in zs_recs
                  if r.uid not in EXCLUDED_UIDS
                  and r.uid != PREVIOUS_ZEISS_UID
                  and r.uid not in FIXED_KEEP_UIDS]
    if not zeiss_pool:
        raise RuntimeError("no eligible Zeiss zero-shot sample available")
    new_zeiss = rng.choice(zeiss_pool)
    print(f"new Zeiss Rice: {new_zeiss.uid}")

    keep_uids = list(FIXED_KEEP_UIDS) + [new_zeiss.uid]
    records = []
    for uid in keep_uids:
        rec = by_uid.get(uid)
        if rec is None:
            print(f"  ⚠ UID not found: {uid}")
            continue
        records.append(rec)

    # Sort by paper species order → microscope → genotype, with a manual
    # override that pins certain samples to the front (currently: Rice
    # wox10-15 + Sorghum SRN39 paired on row 1 per user request).
    species_priority = ["Rice", "Sorghum", "Millet",
                        "Solanum_lycopersicum"]
    def species_rank(sp):
        if sp in species_priority:
            return species_priority.index(sp)
        return len(species_priority) + (0 if sp.startswith("Solanum_") else 1)
    microscope_priority = ["Olympus", "C10", "Zeiss"]
    def mic_rank(m):
        return microscope_priority.index(m) if m in microscope_priority else 99

    PINNED_FIRST = [
        "Rice_Olympus_Exp10_Rhizotron_Wox10-50_3_06",   # wox10-15
        "Sorghum_Olympus_Exp16_2_02",                    # SRN39
    ]
    pinned_rank = {uid: i for i, uid in enumerate(PINNED_FIRST)}
    def sort_key(r):
        if r.uid in pinned_rank:
            return (0, pinned_rank[r.uid], 0, "")
        return (1,
                species_rank(combo_of(r)[0]),
                mic_rank(r.microscope),
                combo_of(r)[2].lower())
    sorted_records = sorted(records, key=sort_key)

    out = []
    for rec in sorted_records:
        split_name = "test" if rec.uid in test_ids else "zero-shot"
        pred_dir = TEST_PRED_DIR if split_name == "test" else ZS_PRED_DIR
        pred_path = pred_dir / f"{rec.uid}.txt"
        gt_path = rec.annotation_path
        out.append((rec, split_name, combo_of(rec), gt_path, pred_path))
    return out


def main():
    samples = select_samples()
    print(f"Selected {len(samples)} samples covering "
          f"{len({s[2] for s in samples})} (species, microscope, genotype) "
          "combos.")
    for rec, sp_name, combo, gt, pred in samples:
        flag = "" if pred.exists() else "  ⚠ missing prediction"
        print(f"  {sp_name:10s} {rec.uid:55s} {combo}{flag}")

    thumb_mm = thumb_size_mm()
    canvas_h_mm = (TOP_PAD_MM + ROWS * thumb_mm
                   + (ROWS - 1) * ROW_GAP_MM + LEGEND_H_MM + BOT_PAD_MM)
    print(f"\ntile size : {thumb_mm:.2f} mm")
    print(f"canvas    : {CANVAS_W_MM:.1f} × {canvas_h_mm:.1f} mm "
          f"({CANVAS_W_MM/MM_PER_INCH*DPI:.0f} × "
          f"{canvas_h_mm/MM_PER_INCH*DPI:.0f} px) at {DPI} dpi")

    fig = plt.figure(figsize=(CANVAS_W_MM / MM_PER_INCH,
                              canvas_h_mm / MM_PER_INCH),
                     dpi=DPI)
    fig.patch.set_facecolor("black")

    def mm_axes(x_mm, y_mm_top, w_mm, h_mm):
        ax = fig.add_axes([x_mm / CANVAS_W_MM,
                           (canvas_h_mm - y_mm_top - h_mm) / canvas_h_mm,
                           w_mm / CANVAS_W_MM,
                           h_mm / canvas_h_mm])
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        return ax

    # Each sample group occupies a [label | Input | GT | Pred] block. Two
    # sample groups per row, separated by INTER_SAMPLE_GAP_MM.
    sample_w_mm = per_sample_w_mm()
    def sample_block_x(sample_col):
        return (SIDE_PAD_MM
                + sample_col * (sample_w_mm + INTER_SAMPLE_GAP_MM))

    # Column headers across the top - "Input Image | Ground Truth Mask |
    # Predicted Mask" centred over each block's three image tiles (skip
    # the label column). Top margin (canvas top → TOP of header text)
    # matches the bottom margin (canvas bottom → BOTTOM of legend
    # swatch), which sits at (0.5 − sw_h/2) × LEGEND_H_MM = 1.65 mm.
    col_label_top_mm = 0.5 * LEGEND_H_MM - 0.45 * LEGEND_H_MM / 2
    col_titles = ["Input Image", "Ground Truth Mask", "Predicted Mask"]
    for sample_col in range(SAMPLES_PER_ROW):
        x0 = sample_block_x(sample_col) + LABEL_W_MM + INTRA_GAP_MM
        for j, title in enumerate(col_titles):
            x_center_mm = x0 + j * (thumb_mm + INTRA_GAP_MM) + thumb_mm / 2
            fig.text(x_center_mm / CANVAS_W_MM,
                     (canvas_h_mm - col_label_top_mm) / canvas_h_mm,
                     title, ha="center", va="top",
                     fontsize=COL_HEADER_PT, fontweight="bold",
                     color="white", family=["Arial", "DejaVu Sans"])

    sample_miou = load_per_sample_miou()

    # 10 sample groups in a 2-per-row × 5-row layout.
    for trio_idx, (rec, split_name, combo, gt_path, pred_path) in enumerate(samples):
        row_i = trio_idx // SAMPLES_PER_ROW
        sample_col_i = trio_idx % SAMPLES_PER_ROW
        x_block = sample_block_x(sample_col_i)
        x_label = x_block
        x_input = x_block + LABEL_W_MM + INTRA_GAP_MM
        x_mms = [x_input + k * (thumb_mm + INTRA_GAP_MM) for k in range(3)]
        y_mm = TOP_PAD_MM + row_i * (thumb_mm + ROW_GAP_MM)

        # Native image dimensions for rasterising polygons.
        hw = get_native_hw(rec)
        if hw is None:
            print(f"  ⚠ no TIF for {rec.uid}")
            continue
        h, w = hw

        # Original channel composite (with per-microscope boost).
        try:
            orig_img = make_channel_composite(rec, rec.microscope)
        except Exception as e:
            print(f"  ⚠ composite failed for {rec.uid}: {e}")
            orig_img = np.zeros((h, w, 3), dtype=np.uint8)

        # GT and prediction colour masks.
        gt_bio = load_polygons_to_bio7(gt_path, h, w)
        pr_bio = load_polygons_to_bio7(pred_path, h, w)
        gt_img = paint_bio7_mask(gt_bio) if gt_bio is not None else np.zeros((h, w, 3), dtype=np.uint8)
        pr_img = (paint_bio7_mask(pr_bio) if pr_bio is not None
                  else np.zeros((h, w, 3), dtype=np.uint8))

        ax_orig = mm_axes(x_mms[0], y_mm, thumb_mm, thumb_mm)
        ax_gt   = mm_axes(x_mms[1], y_mm, thumb_mm, thumb_mm)
        ax_pred = mm_axes(x_mms[2], y_mm, thumb_mm, thumb_mm)
        ax_orig.imshow(orig_img)
        ax_gt.imshow(gt_img)
        ax_pred.imshow(pr_img)

        # 100 µm scale bar on the ORIGINAL tile.
        sb_mm = scale_bar_mm(rec.uid, thumb_mm)
        if sb_mm is not None:
            draw_scale_bar(ax_orig, sb_mm, thumb_mm,
                           label_fontsize=LABEL_PT,
                           label_fontweight="normal")

        # Left-side rotated label cell: species name (top, rotated CCW) +
        # genotype (below) - same convention as supp_annotator_comparison.
        sp_raw, g_raw = load_species_genotype_map().get(
            _norm_uid(rec.uid), ("", ""))
        sp_disp = format_species(sp_raw) or rec.uid.split("_", 1)[0]
        gt_disp = format_genotype(g_raw)
        gt_italic = bool(
            gt_disp
            and gt_disp == gt_disp.lower()
            and re.fullmatch(r"[a-z0-9\-]+", gt_disp)
            and re.search(r"[a-z]", gt_disp))
        # Plain species name (no "(zero-shot)" suffix - the row's
        # geometry / microscope already conveys that context).
        sp_line = sp_disp

        ax_label = mm_axes(x_label, y_mm, LABEL_W_MM, thumb_mm)
        ax_label.set_facecolor("black")
        ax_label.set_xlim(0, 1); ax_label.set_ylim(0, 1)
        # Rotated 90° CCW → x=0.3 ends up on top, x=0.7 on bottom.
        if sp_line:
            ax_label.text(0.3, 0.5, sp_line,
                          transform=ax_label.transAxes,
                          ha="center", va="center",
                          fontsize=LABEL_PT, color="white",
                          family=["Arial", "DejaVu Sans"], rotation=90)
        if gt_disp:
            kw = dict(transform=ax_label.transAxes, ha="center", va="center",
                      fontsize=LABEL_PT, color="white",
                      family=["Arial", "DejaVu Sans"], rotation=90)
            if gt_italic:
                kw["fontstyle"] = "italic"
            ax_label.text(0.7, 0.5, gt_disp, **kw)

        # Sample-level mIoU just below the PREDICTION tile (in the row gap).
        miou = sample_miou.get(rec.uid)
        if miou is not None:
            ax_pred.text(0.5, -0.04, f"mIoU = {miou:.3f}",
                         transform=ax_pred.transAxes,
                         ha="center", va="top",
                         fontsize=LABEL_PT,
                         color="white", family=["Arial", "DejaVu Sans"])

    # Bottom Bio-7 class legend strip - swatches + labels (same format
    # as supp_annotator_comparison_600dpi).
    from matplotlib.patches import Rectangle
    legend_classes = ["Vascular", "Exodermis", "Endodermis",
                      "Cortex", "Epidermis", "Aerenchyma"]
    leg_ax = fig.add_axes([0.0, 0.0, 1.0, LEGEND_H_MM / canvas_h_mm])
    leg_ax.set_facecolor("black")
    leg_ax.set_xlim(0, 1); leg_ax.set_ylim(0, 1)
    leg_ax.set_xticks([]); leg_ax.set_yticks([])
    for s in leg_ax.spines.values():
        s.set_visible(False)
    sw_w = 0.020
    sw_h = 0.45
    gap_label = 0.008
    gap_entry = 0.020
    char_w = 0.0085
    entry_widths = []
    for cls in legend_classes:
        text_w = char_w * len(cls)
        entry_widths.append(sw_w + gap_label + text_w)
    total_w = sum(entry_widths) + gap_entry * (len(legend_classes) - 1)
    x = (1.0 - total_w) / 2
    for cls, ew in zip(legend_classes, entry_widths):
        rgb = tuple(c / 255 for c in PALETTE[cls])
        leg_ax.add_patch(Rectangle(
            (x, 0.5 - sw_h / 2), sw_w, sw_h,
            facecolor=rgb, edgecolor="none"))
        leg_ax.text(x + sw_w + gap_label, 0.5, cls,
                    ha="left", va="center",
                    fontsize=LABEL_PT, color="white", family=["Arial", "DejaVu Sans"])
        x += ew + gap_entry

    plt.savefig(OUT_PATH, dpi=DPI, facecolor="black",
                bbox_inches=None, pad_inches=0)
    plt.close(fig)
    print(f"\n→ {OUT_PATH.relative_to(PROJECT)}  (saved at {DPI} dpi)")


if __name__ == "__main__":
    main()
