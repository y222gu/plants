"""Supplementary figure: ground-truth annotations used for training.

Shows the 18 training-set samples that appear in row 4 and the last two
rows of supp_training_gallery_600dpi.png. Each sample is displayed as
two adjacent tiles: the input channel composite on the left, the
ground-truth Bio-7 anatomy mask on the right. Layout is 6 cols × 6 rows
(3 sample-pairs per row × 6 rows). Same conventions as the test-
predictions figure: 7 pt bold column headers, 6 pt species/genotype
labels overlaid inside the input tile, 100 µm scale bar on the input
tile, paper Bio-7 palette on the mask, black background.

Output: supp_training_examples_600dpi.png in this directory.
"""

import re
import sys
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

# Reuse helpers from the existing supplementary scripts.
from make_supp_training_gallery import (
    FIG1C_DIR, UNCOVERED_DIR, load_species_genotype_map, format_species,
    format_genotype, microscope_for, native_px_for, scale_bar_mm,
    draw_scale_bar, boost_for_microscope, find_image_and_boost,
    _norm_uid,
)
sys.path.insert(0, str(HERE.parent / 'test_prediction_examples'))
from supp_more_test_prediction_examples import (
    paint_bio7_mask, load_polygons_to_bio7, get_native_hw,
)
from make_supp_training_gallery import _uid_to_record


OUT_PATH = HERE / "supp_training_examples_600dpi.png"

# The 18 sample UIDs the user asked for: row 4 of
# supp_training_gallery_600dpi.png plus the last two rows (rows 6 & 7).
CURATED_UIDS = [
    # Row 4 of the training gallery
    "Sorghum_Olympus_Exp32_3_05",
    "Sorghum_Olympus_Exp36_3_01",
    "Sorghum_Olympus_Exp38_3_03",
    "Sorghum_Olympus_Exp42_3_02",
    "Sorghum_Olympus_Exp44_2_01",
    "Sorghum_Olympus_Exp45_4_03",
    # Row 6
    "Sorghum_Olympus_Exp55_2_02",
    "Sorghum_Olympus_Exp56_4_02",
    "Sorghum_Olympus_Exp60_Control_19",
    "Sorghum_Olympus_Exp91_Control_05",
    "Sorghum_C10_Exp5_C7",
    "Tomato_Olympus_Exp5_Solanum_MajkenSlide1_09",
    # Row 7
    "Tomato_Olympus_Exp5_Solanum_MajkenSlide1_44",
    "Tomato_Olympus_Exp10_Suberin_mutants_1-12_05",
    "Tomato_Olympus_Exp9_Suberin_mutants_1-10_30",
    "Tomato_Olympus_Exp1_Lignin_mutants_146",
    "Tomato_Olympus_Exp1_Lignin_mutants_161",
    "Tomato_Olympus_Exp20_WT_-77-82_13",
]
assert len(CURATED_UIDS) == 18

# Canvas geometry - same conventions as supp_test_predictions.
CANVAS_W_MM        = 180.0
SIDE_PAD_MM        = 2.0
TOP_PAD_MM         = 6.0
BOT_PAD_MM         = 2.0
LEGEND_H_MM        = 6.0
SAMPLES_PER_ROW    = 3
ROWS               = 6
LABEL_W_MM         = 6.0
INTRA_GAP_MM       = 1.5
INTER_SAMPLE_GAP_MM = 4.0
ROW_GAP_MM         = 3.0
DPI                = 600
MM_PER_INCH        = 25.4
COL_HEADER_PT      = 7.0
LABEL_PT           = 6.0
CELLS_PER_SAMPLE   = 2     # Input + GT

# TILE_MM derived so the layout fills the 180 mm canvas exactly.
TILE_MM = ((CANVAS_W_MM - 2 * SIDE_PAD_MM
            - (SAMPLES_PER_ROW - 1) * INTER_SAMPLE_GAP_MM
            - SAMPLES_PER_ROW * LABEL_W_MM
            - SAMPLES_PER_ROW * CELLS_PER_SAMPLE * INTRA_GAP_MM)
           / (SAMPLES_PER_ROW * CELLS_PER_SAMPLE))


def per_sample_w_mm() -> float:
    """Width of one sample group = label + 2 tiles + 2 intra-gaps."""
    return LABEL_W_MM + 2 * TILE_MM + 2 * INTRA_GAP_MM


def thumb_size_mm():
    return TILE_MM


def main():
    # Build uid → SampleRecord for GT annotation paths + native TIFs.
    uid_to_record = _uid_to_record()

    sp_gt_map = load_species_genotype_map()

    thumb_mm = thumb_size_mm()
    canvas_h_mm = (TOP_PAD_MM + ROWS * thumb_mm
                   + (ROWS - 1) * ROW_GAP_MM + LEGEND_H_MM + BOT_PAD_MM)
    print(f"tile size : {thumb_mm:.2f} mm")
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

    sample_w_mm = per_sample_w_mm()
    def sample_block_x(sample_col):
        return (SIDE_PAD_MM
                + sample_col * (sample_w_mm + INTER_SAMPLE_GAP_MM))

    # Column headers: "Input Image | Ground Truth Mask" centred over each
    # block's two image tiles. The label column itself has no header.
    # Top margin (canvas top → TOP of header text) matches the bottom
    # margin (canvas bottom → BOTTOM of legend swatch), which sits at
    # (0.5 − sw_h/2) × LEGEND_H_MM = 1.65 mm.
    col_label_top_mm = 0.5 * LEGEND_H_MM - 0.45 * LEGEND_H_MM / 2
    col_titles = ["Input Image", "Ground Truth Mask"]
    for sample_col in range(SAMPLES_PER_ROW):
        x0 = sample_block_x(sample_col) + LABEL_W_MM + INTRA_GAP_MM
        for j, title in enumerate(col_titles):
            x_center_mm = x0 + j * (thumb_mm + INTRA_GAP_MM) + thumb_mm / 2
            fig.text(x_center_mm / CANVAS_W_MM,
                     (canvas_h_mm - col_label_top_mm) / canvas_h_mm,
                     title, ha="center", va="top",
                     fontsize=COL_HEADER_PT, fontweight="bold",
                     color="white", family=["Arial", "DejaVu Sans"])

    missing_pred_or_input = []
    for pair_idx, uid in enumerate(CURATED_UIDS):
        rec = uid_to_record.get(uid)
        if rec is None:
            print(f"  ⚠ no SampleRecord for {uid}")
            continue

        row_i = pair_idx // SAMPLES_PER_ROW
        sample_col_i = pair_idx % SAMPLES_PER_ROW
        x_block = sample_block_x(sample_col_i)
        x_label = x_block
        x_mm_left  = x_block + LABEL_W_MM + INTRA_GAP_MM
        x_mm_right = x_mm_left + thumb_mm + INTRA_GAP_MM
        y_mm       = TOP_PAD_MM + row_i * (thumb_mm + ROW_GAP_MM)

        hw = get_native_hw(rec)
        if hw is None:
            print(f"  ⚠ no TIF for {uid}")
            continue
        h, w = hw

        # Input image - prefer the pre-rendered PNG in fig1c_selections or
        # _uncovered_thumbs (faster and matches the training-gallery boost).
        pil_img, needs_boost = find_image_and_boost(uid)
        if pil_img is None:
            print(f"  ⚠ no input image for {uid}; tile will be blank")
            input_arr = np.zeros((h, w, 3), dtype=np.uint8)
        else:
            input_arr = np.asarray(pil_img)
            if needs_boost:
                input_arr = boost_for_microscope(input_arr, microscope_for(uid))

        # GT mask
        gt_bio = load_polygons_to_bio7(rec.annotation_path, h, w)
        if gt_bio is None:
            print(f"  ⚠ no GT annotations for {uid}")
            gt_img = np.zeros((h, w, 3), dtype=np.uint8)
        else:
            gt_img = paint_bio7_mask(gt_bio)

        ax_input = mm_axes(x_mm_left,  y_mm, thumb_mm, thumb_mm)
        ax_gt    = mm_axes(x_mm_right, y_mm, thumb_mm, thumb_mm)
        ax_input.imshow(input_arr)
        ax_gt.imshow(gt_img)

        # 100 µm scale bar inside the input tile.
        sb_mm = scale_bar_mm(uid, thumb_mm)
        if sb_mm is not None:
            draw_scale_bar(ax_input, sb_mm, thumb_mm,
                           label_fontsize=LABEL_PT,
                           label_fontweight="normal")

        # Left-side rotated label cell (species top / genotype bottom).
        sp_raw, g_raw = sp_gt_map.get(_norm_uid(uid), ("", ""))
        sp_disp = format_species(sp_raw) or uid.split("_", 1)[0]
        gt_disp = format_genotype(g_raw)
        gt_italic = bool(
            gt_disp
            and gt_disp == gt_disp.lower()
            and re.fullmatch(r"[a-z0-9\-]+", gt_disp)
            and re.search(r"[a-z]", gt_disp))
        ax_label = mm_axes(x_label, y_mm, LABEL_W_MM, thumb_mm)
        ax_label.set_facecolor("black")
        ax_label.set_xlim(0, 1); ax_label.set_ylim(0, 1)
        if sp_disp:
            ax_label.text(0.3, 0.5, sp_disp,
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

    # Bottom Bio-7 class legend strip.
    from matplotlib.patches import Rectangle
    # Import the PALETTE from the test-predictions module (same paper palette).
    from supp_more_test_prediction_examples import PALETTE
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
    entry_widths = [sw_w + gap_label + char_w * len(c) for c in legend_classes]
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
    print(f"\n→ {OUT_PATH.relative_to(HERE)}  (saved at {DPI} dpi)")


if __name__ == "__main__":
    main()
