"""Supplementary figure: ground truth vs a second internal annotator vs the
best-model prediction, on the 16 samples re-annotated for human-vs-human
agreement.

Layout: one row per sample, four columns:
    Image (DAPI=cyan, FITC=yellow, TRITC=red)
    Ground Truth (canonical Fig 1a anatomy palette on black)
    Internal Annotator (same palette)
    Model Prediction (same palette)

Source paths:
    hh_variance/gt_annotation/{uid}.txt         ground truth (1st annotator)
    hh_variance/annotator2/{uid}.txt            internal annotator (2nd)
    best-model-run/eval/test/predictions/{uid}.txt   model prediction

Output: figures_for_paper/supplementary/supp_annotator_comparison_600dpi.png
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent.parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(PROJECT))

from src.annotation_utils import parse_yolo_annotations, polygons_to_raw_binary_masks
from src.model_classes import fill_contours, yolo_overlap_false_to_bio7
from src.preprocessing import load_sample_normalized

# Shared per-microscope boost and scale-bar helpers (used elsewhere in the
# supplementary folder, kept consistent here).
from make_supp_training_gallery import (
    boost_for_microscope, scale_bar_mm, draw_scale_bar, _uid_to_record,
)


HH_DIR = HERE.parent / "hh_variance"
GT_DIR = HH_DIR / "gt_annotation"
ANN_DIR = HH_DIR / "annotator2"
PRED_DIR = HERE.parent / "eval" / "radix" / "test" / "predictions"

OUT_PATH = HERE / "supp_annotator_comparison_600dpi.png"

# 8 representative samples (positions 1, 3, 4, 7, 9, 10, 13, 14 of the
# alphabetically-sorted 16-sample set): 1 Millet, 2 Rice, 3 Sorghum, 2 Tomato.
SAMPLE_UIDS = [
    "Millet_Olympus_Exp4_1-2cm__08",
    "Rice_C10_Exp7_PSY17-4-a",
    "Rice_Olympus_Exp10_Rhizotron_Wox10-15_4_06",
    "Sorghum_Olympus_Exp75_PDB161_03",
    "Tomato_Olympus_Exp12_Suberin_mutants_13-24_19",
]

# Species + genotype labels (from figure1/Diversity counts.xlsx).
# (species, genotype) - stacked on two lines in the figure.
SAMPLE_LABELS = {
    "Millet_Olympus_Exp4_1-2cm__08":                 ("Millet",          "AW23"),
    "Rice_C10_Exp7_PSY17-4-a":                       ("Rice",            "psy1-17"),
    "Rice_Olympus_Exp10_Rhizotron_Wox10-15_4_06":    ("Rice",            "wox10-15"),
    "Sorghum_Olympus_Exp16_2_01":                    ("Sorghum",         "SRN39"),
    "Sorghum_Olympus_Exp59_1_07":                    ("Sorghum",         "SQR"),
    "Sorghum_Olympus_Exp75_PDB161_03":               ("Sorghum",         "Teshale"),
    "Tomato_Olympus_Exp12_Suberin_mutants_13-24_19": ("S. lycopersicum", "slasft"),
    "Tomato_Olympus_Exp18_WT_38_07":                 ("S. lycopersicum", "M82"),
}

# Names that should be italicised: Latin binomials + gene/allele names.
# Common names (Millet, Rice, Sorghum) and cultivar/line names (AW23, Teshale)
# stay upright.
ITALIC_NAMES = {"S. lycopersicum", "psy1-17", "wox10-15", "slasft"}

# Fig 1a anatomy palette. Whole Root is not painted (covered by inner classes).
COLOR_MAP = {
    "Epidermis":  "#0a9396",
    "Exodermis":  "#f4a261",
    "Cortex":     "#94d2bd",
    "Endodermis": "#f6e48e",
    "Vascular":   "#e76f61",
    "Aerenchyma": "#264653",
}
PAINT_ORDER = ["Epidermis", "Exodermis", "Cortex", "Endodermis", "Vascular", "Aerenchyma"]
# Classes used for the per-panel mIoU (Whole Root excluded by user request).
MIOU_CLASSES = ["Epidermis", "Exodermis", "Cortex", "Endodermis", "Vascular", "Aerenchyma"]


def per_class_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool); b = b.astype(bool)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return float("nan")
    return float(np.logical_and(a, b).sum() / union)


def mean_iou(gt: dict, pred: dict) -> float:
    """Mean IoU over MIOU_CLASSES, ignoring classes absent from both."""
    vals = [per_class_iou(gt[c], pred[c]) for c in MIOU_CLASSES]
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")

# Render at a moderate size to keep memory low; matplotlib upscales for 600 dpi.
RENDER_SIZE = 512


def hex_to_rgb(h: str):
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def render_composite(sample) -> np.ndarray:
    """Channel composite with the same per-microscope brightness /
    saturation / sharpness boost used in Fig 1c and the other supplementary
    figures."""
    img = load_sample_normalized(sample)
    tritc, fitc, dapi = img[..., 0], img[..., 1], img[..., 2]
    h, w = dapi.shape
    comp = np.zeros((h, w, 3), dtype=np.float32)
    comp[..., 1] += dapi; comp[..., 2] += dapi
    comp[..., 0] += fitc; comp[..., 1] += fitc
    comp[..., 0] += tritc
    comp_u8 = (np.clip(comp, 0, 1) * 255).astype(np.uint8)
    return boost_for_microscope(comp_u8, sample.microscope)


def render_bio7_image(bio7: dict, h: int, w: int) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for cls in PAINT_ORDER:
        mask = bio7.get(cls)
        if mask is None:
            continue
        img[mask.astype(bool)] = hex_to_rgb(COLOR_MAP[cls])
    return img


def load_polygons_to_bio7(txt_path: Path, h: int, w: int) -> dict:
    anns = parse_yolo_annotations(txt_path, w, h)
    raw = polygons_to_raw_binary_masks(anns, h, w)
    raw = {k: fill_contours(v) for k, v in raw.items()}
    return yolo_overlap_false_to_bio7(raw, h, w)


def downscale_square(img: np.ndarray, size: int) -> np.ndarray:
    """Letterbox into a square `size x size` canvas with black padding."""
    import cv2
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
    out = np.zeros((size, size, 3), dtype=img.dtype)
    y0 = (size - new_h) // 2
    x0 = (size - new_w) // 2
    out[y0:y0 + new_h, x0:x0 + new_w] = resized
    return out


def species_microscope(uid: str) -> str:
    parts = uid.split("_")
    return f"{parts[0]} / {parts[1]}"


def short_id(uid: str) -> str:
    parts = uid.split("_")
    # "Sorghum_Olympus_Exp16_2_01" → "Exp16_2_01"
    return "_".join(parts[2:])


def main():
    print(f"Rendering {len(SAMPLE_UIDS)} representative samples")

    by_uid = _uid_to_record()

    panels = []  # (uid, comp, gt, ann, pred, ann_miou, pred_miou)
    for uid in SAMPLE_UIDS:
        sample = by_uid.get(uid)
        if sample is None:
            print(f"  [skip] {uid}: no SampleRecord")
            continue
        comp = render_composite(sample)
        h, w = comp.shape[:2]

        gt_bio7 = load_polygons_to_bio7(GT_DIR / f"{uid}.txt", h, w)
        ann_bio7 = load_polygons_to_bio7(ANN_DIR / f"{uid}.txt", h, w)
        pred_path = PRED_DIR / f"{uid}.txt"
        if not pred_path.exists():
            print(f"  [warn] no prediction for {uid}")
            pred_bio7 = {k: np.zeros((h, w), dtype=np.uint8) for k in gt_bio7}
        else:
            pred_bio7 = load_polygons_to_bio7(pred_path, h, w)

        ann_miou = mean_iou(gt_bio7, ann_bio7)
        pred_miou = mean_iou(gt_bio7, pred_bio7)

        comp_s = downscale_square(comp, RENDER_SIZE)
        gt_s = downscale_square(render_bio7_image(gt_bio7, h, w), RENDER_SIZE)
        ann_s = downscale_square(render_bio7_image(ann_bio7, h, w), RENDER_SIZE)
        pred_s = downscale_square(render_bio7_image(pred_bio7, h, w), RENDER_SIZE)

        panels.append((uid, comp_s, gt_s, ann_s, pred_s, ann_miou, pred_miou))
        print(f"  {uid}: ann mIoU={ann_miou:.3f}, pred mIoU={pred_miou:.3f}")

    n = len(panels)
    n_img_cols = 4  # Image, GT, Annotator, Prediction
    col_titles = ["Input Image", "Ground Truth Mask",
                  "Internal Annotator Annotation", "Model Predicted Mask"]

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 6,
        "text.color": "white",
    })

    # Figure width fixed to 180 mm (Nature 2-column).
    MM_PER_IN = 25.4
    fig_w = 180 / MM_PER_IN
    label_in = 0.30  # narrow column for vertical two-line text
    cell_in = (fig_w - label_in) / n_img_cols
    title_in = 0.30  # column-header strip (no suptitle); leaves room for text height
    legend_in = 0.35  # vertical room for class legend at the bottom
    fig_h = cell_in * n + title_in + legend_in
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="black")

    widths = [label_in] + [cell_in] * n_img_cols
    gs = fig.add_gridspec(
        n, len(widths),
        width_ratios=widths,
        hspace=0.10, wspace=0.04,
        left=0.0, right=1.0,
        top=1 - title_in / fig_h, bottom=legend_in / fig_h,
    )

    for idx, (uid, comp, gt, ann, pred, ann_miou, pred_miou) in enumerate(panels):
        # Label cell
        ax_lbl = fig.add_subplot(gs[idx, 0])
        ax_lbl.set_facecolor("black")
        ax_lbl.set_xticks([]); ax_lbl.set_yticks([])
        for s in ax_lbl.spines.values():
            s.set_visible(False)
        species, genotype = SAMPLE_LABELS.get(uid, (uid, ""))
        # Render species and genotype as two separate rotated strings so each
        # can independently be italic. Species sits on the right side of the
        # cell, genotype on the left; together they read as two lines after
        # 90° CCW rotation.
        # Rotated 90° CCW, so the LEFT edge of the unrotated cell becomes
        # the TOP of the rendered label. Put species at x=0.3 so it sits
        # on top, with the genotype underneath at x=0.7.
        for x, name in [(0.3, species), (0.7, genotype)]:
            # Use Arial so italic subface actually renders - Helvetica.ttc's
            # italic face isn't loaded by matplotlib's TTC loader on macOS.
            ax_lbl.text(
                x, 0.5, name,
                transform=ax_lbl.transAxes,
                ha="center", va="center",
                fontsize=6, color="white",
                fontfamily="Arial",
                fontstyle="italic" if name in ITALIC_NAMES else "normal",
                rotation=90,
            )
        # Per-panel mIoU shown only on Internal Annotator (j=2) and Model
        # Prediction (j=3) columns.
        miou_by_col = {2: ann_miou, 3: pred_miou}
        for j, img in enumerate([comp, gt, ann, pred]):
            ax = fig.add_subplot(gs[idx, j + 1])
            ax.set_facecolor("black")
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            # 100 µm scale bar on the leftmost (Image) panel only - GT,
            # Annotator and Prediction share its physical field of view.
            if j == 0:
                sb_mm = scale_bar_mm(uid, cell_in * MM_PER_IN)
                if sb_mm is not None:
                    draw_scale_bar(ax, sb_mm, cell_in * MM_PER_IN,
                                   label_fontsize=6,
                                   label_fontweight="normal")
            if j in miou_by_col and not np.isnan(miou_by_col[j]):
                # Bold mIoU score directly below the panel, Helvetica 7pt.
                ax.text(
                    0.5, -0.02,
                    f"mIoU = {miou_by_col[j]:.3f}",
                    transform=ax.transAxes,
                    ha="center", va="top",
                    fontsize=6, color="white",
                    fontfamily="Helvetica",
                )

    # Column headers above the grid (just below the suptitle)
    # Text baseline 0.20 in from the top edge so the ~0.08 in tall column
    # header text sits fully inside the canvas (no clipping at the top).
    header_y = 1 - 0.20 / fig_h
    for j, title in enumerate(col_titles):
        ax_ref = fig.add_subplot(gs[0, j + 1])
        bbox = ax_ref.get_position()
        ax_ref.remove()
        cx = (bbox.x0 + bbox.x1) / 2
        # matplotlib's TTC loader only reads face 0 of Helvetica.ttc on macOS,
        # so fontweight="bold" + family="Helvetica" silently renders Regular.
        # Arial Bold is a standalone .ttf that loads correctly and is the
        # documented fallback (Helvetica → Arial → DejaVu Sans).
        fig.text(cx, header_y, title, ha="center", va="bottom",
                 fontsize=7, fontweight="bold", color="white",
                 fontfamily="Arial")

    # Suptitle removed per user request - caption lives in the manuscript.

    # Bottom legend strip - one row of swatch + label entries.
    # Order matches Figure 2 (Vascular outward to Epidermis, then Aerenchyma).
    legend_classes = ["Vascular", "Exodermis", "Endodermis", "Cortex",
                      "Epidermis", "Aerenchyma"]
    leg_ax = fig.add_axes([0.0, 0.0, 1.0, legend_in / fig_h])
    leg_ax.set_facecolor("black")
    leg_ax.set_xlim(0, 1); leg_ax.set_ylim(0, 1)
    leg_ax.set_xticks([]); leg_ax.set_yticks([])
    for s in leg_ax.spines.values():
        s.set_visible(False)

    from matplotlib.patches import Rectangle
    n_entries = len(legend_classes)
    sw_w = 0.020   # swatch width  (figure-fraction units)
    sw_h = 0.45    # swatch height (axes y-units)
    gap_label = 0.008   # gap between swatch and its label
    gap_entry = 0.020   # gap between entries

    # Pre-measure rough entry widths using the rendered font; approximate
    # using a fixed character width since matplotlib needs a draw pass for
    # exact extents. Then evenly center the legend row.
    renderer_chars = {c: 0.0085 for c in
                      set("".join(legend_classes).lower())}
    entry_widths = []
    for cls in legend_classes:
        text_w = sum(renderer_chars.get(c.lower(), 0.0085) for c in cls)
        entry_widths.append(sw_w + gap_label + text_w)
    total_w = sum(entry_widths) + gap_entry * (n_entries - 1)
    x = (1.0 - total_w) / 2

    for cls, ew in zip(legend_classes, entry_widths):
        rgb = tuple(c / 255 for c in hex_to_rgb(COLOR_MAP[cls]))
        leg_ax.add_patch(Rectangle(
            (x, 0.5 - sw_h / 2), sw_w, sw_h,
            facecolor=rgb, edgecolor="none",
        ))
        leg_ax.text(
            x + sw_w + gap_label, 0.5, cls,
            ha="left", va="center", fontsize=6, color="white",
        )
        x += ew + gap_entry

    # Use the full figure canvas (no tight crop) so the bottom legend strip
    # stays inside the black background.
    fig.savefig(OUT_PATH, dpi=600, facecolor="black", edgecolor="black",
                bbox_inches=None, pad_inches=0)
    plt.close(fig)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
