"""Supplementary figure: 8 segmentation model comparison.

Panel a - Training and validation loss curves per model (one mini-plot each).
Panel b - Total parameter count per model (horizontal bar chart).
Panel c - Predictions of all 8 models on 10 random test samples + 5 random
          zero-shot samples, with the input fluorescence composite and the
          ground truth as the first two columns.

Models match Figure 2f. Saved as a 600 dpi PNG to
figures_for_paper/supplementary/.
"""

import glob
import json
import random
import struct
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
from make_supp_training_gallery import _uid_to_record

# Shared per-microscope boost + scale-bar helpers (consistent across supp figs).
from make_supp_training_gallery import (
    boost_for_microscope, scale_bar_mm, draw_scale_bar,
)


EVAL_DIR = HERE.parent / "eval"
LOSS_CSV   = HERE / "model_loss_curves.csv"
PARAMS_CSV = HERE / "model_params.csv"
OUT_PATH = HERE / "model_training_and_prediction_comparison_600dpi.png"
RENDER_SIZE = 384

# 8 models from Figure 2f. `run` is the short subdir name under EVAL_DIR.
MODELS = [
    {"key": "DINOv3 + DPT",       "run": "radix",            "color": "#4e79a7"},
    {"key": "DINOv2 + DPT",       "run": "dinov2_dpt",       "color": "#f28e2b"},
    {"key": "DINOv2 + MS",        "run": "dinov2_ms",        "color": "#59a14f"},
    {"key": "SegDINO",            "run": "segdino",          "color": "#b07aa1"},
    {"key": "ResNet34 + UNet++",  "run": "resnet34_unetpp",  "color": "#76b7b2"},
    {"key": "ResNet50 + UNet++",  "run": "resnet50_unetpp",  "color": "#9c755f"},
    {"key": "MicroSAM + UNETR",   "run": "microsam_unetr",   "color": "#edc948"},
    {"key": "YOLO26m",            "run": "yolo26m",          "color": "#ff9da7"},
]

# Canonical Fig 1a anatomy palette + paint order.
COLOR_MAP = {
    "Epidermis":  "#0a9396",
    "Exodermis":  "#f4a261",
    "Cortex":     "#94d2bd",
    "Endodermis": "#f6e48e",
    "Vascular":   "#e76f61",
    "Aerenchyma": "#264653",
}
PAINT_ORDER = ["Epidermis", "Exodermis", "Cortex", "Endodermis", "Vascular", "Aerenchyma"]
# Classes used for per-panel mIoU (Whole Root excluded).
MIOU_CLASSES = ["Epidermis", "Exodermis", "Cortex", "Endodermis", "Vascular", "Aerenchyma"]

# Loss function used during training, per model. Surfaced in panel a so each
# subplot is self-explanatory; the 7 semantic models share the same custom
# combo, YOLO uses the Ultralytics segmentation losses.
LOSS_NAMES = {
    "DINOv3 + DPT":       "Dice + Focal + wCE + Lovász",
    "DINOv2 + DPT":       "Dice + Focal + wCE + Lovász",
    "DINOv2 + MS":        "Dice + Focal + wCE + Lovász",
    "SegDINO":            "Dice + Focal + wCE + Lovász",
    "ResNet34 + UNet++":  "Dice + Focal + wCE + Lovász",
    "ResNet50 + UNet++":  "Dice + Focal + wCE + Lovász",
    "MicroSAM + UNETR":   "Dice + Focal + wCE + Lovász",
    "YOLO26m":            "Box + Cls + DFL + Seg",
}


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


def hex_to_rgb(h):
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


# ───── tfevents reader (no TF dependency) ───────────────────────────────────

def read_tfevents(path):
    from tensorboard.compat.proto import event_pb2
    with open(path, "rb") as f:
        while True:
            header = f.read(8)
            if len(header) < 8:
                break
            length, = struct.unpack("<Q", header)
            f.read(4)
            payload = f.read(length)
            if len(payload) != length:
                break
            f.read(4)
            ev = event_pb2.Event()
            ev.ParseFromString(payload)
            if ev.summary and ev.summary.value:
                for v in ev.summary.value:
                    if v.HasField("simple_value"):
                        yield ev.step, v.tag, float(v.simple_value)


# ───── Loss curve loaders ───────────────────────────────────────────────────

def load_loss_lightning_csv(run_dir: Path):
    import pandas as pd
    csv_path = run_dir / "logs" / "version_0" / "metrics.csv"
    df = pd.read_csv(csv_path)
    train = df.dropna(subset=["train_loss"]).groupby("epoch")["train_loss"].mean()
    val = df.dropna(subset=["val_loss"]).groupby("epoch")["val_loss"].last()
    epochs = sorted(set(train.index) | set(val.index))
    tl = [float(train[e]) if e in train.index else float("nan") for e in epochs]
    vl = [float(val[e]) if e in val.index else float("nan") for e in epochs]
    return epochs, tl, vl


def load_loss_lightning_tb(run_dir: Path):
    # Lightning sometimes uses version_0, sometimes version_<random>. Scan all.
    files = sorted(glob.glob(str(run_dir / "lightning_logs" / "version_*" / "events.out.tfevents.*")))
    train_pts, val_pts, epoch_map = {}, {}, {}
    for f in files:
        for step, tag, v in read_tfevents(f):
            if tag == "train_loss":
                train_pts.setdefault(step, []).append(v)
            elif tag == "val_loss":
                val_pts.setdefault(step, []).append(v)
            elif tag == "epoch":
                epoch_map[step] = int(round(v))
    if not epoch_map:
        return [], [], []
    # Map step → epoch using nearest previous epoch reading.
    sorted_epochs = sorted(epoch_map.items())
    def step_to_epoch(s):
        e = 0
        for ss, ee in sorted_epochs:
            if ss <= s:
                e = ee
            else:
                break
        return e
    train_by_epoch, val_by_epoch = {}, {}
    for s, vs in train_pts.items():
        e = step_to_epoch(s)
        train_by_epoch.setdefault(e, []).extend(vs)
    for s, vs in val_pts.items():
        e = step_to_epoch(s)
        val_by_epoch.setdefault(e, []).extend(vs)
    epochs = sorted(set(train_by_epoch) | set(val_by_epoch))
    tl = [float(np.mean(train_by_epoch[e])) if e in train_by_epoch else float("nan") for e in epochs]
    vl = [float(np.mean(val_by_epoch[e])) if e in val_by_epoch else float("nan") for e in epochs]
    return epochs, tl, vl


def load_loss_yolo(run_dir: Path):
    files = sorted(glob.glob(str(run_dir / "tensorboard" / "events.out.tfevents.*")))
    train_tags = {"train/train/box_loss", "train/train/cls_loss",
                  "train/train/dfl_loss", "train/train/seg_loss"}
    val_tags = {"ultralytics/val/box_loss", "ultralytics/val/cls_loss",
                "ultralytics/val/dfl_loss", "ultralytics/val/seg_loss"}
    train_by_step, val_by_step = {}, {}
    for f in files:
        for step, tag, v in read_tfevents(f):
            if tag in train_tags:
                train_by_step.setdefault(step, {})[tag] = v
            elif tag in val_tags:
                val_by_step.setdefault(step, {})[tag] = v
    epochs = sorted(set(train_by_step) | set(val_by_step))
    def total(d, e, expected):
        r = d.get(e, {})
        if not r:
            return float("nan")
        return float(sum(r.values()))
    tl = [total(train_by_step, e, len(train_tags)) for e in epochs]
    vl = [total(val_by_step, e, len(val_tags)) for e in epochs]
    return epochs, tl, vl


def load_loss(model):
    rd = RUNS_DIR / model["run"]
    if model["fmt"] == "lightning_csv":
        return load_loss_lightning_csv(rd)
    if model["fmt"] == "lightning_tb":
        return load_loss_lightning_tb(rd)
    if model["fmt"] == "yolo":
        return load_loss_yolo(rd)
    raise ValueError(model["fmt"])


# ───── Param counters ───────────────────────────────────────────────────────

def count_params(model):
    import torch
    rd = RUNS_DIR / model["run"]
    if model["fmt"] in ("lightning_csv", "lightning_tb"):
        ckpts = sorted(glob.glob(str(rd / "checkpoints" / "best-*.ckpt")))
        if not ckpts:
            return float("nan")
        ck = torch.load(ckpts[0], map_location="cpu", weights_only=False)
        sd = ck.get("state_dict", ck)
        return float(sum(v.numel() for v in sd.values() if hasattr(v, "numel"))) / 1e6
    if model["fmt"] == "yolo":
        wt = rd / "weights" / "best.pt"
        ck = torch.load(str(wt), map_location="cpu", weights_only=False)
        mdl = ck.get("model")
        if hasattr(mdl, "parameters"):
            return float(sum(p.numel() for p in mdl.parameters())) / 1e6
        if isinstance(mdl, dict):
            return float(sum(v.numel() for v in mdl.values() if hasattr(v, "numel"))) / 1e6
        return float("nan")
    return float("nan")


# ───── Mask / image rendering ───────────────────────────────────────────────

def load_polygons_to_bio7(txt_path: Path, h: int, w: int) -> dict:
    anns = parse_yolo_annotations(txt_path, w, h)
    raw = polygons_to_raw_binary_masks(anns, h, w)
    raw = {k: fill_contours(v) for k, v in raw.items()}
    return yolo_overlap_false_to_bio7(raw, h, w)


def render_bio7(bio7, h, w):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for cls in PAINT_ORDER:
        m = bio7.get(cls)
        if m is None:
            continue
        img[m.astype(bool)] = hex_to_rgb(COLOR_MAP[cls])
    return img


def render_composite(sample):
    img = load_sample_normalized(sample)
    tritc, fitc, dapi = img[..., 0], img[..., 1], img[..., 2]
    h, w = dapi.shape
    comp = np.zeros((h, w, 3), dtype=np.float32)
    comp[..., 1] += dapi; comp[..., 2] += dapi
    comp[..., 0] += fitc; comp[..., 1] += fitc
    comp[..., 0] += tritc
    comp_u8 = (np.clip(comp, 0, 1) * 255).astype(np.uint8)
    # Same per-microscope brightness/saturation/sharpness boost as Fig 1c
    # and every other panel-b style figure in the supplementary set.
    return boost_for_microscope(comp_u8, sample.microscope)


def downscale_square(img, size):
    import cv2
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh = int(round(h * scale)); nw = int(round(w * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    out = np.zeros((size, size, 3), dtype=img.dtype)
    y0 = (size - nh) // 2; x0 = (size - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = resized
    return out


# ───── Load loss curves + param counts from local CSVs ──────────────────────

def build_cache(force: bool = False):
    cache = {"loss": {}, "params": {}}

    import csv as _csv
    with open(LOSS_CSV) as f:
        for r in _csv.DictReader(f):
            m = r["model"]
            cache["loss"].setdefault(m, {"epoch": [], "train": [], "val": []})
            cache["loss"][m]["epoch"].append(int(r["epoch"]))
            cache["loss"][m]["train"].append(float(r["train_loss"]))
            cache["loss"][m]["val"].append(float(r["val_loss"]))

    with open(PARAMS_CSV) as f:
        for r in _csv.DictReader(f):
            cache["params"][r["model"]] = float(r["params_M"])
    return cache


# ───── Main ─────────────────────────────────────────────────────────────────

def main(force_cache: bool = False):
    cache = build_cache(force=force_cache)

    # Sample selection - hand-picked from the earlier 10-test/5-oneshot random
    # set (the 9th and 10th test samples and the 5th zero-shot sample).
    selected_test = [
        "Tomato_Olympus_Exp1_Lignin_mutants_132",
        "Sorghum_Olympus_Exp88_N3_22",
    ]
    selected_one = [
        "Rice_Zeiss_Exp1_Image_65",
        "Rice_Zeiss_Exp1_Image_34",
        "Rice_Zeiss_Exp1_Image_50",
    ]

    # Species + genotype labels (from figure1/Diversity counts.xlsx). The
    # `sp_italic` flag controls whether the species/genotype line is rendered
    # in italics (Latin binomials only - cultivars stay upright).
    SAMPLE_META = {
        "Tomato_Olympus_Exp1_Lignin_mutants_132": ("S. lycopersicum, M82", True),
        "Sorghum_Olympus_Exp88_N3_22":            ("Sorghum, Teshale",    False),
        "Rice_Zeiss_Exp1_Image_65":               ("Rice, Kitaake",       False),
        "Rice_Zeiss_Exp1_Image_34":               ("Rice, Kitaake",       False),
        "Rice_Zeiss_Exp1_Image_50":               ("Rice, Kitaake",       False),
    }

    by_uid = _uid_to_record()

    print("Rendering panel c samples...")
    rows = []
    for split, uids in [("test", selected_test), ("oneshot", selected_one)]:
        for uid in uids:
            sample = by_uid[uid]
            comp = render_composite(sample); h, w = comp.shape[:2]
            gt_bio7 = load_polygons_to_bio7(sample.annotation_path, h, w)
            preds, mious = [], []
            for m in MODELS:
                pred_txt = EVAL_DIR / m["run"] / split / "predictions" / f"{uid}.txt"
                if pred_txt.exists():
                    pb = load_polygons_to_bio7(pred_txt, h, w)
                else:
                    pb = {k: np.zeros((h, w), dtype=np.uint8) for k in gt_bio7}
                preds.append(pb)
                mious.append(mean_iou(gt_bio7, pb))
            comp_s = downscale_square(comp, RENDER_SIZE)
            gt_s = downscale_square(render_bio7(gt_bio7, h, w), RENDER_SIZE)
            pred_s = [downscale_square(render_bio7(p, h, w), RENDER_SIZE) for p in preds]
            rows.append((uid, split, comp_s, gt_s, pred_s, mious))
            print(f"  {split}: {uid}  mIoU=" +
                  " ".join(f"{m['key'].split()[0]}:{v:.2f}" for m, v in zip(MODELS, mious)))

    # ───── Layout ─────
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 6,
        "text.color": "black",
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
    })

    MM = 25.4
    fig_w_in = 180 / MM
    panel_a_in = 2.4
    n_rows_b = len(rows)
    # Label col only needs to fit the thin grey rectangle (~5mm) so it can be
    # very narrow now that species/genotype lives inside the Image panel.
    label_ratio = 0.25
    n_img_cols = 2 + len(MODELS)
    # Square image cells: width = (figure_w * usable_fraction) / (label + N).
    row_h_in = (180 / MM * 0.99) / (label_ratio + n_img_cols)
    panel_b_in = row_h_in * n_rows_b + 0.7
    # No shared panel titles - just enough headroom for per-subplot bold titles
    # (panel a) and rotated column headers (panel b).
    title_a_in = 0.30
    title_b_in = 0.20
    legend_in = 0.30   # bottom legend strip
    # NOTE: keep this consistent with `gap_in` below - the constant gap between
    # panel a and panel b is allocated here so panel_b doesn't get compressed.
    _gap_between = 0.55
    fig_h_in = title_a_in + panel_a_in + _gap_between + title_b_in + panel_b_in + legend_in

    fig = plt.figure(figsize=(fig_w_in, fig_h_in), facecolor="white")

    # Two separate vertically-stacked gridspecs so panel a can keep room for
    # y-axis labels (left=0.05) while panel b extends nearly edge-to-edge
    # (left=0.005). Both share the same vertical schedule:
    # [ title_a · panel_a · gap · title_b · panel_b ].
    gap_in = 0.55
    top = 0.998
    bottom = 0.005
    panel_a_bottom = top - (title_a_in + panel_a_in) / fig_h_in
    panel_b_top = panel_a_bottom - gap_in / fig_h_in
    panel_b_bottom = bottom

    gs_panel_a = fig.add_gridspec(
        2, 1,
        height_ratios=[title_a_in, panel_a_in],
        hspace=0.0,
        left=0.05, right=0.985,
        top=top, bottom=panel_a_bottom,
    )
    gs_panel_b = fig.add_gridspec(
        2, 1,
        height_ratios=[title_b_in, panel_b_in],
        hspace=0.0,
        left=0.005, right=0.995,
        top=panel_b_top, bottom=panel_b_bottom + legend_in / fig_h_in,
    )

    # Shared panel titles removed per user request - per-plot bold titles
    # below + the figure caption in the manuscript carry the same info.

    # ── Panel a ──
    gs_a = gs_panel_a[1].subgridspec(2, 4, hspace=0.60, wspace=0.45)
    for i, m in enumerate(MODELS):
        r, c = i // 4, i % 4
        ax = fig.add_subplot(gs_a[r, c])
        ax.set_facecolor("white")
        L = cache["loss"][m["key"]]
        ep = L["epoch"]; tl = L["train"]; vl = L["val"]
        if ep:
            ax.plot(ep, tl, color=m["color"], lw=0.9, label="Train")
            ax.plot(ep, vl, color="black", lw=0.7, ls="--", label="Val")
        # In panel a we surface RADIX's full project name; the run still
        # carries the shorter "DINOv3 + DPT" label everywhere else (including
        # the panel-b column header) so it matches Fig 2f.
        title_a = "RADIX (DINOv3 + DPT)" if m["key"] == "DINOv3 + DPT" else m["key"]
        loss_name = LOSS_NAMES.get(m["key"], "")
        # Model name bold (via Arial - matplotlib's TTC loader can't render
        # Helvetica Bold on macOS); loss-function line below in regular weight,
        # with extra vertical gap between the two lines.
        ax.text(0.5, 1.14, title_a,
                transform=ax.transAxes,
                ha="center", va="bottom",
                fontsize=6, fontweight="bold", fontfamily="Arial",
                color="black")
        if loss_name:
            ax.text(0.5, 1.015, f"Loss: {loss_name}",
                    transform=ax.transAxes,
                    ha="center", va="bottom",
                    fontsize=6, color="black")
        ax.tick_params(labelsize=6, colors="black", length=2, pad=1)
        for s in ax.spines.values():
            s.set_color("black"); s.set_linewidth(0.4)
        # X- and Y-axis labels on every subplot.
        ax.set_xlabel("Epoch", fontsize=6, color="black", labelpad=1)
        ax.set_ylabel("Loss", fontsize=6, color="black", labelpad=2)
        # Legend on every subplot: short Train / Val (loss function in title).
        ax.legend(loc="upper right", fontsize=6, frameon=False,
                  labelcolor="black", handlelength=1.4, borderaxespad=0.2)

    pos = gs_a[0, 0].get_position(fig)
    fig.text(0.005, pos.y1 + 0.005, "a", fontsize=12, fontweight="bold",
             color="black", ha="left", va="bottom", fontfamily="Arial")

    # ── Panel b (predictions) ──
    n_cols = 2 + len(MODELS)  # Image, GT, 8 models
    gs_b = gs_panel_b[1].subgridspec(
        n_rows_b + 1, n_cols + 1,
        hspace=0.08, wspace=0.015,
        width_ratios=[label_ratio] + [1] * n_cols,
        height_ratios=[0.30] + [1] * n_rows_b,
    )

    # Per-column header. Model columns get a "Pred." suffix on a new line so
    # each header reads as "{model}\nPred." (RADIX stays on its own line).
    def header_for(name):
        if name == "DINOv3 + DPT":
            return "RADIX\n(DINOv3 + DPT)\nPred. Mask"
        return f"{name}\nPred. Mask"
    col_titles = ["Input Image", "G.T. Mask"] + [header_for(m["key"]) for m in MODELS]
    # Header row - tilted ~45° so labels don't overrun the panel-b title.
    for j, t in enumerate(col_titles):
        ax_h = fig.add_subplot(gs_b[0, j + 1])
        ax_h.set_facecolor("white"); ax_h.set_xticks([]); ax_h.set_yticks([])
        for s in ax_h.spines.values():
            s.set_visible(False)
        ax_h.text(0.5, 0.0, t, transform=ax_h.transAxes,
                  ha="left", va="bottom", fontsize=6, color="black",
                  fontweight="bold", fontfamily="Arial",
                  rotation=45, rotation_mode="anchor",
                  linespacing=1.8)

    # Per-row content: empty label-column cell (just the shared test-split
    # rectangle is drawn later), mask images, mIoU score, and the
    # species/genotype label overlaid on top-center of the Image panel.
    for r, (uid, split, comp, gt, preds, mious) in enumerate(rows):
        ax_lbl = fig.add_subplot(gs_b[r + 1, 0])
        # Transparent so the shared figure-level grey rectangle shows through.
        ax_lbl.patch.set_alpha(0)
        ax_lbl.set_xticks([]); ax_lbl.set_yticks([])
        for s in ax_lbl.spines.values():
            s.set_visible(False)
        species_geno, sp_italic = SAMPLE_META.get(uid, (uid, False))
        for j, img in enumerate([comp, gt] + preds):
            ax = fig.add_subplot(gs_b[r + 1, j + 1])
            # Panel facecolor = white so inter-panel gaps look clean; the
            # rendered mask image itself still has a black canvas inside.
            ax.set_facecolor("white")
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            # Species + genotype overlaid on the Image panel (top-center),
            # plus a 100 µm scale bar at the bottom-right.
            if j == 0:
                ax.text(0.5, 0.97, species_geno,
                        transform=ax.transAxes,
                        ha="center", va="top",
                        fontsize=5, color="white",
                        fontstyle="italic" if sp_italic else "normal")
                pos = gs_b[r + 1, j + 1].get_position(fig)
                cell_w_mm = (pos.x1 - pos.x0) * fig_w_in * 25.4
                # Native composite is letterboxed inside the square render
                # canvas, so compute the actual right/bottom edges of the
                # image content and tuck the scale bar inside them.
                ih, iw = comp.shape[:2]
                content_w_frac = iw / max(ih, iw)
                content_h_frac = ih / max(ih, iw)
                right_edge = 0.5 + content_w_frac / 2
                bottom_edge = 0.5 - content_h_frac / 2
                # Inset the bar 6 % of the cell width inward from the
                # inner right edge of the image - matches the default
                # x_end=0.94 used by the annotator comparison figure so
                # the right-edge gap reads visually the same.
                x_end = right_edge - 0.06
                y_bar = bottom_edge + 0.05
                # The image only fills `content_w_frac` of the cell
                # horizontally, so the physical width represented by the
                # cell width is cell_w_mm / content_w_frac. Feed that to
                # scale_bar_mm so sb_mm matches the actual axes scale.
                effective_tile_mm = cell_w_mm * content_w_frac
                sb_mm = scale_bar_mm(uid, effective_tile_mm)
                if sb_mm is not None:
                    # Pass the full cell width as `tile_mm` so the
                    # axes-fraction (sb_mm / tile_mm) is correct in the
                    # 0..1 axes coordinate system that spans the cell.
                    draw_scale_bar(ax, sb_mm, cell_w_mm,
                                   label_fontsize=5,
                                   label_fontweight="normal",
                                   x_end=x_end,
                                   y_bar=y_bar,
                                   label_align="right")
            # "mIoU =" label under the GT panel only; each model column then
            # shows just the numeric score on the same baseline.
            if j == 1:
                ax.text(
                    0.5, -0.02, "mIoU =",
                    transform=ax.transAxes,
                    ha="center", va="top",
                    fontsize=6, color="black",
                )
            elif j >= 2:
                mi = mious[j - 2]
                if not np.isnan(mi):
                    ax.text(
                        0.5, -0.02, f"{mi:.3f}",
                        transform=ax.transAxes,
                        ha="center", va="top",
                        fontsize=6, color="black",
                    )

    # Shared test-split labels: one per group of consecutive rows that share
    # the same split. Drawn in figure coords so a single label spans rows.
    groups = []
    prev_split = None
    for r, (_, split, *_) in enumerate(rows):
        if split != prev_split:
            groups.append([split, r, r])
            prev_split = split
        else:
            groups[-1][2] = r
    from matplotlib.patches import Rectangle
    for split, r0, r1 in groups:
        label = "Zero-shot Test" if split == "oneshot" else "In-distribution Test"
        # Use the Image column's cell extent (column 1) so the grey
        # rectangle aligns precisely with the top of the first row's
        # image and the bottom of the last row's image in the group.
        p_top = gs_b[r0 + 1, 1].get_position(fig)
        p_bot = gs_b[r1 + 1, 1].get_position(fig)
        y_mid = (p_top.y1 + p_bot.y0) / 2
        # Thin grey rectangle: left edge aligns with the panel-b letter
        # (x=0.005), width just enough for the rotated 6pt test-split label.
        rect_x0 = 0.005
        rect_x1 = 0.025
        rect_y0 = p_bot.y0
        rect_y1 = p_top.y1
        fig.add_artist(Rectangle(
            (rect_x0, rect_y0), rect_x1 - rect_x0, rect_y1 - rect_y0,
            facecolor="#dcdcdc", edgecolor="none",
            transform=fig.transFigure, zorder=0,
        ))
        x_mid = (rect_x0 + rect_x1) / 2
        fig.text(x_mid, y_mid, label,
                 color="black", fontsize=6, fontweight="bold",
                 ha="center", va="center", rotation=90,
                 fontfamily="Arial")

    pos = gs_b[0, 0].get_position(fig)
    fig.text(0.005, pos.y1 + 0.005, "b", fontsize=12, fontweight="bold",
             color="black", ha="left", va="bottom", fontfamily="Arial")

    # Bottom legend strip for the bio-7 anatomy classes (order matches Fig 2).
    legend_classes = ["Vascular", "Exodermis", "Endodermis", "Cortex",
                      "Epidermis", "Aerenchyma"]
    leg_ax = fig.add_axes([0.0, 0.0, 1.0, legend_in / fig_h_in])
    # Transparent - otherwise the white patch hides the mIoU score that the
    # last row of panel b draws just below its panels.
    leg_ax.patch.set_alpha(0)
    leg_ax.set_xlim(0, 1); leg_ax.set_ylim(0, 1)
    leg_ax.set_xticks([]); leg_ax.set_yticks([])
    for s in leg_ax.spines.values():
        s.set_visible(False)
    sw_w = 0.018
    sw_h = 0.45
    gap_label = 0.006
    gap_entry = 0.020
    char_w = 0.0085
    entry_widths = []
    for cls in legend_classes:
        text_w = char_w * len(cls)
        entry_widths.append(sw_w + gap_label + text_w)
    total_w = sum(entry_widths) + gap_entry * (len(legend_classes) - 1)
    x = (1.0 - total_w) / 2
    for cls, ew in zip(legend_classes, entry_widths):
        rgb = tuple(c / 255 for c in hex_to_rgb(COLOR_MAP[cls]))
        leg_ax.add_patch(Rectangle(
            (x, 0.5 - sw_h / 2), sw_w, sw_h,
            facecolor=rgb, edgecolor="none",
        ))
        leg_ax.text(
            x + sw_w + gap_label, 0.5, cls,
            ha="left", va="center", fontsize=6, color="black",
        )
        x += ew + gap_entry

    fig.savefig(OUT_PATH, dpi=600, facecolor="white", edgecolor="white",
                bbox_inches=None, pad_inches=0)
    plt.close(fig)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main(force_cache="--rebuild" in sys.argv)
