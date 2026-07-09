"""Supplementary figure: a curated gallery of additional training samples.

Renders a hand-picked list of training-set UIDs (CURATED_UIDS below)
grouped by species × microscope. Each thumbnail uses the Fig 1c channel
convention (DAPI = cyan, FITC = yellow, TRITC = red) plus the per-microscope
brightness / saturation / sharpness boost.

Image sources, in lookup order:
  1. figure1/fig1c_selections/{uid}.png   -- pre-rendered, NOT yet boosted;
                                              the script applies the boost.
  2. _uncovered_thumbs/{uid}.png          -- rendered by make_uncovered_picker.py
                                              with the boost already baked in.

Output: supp_training_gallery_600dpi.png in this directory.
Canvas: 180 mm wide, height computed from the number of thumbnail rows.
"""

import re
from pathlib import Path
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
from PIL import Image
from scipy.ndimage import convolve

DIVERSITY_CSV = Path(__file__).resolve().parent / "diversity_counts.csv"


# ── Per-microscope image boost (mirrors fig1c / fig2a SVG filters) ──────────
# Olympus: gamma 1.0526 → linear slope 1.84 / intercept -0.03 → saturate 1.2.
# C10:     gamma 1.1111 → linear slope 1.15 / intercept -0.165 → saturate 1.1
#          → light 3×3 sharpening kernel.
_C10_SHARPEN = np.array([
    [ 0.0, -0.2,  0.0],
    [-0.2,  1.8, -0.2],
    [ 0.0, -0.2,  0.0],
], dtype=np.float32)
_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _saturate(rgb01: np.ndarray, factor: float) -> np.ndarray:
    gray = rgb01 @ _LUMA
    return gray[..., None] + factor * (rgb01 - gray[..., None])


def _norm_uid(s: str) -> str:
    return re.sub(r"_+", "_", s) if s else s


def _cell_to_str(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v).strip()


def load_species_genotype_map() -> dict:
    import csv as _csv
    if not DIVERSITY_CSV.exists():
        return {}
    out: dict = {}
    with open(DIVERSITY_CSV) as f:
        for r in _csv.DictReader(f):
            uid = r.get("uid", "")
            if not uid:
                continue
            out[_norm_uid(uid)] = (r.get("species", ""), r.get("genotype", ""))
    return out


def format_species(sp_raw: str) -> str:
    """Solanum_X → 'S. X' (italic genus + species); other species pass through."""
    if not sp_raw:
        return ""
    if sp_raw.startswith("Solanum_"):
        return "S. " + sp_raw.replace("Solanum_", "")
    return sp_raw


def format_genotype(g_raw: str) -> str:
    """Strip the trailing 'mutant' word; apply paper-facing spelling fixes;
    otherwise keep the spreadsheet value.

    Known fixes (per `project_kitaake_spelling` memory):
      - Rice cultivar: spreadsheet says 'Kittake' but the correct paper
        spelling is 'Kitaake' (single t, double a). The rename is applied
        here, in the shared helper, so every figure / table picks it up.
    """
    if not g_raw:
        return ""
    g = re.sub(r"[_\s]*mutants?[_\s]*$", "", g_raw).strip()
    # Paper-facing spelling fixes.
    g = g.replace("Kittake", "Kitaake")
    return g


def compose_label(uid: str, species_genotype_map: dict) -> str:
    key = _norm_uid(uid)
    sp_raw, g_raw = species_genotype_map.get(key, ("", ""))
    sp = format_species(sp_raw)
    gt = format_genotype(g_raw)
    if sp and gt:
        return f"{sp}, {gt}"
    return sp or gt or uid


def species_and_genotype(uid: str, species_genotype_map: dict
                        ) -> tuple[str, str, bool]:
    """Return (species_display, genotype_display, genotype_is_italic) for `uid`.
    A genotype is considered italic (mutant gene/allele name) when it is
    all-lowercase letters/digits/hyphens with at least one letter."""
    key = _norm_uid(uid)
    sp_raw, g_raw = species_genotype_map.get(key, ("", ""))
    sp = format_species(sp_raw)
    gt = format_genotype(g_raw)
    italic = bool(gt
                  and gt == gt.lower()
                  and re.fullmatch(r"[a-z0-9\-]+", gt)
                  and re.search(r"[a-z]", gt))
    return sp, gt, italic


def boost_for_microscope(img_uint8: np.ndarray, microscope: str) -> np.ndarray:
    """Apply the same brightness / saturation / sharpening chain that Fig 1c
    and Fig 2a use, in numpy. Input/output are uint8 HxWx3 arrays."""
    x = img_uint8.astype(np.float32) / 255.0
    if microscope == "Olympus":
        x = np.clip(x, 0, 1) ** 1.0526
        x = 1.84 * x - 0.03
        x = _saturate(x, 1.2)
    elif microscope == "C10":
        x = np.clip(x, 0, 1) ** 1.1111
        x = 1.15 * x - 0.165
        x = _saturate(x, 1.1)
        # Per-channel sharpening.
        sharp = np.empty_like(x)
        for c in range(3):
            sharp[..., c] = convolve(x[..., c], _C10_SHARPEN, mode="reflect")
        x = sharp
    x = np.clip(x, 0, 1)
    return (x * 255).astype(np.uint8)

HERE = Path(__file__).resolve().parent
FIG1C_DIR = HERE / "data"  # legacy alias, kept for downstream-import compat
UNCOVERED_DIR = HERE / "data"
OUT_PATH = HERE / "supp_training_gallery_600dpi.png"

# Final curated list of 48 training-set UIDs to show in the supplementary
# gallery, ordered Rice → Sorghum → Solanums (= "Tomato" in raw UIDs) so
# the 6 × 8 grid below fills row-major by species block:
#   Row 1: 5 Rice (Olympus) + 1 Rice (C10)
#   Rows 2-6: 30 Sorghum (Olympus)
#   Row 7: 4 Sorghum (Olympus) + 1 Sorghum (C10) + 1 Solanums (Olympus)
#   Row 8: 6 Solanums (Olympus)
CURATED_UIDS = [
    # Rice / Olympus  (psy1-17, psy1-9, Kittake / WT dropped)
    "Rice_Olympus_Exp6_Falcon_Wox10-50_1_02",
    # Sorghum / Olympus  (235921 dropped; ETSL 101851, 101845, Dekeba,
    # ETSL 100307, ETSL 100726 restored)
    "Sorghum_Olympus_Exp1_2_03",
    "Sorghum_Olympus_Exp2_1_02",
    "Sorghum_Olympus_Exp3_3_03",
    "Sorghum_Olympus_Exp5_1_05",
    "Sorghum_Olympus_Exp7_2_05",
    "Sorghum_Olympus_Exp9_2_04",
    "Sorghum_Olympus_Exp10_3_05",
    "Sorghum_Olympus_Exp13_1_01",
    "Sorghum_Olympus_Exp14_2_03",
    "Sorghum_Olympus_Exp18_3_03",
    "Sorghum_Olympus_Exp21_3_04",
    "Sorghum_Olympus_Exp22_1_04",
    "Sorghum_Olympus_Exp25_1_06",
    "Sorghum_Olympus_Exp27_1_02",
    "Sorghum_Olympus_Exp30_1_04",
    "Sorghum_Olympus_Exp31_2_01",
    "Sorghum_Olympus_Exp32_2_01",
    "Sorghum_Olympus_Exp32_3_05",
    "Sorghum_Olympus_Exp36_3_01",
    "Sorghum_Olympus_Exp38_3_03",
    "Sorghum_Olympus_Exp42_3_02",
    "Sorghum_Olympus_Exp44_2_01",
    "Sorghum_Olympus_Exp45_4_03",
    "Sorghum_Olympus_Exp46_2_02",
    "Sorghum_Olympus_Exp48_1_04",
    "Sorghum_Olympus_Exp49_3_02",
    "Sorghum_Olympus_Exp50_3_03",
    "Sorghum_Olympus_Exp52_2_05",
    "Sorghum_Olympus_Exp53_4_03",
    "Sorghum_Olympus_Exp55_2_02",
    "Sorghum_Olympus_Exp56_4_02",
    "Sorghum_Olympus_Exp60_Control_19",
    "Sorghum_Olympus_Exp91_Control_05",
    # Sorghum / C10
    "Sorghum_C10_Exp5_C7",
    # Solanums / Olympus (slasft restored)
    "Tomato_Olympus_Exp5_Solanum_MajkenSlide1_09",
    "Tomato_Olympus_Exp5_Solanum_MajkenSlide1_44",
    "Tomato_Olympus_Exp10_Suberin_mutants_1-12_05",
    "Tomato_Olympus_Exp9_Suberin_mutants_1-10_30",
    "Tomato_Olympus_Exp1_Lignin_mutants_146",
    "Tomato_Olympus_Exp1_Lignin_mutants_161",
    "Tomato_Olympus_Exp20_WT_-77-82_13",
]
assert len(CURATED_UIDS) == 42, f"expected 42 UIDs, got {len(CURATED_UIDS)}"

# Twelve sample UIDs already used in Figure 1c (see
# figures_for_paper/figure1/make_fig1c_gallery.py).
FIG1C_USED = {
    "Millet_Olympus_Exp5_1-2cm__71",
    "Rice_Olympus_Exp13_P-17-3_08",
    "Rice_Olympus_Exp8_Pouch_WOX10-15_3_03",
    "Rice_Olympus_Exp9_Pouch_WT_4_04",
    "Sorghum_Olympus_Exp103_Hypocotyl_AF_15",
    "Sorghum_Olympus_Exp95_Control_12",
    "Sorghum_C10_Exp5_F10",
    "Rice_C10_Exp1_PSY9-1-c",
    "Tomato_Olympus_Exp20_WT_-95-97_07",
    "Tomato_Olympus_Exp5_Solanum_MajkenSlide1_11",
    "Tomato_Olympus_Exp5_Solanum_MajkenSlide1_28",
    "Tomato_C10_Exp3_M82_WT_E4ROI1",
}

# Group order = the species / microscope order used elsewhere in the paper.
# "Tomato" in raw file names → "Solanum" lycopersicum and relatives in the
# paper-facing label; underscores in genus binomials are rendered as spaces.
GROUP_ORDER = [
    ("Millet",  "Olympus"),
    ("Rice",    "Olympus"),
    ("Rice",    "C10"),
    ("Sorghum", "Olympus"),
    ("Sorghum", "C10"),
    ("Tomato",  "Olympus"),
    ("Tomato",  "C10"),
]
GROUP_LABEL = {
    ("Millet",  "Olympus"): "Millet\nOlympus",
    ("Rice",    "Olympus"): "Rice\nOlympus",
    ("Rice",    "C10"):     "Rice\nC10",
    ("Sorghum", "Olympus"): "Sorghum\nOlympus",
    ("Sorghum", "C10"):     "Sorghum\nC10",
    ("Tomato",  "Olympus"): "Solanums\nOlympus",
    ("Tomato",  "C10"):     "Solanums\nC10",
}


# ── Canvas geometry (mm) ────────────────────────────────────────────────────
CANVAS_W_MM   = 180.0     # Nature 2-column figure width
SIDE_PAD_MM   = 2.0       # left + right canvas margin
TOP_PAD_MM    = 4.5       # room above the first row for its top labels
BOT_PAD_MM    = 2.0
COLS          = 6
ROWS          = 7
COL_GAP_MM    = 1.5       # horizontal gap between thumbnails
ROW_GAP_MM    = 4.5       # vertical gap between rows (just big enough for the
                          # two-line label sitting above each thumbnail)
DPI           = 600
MM_PER_INCH   = 25.4


def thumb_size_mm():
    """Edge length of each square thumbnail so the 6 columns fill 180 mm."""
    avail = CANVAS_W_MM - 2 * SIDE_PAD_MM
    return (avail - (COLS - 1) * COL_GAP_MM) / COLS


def find_image_and_boost(uid: str):
    """Return (PIL.Image, needs_boost) for `uid`. Renders a paper-style
    composite (DAPI=cyan, FITC=yellow, TRITC=red) from the local TIFs in
    supplementary/data/<UID>/."""
    rec = _uid_to_record().get(uid)
    if rec is None:
        return None, False
    import sys as _sys
    _sys.path.insert(0, str(HERE.parent.parent))
    from src.preprocessing import load_sample_normalized
    img = load_sample_normalized(rec)
    tritc, fitc, dapi = img[..., 0], img[..., 1], img[..., 2]
    comp = np.zeros_like(img)
    comp[..., 1] += dapi; comp[..., 2] += dapi
    comp[..., 0] += fitc; comp[..., 1] += fitc
    comp[..., 0] += tritc
    arr = (np.clip(comp, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(arr), True


def microscope_for(uid: str) -> str:
    if "_Olympus_" in uid:
        return "Olympus"
    if "_C10_" in uid:
        return "C10"
    if "_Zeiss_" in uid:
        return "Zeiss"
    return ""


# Microscope-native pixel calibration: how many native-TIF pixels correspond
# to a 100 µm scale bar (provided by the lab). The same physical scale is
# preserved through every rescale step because each thumbnail shows the
# entire root cross-section that the raw TIF captured; the rescale ratio
# (display_px / native_px) is folded in at draw time below.
PX_PER_100UM = {"Olympus": 308, "C10": 308, "Zeiss": 217}

_native_size_cache: dict[str, int] = {}
_uid_to_record_cache: dict | None = None


def _uid_to_record():
    """UID -> SampleRecord backed by the local supplementary/data/ folder."""
    global _uid_to_record_cache
    if _uid_to_record_cache is None:
        import sys
        sys.path.insert(0, str(HERE.parent.parent))
        from src.config import SampleRecord
        recs = {}
        data_dir = HERE / "data"
        if data_dir.exists():
            for sub in data_dir.iterdir():
                if not sub.is_dir():
                    continue
                dapi = list(sub.glob("*_DAPI.tif"))
                if not dapi:
                    continue
                sample_name = dapi[0].stem[: -len("_DAPI")]
                parts = sub.name.split("_")
                sp = parts[0] if len(parts) > 0 else ""
                sc = parts[1] if len(parts) > 1 else ""
                ex = parts[2] if len(parts) > 2 else ""
                recs[sub.name] = SampleRecord(
                    species=sp, microscope=sc, experiment=ex,
                    sample_name=sample_name,
                    image_dir=sub,
                    annotation_path=sub / "gt.txt",
                )
        _uid_to_record_cache = recs
    return _uid_to_record_cache


def native_px_for(uid: str) -> int | None:
    """Width (pixels) of the raw native TIF for `uid`. Each raw TIF is a
    square crop of the root cross-section; native dimensions vary per
    sample. Cached so we only stat each sample's image_dir once."""
    if uid in _native_size_cache:
        return _native_size_cache[uid]
    rec = _uid_to_record().get(uid)
    if rec is None:
        return None
    tifs = list(rec.image_dir.glob("*.tif"))
    if not tifs:
        return None
    w, _h = Image.open(tifs[0]).size
    _native_size_cache[uid] = w
    return w


def scale_bar_mm(uid: str, tile_w_mm: float) -> float | None:
    """Length in mm of a 100 µm scale bar inside a `tile_w_mm` thumbnail
    for the given sample. The image fills the whole tile, so
        physical width of the image = native_px / cal_px_per_100um × 100 µm
    and a 100 µm bar takes (tile_w / physical_width) of the tile width.
    """
    mic = microscope_for(uid)
    cal = PX_PER_100UM.get(mic)
    if cal is None:
        return None
    n = native_px_for(uid)
    if not n:
        return None
    return tile_w_mm * cal / n


def draw_scale_bar(ax, sb_mm: float, tile_mm: float,
                    label_fontsize: float = 4.0,
                    label_fontweight: str = "bold",
                    x_end: float = 0.94,
                    y_bar: float = 0.07,
                    label_align: str = "center"):
    """Draw a white 100 µm scale bar with a "100 µm" label at the bottom-
    right of the axes. All positions in axes coords. `label_fontsize` is
    the text size in points (default 4 pt; callers wanting matched-size
    labels can override). `label_fontweight` selects bold vs normal weight.
    `x_end` / `y_bar` let callers nudge the bar inward when the image is
    letterboxed (e.g. portrait thumbnails in a square axes).
    `label_align="right"` aligns the label's right edge to `x_end` (used
    when the bar is short enough that a centered label overflows the
    image content)."""
    frac = sb_mm / tile_mm
    # Don't let the bar exceed 80 % of the tile width.
    frac = min(frac, 0.80)
    x_start = x_end - frac
    ax.plot([x_start, x_end], [y_bar, y_bar],
            color="white", linewidth=1.2, solid_capstyle="butt",
            transform=ax.transAxes,
            path_effects=[path_effects.withStroke(linewidth=2.0,
                                                  foreground="black")])
    if label_align == "right":
        label_x, label_ha = x_end, "right"
    else:
        label_x, label_ha = (x_start + x_end) / 2, "center"
    ax.text(label_x, y_bar + 0.04, "100 μm",
            color="white", fontsize=label_fontsize, fontweight=label_fontweight,
            ha=label_ha, va="bottom",
            family=["Arial", "DejaVu Sans"],
            transform=ax.transAxes,
            path_effects=[path_effects.withStroke(linewidth=0.7,
                                                  foreground="black")])


def main():
    sp_gt_map = load_species_genotype_map()
    thumb_mm = thumb_size_mm()
    canvas_h_mm = (TOP_PAD_MM + ROWS * thumb_mm
                   + (ROWS - 1) * ROW_GAP_MM + BOT_PAD_MM)

    print(f"thumb size: {thumb_mm:.2f} mm")
    print(f"canvas   : {CANVAS_W_MM:.1f} × {canvas_h_mm:.1f} mm "
          f"({CANVAS_W_MM/MM_PER_INCH*DPI:.0f} × "
          f"{canvas_h_mm/MM_PER_INCH*DPI:.0f} px) "
          f"at {DPI} dpi")

    fig_w_in = CANVAS_W_MM / MM_PER_INCH
    fig_h_in = canvas_h_mm / MM_PER_INCH
    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=DPI)
    fig.patch.set_facecolor("black")

    def mm_axes(x_mm, y_mm_top, w_mm, h_mm):
        x_frac = x_mm / CANVAS_W_MM
        y_frac = (canvas_h_mm - y_mm_top - h_mm) / canvas_h_mm
        ax = fig.add_axes([x_frac, y_frac, w_mm / CANVAS_W_MM,
                           h_mm / canvas_h_mm])
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        return ax

    missing = []
    for idx, uid in enumerate(CURATED_UIDS):
        row_i = idx // COLS
        col_i = idx %  COLS
        x_mm = SIDE_PAD_MM + col_i * (thumb_mm + COL_GAP_MM)
        y_mm = TOP_PAD_MM  + row_i * (thumb_mm + ROW_GAP_MM)
        ax = mm_axes(x_mm, y_mm, thumb_mm, thumb_mm)

        pil_img, needs_boost = find_image_and_boost(uid)
        if pil_img is None:
            ax.set_facecolor("#222")
            ax.text(0.5, 0.5, f"missing\n{uid}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=4, color="red")
            missing.append(uid)
            continue

        arr = np.asarray(pil_img)
        if needs_boost:
            arr = boost_for_microscope(arr, microscope_for(uid))
        ax.imshow(arr)

        # Labels go *above* the thumbnail, in the row gap. y is in axes
        # coordinates: y = 1.0 is the top of the thumbnail. Two lines are
        # placed above: species/microscope on top, genotype just above the
        # image. White Helvetica bold; genotype italic if it looks like a
        # gene/allele name.
        sp_disp, gt_disp, gt_italic = species_and_genotype(uid, sp_gt_map)
        mic = microscope_for(uid)
        sp_mic = f"{sp_disp} / {mic}" if sp_disp and mic else (sp_disp or mic)
        common = dict(transform=ax.transAxes, ha="center", va="bottom",
                      fontsize=4.6, color="white", fontweight="bold",
                      family=["Arial", "DejaVu Sans"])
        if gt_disp:
            kw = dict(common)
            if gt_italic:
                kw["fontstyle"] = "italic"
            ax.text(0.5, 1.02, gt_disp, **kw)
        if sp_mic:
            ax.text(0.5, 1.10, sp_mic, **common)

        # 100 µm scale bar at the bottom-right of the thumbnail.
        sb_mm_len = scale_bar_mm(uid, thumb_mm)
        if sb_mm_len is not None:
            draw_scale_bar(ax, sb_mm_len, thumb_mm)
        else:
            print(f"  ⚠ no scale-bar calibration for {uid}")

    if missing:
        print(f"  ⚠ {len(missing)} images not found:")
        for u in missing:
            print(f"    - {u}")

    plt.savefig(OUT_PATH, dpi=DPI, facecolor="black",
                bbox_inches=None, pad_inches=0)
    plt.close(fig)
    print(f"→ {OUT_PATH.relative_to(HERE.parent.parent)}  "
          f"(saved at {DPI} dpi)")


if __name__ == "__main__":
    main()
