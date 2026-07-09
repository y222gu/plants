# Figure 4a

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
sys.path.insert(0, str(PROJECT))

from src.config import SampleRecord
from src.annotation_utils import parse_yolo_annotations, polygons_to_raw_binary_masks
from src.model_classes import fill_contours, yolo_overlap_false_to_bio7
from src.preprocessing import load_sample_normalized, load_sample_raw

UID = "Sorghum_Olympus_Exp88_N3_02"
OUT_DIR = HERE / "4a"

OVERLAY_ALPHA = 0.85
AER_OVERLAY_ALPHA = 0.95   # aer mask is opaque-ish over the dim DAPI base
# Rotate the main full-image PNG by 90° counter-clockwise (k=1). Crop assets keep
# their original orientation; only the box coordinates on the full image are
# remapped into rotated coords so the boxes still mark the correct region.
ROTATE_MAIN_K_CCW = 0

CROP_SIZE = 128
# Target fraction of crop occupied by the class - picks a crop where the ring
# crosses near the middle (rather than fully inside or fully outside).
CROP_TARGET_FRAC = 0.40

# Aerenchyma rectangular crop (4:1 aspect) - picks the densest window
AER_CROP_W = 512
AER_CROP_H = 128

# Per-(class, channel) intensity keep-ranges (raw uint16 scale).
# Pixels in [low, high] are kept; None means open-ended on that side.
# Source: user's analysis CLI flags
#   --threshold Endodermis:FITC=14000-max
#   --threshold Exodermis:TRITC=7000-max
INTENSITY_THRESHOLDS = {
    "Endodermis": {"FITC":  (14000, None)},
    "Exodermis":  {"TRITC": (7000,  None)},
}
CHANNEL_IDX = {"TRITC": 0, "FITC": 1, "DAPI": 2}

# Endo and exo crop quadrants (relative to the ring centroid).
CROP_QUADRANT = {"endo": "BR", "exo": None}  # exo: free search

# Manual crop overrides as (y, x) top-left in the un-rotated image. When set,
# they bypass the auto-search for that tag. Pick coords with fig4a_pick_crop.html
# and paste them here. Set to None to use the auto search again.
CROP_OVERRIDES: dict[str, tuple[int, int] | None] = {
    "endo": None,
    "exo":  None,
    "aer":  None,   # (y, x) top-left for the AER_CROP_W × AER_CROP_H rectangle
}

COLOR_MAP = {
    "Epidermis":  "#0a9396",
    "Exodermis":  "#f4a261",
    "Cortex":     "#94d2bd",
    "Endodermis": "#f6e48e",
    "Vascular":   "#e76f61",
    "Aerenchyma": "#355c6b",   # subtle lift from #264653
}
PAINT_ORDER = ["Epidermis", "Exodermis", "Cortex", "Endodermis", "Vascular", "Aerenchyma"]


def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def render_composite(sample) -> np.ndarray:
    img = load_sample_normalized(sample)
    tritc, fitc, dapi = img[..., 0], img[..., 1], img[..., 2]
    h, w = dapi.shape
    comp = np.zeros((h, w, 3), dtype=np.float32)
    comp[..., 1] += dapi; comp[..., 2] += dapi
    comp[..., 0] += fitc; comp[..., 1] += fitc
    comp[..., 0] += tritc
    return (np.clip(comp, 0, 1) * 255).astype(np.uint8)


def render_single_channel(sample, channel: str) -> np.ndarray:
    """Single-channel render as grayscale (intensity replicated to all 3 RGB channels).
    channel ∈ {'FITC','TRITC','DAPI'}."""
    img = load_sample_normalized(sample)
    tritc, fitc, dapi = img[..., 0], img[..., 1], img[..., 2]
    arr = {"DAPI": dapi, "FITC": fitc, "TRITC": tritc}.get(channel)
    if arr is None:
        raise ValueError(channel)
    out = np.repeat(arr[..., None], 3, axis=2)
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def render_bio7_mask(bio7: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return (rgb, painted_mask). painted_mask=True where any class is painted."""
    h, w = next(iter(bio7.values())).shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    painted = np.zeros((h, w), dtype=bool)
    for cls in PAINT_ORDER:
        if cls not in bio7:
            continue
        m = bio7[cls].astype(bool)
        rgb[m] = hex_to_rgb(COLOR_MAP[cls])
        painted |= m
    return rgb, painted


def find_crop_box(mask: np.ndarray, size: int, target_frac: float,
                  roi: tuple[int, int, int, int] | None = None) -> tuple[int, int]:
    """Return (y, x) top-left of a (size x size) crop where the fraction of
    True pixels is closest to `target_frac`. Optional `roi=(y_min, y_max, x_min, x_max)`
    restricts the search domain for the top-left corner."""
    H, W = mask.shape
    if H < size or W < size:
        raise ValueError(f"image {H}x{W} smaller than crop size {size}")
    ii = np.zeros((H + 1, W + 1), dtype=np.int64)
    ii[1:, 1:] = np.cumsum(np.cumsum(mask.astype(np.int64), axis=0), axis=1)
    A = ii[size:, size:] - ii[:-size, size:] - ii[size:, :-size] + ii[:-size, :-size]
    target = target_frac * size * size
    diff = np.abs(A.astype(np.float64) - target)
    if roi is not None:
        y0, y1, x0, x1 = roi
        y0 = max(0, y0); x0 = max(0, x0)
        y1 = min(diff.shape[0], y1); x1 = min(diff.shape[1], x1)
        masked = np.full_like(diff, np.inf)
        masked[y0:y1, x0:x1] = diff[y0:y1, x0:x1]
        diff = masked
    flat = np.argmin(diff)
    y, x = np.unravel_index(flat, A.shape)
    return int(y), int(x)


def rotate_image_ccw(arr: np.ndarray, k: int) -> np.ndarray:
    """Rotate 90°×k counter-clockwise. k=0 is identity."""
    return np.rot90(arr, k=k) if k else arr


def remap_box_ccw(x: int, y: int, w: int, h: int,
                  img_w: int, img_h: int, k: int) -> tuple[int, int, int, int, int, int]:
    """Remap a box (top-left x,y, width w, height h) under k×90° CCW rotation of an
    img_w×img_h image. Returns (new_x, new_y, new_w, new_h, new_img_w, new_img_h)."""
    k %= 4
    for _ in range(k):
        # 90° CCW: new_x = y, new_y = (img_w - x - w), new_w = h, new_h = w
        x, y, w, h = y, img_w - x - w, h, w
        img_w, img_h = img_h, img_w
    return x, y, w, h, img_w, img_h


def find_intensity_match_box(
    primary_mask: np.ndarray,
    other_mask: np.ndarray,
    raw_channel: np.ndarray,
    size: int,
    min_primary_px: int = 200,
    min_other_px: int = 200,
) -> tuple[int, int]:
    """Find the (size×size) window minimising |mean(primary) - mean(other)| of
    `raw_channel` (typically TRITC). Both masks must have ≥ min_*_px pixels in
    the window. Uses integral images for O(1) per-window stats."""
    H, W = primary_mask.shape
    if H < size or W < size:
        raise ValueError(f"image {H}x{W} smaller than crop {size}")

    def integral(arr: np.ndarray) -> np.ndarray:
        ii = np.zeros((H + 1, W + 1), dtype=np.float64)
        ii[1:, 1:] = np.cumsum(np.cumsum(arr.astype(np.float64), axis=0), axis=1)
        return ii

    def window_sum(ii: np.ndarray) -> np.ndarray:
        return ii[size:, size:] - ii[:-size, size:] - ii[size:, :-size] + ii[:-size, :-size]

    raw = raw_channel.astype(np.float64)
    p_count = window_sum(integral(primary_mask.astype(bool)))
    o_count = window_sum(integral(other_mask.astype(bool)))
    p_sum   = window_sum(integral(raw * primary_mask.astype(bool)))
    o_sum   = window_sum(integral(raw * other_mask.astype(bool)))

    eligible = (p_count >= min_primary_px) & (o_count >= min_other_px)
    if not eligible.any():
        raise ValueError("no window has enough pixels of both classes")

    p_mean = np.full_like(p_sum, np.nan)
    o_mean = np.full_like(o_sum, np.nan)
    np.divide(p_sum, p_count, out=p_mean, where=p_count > 0)
    np.divide(o_sum, o_count, out=o_mean, where=o_count > 0)
    diff = np.abs(p_mean - o_mean)
    diff = np.where(eligible, diff, np.inf)
    flat = int(np.argmin(diff))
    y, x = np.unravel_index(flat, diff.shape)
    return int(y), int(x)


def find_densest_rect(mask: np.ndarray, h_size: int, w_size: int) -> tuple[int, int]:
    """Return top-left (y, x) of the (h_size × w_size) window with the maximum
    True-pixel count in `mask`. Uses an integral image for O(1) per-window sums."""
    H, W = mask.shape
    if H < h_size or W < w_size:
        raise ValueError(f"image {H}x{W} smaller than crop {h_size}x{w_size}")
    ii = np.zeros((H + 1, W + 1), dtype=np.int64)
    ii[1:, 1:] = np.cumsum(np.cumsum(mask.astype(np.int64), axis=0), axis=1)
    A = (ii[h_size:, w_size:] - ii[:-h_size, w_size:]
         - ii[h_size:, :-w_size] + ii[:-h_size, :-w_size])
    flat = int(np.argmax(A))
    y, x = np.unravel_index(flat, A.shape)
    return int(y), int(x)


def quadrant_roi(mask: np.ndarray, size: int, quadrant: str) -> tuple[int, int, int, int]:
    """ROI for the top-left of a (size×size) crop so its centre lands on a
    specified quadrant arc of a roughly-circular ring `mask`.
    quadrant ∈ {'TL','TR','BL','BR'} (top-left, top-right, bottom-left, bottom-right)."""
    ys, xs = np.where(mask)
    cy, cx = int(np.median(ys)), int(np.median(xs))
    y_min_b, y_max_b = int(ys.min()), int(ys.max())
    x_min_b, x_max_b = int(xs.min()), int(xs.max())
    half = size // 2
    if quadrant == "TR":
        y0, y1 = max(0, y_min_b - 16), max(1, cy - half)
        x0, x1 = max(0, cx - half),    max(1, x_max_b - half)
    elif quadrant == "BR":
        y0, y1 = max(0, cy - half),    max(1, y_max_b - half + 16)
        x0, x1 = max(0, cx - half),    max(1, x_max_b - half)
    elif quadrant == "TL":
        y0, y1 = max(0, y_min_b - 16), max(1, cy - half)
        x0, x1 = max(0, x_min_b - 16), max(1, cx - half)
    elif quadrant == "BL":
        y0, y1 = max(0, cy - half),    max(1, y_max_b - half + 16)
        x0, x1 = max(0, x_min_b - 16), max(1, cx - half)
    else:
        raise ValueError(quadrant)
    return y0, max(y0 + 1, y1), x0, max(x0 + 1, x1)


def crop_array(arr: np.ndarray, y: int, x: int, size: int) -> np.ndarray:
    return arr[y:y + size, x:x + size]


def load_polygons_to_bio7(txt_path: Path, h: int, w: int) -> dict[str, np.ndarray]:
    anns = parse_yolo_annotations(txt_path, w, h)
    raw = polygons_to_raw_binary_masks(anns, h, w)
    raw = {k: fill_contours(v) for k, v in raw.items()}
    return yolo_overlap_false_to_bio7(raw, h, w)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    image_dir = OUT_DIR / UID
    dapi_tifs = list(image_dir.glob("*_DAPI.tif"))
    if not dapi_tifs:
        raise SystemExit(f"No *_DAPI.tif in {image_dir}")
    sample_name = dapi_tifs[0].stem[: -len("_DAPI")]
    sample = SampleRecord(
        species="", microscope="", experiment="",
        sample_name=sample_name,
        image_dir=image_dir,
        annotation_path=image_dir / "gt.txt",
    )

    comp = render_composite(sample)
    h, w = comp.shape[:2]
    comp_rot = rotate_image_ccw(comp, ROTATE_MAIN_K_CCW)
    rh, rw = comp_rot.shape[:2]
    Image.fromarray(comp_rot).save(OUT_DIR / f"{UID}_image.png")
    print(f"  {UID}_image.png  ({rw}×{rh})  rotated {ROTATE_MAIN_K_CCW * 90}° CCW")

    pred_txt = image_dir / "pred.txt"
    if not pred_txt.exists():
        raise SystemExit(f"missing prediction: {pred_txt}")
    pred_bio7 = load_polygons_to_bio7(pred_txt, h, w)
    pred_rgb, painted = render_bio7_mask(pred_bio7)

    overlay = comp.astype(np.float32).copy()
    a = OVERLAY_ALPHA
    overlay[painted] = (1 - a) * comp[painted] + a * pred_rgb[painted]
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # Pre-render full-image grayscale single-channel views and per-class single-channel
    # overlays where only the matching class is alpha-blended on top.
    fitc_full  = render_single_channel(sample, "FITC")
    tritc_full = render_single_channel(sample, "TRITC")

    # Raw uint16 image used for intensity-based threshold filtering.
    raw_image = load_sample_raw(sample)  # shape (h, w, 3); channel order = (TRITC, FITC, DAPI)

    def threshold_mask(cls_mask: np.ndarray, channel: str, cls: str) -> np.ndarray:
        """Apply the per-(cls, channel) intensity keep-range to `cls_mask`."""
        rng = INTENSITY_THRESHOLDS.get(cls, {}).get(channel)
        if rng is None:
            return cls_mask
        low, high = rng
        ch = raw_image[..., CHANNEL_IDX[channel]]
        out = cls_mask.copy().astype(bool)
        if low  is not None: out &= (ch >= low)
        if high is not None: out &= (ch <= high)
        return out

    def blend_with_mask(base: np.ndarray, mask: np.ndarray, cls: str,
                        alpha: float | None = None) -> np.ndarray:
        if mask is None or not mask.any():
            return base.copy()
        a_use = a if alpha is None else alpha
        color = np.array(hex_to_rgb(COLOR_MAP[cls]), dtype=np.float32)
        out = base.astype(np.float32).copy()
        out[mask] = (1 - a_use) * base[mask] + a_use * color
        return np.clip(out, 0, 255).astype(np.uint8)

    # Full-image variants for the interactive crop-picker page AND for client-side
    # cropping in fig4a_overlay.html. All saved un-rotated so coords from the
    # picker (and from python's CROP_OVERRIDES) line up.
    endo_mask = pred_bio7.get("Endodermis", np.zeros((h, w), dtype=bool)).astype(bool)
    exo_mask  = pred_bio7.get("Exodermis",  np.zeros((h, w), dtype=bool)).astype(bool)

    endo_ovl_full     = blend_with_mask(fitc_full,  endo_mask, "Endodermis") if endo_mask.any() else fitc_full
    exo_ovl_full      = blend_with_mask(tritc_full, exo_mask,  "Exodermis")  if exo_mask.any()  else tritc_full
    endo_ovl_thr_full = blend_with_mask(fitc_full,
                                        threshold_mask(endo_mask, "FITC",  "Endodermis"),
                                        "Endodermis") if endo_mask.any() else fitc_full
    exo_ovl_thr_full  = blend_with_mask(tritc_full,
                                        threshold_mask(exo_mask,  "TRITC", "Exodermis"),
                                        "Exodermis")  if exo_mask.any()  else tritc_full

    Image.fromarray(fitc_full).save(        OUT_DIR / f"{UID}_full_endo_fitc.png")
    Image.fromarray(endo_ovl_full).save(    OUT_DIR / f"{UID}_full_endo_fitc_overlay.png")
    Image.fromarray(endo_ovl_thr_full).save(OUT_DIR / f"{UID}_full_endo_fitc_overlay_thr.png")
    Image.fromarray(tritc_full).save(       OUT_DIR / f"{UID}_full_exo_tritc.png")
    Image.fromarray(exo_ovl_full).save(     OUT_DIR / f"{UID}_full_exo_tritc_overlay.png")
    Image.fromarray(exo_ovl_thr_full).save( OUT_DIR / f"{UID}_full_exo_tritc_overlay_thr.png")

    crops_meta = {}
    for tag, cls in [("endo", "Endodermis"), ("exo", "Exodermis")]:
        m = pred_bio7.get(cls)
        if m is None or not m.any():
            print(f"  [warn] no {cls} predicted - skipping {tag} crop")
            continue
        ovr = CROP_OVERRIDES.get(tag)
        if ovr is not None:
            y, x = ovr
            quad = "manual"
        elif tag == "exo":
            # Pick a window where exodermis & epidermis have similar TRITC mean
            # intensity - visualises the model separating two stains-alike rings.
            epi = pred_bio7.get("Epidermis")
            tritc_raw = raw_image[..., CHANNEL_IDX["TRITC"]]
            if epi is not None and epi.any():
                y, x = find_intensity_match_box(
                    m.astype(bool), epi.astype(bool), tritc_raw, CROP_SIZE,
                )
                quad = "intensity_match"
            else:
                quad = CROP_QUADRANT.get(tag)
                roi = quadrant_roi(m, CROP_SIZE, quad) if quad else None
                y, x = find_crop_box(m, CROP_SIZE, CROP_TARGET_FRAC, roi=roi)
        else:
            quad = CROP_QUADRANT.get(tag)
            roi = quadrant_roi(m, CROP_SIZE, quad) if quad else None
            y, x = find_crop_box(m, CROP_SIZE, CROP_TARGET_FRAC, roi=roi)

        cls_mask = m.astype(bool)
        wanted_ch = {"endo": "fitc", "exo": "tritc"}[tag]
        ch_full = {"fitc": fitc_full, "tritc": tritc_full}[wanted_ch]

        ch_crop = crop_array(ch_full, y, x, CROP_SIZE)
        Image.fromarray(ch_crop).save(OUT_DIR / f"{UID}_crop_{tag}_{wanted_ch}.png")

        ovl_full = blend_with_mask(ch_full, cls_mask, cls)
        Image.fromarray(crop_array(ovl_full, y, x, CROP_SIZE)).save(
            OUT_DIR / f"{UID}_crop_{tag}_{wanted_ch}_overlay.png")

        thr_mask = threshold_mask(cls_mask, wanted_ch.upper(), cls)
        thr_full = blend_with_mask(ch_full, thr_mask, cls)
        Image.fromarray(crop_array(thr_full, y, x, CROP_SIZE)).save(
            OUT_DIR / f"{UID}_crop_{tag}_{wanted_ch}_overlay_thr.png")

        frac = float(m[y:y + CROP_SIZE, x:x + CROP_SIZE].mean())
        crops_meta[tag] = {"class": cls, "x": x, "y": y, "size": CROP_SIZE, "frac": frac,
                           "quadrant": quad}
        print(f"  crop_{tag} ({cls}): top-left=({x},{y})  frac={frac:.2f}  quad={quad}")

    # Aerenchyma rectangular crop (2:1 aspect): max-density 256×128 window.
    # Base channel = DAPI (cell walls) so the air-filled aer cavities show as
    # low-signal voids that the model fills with the aer color.
    aer_mask = pred_bio7.get("Aerenchyma")
    if aer_mask is not None and aer_mask.any():
        dapi_full = render_single_channel(sample, "DAPI")
        aer_only_overlay = blend_with_mask(
            dapi_full, aer_mask.astype(bool), "Aerenchyma",
            alpha=AER_OVERLAY_ALPHA,
        )
        # Plain DAPI grayscale (used by the fig4a aer cell - no overlay paint).
        Image.fromarray(dapi_full).save(OUT_DIR / f"{UID}_full_aer_dapi.png")
        # DAPI + aer overlay (used by fig4a_pick_crop.html stage). Filename kept
        # as `_composite_overlay` for backwards compat with the picker HTML.
        Image.fromarray(aer_only_overlay).save(OUT_DIR / f"{UID}_full_aer_composite_overlay.png")
        aer_ovr = CROP_OVERRIDES.get("aer")
        if aer_ovr is not None:
            ay, ax = aer_ovr
            aer_quad = "manual"
        else:
            ay, ax = find_densest_rect(aer_mask.astype(bool), AER_CROP_H, AER_CROP_W)
            aer_quad = "max_density"
        dapi_aer = dapi_full[ay:ay + AER_CROP_H, ax:ax + AER_CROP_W]
        Image.fromarray(dapi_aer).save(OUT_DIR / f"{UID}_crop_aer_image.png")
        ovl_aer = aer_only_overlay[ay:ay + AER_CROP_H, ax:ax + AER_CROP_W]
        Image.fromarray(ovl_aer).save(OUT_DIR / f"{UID}_crop_aer_overlay.png")
        aer_frac = float(aer_mask[ay:ay + AER_CROP_H, ax:ax + AER_CROP_W].mean())
        crops_meta["aer"] = {"class": "Aerenchyma", "x": ax, "y": ay,
                             "w": AER_CROP_W, "h": AER_CROP_H, "frac": aer_frac,
                             "quadrant": aer_quad}
        print(f"  crop_aer (Aerenchyma): top-left=({ax},{ay})  frac={aer_frac:.2f}  "
              f"size={AER_CROP_W}×{AER_CROP_H}  quad={aer_quad}")
    else:
        print("  [warn] no Aerenchyma predicted - skipping aer crop")

    # Remap crop-box positions to the rotated full-image coordinate system so
    # the boxes/connectors in the HTML line up with the rotated main image.
    crops_meta_rot = {}
    for tag, c in crops_meta.items():
        cw = c.get("w", c.get("size"))
        ch = c.get("h", c.get("size"))
        nx, ny, nw, nh, _, _ = remap_box_ccw(c["x"], c["y"], cw, ch, w, h, ROTATE_MAIN_K_CCW)
        new = dict(c)
        new["x"], new["y"] = nx, ny
        if "w" in c or "h" in c:
            new["w"], new["h"] = nw, nh
        else:
            # Square crops keep "size", but after rotation w==h still, so leave it
            new["size"] = nw  # nw == nh for squares
        crops_meta_rot[tag] = new

    with open(OUT_DIR / f"{UID}_crops.json", "w") as f:
        json.dump({"image_size": [rw, rh], "crops": crops_meta_rot,
                   "rotation_k_ccw": ROTATE_MAIN_K_CCW}, f, indent=2)


if __name__ == "__main__":
    main()
