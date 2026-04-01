"""Shared visualization utilities for prediction and evaluation scripts."""

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from .config import TARGET_CLASS_COLORS_RGB, get_target_classes

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Font helpers ──────────────────────────────────────────────────────────────

def load_font(size: int) -> ImageFont.FreeTypeFont:
    """Load a TrueType font with graceful fallback."""
    for name in ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def pil_text(img: np.ndarray, text: str, xy: tuple,
             font: ImageFont.FreeTypeFont, fill: tuple,
             outline: tuple = None) -> np.ndarray:
    """Draw text on a numpy RGB image using PIL for clean rendering."""
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    if outline:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx != 0 or dy != 0:
                    draw.text((xy[0] + dx, xy[1] + dy), text, font=font, fill=outline)
    draw.text(xy, text, font=font, fill=fill)
    return np.array(pil_img)


# ── Mask overlay ──────────────────────────────────────────────────────────────

def draw_masks_overlay(img_uint8: np.ndarray, masks: np.ndarray,
                       labels: np.ndarray, scores: np.ndarray = None,
                       alpha: float = 0.45) -> np.ndarray:
    """Draw instance masks with semi-transparent fill + contours.

    Draws largest masks first so smaller ones appear on top.
    """
    result = img_uint8.copy()
    if len(masks) == 0:
        return result

    areas = [m.sum() for m in masks]
    order = np.argsort(areas)[::-1]

    overlay = result.copy()
    for idx in order:
        mask = masks[idx]
        cls_id = int(labels[idx])
        color = TARGET_CLASS_COLORS_RGB.get(cls_id, (128, 128, 128))
        overlay[mask > 0] = color

    result = cv2.addWeighted(result, 1 - alpha, overlay, alpha, 0)

    for idx in order:
        mask = masks[idx]
        cls_id = int(labels[idx])
        color = TARGET_CLASS_COLORS_RGB.get(cls_id, (128, 128, 128))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, color, 2)

    return result


# ── Legend bar ────────────────────────────────────────────────────────────────

def make_legend_bar(width: int, height: int = 32, num_classes: int = 5) -> np.ndarray:
    """Create a class-legend bar image (RGB uint8)."""
    legend_font = load_font(14)
    legend = np.zeros((height, width, 3), dtype=np.uint8)
    pil_legend = Image.fromarray(legend)
    draw = ImageDraw.Draw(pil_legend)
    x_offset = 10
    classes = get_target_classes(num_classes)
    for cls_id, cls_name in classes.items():
        color = TARGET_CLASS_COLORS_RGB.get(cls_id, (128, 128, 128))
        draw.rectangle([x_offset, 6, x_offset + 20, 26], fill=color)
        draw.text((x_offset + 25, 8), cls_name, font=legend_font, fill=(255, 255, 255))
        bbox = legend_font.getbbox(cls_name)
        text_w = bbox[2] - bbox[0]
        x_offset += 25 + text_w + 20
    return np.array(pil_legend)


# ── Resize helpers ────────────────────────────────────────────────────────────

def downscale_for_vis(img_uint8: np.ndarray, masks: np.ndarray,
                      max_dim: int = 800):
    """Downscale image and masks for visualization.

    Returns (img_small, masks_small, new_h, new_w).
    """
    h, w = img_uint8.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        new_h, new_w = int(h * scale), int(w * scale)
        img_small = cv2.resize(img_uint8, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        if len(masks) > 0:
            masks_small = np.zeros((len(masks), new_h, new_w), dtype=np.uint8)
            for i in range(len(masks)):
                masks_small[i] = cv2.resize(masks[i], (new_w, new_h),
                                            interpolation=cv2.INTER_NEAREST)
        else:
            masks_small = masks
    else:
        img_small = img_uint8
        new_h, new_w = h, w
        masks_small = masks
    return img_small, masks_small, new_h, new_w


# ── Publication-quality matplotlib setup ──────────────────────────────────────

# Color-blind-friendly palette (adapted from Wong 2011, Nature Methods)
PUB_CLASS_COLORS = {
    "Whole Root":   "#0072B2",
    "Aerenchyma":   "#E69F00",
    "Endodermis":   "#009E73",
    "Vascular":     "#CC79A7",
    "Exodermis":    "#56B4E9",
}
PUB_HATCHES = {
    "Whole Root":   "",
    "Aerenchyma":   "//",
    "Endodermis":   "\\\\",
    "Vascular":     "xx",
    "Exodermis":    "..",
}


def setup_pub_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
