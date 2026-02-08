"""Image loading, normalization, and resize/padding utilities."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import tifffile

from .config import CHANNELS, OUTPUT_DIR, SampleRecord


@dataclass
class PadMeta:
    """Metadata for reversing resize + padding."""
    original_h: int
    original_w: int
    scale: float
    scaled_h: int
    scaled_w: int
    pad_bottom: int
    pad_right: int
    target_size: int


def load_channel(path: Path) -> np.ndarray:
    """Load a single TIF channel, return as 2D float32 array."""
    img = tifffile.imread(str(path))
    # Handle multi-dimensional (take first slice)
    if img.ndim > 2:
        img = img[0] if img.shape[0] < img.shape[-1] else img[..., 0]
    return img.astype(np.float32)


def load_sample_raw(sample: SampleRecord) -> np.ndarray:
    """Load 3-channel image as (H, W, 3) float32 without normalization.

    Channel order: R=TRITC, G=FITC, B=DAPI.
    Values remain in original range (uint16 or float32).
    """
    channels = []
    for ch_name in ["TRITC", "FITC", "DAPI"]:  # RGB order
        ch = load_channel(sample.channel_path(ch_name))
        channels.append(ch)
    return np.stack(channels, axis=-1)


def normalize_percentile(
    img: np.ndarray,
    p_low: float = 1.0,
    p_high: float = 99.5,
) -> np.ndarray:
    """Percentile-based normalization per channel → [0, 1] float32.

    Args:
        img: (H, W, C) float32 image.
        p_low: Lower percentile for clipping.
        p_high: Upper percentile for clipping.

    Returns:
        (H, W, C) float32 image in [0, 1].
    """
    out = np.empty_like(img)
    for c in range(img.shape[-1]):
        ch = img[..., c]
        lo = np.percentile(ch, p_low)
        hi = np.percentile(ch, p_high)
        if hi > lo:
            out[..., c] = np.clip((ch - lo) / (hi - lo), 0, 1)
        else:
            out[..., c] = 0.0
    return out


def load_sample_normalized(sample: SampleRecord) -> np.ndarray:
    """Load sample and normalize to [0, 1] float32, shape (H, W, 3)."""
    raw = load_sample_raw(sample)
    return normalize_percentile(raw)


def resize_with_padding(
    img: np.ndarray,
    target_size: int = 1024,
    masks: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], PadMeta]:
    """Resize image (and optional masks) preserving aspect ratio, pad to square.

    Padding is added to right/bottom only with zeros.
    Masks use nearest-neighbor interpolation.

    Args:
        img: (H, W, C) float32 image.
        target_size: Target square dimension.
        masks: Optional (N, H, W) binary masks.

    Returns:
        (resized_img, resized_masks, pad_meta)
    """
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))

    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_bottom = target_size - new_h
    pad_right = target_size - new_w
    padded_img = np.zeros((target_size, target_size, img.shape[2]), dtype=img.dtype)
    padded_img[:new_h, :new_w] = resized_img

    meta = PadMeta(
        original_h=h, original_w=w,
        scale=scale, scaled_h=new_h, scaled_w=new_w,
        pad_bottom=pad_bottom, pad_right=pad_right,
        target_size=target_size,
    )

    resized_masks = None
    if masks is not None and len(masks) > 0:
        n = masks.shape[0]
        resized_masks = np.zeros((n, target_size, target_size), dtype=masks.dtype)
        for i in range(n):
            m = cv2.resize(
                masks[i].astype(np.uint8), (new_w, new_h),
                interpolation=cv2.INTER_NEAREST,
            )
            resized_masks[i, :new_h, :new_w] = m

    return padded_img, resized_masks, meta


def unpad_predictions(
    pred: np.ndarray,
    meta: PadMeta,
) -> np.ndarray:
    """Remove padding and resize predictions back to original image size.

    Args:
        pred: (H_pad, W_pad) or (N, H_pad, W_pad) prediction array.
        meta: PadMeta from resize_with_padding.

    Returns:
        Prediction resized to (original_h, original_w).
    """
    if pred.ndim == 2:
        cropped = pred[:meta.scaled_h, :meta.scaled_w]
        return cv2.resize(
            cropped.astype(np.uint8), (meta.original_w, meta.original_h),
            interpolation=cv2.INTER_NEAREST,
        )
    else:
        result = []
        for i in range(pred.shape[0]):
            cropped = pred[i, :meta.scaled_h, :meta.scaled_w]
            r = cv2.resize(
                cropped.astype(np.uint8), (meta.original_w, meta.original_h),
                interpolation=cv2.INTER_NEAREST,
            )
            result.append(r)
        return np.stack(result)


def to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert [0, 1] float32 image to uint8."""
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def compute_dataset_stats(
    samples: List[SampleRecord],
    cache_path: Optional[Path] = None,
) -> Dict[str, List[float]]:
    """Compute per-channel mean and std over dataset (after percentile normalization).

    Results are cached to disk if cache_path is provided.
    """
    if cache_path is None:
        cache_path = OUTPUT_DIR / "dataset_stats.json"

    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    print(f"Computing dataset stats over {len(samples)} samples...")
    running_sum = np.zeros(3, dtype=np.float64)
    running_sq_sum = np.zeros(3, dtype=np.float64)
    n_pixels = 0

    for sample in samples:
        img = load_sample_normalized(sample)
        h, w = img.shape[:2]
        running_sum += img.reshape(-1, 3).sum(axis=0)
        running_sq_sum += (img.reshape(-1, 3) ** 2).sum(axis=0)
        n_pixels += h * w

    mean = (running_sum / n_pixels).tolist()
    std = np.sqrt(running_sq_sum / n_pixels - np.array(mean) ** 2).tolist()

    stats = {"mean": mean, "std": std}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats cached to {cache_path}")

    return stats
