"""Downstream analysis: aerenchyma ratio, channel intensities, counts."""

from typing import Dict, List

import numpy as np

from .config import CHANNELS


def aerenchyma_ratio(
    masks: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Compute total aerenchyma area / whole root area.

    Args:
        masks: (N, H, W) binary instance masks.
        labels: (N,) class labels (0=root, 1=aerenchyma).

    Returns:
        Ratio in [0, 1], or 0 if no whole root.
    """
    root_idx = np.where(labels == 0)[0]
    aer_idx = np.where(labels == 1)[0]

    if len(root_idx) == 0:
        return 0.0

    root_area = np.clip(masks[root_idx].sum(axis=0), 0, 1).sum()
    if root_area == 0:
        return 0.0

    # Merge into a single binary mask to avoid double-counting overlapping instances
    aer_area = (np.clip(masks[aer_idx].sum(axis=0), 0, 1).sum()
                if len(aer_idx) > 0 else 0)
    return float(aer_area / root_area)


def region_channel_intensity(
    image: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, float]:
    """Compute average intensity per channel in a masked region.

    Args:
        image: (H, W, 3) float32 raw image (R=TRITC, G=FITC, B=DAPI).
        mask: (H, W) binary mask.

    Returns:
        Dict with TRITC, FITC, DAPI mean intensities.
    """
    channel_names = ["TRITC", "FITC", "DAPI"]
    result = {}
    pixels = mask.astype(bool)
    n_pixels = pixels.sum()

    if n_pixels == 0:
        return {ch: 0.0 for ch in channel_names}

    for i, ch_name in enumerate(channel_names):
        result[ch_name] = float(image[..., i][pixels].mean())

    return result


def endodermis_intensity(
    image: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """Average per-channel intensity in the endodermis ring (class 2)."""
    endo_idx = np.where(labels == 2)[0]
    if len(endo_idx) == 0:
        return {ch: 0.0 for ch in ["TRITC", "FITC", "DAPI"]}
    endo_mask = np.clip(masks[endo_idx].sum(axis=0), 0, 1).astype(np.uint8)
    return region_channel_intensity(image, endo_mask)


def vascular_intensity(
    image: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """Average per-channel intensity in the vascular region (class 3)."""
    vasc_idx = np.where(labels == 3)[0]
    if len(vasc_idx) == 0:
        return {ch: 0.0 for ch in ["TRITC", "FITC", "DAPI"]}
    vasc_mask = np.clip(masks[vasc_idx].sum(axis=0), 0, 1).astype(np.uint8)
    return region_channel_intensity(image, vasc_mask)


def aerenchyma_count(labels: np.ndarray) -> int:
    """Count number of aerenchyma instances."""
    return int((labels == 1).sum())


def analyze_sample(
    image: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    sample_id: str = "",
) -> Dict:
    """Run all downstream analyses for a single sample.

    Args:
        image: (H, W, 3) float32 raw image (no normalization).
        masks: (N, H, W) binary instance masks.
        labels: (N,) target class labels.
        sample_id: Optional identifier.

    Returns:
        Dict with all computed metrics.
    """
    return {
        "sample_id": sample_id,
        "aerenchyma_ratio": aerenchyma_ratio(masks, labels),
        "aerenchyma_count": aerenchyma_count(labels),
        "endodermis_intensity": endodermis_intensity(image, masks, labels),
        "vascular_intensity": vascular_intensity(image, masks, labels),
    }
