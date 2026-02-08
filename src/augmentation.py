"""Albumentations pipelines for fluorescence microscopy augmentation."""

from typing import Dict, List, Optional, Tuple

import albumentations as A
import numpy as np


def get_train_transform(
    img_size: int = 1024,
    p_channel_dropout: float = 0.2,
    p_channel_shuffle: float = 0.2,
) -> A.Compose:
    """Training augmentation pipeline for fluorescence microscopy.

    Does NOT use hue/saturation jitter (meaningless for fluorescence).
    Includes channel dropout and shuffle for microscope generalization.
    """
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            scale=(0.9, 1.1),
            rotate=(-15, 15),
            mode=0, cval=0, p=0.5,
        ),
        A.ElasticTransform(
            alpha=50, sigma=10, border_mode=0, p=0.2,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.15, contrast_limit=0.15, p=0.5,
        ),
        A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
        A.ChannelDropout(
            channel_drop_range=(1, 1), p=p_channel_dropout,
        ),
        A.ChannelShuffle(p=p_channel_shuffle),
        A.Resize(img_size, img_size),
    ])


def get_val_transform(img_size: int = 1024) -> A.Compose:
    """Validation/test transform — resize only, no augmentation."""
    return A.Compose([
        A.Resize(img_size, img_size),
    ])


def apply_transform_with_masks(
    transform: A.Compose,
    image: np.ndarray,
    masks: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply albumentations transform to image and multiple instance masks.

    Args:
        transform: Albumentations Compose pipeline.
        image: (H, W, 3) float32 image.
        masks: (N, H, W) uint8 binary instance masks.

    Returns:
        (transformed_image, transformed_masks)
    """
    if len(masks) == 0:
        result = transform(image=image)
        return result["image"], masks

    # Albumentations handles additional_targets for multiple masks
    # We pass each mask as a separate key
    additional_targets = {}
    mask_keys = []
    for i in range(len(masks)):
        key = f"mask{i}"
        additional_targets[key] = "mask"
        mask_keys.append(key)

    aug = A.Compose(
        transform.transforms,
        additional_targets=additional_targets,
    )

    kwargs = {"image": image}
    for i, key in enumerate(mask_keys):
        kwargs[key] = masks[i]

    result = aug(**kwargs)

    transformed_masks = np.stack([result[key] for key in mask_keys])
    return result["image"], transformed_masks


def apply_transform_semantic(
    transform: A.Compose,
    image: np.ndarray,
    mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply transform to image and semantic mask.

    Args:
        transform: Albumentations pipeline.
        image: (H, W, 3) float32 image.
        mask: (H, W) int32 semantic mask.

    Returns:
        (transformed_image, transformed_mask)
    """
    result = transform(image=image, mask=mask)
    return result["image"], result["mask"]
