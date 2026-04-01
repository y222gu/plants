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
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            scale=(0.7, 1.3),
            rotate=(-45, 45),
            shear=(-10, 10),
            border_mode=0, p=0.7,
        ),
        A.ElasticTransform(
            alpha=120, sigma=12, border_mode=0, p=0.3,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.3, contrast_limit=0.3, p=0.6,
        ),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.GaussNoise(std_range=(0.01, 0.08), p=0.4),
        A.RandomGamma(gamma_limit=(70, 150), p=0.3),
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
