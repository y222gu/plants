"""Visualize augmentation pipelines side-by-side for all models.

Picks one random training sample, augments it 20 times with each model's
pipeline, and saves comparison grids.

Usage:
    python visualize_augmentation.py
"""

import random
from pathlib import Path

import albumentations as A
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile


# ── Load a random sample ────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent / "data"
IMG_SIZE = 1024
N_AUGMENTS = 20
SEED = 42


def find_sample_dirs(base):
    """Find all sample directories (contain *_DAPI.tif)."""
    return sorted(set(p.parent for p in base.rglob("*_DAPI.tif")))


def load_and_normalize(sample_dir):
    """Load 3-channel fluorescence image, percentile-normalize to [0,1]."""
    channels = []
    for ch in ["TRITC", "FITC", "DAPI"]:
        tif = list(sample_dir.glob(f"*_{ch}.tif"))[0]
        img = tifffile.imread(str(tif)).astype(np.float32)
        if img.ndim > 2:
            img = img[0] if img.shape[0] < img.shape[-1] else img[..., 0]
        lo, hi = np.percentile(img, 1.0), np.percentile(img, 99.5)
        if hi > lo:
            img = np.clip((img - lo) / (hi - lo), 0, 1)
        else:
            img = np.zeros_like(img)
        channels.append(img)
    return np.stack(channels, axis=-1)  # (H, W, 3) float32 [0,1]


# ── Augmentation pipelines ──────────────────────────────────────────────────

def get_unet_sam_transform(img_size=IMG_SIZE):
    """Shared albumentations pipeline (U-Net++ and SAM)."""
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
        A.ElasticTransform(alpha=120, sigma=12, border_mode=0, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.GaussNoise(std_range=(0.01, 0.08), p=0.4),
        A.RandomGamma(gamma_limit=(70, 150), p=0.3),
        A.ChannelDropout(channel_drop_range=(1, 1), p=0.2),
        A.ChannelShuffle(p=0.2),
        A.Resize(img_size, img_size),
    ])


def get_yolo_transform(img_size=IMG_SIZE):
    """Replicate YOLO Ultralytics augmentation with albumentations.

    Matches: degrees=45, translate=0.1, scale=0.3 (0.7-1.3),
    shear=10, flipud=0.5, fliplr=0.5, bgr=0.2, hsv_v=0.2.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            scale=(0.7, 1.3),
            rotate=(-45, 45),
            shear=(-10, 10),
            border_mode=0, p=1.0,
        ),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.0, p=0.5),
        A.ChannelShuffle(p=0.2),  # approximates bgr=0.2
        A.Resize(img_size, img_size),
    ])


def get_cellpose_transform(img_size=512):
    """Replicate Cellpose random_rotate_and_resize() with albumentations.

    Cellpose: random rotation 0-360, scale 0.75-1.25, hflip 50%,
    then center-crop to output size. The actual Cellpose implementation
    crops around the image center with a small random offset.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Affine(
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            rotate=(-180, 180),
            scale=(0.7, 1.3),
            shear=(-10, 10),
            border_mode=0, p=1.0,
        ),
        A.Resize(img_size, img_size),
    ])


# ── Generate augmented grid ─────────────────────────────────────────────────

def augment_n_times(image, transform, n=N_AUGMENTS):
    """Apply transform n times, return list of augmented images."""
    results = []
    for _ in range(n):
        aug = transform(image=image)["image"]
        results.append(aug)
    return results


def make_grid(images, ncols=5, title=""):
    """Create a grid figure from a list of images."""
    nrows = (len(images) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i < len(images):
            img = images[i]
            img_display = np.clip(img, 0, 1)
            ax.imshow(img_display)
            ax.set_title(f"#{i+1}", fontsize=8)
        ax.axis("off")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    random.seed(SEED)
    np.random.seed(SEED)

    # Find a random training sample
    train_dirs = find_sample_dirs(DATA_DIR / "train" / "image")
    if not train_dirs:
        print("No training samples found in data/train/image/")
        return

    sample_dir = random.choice(train_dirs)
    sample_name = sample_dir.name
    print(f"Selected sample: {sample_dir}")

    # Load and normalize
    img = load_and_normalize(sample_dir)
    print(f"Image shape: {img.shape}, range: [{img.min():.3f}, {img.max():.3f}]")

    # Resize to 1024 for U-Net/SAM/YOLO base
    img_1024 = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

    # For cellpose: resize to larger than 512 so RandomCrop works
    cellpose_size = 512
    img_cellpose = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

    # Output directory
    out_dir = Path(__file__).parent / "output" / "augmentation_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save original
    fig_orig, ax_orig = plt.subplots(1, 1, figsize=(5, 5))
    ax_orig.imshow(np.clip(img_1024, 0, 1))
    ax_orig.set_title(f"Original: {sample_name}", fontsize=10)
    ax_orig.axis("off")
    fig_orig.tight_layout()
    fig_orig.savefig(out_dir / "00_original.png", dpi=150, bbox_inches="tight")
    plt.close(fig_orig)
    print(f"Saved original to {out_dir / '00_original.png'}")

    # Generate augmented images for each pipeline
    pipelines = {
        "UNet_SAM": (get_unet_sam_transform(IMG_SIZE), img_1024),
        "YOLO": (get_yolo_transform(IMG_SIZE), img_1024),
        "Cellpose": (get_cellpose_transform(cellpose_size), img_cellpose),
    }

    for name, (transform, base_img) in pipelines.items():
        print(f"Generating {N_AUGMENTS} augmented images for {name}...")
        augmented = augment_n_times(base_img, transform, N_AUGMENTS)
        fig = make_grid(augmented, ncols=5, title=f"{name} Augmentation ({sample_name})")
        save_path = out_dir / f"{name}_augmented.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved to {save_path}")

    print(f"\nAll outputs saved to {out_dir}/")


if __name__ == "__main__":
    main()
