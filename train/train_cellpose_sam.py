"""Fine-tune Cellpose-SAM for per-class instance segmentation.

Uses Cellpose's native train_seg() which efficiently handles flow computation,
augmentation (rotation + scale + crop), and training. Images are resized to
256×256 so the model sees the full root at native Cellpose-SAM resolution.

Trains one model per raw annotation class (6 classes total).

Usage:
    python train/train_cellpose_sam.py --class-id 0         # Whole Root
    python train/train_cellpose_sam.py --class-id 1         # Aerenchyma
    python train/train_cellpose_sam.py --all-classes        # All 6 sequentially
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import time

import numpy as np
import cv2
import yaml

from cellpose import models, train

from src.config import ANNOTATED_CLASSES, OUTPUT_DIR, make_run_subfolder
from src.preprocessing import load_sample_normalized
from src.annotation_utils import load_sample_annotations
from src.splits import get_split, print_split_summary


def preprocess_maxproj_unsharp(img):
    """Max projection + unsharp mask → single channel float32 [0, 1]."""
    dapi, fitc, tritc = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    max_proj = np.maximum(np.maximum(dapi, fitc), tritc)
    # Percentile normalize
    lo = np.percentile(max_proj, 1)
    hi = np.percentile(max_proj, 99.8)
    if hi - lo < 1e-6:
        mp_norm = np.zeros_like(max_proj, dtype=np.float32)
    else:
        mp_norm = np.clip((max_proj - lo) / (hi - lo), 0, 1).astype(np.float32)
    # Unsharp mask
    blur = cv2.GaussianBlur(mp_norm, (0, 0), sigmaX=3)
    unsharp = np.clip(cv2.addWeighted(mp_norm, 1.5, blur, -0.5, 0), 0, 1).astype(np.float32)
    return unsharp


def prepare_data(samples, class_id, img_size=256):
    """Prepare images and instance labels for a single class.

    Preprocessing: max projection + unsharp mask → 1 channel, stacked to 3ch
    for Cellpose (which expects 3-channel input).

    Returns:
        images: list of (3, H, W) float32 arrays
        labels: list of (H, W) int32 instance label arrays
    """
    images, labels = [], []

    for sample in samples:
        img = load_sample_normalized(sample)  # (H, W, 3) float32 [0, 1]
        h, w = img.shape[:2]

        # Preprocessing: max projection + unsharp → 1 channel
        single_ch = preprocess_maxproj_unsharp(img)

        # Build instance mask for this class
        anns = load_sample_annotations(sample, h, w, raw_classes=True)
        instance_mask = np.zeros((h, w), dtype=np.int32)
        inst_id = 0
        for i in range(len(anns["labels"])):
            if anns["labels"][i] == class_id:
                inst_id += 1
                instance_mask[anns["masks"][i] > 0] = inst_id

        # Resize to target
        img_resized = cv2.resize(single_ch, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(instance_mask, (img_size, img_size),
                                   interpolation=cv2.INTER_NEAREST)

        # Cellpose expects (3, H, W) — stack single channel 3x
        img_chw = np.stack([img_resized, img_resized, img_resized], axis=0).astype(np.float32)

        images.append(img_chw)
        labels.append(mask_resized)

    n_with_instances = sum(1 for l in labels if l.max() > 0)
    print(f"  Class {class_id} ({ANNOTATED_CLASSES[class_id]}): "
          f"{len(images)} samples, {n_with_instances} with instances")
    return images, labels


def train_single_class(class_id, split, args):
    """Train Cellpose-SAM for a single annotation class."""
    class_name = ANNOTATED_CLASSES[class_id]
    print(f"\n{'='*60}")
    print(f"Training class {class_id}: {class_name}")
    print(f"{'='*60}")

    # Prepare data
    print("  Loading training data...")
    train_imgs, train_labels = prepare_data(split["train"], class_id, args.img_size)
    print("  Loading validation data...")
    val_imgs, val_labels = prepare_data(split["val"], class_id, args.img_size)

    if len(train_imgs) == 0:
        print(f"  No training data for class {class_id}, skipping")
        return None

    # Output directory
    pretrained_tag = "cpsam" if not args.no_pretrained else "scratch"
    run_name = f"cellpose_{pretrained_tag}_class{class_id}_{args.strategy}"
    base_dir = OUTPUT_DIR / "runs" / "cellpose_sam" / run_name
    run_dir = make_run_subfolder(base_dir)
    print(f"  Output: {run_dir}")

    # Save hyperparameters
    hparams = {
        "class_id": class_id,
        "class_name": class_name,
        "pretrained": not args.no_pretrained,
        "n_epochs": args.epochs,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "img_size": args.img_size,
        "n_train": len(train_imgs),
        "n_val": len(val_imgs),
        "strategy": args.strategy,
        "seed": args.seed,
    }
    with open(run_dir / "hparams.yaml", "w") as f:
        yaml.dump(hparams, f, sort_keys=True)

    # Load model
    print(f"  Loading Cellpose-SAM (cpsam)...")
    model = models.CellposeModel(gpu=True, pretrained_model="cpsam")

    # Train using Cellpose's native training loop
    print(f"  Training: {len(train_imgs)} train, {len(val_imgs)} val, "
          f"{args.epochs} epochs, lr={args.lr}, batch={args.batch_size}, "
          f"img_size={args.img_size}")

    np.random.seed(args.seed)
    t0 = time.time()

    model_path, train_losses, test_losses = train.train_seg(
        net=model.net,
        train_data=train_imgs,
        train_labels=train_labels,
        test_data=val_imgs,
        test_labels=val_labels,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        n_epochs=args.epochs,
        weight_decay=args.weight_decay,
        normalize=False,  # Already normalized
        compute_flows=True,  # Precompute flows for efficiency
        save_path=str(run_dir),
        save_every=args.save_every,
        save_each=True,
        bsize=args.img_size,  # Use full image, no cropping
        min_train_masks=0,
        model_name=f"cellpose_sam_class{class_id}",
        rescale=False,
    )

    elapsed = time.time() - t0
    print(f"  Training completed in {elapsed/60:.1f} min")

    # Save training history
    history = {
        "train_losses": train_losses.tolist(),
        "test_losses": test_losses.tolist(),
        "n_epochs": args.epochs,
        "elapsed_seconds": elapsed,
        "model_path": str(model_path),
    }
    with open(run_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Plot training curves
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        epochs = np.arange(len(train_losses))
        ax.plot(epochs, train_losses, label="Train Loss", alpha=0.7)

        # Test loss computed every 10 epochs
        test_mask = test_losses > 0
        if test_mask.any():
            ax.plot(epochs[test_mask], test_losses[test_mask], "o-",
                    label="Val Loss", markersize=4, alpha=0.7)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (MSE flows + BCE cellprob)")
        ax.set_title(f"Cellpose-SAM — Class {class_id} ({class_name})")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add final metrics text
        ax.text(0.98, 0.98, f"Final train: {train_losses[-1]:.4f}\n"
                f"Final val: {test_losses[test_mask][-1]:.4f}" if test_mask.any() else "",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        fig.tight_layout()
        fig.savefig(run_dir / "loss_curve.png", dpi=150)
        plt.close(fig)
        print(f"  Loss curve saved")
    except Exception as e:
        print(f"  Warning: could not plot: {e}")

    print(f"  Model saved to {model_path}")
    print(f"  Final train loss: {train_losses[-1]:.4f}")

    return run_dir


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Cellpose-SAM per-class")
    parser.add_argument("--class-id", type=int, default=None,
                        help="Single class to train (0-5)")
    parser.add_argument("--all-classes", action="store_true",
                        help="Train all 6 classes sequentially")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--img-size", type=int, default=256,
                        help="Resize images to this size (default: 256)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strategy", default="A")
    args = parser.parse_args()

    if args.class_id is None and not args.all_classes:
        parser.error("Must specify --class-id or --all-classes")

    split = get_split(args.strategy)
    print_split_summary(split)

    classes = list(range(6)) if args.all_classes else [args.class_id]

    results = {}
    for cls_id in classes:
        run_dir = train_single_class(cls_id, split, args)
        results[cls_id] = str(run_dir) if run_dir else "skipped"

    print(f"\n{'='*60}")
    print("All training complete:")
    for cls_id, path in results.items():
        print(f"  Class {cls_id} ({ANNOTATED_CLASSES[cls_id]}): {path}")


if __name__ == "__main__":
    main()
