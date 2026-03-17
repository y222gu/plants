"""Cellpose training script — compare v2 and v3 models.

Usage:
    python train_cellpose.py --strategy A --class-id 1
    python train_cellpose.py --strategy A --all-classes
    python train_cellpose.py --version 3 --strategy A --class-id 1
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
from typing import Dict, List, Optional

import numpy as np
from cellpose import models, train

from src.config import DEFAULT_IMG_SIZE, OUTPUT_DIR, get_target_classes, make_run_subfolder, save_hparams
from src.dataset import SampleRegistry
from src.models.cellpose_utils import prepare_cellpose_data
from src.splits import get_split, print_split_summary


def train_cellpose_model(
    train_images: List[np.ndarray],
    train_labels: List[np.ndarray],
    val_images: List[np.ndarray],
    val_labels: List[np.ndarray],
    model_dir: Path,
    version: int = 2,
    n_epochs: int = 100,
    learning_rate: float = 0.01,
    weight_decay: float = 0.01,
    batch_size: int = 8,
    model_type: str = "cyto2",
    rescale: bool = True,
    scale_range: float = 0.5,
    nimg_per_epoch: Optional[int] = None,
) -> Path:
    """Train a Cellpose model.

    Args:
        train_images: List of (H,W,C) uint8 images.
        train_labels: List of (H,W) int32 instance masks.
        val_images: Validation images.
        val_labels: Validation masks.
        model_dir: Directory to save model.
        version: Cellpose version (2 or 3).
        n_epochs: Number of epochs.
        learning_rate: Learning rate.
        batch_size: Batch size.
        model_type: Base model type (e.g. "cyto2", "cyto3").

    Returns:
        Path to saved model.
    """
    model_dir.mkdir(parents=True, exist_ok=True)

    if version == 3:
        model_type = "cyto3"

    # Initialize model — verify CUDA
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Cellpose training requires a GPU.")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    model = models.CellposeModel(
        gpu=True,
        model_type=model_type,
    )
    print(f"Cellpose model device: {model.device}")

    # Train
    nimg_epoch = nimg_per_epoch if nimg_per_epoch is not None else len(train_images)
    model_path, train_losses, test_losses = train.train_seg(
        model.net,
        train_data=train_images,
        train_labels=train_labels,
        test_data=val_images if val_images else None,
        test_labels=val_labels if val_labels else None,
        save_path=str(model_dir),
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        nimg_per_epoch=nimg_epoch,
        min_train_masks=1,
        rescale=rescale,
        scale_range=scale_range,
    )

    # Save training history
    history = {
        "train_losses": [float(x) for x in train_losses] if train_losses is not None and len(train_losses) > 0 else [],
        "test_losses": [float(x) for x in test_losses] if test_losses is not None and len(test_losses) > 0 else [],
    }
    with open(model_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Plot loss curves
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history["train_losses"], label="Train Loss")
    if history["test_losses"]:
        # Cellpose only computes val loss at epoch 0, 5, and every 10 epochs;
        # intermediate epochs are logged as 0.0 — filter those out for plotting.
        val_epochs = [i for i, v in enumerate(history["test_losses"]) if v > 0]
        val_values = [history["test_losses"][i] for i in val_epochs]
        ax.plot(val_epochs, val_values, "o-", label="Val Loss", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Cellpose v{version} Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(model_dir / "loss_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Loss curve saved to {model_dir / 'loss_curve.png'}")

    print(f"Model saved to {model_path}")
    return Path(model_path)


def evaluate_cellpose(
    model_path: Path,
    test_images: List[np.ndarray],
    test_labels: List[np.ndarray],
) -> Dict:
    """Evaluate Cellpose model on test set."""
    model = models.CellposeModel(
        gpu=True,
        pretrained_model=str(model_path),
    )

    # Run inference
    pred_masks = model.eval(
        test_images,
        channels=[0, 0],
        diameter=None,  # auto-detect
    )[0]

    # Compute metrics
    from cellpose import metrics as cp_metrics

    ap_scores = cp_metrics.average_precision(test_labels, pred_masks)[0]
    # ap_scores may be multi-dimensional (samples x IoU thresholds);
    # average per sample across thresholds if needed
    ap_scores = np.array(ap_scores)
    if ap_scores.ndim > 1:
        per_sample_ap = ap_scores.mean(axis=-1)
    else:
        per_sample_ap = ap_scores
    mean_ap = float(np.mean(per_sample_ap))

    results = {
        "mean_AP": mean_ap,
        "per_sample_AP": [float(x) for x in per_sample_ap],
        "n_samples": len(test_images),
    }

    print(f"  Mean AP: {mean_ap:.4f}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Cellpose training")
    parser.add_argument("--strategy", default="A",
                        choices=["A", "B", "C"])
    parser.add_argument("--species", default=None)
    parser.add_argument("--num-classes", type=int, default=4, choices=[4, 5],
                        help="Number of target classes (4=standard, 5=with exodermis)")
    parser.add_argument("--class-id", type=int, default=None,
                        help="Target class ID (0-3 or 0-4). If None, train on all classes.")
    parser.add_argument("--all-classes", action="store_true",
                        help="Train separate models for each class")
    parser.add_argument("--version", type=int, default=2, choices=[2, 3],
                        help="Cellpose version")
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--rescale", action="store_true", default=True,
                        help="Enable diameter-based rescaling during training")
    parser.add_argument("--no-rescale", dest="rescale", action="store_false")
    parser.add_argument("--scale-range", type=float, default=0.5,
                        help="Random rescaling range for augmentation")
    parser.add_argument("--nimg-per-epoch", type=int, default=None,
                        help="Images per epoch (None=use all)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Setup
    registry = SampleRegistry()
    split = get_split(args.strategy, registry, seed=args.seed, species=args.species)
    print_split_summary(split)

    # Determine classes to train
    target_classes = get_target_classes(args.num_classes)
    if args.all_classes:
        class_ids = list(range(args.num_classes))
    elif args.class_id is not None:
        class_ids = [args.class_id]
    else:
        class_ids = [None]  # all classes combined

    # Pre-load images and annotations once, then build per-class labels
    # without reloading from disk for each class.
    from src.models.cellpose_utils import preload_cellpose_data, build_class_labels

    print("Pre-loading images and annotations (shared across all classes)...")
    train_cache = preload_cellpose_data(split["train"], args.img_size, args.num_classes)
    val_cache = preload_cellpose_data(split["val"], args.img_size, args.num_classes)
    test_cache = preload_cellpose_data(split["test"], args.img_size, args.num_classes)

    for class_id in class_ids:
        class_name = target_classes.get(class_id, "all") if class_id is not None else "all"
        print(f"\n{'='*60}")
        print(f"Training Cellpose v{args.version} for class: {class_name}")

        # Build per-class labels from cached data (no disk I/O)
        train_imgs, train_lbls = build_class_labels(train_cache, class_id, args.img_size)
        val_imgs, val_lbls = build_class_labels(val_cache, class_id, args.img_size)
        test_imgs, test_lbls = build_class_labels(test_cache, class_id, args.img_size)

        # Filter out samples with no instances
        train_pairs = [(i, l) for i, l in zip(train_imgs, train_lbls) if l.max() > 0]
        val_pairs = [(i, l) for i, l in zip(val_imgs, val_lbls) if l.max() > 0]

        if not train_pairs:
            print(f"  No training samples with class {class_name}. Skipping.")
            continue

        train_imgs, train_lbls = zip(*train_pairs)
        train_imgs, train_lbls = list(train_imgs), list(train_lbls)
        if val_pairs:
            val_imgs, val_lbls = zip(*val_pairs)
            val_imgs, val_lbls = list(val_imgs), list(val_lbls)
        else:
            val_imgs, val_lbls = [], []

        print(f"  Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}")

        # Train
        run_name = f"cellpose_v{args.version}_{class_name}_{args.strategy}_c{args.num_classes}"
        base_model_dir = OUTPUT_DIR / "runs" / "cellpose" / run_name
        model_dir = make_run_subfolder(base_model_dir)
        save_hparams(model_dir, args)

        model_path = train_cellpose_model(
            train_imgs, train_lbls,
            val_imgs, val_lbls,
            model_dir,
            version=args.version,
            n_epochs=args.epochs,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            rescale=args.rescale,
            scale_range=args.scale_range,
            nimg_per_epoch=args.nimg_per_epoch,
        )

        # Evaluate
        print(f"\nEvaluating on test set ({len(test_imgs)} samples):")
        test_results = evaluate_cellpose(model_path, test_imgs, test_lbls)

        with open(model_dir / "test_results.json", "w") as f:
            json.dump(test_results, f, indent=2)


if __name__ == "__main__":
    main()
