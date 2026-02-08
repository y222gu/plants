"""Cellpose training script — compare v2 and v3 models.

Usage:
    python train_cellpose.py --strategy strategy1 --class-id 1
    python train_cellpose.py --strategy strategy1 --all-classes
    python train_cellpose.py --version 3 --strategy strategy1 --class-id 1
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from cellpose import models, train

from src.config import DEFAULT_IMG_SIZE, NUM_CLASSES, OUTPUT_DIR, TARGET_CLASSES
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
    learning_rate: float = 0.1,
    batch_size: int = 8,
    model_type: str = "cyto2",
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

    # Initialize model
    model = models.CellposeModel(
        gpu=True,
        model_type=model_type,
    )

    # Train
    model_path, train_losses, test_losses = train.train_seg(
        model.net,
        train_data=train_images,
        train_labels=train_labels,
        test_data=val_images if val_images else None,
        test_labels=val_labels if val_labels else None,
        channels=[0, 0],  # grayscale-like (use all channels)
        save_path=str(model_dir),
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        nimg_per_epoch=len(train_images),
    )

    # Save training history
    history = {
        "train_losses": [float(x) for x in train_losses],
        "test_losses": [float(x) for x in test_losses] if test_losses else [],
    }
    with open(model_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

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
    mean_ap = float(np.mean(ap_scores))

    results = {
        "mean_AP": mean_ap,
        "per_sample_AP": [float(x) for x in ap_scores],
        "n_samples": len(test_images),
    }

    print(f"  Mean AP: {mean_ap:.4f}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Cellpose training")
    parser.add_argument("--strategy", default="strategy1",
                        choices=["strategy1", "strategy2", "strategy3"])
    parser.add_argument("--species", default=None)
    parser.add_argument("--class-id", type=int, default=None,
                        help="Target class ID (0-3). If None, train on all classes.")
    parser.add_argument("--all-classes", action="store_true",
                        help="Train separate models for each class")
    parser.add_argument("--version", type=int, default=2, choices=[2, 3],
                        help="Cellpose version")
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Setup
    registry = SampleRegistry()
    split = get_split(args.strategy, registry, seed=args.seed, species=args.species)
    print_split_summary(split)

    # Determine classes to train
    if args.all_classes:
        class_ids = list(range(NUM_CLASSES))
    elif args.class_id is not None:
        class_ids = [args.class_id]
    else:
        class_ids = [None]  # all classes combined

    for class_id in class_ids:
        class_name = TARGET_CLASSES.get(class_id, "all") if class_id is not None else "all"
        print(f"\n{'='*60}")
        print(f"Training Cellpose v{args.version} for class: {class_name}")

        # Prepare data
        train_imgs, train_lbls = prepare_cellpose_data(
            split["train"], img_size=args.img_size, target_class=class_id,
        )
        val_imgs, val_lbls = prepare_cellpose_data(
            split["val"], img_size=args.img_size, target_class=class_id,
        )
        test_imgs, test_lbls = prepare_cellpose_data(
            split["test"], img_size=args.img_size, target_class=class_id,
        )

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
        run_name = f"cellpose_v{args.version}_{class_name}_{args.strategy}"
        model_dir = OUTPUT_DIR / "runs" / "cellpose" / run_name

        model_path = train_cellpose_model(
            train_imgs, train_lbls,
            val_imgs, val_lbls,
            model_dir,
            version=args.version,
            n_epochs=args.epochs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
        )

        # Evaluate
        print(f"\nEvaluating on test set ({len(test_imgs)} samples):")
        test_results = evaluate_cellpose(model_path, test_imgs, test_lbls)

        with open(model_dir / "test_results.json", "w") as f:
            json.dump(test_results, f, indent=2)


if __name__ == "__main__":
    main()
