"""Run training for multiple models sequentially with default settings.

Trains YOLO, U-Net++ multilabel, and U-Net++ semantic one by one.
Each model uses its default hyperparameters. Override any setting via CLI.

Usage:
    # Train all 3 models with defaults
    python train/run_grid_training.py

    # Train specific models only
    python train/run_grid_training.py --models yolo unet_multilabel

    # Override shared settings
    python train/run_grid_training.py --epochs 100 --batch-size 8
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

TRAIN_DIR = Path(__file__).resolve().parent

MODELS = {
    "yolo": {
        "script": TRAIN_DIR / "train_yolo.py",
        "description": "YOLO26m-seg (instance segmentation, 6 raw classes)",
    },
    "unet_multilabel": {
        "script": TRAIN_DIR / "train_unet_binary.py",
        "description": "U-Net++ multilabel (6 sigmoid channels, raw classes)",
    },
    "unet_semantic": {
        "script": TRAIN_DIR / "train_unet_semantic.py",
        "description": "U-Net++ semantic (7-class softmax)",
    },
}

DEFAULT_ORDER = ["yolo", "unet_multilabel", "unet_semantic"]


def run_model(name: str, extra_args: list) -> bool:
    """Run a single model training. Returns True if successful."""
    info = MODELS[name]
    script = str(info["script"])

    print(f"\n{'=' * 70}")
    print(f"  TRAINING: {name}")
    print(f"  {info['description']}")
    print(f"  Script: {script}")
    if extra_args:
        print(f"  Extra args: {' '.join(extra_args)}")
    print(f"{'=' * 70}\n")

    cmd = [sys.executable, script] + extra_args
    start = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start

    hours, remainder = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(remainder, 60)

    if result.returncode == 0:
        print(f"\n  {name} COMPLETED in {hours}h {minutes}m {seconds}s")
    else:
        print(f"\n  {name} FAILED (exit code {result.returncode}) after {hours}h {minutes}m {seconds}s")

    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Train multiple models sequentially")
    parser.add_argument("--models", nargs="+", default=DEFAULT_ORDER,
                        choices=list(MODELS.keys()),
                        help=f"Models to train (default: {DEFAULT_ORDER})")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epochs for all models")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size for all models")
    parser.add_argument("--img-size", type=int, default=None,
                        help="Override image size for all models")
    parser.add_argument("--patience", type=int, default=None,
                        help="Override early stopping patience")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override random seed")
    parser.add_argument("--strategy", default=None,
                        help="Override split strategy")
    args, unknown_args = parser.parse_known_args()

    # Build shared extra args from overrides
    extra_args = list(unknown_args)
    if args.epochs is not None:
        extra_args += ["--epochs", str(args.epochs)]
    if args.batch_size is not None:
        extra_args += ["--batch-size", str(args.batch_size)]
    if args.img_size is not None:
        extra_args += ["--img-size", str(args.img_size)]
    if args.patience is not None:
        extra_args += ["--patience", str(args.patience)]
    if args.seed is not None:
        extra_args += ["--seed", str(args.seed)]
    if args.strategy is not None:
        extra_args += ["--strategy", str(args.strategy)]

    print(f"Training {len(args.models)} models: {args.models}")
    if extra_args:
        print(f"Shared overrides: {' '.join(extra_args)}")

    results = {}
    total_start = time.time()

    for name in args.models:
        success = run_model(name, extra_args)
        results[name] = success
        if not success:
            print(f"\n  WARNING: {name} failed. Continuing with next model...")

    total_elapsed = time.time() - total_start
    hours, remainder = divmod(int(total_elapsed), 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"\n{'=' * 70}")
    print(f"  GRID TRAINING COMPLETE — {hours}h {minutes}m {seconds}s total")
    print(f"{'=' * 70}")
    for name, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  {name}: {status}")


if __name__ == "__main__":
    main()
