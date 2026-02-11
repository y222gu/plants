"""Grid training: run all model/mode/strategy combinations sequentially.

Each run trains a model and then evaluates on the test set.

Usage:
    python run_grid_training.py                     # Run all 6 combinations
    python run_grid_training.py --only 1 3          # Run only runs #1 and #3
    python run_grid_training.py --skip 5 6          # Skip YOLO runs
    python run_grid_training.py --epochs 5          # Quick test with 5 epochs
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from src.config import OUTPUT_DIR


# Grid definition: (name, train_cmd, eval_cmd)
GRID = [
    {
        "id": 1,
        "name": "UNet Multilabel Strategy1",
        "train": [
            sys.executable, "train_unet.py",
            "--mode", "multilabel", "--strategy", "strategy1",
        ],
        "eval": [
            sys.executable, "evaluate.py",
            "--model", "unet", "--unet-mode", "multilabel",
            "--strategy", "strategy1", "--no-vis",
        ],
        "checkpoint_pattern": "output/runs/unet/unet_resnet34_strategy1_multilabel/checkpoints/best-*.ckpt",
    },
    {
        "id": 2,
        "name": "UNet Multilabel Strategy2",
        "train": [
            sys.executable, "train_unet.py",
            "--mode", "multilabel", "--strategy", "strategy2",
        ],
        "eval": [
            sys.executable, "evaluate.py",
            "--model", "unet", "--unet-mode", "multilabel",
            "--strategy", "strategy2", "--no-vis",
        ],
        "checkpoint_pattern": "output/runs/unet/unet_resnet34_strategy2_multilabel/checkpoints/best-*.ckpt",
    },
    {
        "id": 3,
        "name": "UNet Semantic Strategy1",
        "train": [
            sys.executable, "train_unet.py",
            "--mode", "semantic", "--strategy", "strategy1",
        ],
        "eval": [
            sys.executable, "evaluate.py",
            "--model", "unet", "--unet-mode", "semantic",
            "--strategy", "strategy1", "--no-vis",
        ],
        "checkpoint_pattern": "output/runs/unet/unet_resnet34_strategy1_semantic/checkpoints/best-*.ckpt",
    },
    {
        "id": 4,
        "name": "UNet Semantic Strategy2",
        "train": [
            sys.executable, "train_unet.py",
            "--mode", "semantic", "--strategy", "strategy2",
        ],
        "eval": [
            sys.executable, "evaluate.py",
            "--model", "unet", "--unet-mode", "semantic",
            "--strategy", "strategy2", "--no-vis",
        ],
        "checkpoint_pattern": "output/runs/unet/unet_resnet34_strategy2_semantic/checkpoints/best-*.ckpt",
    },
    {
        "id": 5,
        "name": "YOLO Strategy1",
        "train": [
            sys.executable, "train_yolo.py",
            "--strategy", "strategy1",
        ],
        "eval": [
            sys.executable, "evaluate.py",
            "--model", "yolo",
            "--strategy", "strategy1", "--no-vis",
        ],
        "checkpoint_pattern": "output/runs/yolo/yolo11m-seg_strategy1/weights/best.pt",
    },
    {
        "id": 6,
        "name": "YOLO Strategy2",
        "train": [
            sys.executable, "train_yolo.py",
            "--strategy", "strategy2",
        ],
        "eval": [
            sys.executable, "evaluate.py",
            "--model", "yolo",
            "--strategy", "strategy2", "--no-vis",
        ],
        "checkpoint_pattern": "output/runs/yolo/yolo11m-seg_strategy2/weights/best.pt",
    },
]


def find_checkpoint(pattern: str) -> str:
    """Find the best checkpoint file matching a glob pattern."""
    import glob
    base = Path(__file__).parent
    matches = sorted(glob.glob(str(base / pattern)))
    if not matches:
        # Also try "last.ckpt" for Lightning
        last = Path(pattern).parent / "last.ckpt"
        if (base / last).exists():
            return str(base / last)
        return ""
    return matches[-1]


def run_command(cmd, label, extra_args=None):
    """Run a command and return (success, elapsed_seconds, output)."""
    if extra_args:
        cmd = cmd + extra_args
    print(f"\n{'=' * 70}")
    print(f"RUNNING: {label}")
    print(f"CMD: {' '.join(cmd)}")
    print(f"{'=' * 70}\n")

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(Path(__file__).parent),
            timeout=86400,  # 24h max per run
        )
        elapsed = time.time() - start
        success = result.returncode == 0
        return success, elapsed
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        print(f"TIMEOUT after {elapsed:.0f}s")
        return False, elapsed
    except Exception as e:
        elapsed = time.time() - start
        print(f"ERROR: {e}")
        return False, elapsed


def format_time(seconds):
    """Format seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def main():
    parser = argparse.ArgumentParser(description="Grid training for all model combinations")
    parser.add_argument("--only", nargs="+", type=int, default=None,
                        help="Only run these run IDs (1-6)")
    parser.add_argument("--skip", nargs="+", type=int, default=None,
                        help="Skip these run IDs")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epochs for all runs")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size for all runs")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, only run evaluation on existing checkpoints")
    parser.add_argument("--train-only", action="store_true",
                        help="Skip evaluation, only run training")
    args = parser.parse_args()

    # Filter runs
    runs = GRID[:]
    if args.only:
        runs = [r for r in runs if r["id"] in args.only]
    if args.skip:
        runs = [r for r in runs if r["id"] not in args.skip]

    # Build extra args
    extra_train_args = []
    if args.epochs is not None:
        extra_train_args.extend(["--epochs", str(args.epochs)])
    if args.batch_size is not None:
        extra_train_args.extend(["--batch-size", str(args.batch_size)])

    print(f"Grid Training: {len(runs)} runs planned")
    for r in runs:
        print(f"  #{r['id']}: {r['name']}")
    print()

    results = []
    total_start = time.time()

    for run in runs:
        run_result = {
            "id": run["id"],
            "name": run["name"],
            "train_ok": None,
            "eval_ok": None,
            "train_time": 0,
            "eval_time": 0,
        }

        # Training
        if not args.eval_only:
            ok, elapsed = run_command(
                run["train"], f"TRAIN #{run['id']}: {run['name']}",
                extra_args=extra_train_args,
            )
            run_result["train_ok"] = ok
            run_result["train_time"] = elapsed
            if not ok:
                print(f"Training FAILED for #{run['id']}: {run['name']}")
                results.append(run_result)
                continue

        # Evaluation
        if not args.train_only:
            checkpoint = find_checkpoint(run["checkpoint_pattern"])
            if not checkpoint:
                print(f"No checkpoint found for #{run['id']}: {run['name']}")
                print(f"  Pattern: {run['checkpoint_pattern']}")
                run_result["eval_ok"] = False
                results.append(run_result)
                continue

            eval_cmd = run["eval"] + ["--checkpoint", checkpoint]
            ok, elapsed = run_command(
                eval_cmd, f"EVAL #{run['id']}: {run['name']}",
            )
            run_result["eval_ok"] = ok
            run_result["eval_time"] = elapsed

        results.append(run_result)

    total_elapsed = time.time() - total_start

    # Summary table
    print("\n" + "=" * 80)
    print("GRID TRAINING SUMMARY")
    print("=" * 80)
    print(f"{'#':<3} {'Name':<30} {'Train':<8} {'T.Time':<10} {'Eval':<8} {'E.Time':<10}")
    print("-" * 80)
    for r in results:
        train_str = "OK" if r["train_ok"] else ("FAIL" if r["train_ok"] is False else "SKIP")
        eval_str = "OK" if r["eval_ok"] else ("FAIL" if r["eval_ok"] is False else "SKIP")
        print(f"{r['id']:<3} {r['name']:<30} {train_str:<8} "
              f"{format_time(r['train_time']):<10} {eval_str:<8} "
              f"{format_time(r['eval_time']):<10}")
    print("-" * 80)
    print(f"Total time: {format_time(total_elapsed)}")

    # Check for failures
    failures = [r for r in results if r["train_ok"] is False or r["eval_ok"] is False]
    if failures:
        print(f"\n{len(failures)} run(s) had failures.")
        sys.exit(1)
    else:
        print("\nAll runs completed successfully.")


if __name__ == "__main__":
    main()
