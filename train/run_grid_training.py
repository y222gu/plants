"""Grid training: Strategy A benchmark runs (1-5) sequentially.

Each run trains a model and then evaluates on the test set.

Hyperparameter Table (Runs 1-5, Strategy A Benchmark):
┌─────┬────────────────────────┬────────┬────┬─────┬──────┬─────────┬──────────┬─────┬────────┬─────────┬────────┬───────────┐
│ Run │ Model                  │ Arch   │ NC │ MLos│ ImgSz│ Epochs  │ Patience │ BS  │ LR     │ Backbone│ WD     │ Params    │
├─────┼────────────────────────┼────────┼────┼─────┼──────┼─────────┼──────────┼─────┼────────┼─────────┼────────┼───────────┤
│ 1   │ YOLO11m-seg            │ —      │  4 │ —   │ 1024 │ 300     │ 15       │ 16  │ (auto) │ (auto)  │ 5e-4   │ 22.4M     │
│ 2   │ U-Net++ multilabel     │ res34  │  4 │ No  │ 1024 │ 300     │ 15       │ 16  │ 1e-4   │ 1e-5    │ 1e-4   │ 24.4M     │
│ 3   │ U-Net++ multilabel     │ res34  │  5 │ Yes │ 1024 │ 300     │ 15       │ 16  │ 1e-4   │ 1e-5    │ 1e-4   │ 24.4M     │
│ 4   │ SAM vit_b              │ —      │  5 │ —   │ 1024 │ 300     │ 15       │ 8   │ 1e-4   │ frozen  │ 1e-4   │ 4.1M/93.7M│
│ 5   │ Cellpose v3 per-class  │ cyto3  │  5 │ —   │ 1024 │ 150     │ —        │ 8   │ 0.1    │ —       │ 1e-4   │ ~13M      │
└─────┴────────────────────────┴────────┴────┴─────┴──────┴─────────┴──────────┴─────┴────────┴─────────┴────────┴───────────┘

NC = num_classes, MLos = masked loss, BS = batch size, WD = weight decay, res34 = resnet34
Params = trainable parameters (SAM: trainable/total). YOLO and U-Net++ are fully fine-tuned.
U-Net++ BCE pos_weight: [1, 2, 5, 1] (4c) or [1, 2, 5, 1, 5] (5c). BCE:Dice ratio = 1:1.
SAM image encoder is frozen; only mask decoder (+ optionally prompt encoder) is trained.

Training Config (Optimizer, Scheduler, Checkpointing, Outputs):
┌─────┬───────┬──────────────────┬──────────┬──────────────────┬─────────┬──────────────────────────────────┐
│ Run │ Optim │ Scheduler        │ Loss     │ Early Stop       │ SaveEvr │ Output Files                     │
├─────┼───────┼──────────────────┼──────────┼──────────────────┼─────────┼──────────────────────────────────┤
│ 1   │ SGD   │ (auto/YOLO)      │ YOLO     │ mAP (Ultralytics)│ 50 ep   │ confusion_matrix, PR curves,     │
│     │       │                  │          │                  │         │ loss curves, label plots (auto)  │
│ 2-3 │ AdamW │ Cosine (1e-7)    │ BCE+Dice │ val_loss (PL)    │ 50 ep   │ metrics.csv, loss_curve.png      │
│     │       │                  │          │                  │         │ best + last + periodic ckpts     │
│ 4   │ AdamW │ Cosine (1e-7)    │ BCE+Dice │ val_loss (manual)│ 50 ep   │ training_history.json,           │
│     │       │                  │          │                  │         │ loss_curve.png, best + periodic  │
│ 5   │ AdamW │ (auto/Cellpose)  │ Flow     │ None (fixed ep.) │ (auto)  │ training_history.json,           │
│     │       │                  │          │                  │         │ loss_curve.png, test_results.json│
└─────┴───────┴──────────────────┴──────────┴──────────────────┴─────────┴──────────────────────────────────┘

evaluate.py outputs (per run): metrics.json, per_sample.csv, per_class_comparison.[png|pdf],
summary_comparison.[png|pdf], species_microscope_comparison.[png|pdf], vis/ overlay PNGs.
PL = PyTorch Lightning. Cosine = CosineAnnealingLR. SaveEvr = --save-every (periodic checkpoints).

Usage:
    python run_grid_training.py                     # Run all 5 benchmark runs
    python run_grid_training.py --only 1 3          # Run only runs #1 and #3
    python run_grid_training.py --skip 4 5          # Skip SAM and Cellpose
    python run_grid_training.py --epochs 5          # Quick test with 5 epochs
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_TRAIN_DIR = str(Path(__file__).resolve().parent)
_PROJECT_DIR = str(Path(__file__).resolve().parent.parent)

# ── Strategy A Benchmark Grid (Runs 1-5) ─────────────────────────────────────

GRID = [
    {
        "id": 1,
        "name": "YOLO11m-seg (4-class)",
        "train": [
            sys.executable, f"{_TRAIN_DIR}/train_yolo.py",
            "--strategy", "A",
            "--model", "yolo11m-seg",
            "--num-classes", "4",
            "--img-size", "1024",
            "--batch-size", "16",
            "--epochs", "300",
            "--patience", "15",
            "--save-every", "50",
        ],
        "eval": [
            sys.executable, f"{_PROJECT_DIR}/evaluate.py",
            "--model", "yolo",
            "--strategy", "A",
            "--num-classes", "4",
            "--no-vis",
        ],
        "checkpoint_pattern": "output/runs/yolo/yolo11m-seg_A/weights/best.pt",
    },
    {
        "id": 2,
        "name": "U-Net++ multilabel 4-class",
        "train": [
            sys.executable, f"{_TRAIN_DIR}/train_unet.py",
            "--mode", "multilabel",
            "--arch", "unetplusplus",
            "--encoder", "resnet34",
            "--strategy", "A",
            "--num-classes", "4",
            "--img-size", "1024",
            "--batch-size", "16",
            "--epochs", "300",
            "--lr", "1e-4",
            "--backbone-lr", "1e-5",
            "--weight-decay", "1e-4",
            "--patience", "15",
            "--save-every", "50",
        ],
        "eval": [
            sys.executable, f"{_PROJECT_DIR}/evaluate.py",
            "--model", "unet", "--unet-mode", "multilabel",
            "--strategy", "A",
            "--num-classes", "4",
            "--no-vis",
        ],
        "checkpoint_pattern": "output/runs/unet/unetplusplus_resnet34_A_multilabel_c4/checkpoints/best-*.ckpt",
    },
    {
        "id": 3,
        "name": "U-Net++ multilabel 5-class masked",
        "train": [
            sys.executable, f"{_TRAIN_DIR}/train_unet.py",
            "--mode", "multilabel",
            "--arch", "unetplusplus",
            "--encoder", "resnet34",
            "--strategy", "A",
            "--num-classes", "5",
            "--mask-missing",
            "--img-size", "1024",
            "--batch-size", "16",
            "--epochs", "300",
            "--lr", "1e-4",
            "--backbone-lr", "1e-5",
            "--weight-decay", "1e-4",
            "--patience", "15",
            "--save-every", "50",
        ],
        "eval": [
            sys.executable, f"{_PROJECT_DIR}/evaluate.py",
            "--model", "unet", "--unet-mode", "multilabel",
            "--strategy", "A",
            "--num-classes", "5",
            "--no-vis",
        ],
        "checkpoint_pattern": "output/runs/unet/unetplusplus_resnet34_A_multilabel_c5_masked/checkpoints/best-*.ckpt",
    },
    {
        "id": 4,
        "name": "SAM vit_b (5-class)",
        "train": [
            sys.executable, f"{_TRAIN_DIR}/train_sam.py",
            "--strategy", "A",
            "--sam-type", "vit_b",
            "--num-classes", "5",
            "--img-size", "1024",
            "--batch-size", "16",
            "--epochs", "300",
            "--lr", "1e-4",
            "--weight-decay", "1e-4",
            "--patience", "15",
            "--save-every", "50",
        ],
        "eval": [
            sys.executable, f"{_PROJECT_DIR}/evaluate.py",
            "--model", "sam",
            "--sam-type", "vit_b",
            "--strategy", "A",
            "--num-classes", "5",
            "--no-vis",
        ],
        "checkpoint_pattern": "output/runs/sam/sam_vit_b_A_c5/best.pth",
    },
    {
        "id": 5,
        "name": "Cellpose v3 per-class (5-class)",
        "train": [
            sys.executable, f"{_TRAIN_DIR}/train_cellpose.py",
            "--strategy", "A",
            "--version", "3",
            "--all-classes",
            "--num-classes", "5",
            "--img-size", "1024",
            "--batch-size", "16",
            "--epochs", "150",
            "--lr", "0.1",
        ],
        "eval": [
            sys.executable, f"{_PROJECT_DIR}/evaluate.py",
            "--model", "cellpose",
            "--strategy", "A",
            "--num-classes", "5",
            "--no-vis",
        ],
        "checkpoint_pattern": "output/runs/cellpose/",
    },
]


def find_checkpoint(pattern: str) -> str:
    """Find the best checkpoint file or directory matching a glob pattern."""
    if pattern is None:
        return ""
    base = Path(__file__).parent.parent
    full_path = base / pattern
    # If pattern points to a directory (e.g. Cellpose), return it directly
    if full_path.is_dir():
        return str(full_path)
    import glob
    matches = sorted(glob.glob(str(full_path)))
    if not matches:
        # Also try "last.ckpt" for Lightning
        last = Path(pattern).parent / "last.ckpt"
        if (base / last).exists():
            return str(base / last)
        return ""
    return matches[-1]


def run_command(cmd, label, extra_args=None):
    """Run a command and return (success, elapsed_seconds)."""
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
            cwd=str(Path(__file__).parent.parent),
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
    parser = argparse.ArgumentParser(description="Grid training for Strategy A benchmark (runs 1-5)")
    parser.add_argument("--only", nargs="+", type=int, default=None,
                        help="Only run these run IDs (1-5)")
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

        if not args.train_only and run["eval"] is not None:
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
        elif run["eval"] is None:
            run_result["eval_ok"] = None

        results.append(run_result)

    total_elapsed = time.time() - total_start

    # Summary table
    print("\n" + "=" * 80)
    print("GRID TRAINING SUMMARY")
    print("=" * 80)
    print(f"{'#':<3} {'Name':<38} {'Train':<8} {'T.Time':<10} {'Eval':<8} {'E.Time':<10}")
    print("-" * 80)
    for r in results:
        train_str = "OK" if r["train_ok"] else ("FAIL" if r["train_ok"] is False else "SKIP")
        eval_str = "OK" if r["eval_ok"] else ("FAIL" if r["eval_ok"] is False else ("N/A" if r["eval_ok"] is None else "SKIP"))
        print(f"{r['id']:<3} {r['name']:<38} {train_str:<8} "
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
