"""YOLOv8/v11-seg training script.

Usage:
    python train_yolo.py --model yolo11m-seg
    python train_yolo.py --model yolo11m-seg --epochs 150 --batch-size 2
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

from ultralytics import YOLO

from src.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_IMG_SIZE,
    DEFAULT_PATIENCE,
    OUTPUT_DIR,
    make_run_subfolder,
    save_hparams,
)
from src.formats.yolo_format import export_yolo_dataset
from src.splits import get_split, print_split_summary


def main():
    parser = argparse.ArgumentParser(description="YOLO instance segmentation training")
    parser.add_argument("--model", default="yolo26m-seg",
                        help="YOLO model variant (e.g. yolo11m-seg, yolo26m-seg)")
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--num-classes", type=int, default=6, choices=[4, 6],
                        help="Number of raw annotation classes (6=all classes, 4=cereals only)")
    parser.add_argument("--save-every", type=int, default=10,
                        help="Save periodic checkpoint every N epochs (-1 to disable)")
    parser.add_argument("--gpus", type=int, default=1,
                        help="Number of GPUs (Ultralytics handles DDP internally)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--export-only", action="store_true",
                        help="Only export dataset, don't train")
    parser.add_argument("--force-export", action="store_true",
                        help="Force re-export even if dataset already exists")
    parser.add_argument("--strategy", default="A", help="Split strategy")
    args = parser.parse_args()

    # Setup
    split = get_split(strategy=args.strategy)
    print_split_summary(split)

    # Export to YOLO format (skip if already exported with matching counts)
    run_name = f"{args.model}"
    export_dir = OUTPUT_DIR / "yolo_dataset" / run_name
    yaml_path = export_dir / "data.yaml"

    if not args.force_export and yaml_path.exists():
        # Check if exported image counts match the split
        counts_match = True
        for subset, samples in split.items():
            img_dir = export_dir / "images" / subset
            if not img_dir.exists():
                counts_match = False
                break
            exported_count = len(list(img_dir.glob("*.png")))
            if exported_count != len(samples):
                print(f"  {subset}: expected {len(samples)}, found {exported_count}")
                counts_match = False
                break

        if counts_match:
            print(f"YOLO dataset already exported at {export_dir} (counts match). Skipping export.")
            print("  Use --force-export to re-export.")
        else:
            print("Existing export has mismatched counts. Re-exporting...")
            yaml_path = export_yolo_dataset(split, export_dir, img_size=args.img_size,
                                            num_classes=args.num_classes)
    else:
        yaml_path = export_yolo_dataset(split, export_dir, img_size=args.img_size,
                                        num_classes=args.num_classes)

    if args.export_only:
        print("Export complete. Exiting.")
        return

    # Train
    project_dir = OUTPUT_DIR / "runs" / "yolo" / run_name
    run_dir = make_run_subfolder(project_dir)
    dated_name = run_dir.name
    save_hparams(run_dir, args)

    model = YOLO(args.model)

    if args.resume:
        model = YOLO(args.resume)

    # Multi-GPU: pass list of device IDs for DDP, single int for single GPU
    device = list(range(args.gpus)) if args.gpus > 1 else 0

    results = model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch_size,
        patience=args.patience,
        device=device,
        project=str(project_dir),
        name=dated_name,
        save=True,
        save_period=args.save_every,
        amp=True,
        seed=args.seed,
        workers=8,
        exist_ok=True,
        plots=True,
        cos_lr=True,              # cosine annealing (better than linear decay)
        mask_ratio=1,             # mask resolution = img_size/1 (full resolution)
        # Augmentation — match shared albumentations pipeline
        hsv_h=0.0,               # no hue augmentation (fluorescence)
        hsv_s=0.0,               # no saturation augmentation (fluorescence)
        hsv_v=0.2,               # mild brightness
        degrees=45.0,            # rotation ±45°
        translate=0.1,           # translation ±10%
        scale=0.3,               # scale 0.7-1.3
        shear=10.0,              # shear ±10°
        flipud=0.5,
        fliplr=0.5,
        bgr=0.2,                 # BGR channel swap (similar to ChannelShuffle)
        mosaic=0.0,              # disabled for fair comparison
        mixup=0.0,               # disabled for fair comparison
    )

    # Plot loss curves from YOLO results CSV
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    results_csv = run_dir / "results.csv"
    if results_csv.exists():
        df = pd.read_csv(results_csv)
        df.columns = df.columns.str.strip()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Box + Seg loss
        loss_cols_train = [c for c in df.columns if "train" in c.lower() and "loss" in c.lower()]
        loss_cols_val = [c for c in df.columns if "val" in c.lower() and "loss" in c.lower()]
        for col in loss_cols_train:
            axes[0].plot(df["epoch"], df[col], label=col)
        for col in loss_cols_val:
            axes[0].plot(df["epoch"], df[col], label=col, linestyle="--")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title(f"{args.model} — Loss")
        axes[0].legend(fontsize=7)
        axes[0].grid(True, alpha=0.3)

        # mAP metrics
        map_cols = [c for c in df.columns if "map" in c.lower() or "mAP" in c]
        for col in map_cols:
            axes[1].plot(df["epoch"], df[col], label=col)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("mAP")
        axes[1].set_title(f"{args.model} — mAP")
        axes[1].legend(fontsize=7)
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        plot_path = run_dir / "loss_curve.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Loss curve saved to {plot_path}")

    # Evaluate on test set
    print("\n" + "=" * 60)
    print("EVALUATING ON TEST SET")
    best_model_path = run_dir / "weights" / "best.pt"
    if best_model_path.exists():
        best_model = YOLO(str(best_model_path))
        test_results = best_model.val(
            data=str(yaml_path),
            split="test",
            imgsz=args.img_size,
            batch=args.batch_size,
            project=str(project_dir),
            name=f"{dated_name}_test",
            exist_ok=True,
        )
        print(f"\nTest mAP@0.5: {test_results.seg.map50:.4f}")
        print(f"Test mAP@0.5:0.95: {test_results.seg.map:.4f}")
        print(f"Per-class AP@0.5: {test_results.seg.ap50}")
    else:
        print(f"Best model not found at {best_model_path}")


if __name__ == "__main__":
    main()
