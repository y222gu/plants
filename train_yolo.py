"""YOLOv8/v11-seg training script.

Usage:
    python train_yolo.py --model yolov8m-seg --strategy strategy1
    python train_yolo.py --model yolo11m-seg --strategy strategy2 --img-size 640
    python train_yolo.py --model yolov8l-seg --epochs 150 --batch-size 2
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

from src.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_IMG_SIZE,
    DEFAULT_PATIENCE,
    OUTPUT_DIR,
)
from src.dataset import SampleRegistry
from src.formats.yolo_format import export_yolo_dataset
from src.splits import get_split, print_split_summary


def main():
    parser = argparse.ArgumentParser(description="YOLO instance segmentation training")
    parser.add_argument("--model", default="yolov8m-seg",
                        help="YOLO model variant (e.g. yolov8m-seg, yolo11m-seg)")
    parser.add_argument("--strategy", default="strategy1",
                        choices=["strategy1", "strategy2", "strategy3"])
    parser.add_argument("--species", default=None,
                        help="Species for strategy3")
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--export-only", action="store_true",
                        help="Only export dataset, don't train")
    args = parser.parse_args()

    # Setup
    registry = SampleRegistry()
    print(registry.summary())

    split = get_split(args.strategy, registry, seed=args.seed, species=args.species)
    print_split_summary(split)

    # Export to YOLO format
    run_name = f"{args.model}_{args.strategy}"
    if args.species:
        run_name += f"_{args.species}"
    export_dir = OUTPUT_DIR / "yolo_dataset" / run_name
    yaml_path = export_yolo_dataset(split, export_dir, img_size=args.img_size)

    if args.export_only:
        print("Export complete. Exiting.")
        return

    # Train
    project_dir = OUTPUT_DIR / "runs" / "yolo"
    model = YOLO(args.model)

    if args.resume:
        model = YOLO(args.resume)

    results = model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch_size,
        patience=args.patience,
        project=str(project_dir),
        name=run_name,
        save=True,
        save_period=-1,  # only save best
        amp=True,
        seed=args.seed,
        workers=4,
        exist_ok=True,
        plots=True,
        # Augmentation (most handled by Ultralytics, but disable hue)
        hsv_h=0.0,  # no hue augmentation for fluorescence
        hsv_s=0.0,  # no saturation augmentation
        hsv_v=0.2,  # mild value/brightness
        flipud=0.5,
        fliplr=0.5,
        mosaic=0.5,
        mixup=0.1,
    )

    # Evaluate on test set
    print("\n" + "=" * 60)
    print("EVALUATING ON TEST SET")
    best_model_path = project_dir / run_name / "weights" / "best.pt"
    if best_model_path.exists():
        best_model = YOLO(str(best_model_path))
        test_results = best_model.val(
            data=str(yaml_path),
            split="test",
            imgsz=args.img_size,
            batch=args.batch_size,
            project=str(project_dir),
            name=f"{run_name}_test",
            exist_ok=True,
        )
        print(f"\nTest mAP@0.5: {test_results.seg.map50:.4f}")
        print(f"Test mAP@0.5:0.95: {test_results.seg.map:.4f}")
        print(f"Per-class AP@0.5: {test_results.seg.ap50}")
    else:
        print(f"Best model not found at {best_model_path}")


if __name__ == "__main__":
    main()
