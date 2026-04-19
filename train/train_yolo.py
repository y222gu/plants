"""YOLO instance segmentation training.

Usage:
    python train/train_yolo.py --model yolo26m-seg --epochs 200 --batch-size 32
    python train/train_yolo.py --model yolo26m-seg --strategy B-mono --force-export

Monitor training:
    tensorboard --logdir output/runs/yolo/ --bind_all
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse, csv, cv2, numpy as np, random
from ultralytics import YOLO
from torch.utils.tensorboard import SummaryWriter

from src.config import ANNOTATED_CLASSES, DEFAULT_EPOCHS, DEFAULT_IMG_SIZE, OUTPUT_DIR, make_run_subfolder, save_hparams
from src.yolo_dataset import export_yolo_dataset
from src.preprocessing import load_sample_normalized, to_uint8
from src.annotation_utils import load_sample_annotations
from src.splits import get_split, print_split_summary


from src.model_classes import fill_contours


def compute_val_iou_dice(model, val_samples, img_size):
    """Compute per-class pixel IoU and Dice on a set of samples."""
    per_class = {c: {"inter": 0, "union": 0, "pred": 0, "gt": 0} for c in ANNOTATED_CLASSES}
    for sample in val_samples:
        img = load_sample_normalized(sample)
        h, w = img.shape[:2]
        img_bgr = cv2.cvtColor(to_uint8(img), cv2.COLOR_RGB2BGR)
        results = model(img_bgr, imgsz=img_size, verbose=False)[0]
        gt = load_sample_annotations(sample, h, w, raw_classes=True)
        if results.masks is not None:
            masks = results.masks.data.cpu().numpy().astype(np.uint8)
            labels = results.boxes.cls.cpu().numpy().astype(np.int32)
            resized = np.zeros((len(masks), h, w), dtype=np.uint8)
            for i in range(len(masks)):
                smooth = cv2.resize(masks[i].astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
                resized[i] = fill_contours((smooth > 0.5).astype(np.uint8))
        else:
            resized = np.zeros((0, h, w), dtype=np.uint8)
            labels = np.zeros(0, dtype=np.int32)
        for cls_id in ANNOTATED_CLASSES:
            gt_idx = np.where(gt["labels"] == cls_id)[0]
            pred_idx = np.where(labels == cls_id)[0]
            g = np.clip(gt["masks"][gt_idx].sum(axis=0), 0, 1).astype(bool) if len(gt_idx) > 0 else np.zeros((h, w), dtype=bool)
            p = np.clip(resized[pred_idx].sum(axis=0), 0, 1).astype(bool) if len(pred_idx) > 0 else np.zeros((h, w), dtype=bool)
            per_class[cls_id]["inter"] += int(np.logical_and(g, p).sum())
            per_class[cls_id]["union"] += int(np.logical_or(g, p).sum())
            per_class[cls_id]["pred"] += int(p.sum())
            per_class[cls_id]["gt"] += int(g.sum())
    iou, dice = {}, {}
    for cls_id, name in ANNOTATED_CLASSES.items():
        s = per_class[cls_id]
        if s["union"] > 0:
            iou[name] = s["inter"] / s["union"]
        else:
            iou[name] = float("nan")  # both empty → exclude from mean
        denom = s["pred"] + s["gt"]
        if denom > 0:
            dice[name] = 2 * s["inter"] / denom
        else:
            dice[name] = float("nan")
    return iou, dice


def main():
    parser = argparse.ArgumentParser(description="YOLO instance segmentation training")
    parser.add_argument("--model", default="yolo26m-seg",
                        help="YOLO model variant (e.g. yolo11m-seg, yolo26m-seg)")
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--num-classes", type=int, default=6, choices=[4, 6],
                        help="Number of raw annotation classes (6=all, 4=cereals only)")
    parser.add_argument("--save-every", type=int, default=10,
                        help="Save periodic checkpoint every N epochs")
    parser.add_argument("--val-iou-every", type=int, default=10,
                        help="Compute val IoU/Dice every N epochs (0=disable)")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--export-only", action="store_true",
                        help="Only export dataset, don't train")
    parser.add_argument("--force-export", action="store_true",
                        help="Force re-export even if dataset exists")
    parser.add_argument("--strategy", default="A", help="Split strategy (A, B-mono, B-dico)")
    parser.add_argument("--channel-dropout", type=float, default=0.0,
                        help="Probability of zeroing a random channel per image (0=off)")
    args = parser.parse_args()

    # ── Data split ──────────────────────────────────────────────────────────
    split = get_split(strategy=args.strategy)
    print_split_summary(split)

    # ── Export to YOLO format ───────────────────────────────────────────────
    # Include strategy in path so different splits don't collide
    run_name = args.model
    dataset_name = run_name if args.strategy == "A" else f"{run_name}_{args.strategy}"
    export_dir = OUTPUT_DIR / "yolo_dataset" / dataset_name
    yaml_path = export_dir / "data.yaml"

    if not args.force_export and yaml_path.exists():
        counts_match = True
        for subset, samples in split.items():
            img_dir = export_dir / "images" / subset
            if not img_dir.exists():
                counts_match = False
                break
            if len(list(img_dir.glob("*.png"))) != len(samples):
                counts_match = False
                break
        if counts_match:
            print(f"Dataset already exported at {export_dir}. Use --force-export to re-export.")
        else:
            yaml_path = export_yolo_dataset(split, export_dir, img_size=args.img_size,
                                            num_classes=args.num_classes)
    else:
        yaml_path = export_yolo_dataset(split, export_dir, img_size=args.img_size,
                                        num_classes=args.num_classes)

    if args.export_only:
        print("Export complete. Exiting.")
        return

    # ── Training setup ──────────────────────────────────────────────────────
    project_dir = OUTPUT_DIR / "runs" / "yolo" / run_name
    run_dir = make_run_subfolder(project_dir)
    dated_name = run_dir.name
    save_hparams(run_dir, args)

    model = YOLO(args.resume if args.resume else args.model)
    device = list(range(args.gpus)) if args.gpus > 1 else 0

    # ── Channel dropout via monkey-patching (YOLO has no built-in support) ──
    if args.channel_dropout > 0:
        from ultralytics.models.yolo.segment.train import SegmentationTrainer
        _orig_preprocess = SegmentationTrainer.preprocess_batch
        _dropout_p = args.channel_dropout
        def _preprocess_with_dropout(self, batch):
            batch = _orig_preprocess(self, batch)
            if "img" in batch:
                imgs = batch["img"]
                B, C = imgs.shape[0], imgs.shape[1]
                for i in range(B):
                    if random.random() < _dropout_p:
                        ch = random.randint(0, C - 1)
                        imgs[i, ch] = 0
            return batch
        SegmentationTrainer.preprocess_batch = _preprocess_with_dropout
        print(f"Channel dropout enabled: p={args.channel_dropout}")

    # ── TensorBoard + custom IoU/Dice callback ──────────────────────────────
    tb_writer = SummaryWriter(str(run_dir / "tensorboard"))
    print(f"TensorBoard logs: {run_dir / 'tensorboard'}")
    print(f"  Monitor with: tensorboard --logdir {project_dir} --bind_all")

    val_samples = split.get("val", [])
    iou_csv_path = run_dir / "val_iou_dice.csv"
    iou_csv_header_written = False

    def on_fit_epoch_end(trainer):
        nonlocal iou_csv_header_written
        epoch = trainer.epoch

        # Log Ultralytics built-in metrics to TensorBoard
        for key, val in trainer.metrics.items():
            tb_writer.add_scalar(f"ultralytics/{key}", val, epoch)
        for key, val in trainer.label_loss_items(trainer.tloss, prefix="train").items():
            tb_writer.add_scalar(f"train/{key}", val, epoch)

        # Compute custom val IoU/Dice periodically
        if args.val_iou_every > 0 and val_samples and (epoch + 1) % args.val_iou_every == 0:
            print(f"\n  Computing val IoU/Dice (epoch {epoch + 1})...")
            eval_model = YOLO(trainer.best if Path(trainer.best).exists() else trainer.last)
            iou, dice = compute_val_iou_dice(eval_model, val_samples, args.img_size)
            mean_iou = np.nanmean(list(iou.values()))
            mean_dice = np.nanmean(list(dice.values()))
            print(f"  Val Mean IoU: {mean_iou:.4f}  Mean Dice: {mean_dice:.4f}")
            for cls_name in iou:
                print(f"    {cls_name:25s}  IoU={iou[cls_name]:.4f}  Dice={dice[cls_name]:.4f}")
                tb_writer.add_scalar(f"val_IoU/{cls_name}", iou[cls_name], epoch)
                tb_writer.add_scalar(f"val_Dice/{cls_name}", dice[cls_name], epoch)
            tb_writer.add_scalar("val_IoU/mean", mean_iou, epoch)
            tb_writer.add_scalar("val_Dice/mean", mean_dice, epoch)
            tb_writer.flush()

            # Save to CSV
            row = {"epoch": epoch + 1}
            for cls_name in iou:
                row[f"{cls_name}_IoU"] = round(iou[cls_name], 4)
                row[f"{cls_name}_Dice"] = round(dice[cls_name], 4)
            row["mean_IoU"] = round(mean_iou, 4)
            row["mean_Dice"] = round(mean_dice, 4)
            with open(iou_csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                if not iou_csv_header_written:
                    writer.writeheader()
                    iou_csv_header_written = True
                writer.writerow(row)

    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

    # ── Train ───────────────────────────────────────────────────────────────
    model.train(
        resume=bool(args.resume),
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
        cos_lr=True,
        # ── Mask settings ──
        overlap_mask=False,       # each instance gets own GT mask (required for overlapping annotations)
        mask_ratio=1,             # full-resolution masks
        # ── Augmentation (fluorescence-appropriate) ──
        hsv_h=0.0,               # no hue augmentation (fluorescence channels have fixed meaning)
        hsv_s=0.0,               # no saturation augmentation
        hsv_v=0.2,               # mild brightness
        degrees=45.0,            # rotation ±45°
        translate=0.1,           # translation ±10%
        scale=0.3,               # scale 0.7-1.3
        shear=10.0,              # shear ±10°
        flipud=0.5,
        fliplr=0.5,
        bgr=0.2,                 # BGR channel swap (encourages channel-invariant features)
        mosaic=0.0,              # disabled for fair comparison
        mixup=0.0,               # disabled for fair comparison
    )

    tb_writer.close()

    # ── Post-training: plot loss curves ─────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    results_csv = run_dir / "results.csv"
    if results_csv.exists():
        df = pd.read_csv(results_csv)
        df.columns = df.columns.str.strip()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

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

    # Plot IoU/Dice curves
    if iou_csv_path.exists():
        df_iou = pd.read_csv(iou_csv_path)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        iou_cols = [c for c in df_iou.columns if c.endswith("_IoU") and c != "mean_IoU"]
        dice_cols = [c for c in df_iou.columns if c.endswith("_Dice") and c != "mean_Dice"]

        for col in iou_cols:
            axes[0].plot(df_iou["epoch"], df_iou[col], label=col.replace("_IoU", ""))
        axes[0].plot(df_iou["epoch"], df_iou["mean_IoU"], label="Mean", linewidth=2, color="black", linestyle="--")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("IoU")
        axes[0].set_title("Val Per-class IoU")
        axes[0].legend(fontsize=7)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 1)

        for col in dice_cols:
            axes[1].plot(df_iou["epoch"], df_iou[col], label=col.replace("_Dice", ""))
        axes[1].plot(df_iou["epoch"], df_iou["mean_Dice"], label="Mean", linewidth=2, color="black", linestyle="--")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Dice")
        axes[1].set_title("Val Per-class Dice")
        axes[1].legend(fontsize=7)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 1)

        fig.tight_layout()
        plot_path = run_dir / "val_iou_dice_curve.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Val IoU/Dice curve saved to {plot_path}")

    # ── Post-training: test evaluation ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("EVALUATING ON TEST SET")
    best_path = run_dir / "weights" / "best.pt"
    test_samples = split.get("test", [])
    if best_path.exists() and test_samples:
        best_model = YOLO(str(best_path))
        iou, dice = compute_val_iou_dice(best_model, test_samples, args.img_size)
        mean_iou = np.mean(list(iou.values()))
        mean_dice = np.mean(list(dice.values()))
        print(f"Test Mean IoU: {mean_iou:.4f}  Mean Dice: {mean_dice:.4f}")
        for cls_name in iou:
            print(f"  {cls_name:25s}  IoU={iou[cls_name]:.4f}  Dice={dice[cls_name]:.4f}")
    else:
        print(f"Best model not found at {best_path}")


if __name__ == "__main__":
    main()
