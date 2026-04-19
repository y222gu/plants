"""YOLO ablation study: channel dropout + channel shuffle (bgr)."""
import sys, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse, csv, cv2, numpy as np, torch
from ultralytics import YOLO
from torch.utils.tensorboard import SummaryWriter

from src.config import ANNOTATED_CLASSES, DEFAULT_IMG_SIZE, OUTPUT_DIR, save_hparams
from src.yolo_dataset import export_yolo_dataset
from src.preprocessing import load_sample_normalized, to_uint8
from src.annotation_utils import load_sample_annotations
from src.splits import get_split, print_split_summary


from src.model_classes import fill_contours


def compute_val_iou_dice(model, val_samples, img_size):
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
        iou[name] = s["inter"] / s["union"] if s["union"] > 0 else 0.0
        denom = s["pred"] + s["gt"]
        dice[name] = 2 * s["inter"] / denom if denom > 0 else 0.0
    return iou, dice


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo26m-seg")
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-iou-every", type=int, default=10)
    parser.add_argument("--channel-dropout", type=float, default=0.0)
    parser.add_argument("--bgr", type=float, default=0.0)
    parser.add_argument("--condition-name", default="custom")
    parser.add_argument("--force-export", action="store_true")
    args = parser.parse_args()

    split = get_split(strategy="A")
    print_split_summary(split)

    export_dir = OUTPUT_DIR / "yolo_dataset" / args.model
    yaml_path = export_dir / "data.yaml"
    if not yaml_path.exists() or args.force_export:
        yaml_path = export_yolo_dataset(split, export_dir, img_size=args.img_size, num_classes=6)
    else:
        print(f"Dataset already at {export_dir}")

    # Let YOLO create the run directory via project + name
    project_dir = str(OUTPUT_DIR / "runs" / "yolo" / f"{args.model}_ablation")
    run_name = args.condition_name

    model = YOLO(args.model)
    dropout_p = args.channel_dropout

    def on_train_batch_start_DISABLED(trainer):
        if dropout_p > 0 and "img" in trainer.batch:
            imgs = trainer.batch["img"]
            B, C = imgs.shape[0], imgs.shape[1]
            for i in range(B):
                if random.random() < dropout_p:
                    ch = random.randint(0, C - 1)
                    imgs[i, ch] = 0

    # Monkey-patch preprocess_batch to add channel dropout
    if dropout_p > 0:
        from ultralytics.models.yolo.segment.train import SegmentationTrainer
        _orig_preprocess = SegmentationTrainer.preprocess_batch
        def _preprocess_with_dropout(self, batch):
            batch = _orig_preprocess(self, batch)
            if "img" in batch:
                imgs = batch["img"]
                B, C = imgs.shape[0], imgs.shape[1]
                for i in range(B):
                    if random.random() < dropout_p:
                        ch = random.randint(0, C - 1)
                        imgs[i, ch] = 0
            return batch
        SegmentationTrainer.preprocess_batch = _preprocess_with_dropout

    # TensorBoard + val IoU
    # We'll set up TB after train creates the directory
    val_samples = split.get("val", [])
    iou_csv_written = [False]
    tb_writer = [None]

    def on_fit_epoch_end(trainer):
        run_dir = Path(trainer.save_dir)
        if tb_writer[0] is None:
            tb_writer[0] = SummaryWriter(str(run_dir / "tensorboard"))
        for key, val in trainer.metrics.items():
            tb_writer[0].add_scalar(f"ultralytics/{key}", val, trainer.epoch)
        for key, val in trainer.label_loss_items(trainer.tloss, prefix="train").items():
            tb_writer[0].add_scalar(f"train/{key}", val, trainer.epoch)
        if args.val_iou_every > 0 and val_samples and (trainer.epoch + 1) % args.val_iou_every == 0:
            print(f"\n  Computing val IoU/Dice (epoch {trainer.epoch + 1})...")
            eval_model = YOLO(trainer.best if Path(trainer.best).exists() else trainer.last)
            iou, dice = compute_val_iou_dice(eval_model, val_samples, args.img_size)
            mean_iou = np.mean(list(iou.values()))
            print(f"  Val Mean IoU: {mean_iou:.4f}")
            for cls_name in iou:
                print(f"    {cls_name:25s}  IoU={iou[cls_name]:.4f}  Dice={dice[cls_name]:.4f}")
                tb_writer[0].add_scalar(f"val_IoU/{cls_name}", iou[cls_name], trainer.epoch)
            tb_writer[0].add_scalar("val_IoU/mean", mean_iou, trainer.epoch)
            tb_writer[0].flush()
            iou_path = run_dir / "val_iou_dice.csv"
            row = {"epoch": trainer.epoch + 1, "mean_IoU": round(mean_iou, 4)}
            for cls_name in iou:
                row[f"{cls_name}_IoU"] = round(iou[cls_name], 4)
                row[f"{cls_name}_Dice"] = round(dice[cls_name], 4)
            with open(iou_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                if not iou_csv_written[0]:
                    writer.writeheader()
                    iou_csv_written[0] = True
                writer.writerow(row)

    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

    print(f"\n{'='*60}")
    print(f"ABLATION: {args.condition_name}")
    print(f"  channel_dropout={dropout_p}, bgr={args.bgr}")
    print(f"{'='*60}\n")

    model.train(
        data=str(yaml_path), epochs=args.epochs, imgsz=args.img_size,
        batch=args.batch_size, patience=args.patience, device=0,
        project=project_dir, name=run_name,
        save=True, save_period=10, amp=True, seed=args.seed,
        workers=8, exist_ok=True, plots=True, cos_lr=True,
        overlap_mask=False, mask_ratio=1,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.2, degrees=45.0, translate=0.1,
        scale=0.3, shear=10.0, flipud=0.5, fliplr=0.5,
        bgr=args.bgr, mosaic=0.0, mixup=0.0,
    )
    if tb_writer[0]:
        tb_writer[0].close()

    # Save hparams
    run_dir = Path(project_dir) / run_name
    save_hparams(run_dir, args)

    # Test eval
    best_path = run_dir / "weights" / "best.pt"
    test_samples = split.get("test", [])
    if best_path.exists() and test_samples:
        print("\nEVALUATING ON TEST SET")
        best_model = YOLO(str(best_path))
        iou, dice = compute_val_iou_dice(best_model, test_samples, args.img_size)
        mean_iou = np.mean(list(iou.values()))
        mean_dice = np.mean(list(dice.values()))
        print(f"Test Mean IoU: {mean_iou:.4f}  Mean Dice: {mean_dice:.4f}")
        for cls_name in iou:
            print(f"  {cls_name:25s}  IoU={iou[cls_name]:.4f}  Dice={dice[cls_name]:.4f}")

if __name__ == "__main__":
    main()
