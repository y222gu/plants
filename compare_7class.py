"""Evaluate YOLO checkpoints on 7 derived biological classes."""
import sys, cv2, numpy as np, argparse
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

from src.splits import get_split
from src.annotation_utils import load_sample_annotations
from src.preprocessing import load_sample_normalized, to_uint8
from src.config import ANNOTATED_CLASSES

BIO_CLASSES = ["Whole Root", "Epidermis", "Exodermis", "Cortex", "Aerenchyma", "Endodermis", "Vascular"]

def fill_contours(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    return filled

def get_filled_classes(masks, labels, h, w):
    filled = {}
    for cls_id in range(6):
        idx = np.where(labels == cls_id)[0]
        if len(idx) > 0:
            merged = np.clip(masks[idx].sum(axis=0), 0, 1).astype(np.uint8)
            filled[cls_id] = fill_contours(merged)
        else:
            filled[cls_id] = np.zeros((h, w), dtype=np.uint8)
    return filled

def derive_bio(filled):
    sub = lambda a, b: np.clip(a.astype(np.int8) - b.astype(np.int8), 0, 1).astype(np.uint8)
    return {
        "Whole Root": filled[0],
        "Epidermis": sub(filled[0], filled[4]),
        "Exodermis": sub(filled[4], filled[5]),
        "Cortex": sub(filled[5], filled[2]),
        "Aerenchyma": filled[1],
        "Endodermis": sub(filled[2], filled[3]),
        "Vascular": filled[3],
    }

def evaluate_checkpoint(ckpt, samples, label, img_size=1024):
    model = YOLO(ckpt)
    # 6 raw class stats
    raw_stats = {c: {"inter": 0, "union": 0, "pred": 0, "gt": 0} for c in range(6)}
    # 7 bio class stats
    bio_stats = {c: {"inter": 0, "union": 0, "pred": 0, "gt": 0} for c in BIO_CLASSES}

    for sample in tqdm(samples, desc=f"{label}"):
        img = load_sample_normalized(sample)
        h, w = img.shape[:2]
        img_bgr = cv2.cvtColor(to_uint8(img), cv2.COLOR_RGB2BGR)
        results = model(img_bgr, imgsz=img_size, verbose=False, retina_masks=True)[0]

        if results.masks is not None:
            masks = results.masks.data.cpu().numpy().astype(np.uint8)
            labels = results.boxes.cls.cpu().numpy().astype(np.int32)
        else:
            masks = np.zeros((0, h, w), dtype=np.uint8)
            labels = np.zeros(0, dtype=np.int32)

        gt_data = load_sample_annotations(sample, h, w, raw_classes=True)

        # Fill and compute 6 raw classes
        pred_filled = get_filled_classes(masks, labels, h, w)
        gt_filled = get_filled_classes(gt_data["masks"], gt_data["labels"], h, w)

        for cls_id in range(6):
            g = gt_filled[cls_id].astype(bool)
            p = pred_filled[cls_id].astype(bool)
            raw_stats[cls_id]["inter"] += int(np.logical_and(g, p).sum())
            raw_stats[cls_id]["union"] += int(np.logical_or(g, p).sum())
            raw_stats[cls_id]["pred"] += int(p.sum())
            raw_stats[cls_id]["gt"] += int(g.sum())

        # Derive and compute 7 bio classes
        pred_bio = derive_bio(pred_filled)
        gt_bio = derive_bio(gt_filled)

        for name in BIO_CLASSES:
            g = gt_bio[name].astype(bool)
            p = pred_bio[name].astype(bool)
            bio_stats[name]["inter"] += int(np.logical_and(g, p).sum())
            bio_stats[name]["union"] += int(np.logical_or(g, p).sum())
            bio_stats[name]["pred"] += int(p.sum())
            bio_stats[name]["gt"] += int(g.sum())

    # Compute IoU/Dice
    def compute(stats):
        iou, dice = {}, {}
        for k, s in stats.items():
            iou[k] = s["inter"] / s["union"] if s["union"] > 0 else 0.0
            denom = s["pred"] + s["gt"]
            dice[k] = 2 * s["inter"] / denom if denom > 0 else 0.0
        return iou, dice

    raw_iou, raw_dice = compute(raw_stats)
    bio_iou, bio_dice = compute(bio_stats)
    return raw_iou, raw_dice, bio_iou, bio_dice

def print_table(title, names, iou, dice):
    print(f"\n{title}")
    print(f"  {'Class':25s}  {'IoU':>8s}  {'Dice':>8s}")
    print(f"  {'-'*45}")
    for name in names:
        print(f"  {str(name):25s}  {iou[name]:8.4f}  {dice[name]:8.4f}")
    mean_iou = np.mean(list(iou.values()))
    mean_dice = np.mean(list(dice.values()))
    print(f"  {'-'*45}")
    print(f"  {'Mean':25s}  {mean_iou:8.4f}  {mean_dice:8.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--strategy", default="A")
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    split = get_split(strategy=args.strategy)
    samples = split["test"]
    label = args.label or f"{args.strategy} test ({len(samples)} samples)"

    print(f"\nEvaluating: {args.checkpoint}")
    print(f"Test set: {label}")

    raw_iou, raw_dice, bio_iou, bio_dice = evaluate_checkpoint(args.checkpoint, samples, label)

    raw_names = {v: k for k, v in ANNOTATED_CLASSES.items()}
    print_table(f"6 Raw Classes ({label})",
                [ANNOTATED_CLASSES[i] for i in range(6)],
                {ANNOTATED_CLASSES[k]: v for k, v in raw_iou.items()},
                {ANNOTATED_CLASSES[k]: v for k, v in raw_dice.items()})
    print_table(f"7 Derived Bio Classes ({label})", BIO_CLASSES, bio_iou, bio_dice)
