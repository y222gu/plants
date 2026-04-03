"""Compare overlap_mask=True vs False on 7 derived biological classes."""

import sys, cv2, numpy as np
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

from src.splits import get_split
from src.annotation_utils import load_sample_annotations
from src.preprocessing import load_sample_normalized, to_uint8
from src.config import ANNOTATED_CLASSES

BIO_CLASSES = {
    0: "Whole Root",
    1: "Epidermis",
    2: "Exodermis",
    3: "Cortex",
    4: "Aerenchyma",
    5: "Endodermis",
    6: "Vascular",
}

BIO_COLORS = {
    0: (0, 0, 255),       # Blue
    1: (255, 165, 0),     # Orange
    2: (0, 255, 255),     # Cyan
    3: (0, 200, 0),       # Green
    4: (255, 255, 0),     # Yellow
    5: (0, 255, 0),       # Lime
    6: (255, 0, 0),       # Red
}


def fill_contours(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    return filled


def raw_to_filled(masks, labels, h, w):
    """Get filled polygon per raw class from YOLO predictions."""
    filled = {}
    for cls_id in range(6):
        idx = np.where(labels == cls_id)[0]
        if len(idx) > 0:
            merged = np.clip(masks[idx].sum(axis=0), 0, 1).astype(np.uint8)
            filled[cls_id] = fill_contours(merged)
        else:
            filled[cls_id] = np.zeros((h, w), dtype=np.uint8)
    return filled


def derive_7_classes(filled):
    """Derive 7 biological classes from 6 filled raw-class masks."""
    bio = {}
    bio[0] = filled[0]                                                         # Whole Root
    bio[1] = np.clip(filled[0].astype(np.int8) - filled[4].astype(np.int8), 0, 1).astype(np.uint8)  # Epidermis = root - outer exo
    bio[2] = np.clip(filled[4].astype(np.int8) - filled[5].astype(np.int8), 0, 1).astype(np.uint8)  # Exodermis ring
    cortex = np.clip(filled[5].astype(np.int8) - filled[2].astype(np.int8), 0, 1).astype(np.uint8)  # inner exo - outer endo
    cortex = np.clip(cortex.astype(np.int8) - filled[1].astype(np.int8), 0, 1).astype(np.uint8)     # minus aerenchyma
    bio[3] = cortex                                                            # Cortex
    bio[4] = filled[1]                                                         # Aerenchyma
    bio[5] = np.clip(filled[2].astype(np.int8) - filled[3].astype(np.int8), 0, 1).astype(np.uint8)  # Endodermis ring
    bio[6] = filled[3]                                                         # Vascular
    return bio


def predict_sample(model, sample, img_size=1024):
    img = load_sample_normalized(sample)
    h, w = img.shape[:2]
    img_uint8 = to_uint8(img)
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    results = model(img_bgr, imgsz=img_size, verbose=False, retina_masks=True)[0]
    if results.masks is not None:
        masks = results.masks.data.cpu().numpy().astype(np.uint8)
        labels = results.boxes.cls.cpu().numpy().astype(np.int32)
    else:
        masks = np.zeros((0, h, w), dtype=np.uint8)
        labels = np.zeros(0, dtype=np.int32)
    return masks, labels, img_uint8, h, w


def compute_metrics(ckpt_path, samples, label, img_size=1024):
    model = YOLO(ckpt_path)
    stats = {c: {"inter": 0, "union": 0, "pred": 0, "gt": 0} for c in BIO_CLASSES}

    for sample in tqdm(samples, desc=f"{label} inference"):
        masks, labels, _, h, w = predict_sample(model, sample, img_size)
        gt_data = load_sample_annotations(sample, h, w, raw_classes=True)
        gt_filled = raw_to_filled(gt_data["masks"], gt_data["labels"], h, w)
        gt_bio = derive_7_classes(gt_filled)
        pred_filled = raw_to_filled(masks, labels, h, w)
        pred_bio = derive_7_classes(pred_filled)

        for cls_id in BIO_CLASSES:
            g = gt_bio[cls_id].astype(bool)
            p = pred_bio[cls_id].astype(bool)
            stats[cls_id]["inter"] += int(np.logical_and(g, p).sum())
            stats[cls_id]["union"] += int(np.logical_or(g, p).sum())
            stats[cls_id]["pred"] += int(p.sum())
            stats[cls_id]["gt"] += int(g.sum())

    iou, dice = {}, {}
    for cls_id, name in BIO_CLASSES.items():
        s = stats[cls_id]
        iou[name] = s["inter"] / s["union"] if s["union"] > 0 else 0.0
        denom = s["pred"] + s["gt"]
        dice[name] = 2 * s["inter"] / denom if denom > 0 else 0.0
    return iou, dice


def save_vis(ckpt_path, samples, out_dir, label, n=5, img_size=1024):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(ckpt_path)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for sample in samples[:n]:
        masks, labels, img_uint8, h, w = predict_sample(model, sample, img_size)
        gt_data = load_sample_annotations(sample, h, w, raw_classes=True)
        gt_filled = raw_to_filled(gt_data["masks"], gt_data["labels"], h, w)
        gt_bio = derive_7_classes(gt_filled)
        pred_filled = raw_to_filled(masks, labels, h, w)
        pred_bio = derive_7_classes(pred_filled)

        # 7 classes in a grid: 2 rows x 4 cols (last cell = original image)
        cell_h, cell_w = 280, 280
        pw = cell_w * 2 + 3
        rows, cols = 2, 4
        grid = np.zeros((rows * (cell_h + 3) + 40, cols * (pw + 3), 3), dtype=np.uint8)

        items = list(BIO_CLASSES.items()) + [(-1, "Original")]
        for ci, (cls_id, cls_name) in enumerate(items):
            r, c = ci // cols, ci % cols
            y0 = r * (cell_h + 3)
            x0 = c * (pw + 3)

            if cls_id == -1:
                small = cv2.resize(img_uint8, (pw, cell_h))
                cv2.putText(small, "Original", (5, 20), font, 0.5, (255, 255, 255), 1)
                grid[y0:y0 + cell_h, x0:x0 + pw] = small
                continue

            gt_m = gt_bio[cls_id]
            pred_m = pred_bio[cls_id]
            color = BIO_COLORS[cls_id]

            def overlay(base, mask):
                vis = base.copy()
                ov = vis.copy()
                ov[mask > 0] = color
                vis = cv2.addWeighted(vis, 0.5, ov, 0.5, 0)
                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, cnts, -1, color, 2)
                return cv2.resize(vis, (cell_w, cell_h))

            gt_s = overlay(img_uint8, gt_m)
            pred_s = overlay(img_uint8, pred_m)

            gt_b, pred_b = gt_m.astype(bool), pred_m.astype(bool)
            union = np.logical_or(gt_b, pred_b).sum()
            iou_val = np.logical_and(gt_b, pred_b).sum() / max(union, 1)
            iou_color = (100, 255, 100) if iou_val > 0.5 else (100, 100, 255)

            cv2.putText(gt_s, "GT", (5, 20), font, 0.5, (255, 255, 255), 1)
            cv2.putText(pred_s, f"Pred IoU:{iou_val:.2f}", (5, 20), font, 0.45, iou_color, 1)
            cv2.putText(gt_s, cls_name, (5, cell_h - 10), font, 0.45, color, 1)

            div = np.full((cell_h, 3, 3), 80, dtype=np.uint8)
            cell = np.concatenate([gt_s, div, pred_s], axis=1)
            grid[y0:y0 + cell_h, x0:x0 + pw] = cell

        title = np.zeros((35, grid.shape[1], 3), dtype=np.uint8)
        cv2.putText(title, f"{label}: {sample.uid}", (10, 25), font, 0.6, (255, 255, 255), 2)
        grid = np.concatenate([title, grid], axis=0)

        cv2.imwrite(str(out_dir / f"{label}_{sample.uid}.png"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"Saved {min(n, len(samples))} vis to {out_dir}")


if __name__ == "__main__":
    split = get_split()
    test_samples = split["test"]

    ckpt_v1 = "lambda_output/runs/yolo/yolo26m-seg/2026-04-02_001/weights/best.pt"
    ckpt_v2 = "lambda_output/runs/yolo/yolo26m-seg/2026-04-02_002/weights/best.pt"

    print("=" * 70)
    print("Computing 7-class bio metrics: overlap_mask=True (v1)")
    iou_v1, dice_v1 = compute_metrics(ckpt_v1, test_samples, "v1_overlap_true")

    print("\n" + "=" * 70)
    print("Computing 7-class bio metrics: overlap_mask=False (v2)")
    iou_v2, dice_v2 = compute_metrics(ckpt_v2, test_samples, "v2_overlap_false")

    print("\n" + "=" * 70)
    print(f"{'Class':15s}  {'v1 IoU':>8s}  {'v2 IoU':>8s}  {'v1 Dice':>8s}  {'v2 Dice':>8s}")
    print("-" * 60)
    for name in BIO_CLASSES.values():
        print(f"{name:15s}  {iou_v1[name]:8.4f}  {iou_v2[name]:8.4f}  {dice_v1[name]:8.4f}  {dice_v2[name]:8.4f}")
    print("-" * 60)
    mean_iou_v1 = np.mean(list(iou_v1.values()))
    mean_iou_v2 = np.mean(list(iou_v2.values()))
    mean_dice_v1 = np.mean(list(dice_v1.values()))
    mean_dice_v2 = np.mean(list(dice_v2.values()))
    print(f"{'Mean':15s}  {mean_iou_v1:8.4f}  {mean_iou_v2:8.4f}  {mean_dice_v1:8.4f}  {mean_dice_v2:8.4f}")

    # Vis for a few diverse samples
    diverse = []
    seen = set()
    for s in test_samples:
        if s.species not in seen:
            diverse.append(s)
            seen.add(s.species)
        if len(diverse) == 4:
            break

    save_vis(ckpt_v1, diverse, "output/compare_7class", "v1_overlap_true", n=4)
    save_vis(ckpt_v2, diverse, "output/compare_7class", "v2_overlap_false", n=4)
    print("\nDone!")
