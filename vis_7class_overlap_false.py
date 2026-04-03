"""Evaluate YOLO on 7 derived biological classes: vis + per-sample CSV."""
import sys, csv, cv2, numpy as np, argparse
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

from src.splits import get_split
from src.annotation_utils import load_sample_annotations
from src.preprocessing import load_sample_normalized, to_uint8
from src.config import ANNOTATED_CLASSES

BIO_NAMES = ["Whole Root", "Epidermis", "Exodermis", "Cortex", "Aerenchyma", "Endodermis", "Vascular"]
BIO_COLORS = {
    "Whole Root":  (0, 0, 255),
    "Epidermis":   (255, 165, 0),
    "Exodermis":   (0, 255, 255),
    "Cortex":      (0, 200, 0),
    "Aerenchyma":  (255, 255, 0),
    "Endodermis":  (0, 255, 0),
    "Vascular":    (255, 0, 0),
}
RAW_NAMES = [ANNOTATED_CLASSES[i] for i in range(6)]


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


def iou_dice(gt_mask, pred_mask):
    g, p = gt_mask.astype(bool), pred_mask.astype(bool)
    inter = int(np.logical_and(g, p).sum())
    union = int(np.logical_or(g, p).sum())
    pred_sum, gt_sum = int(p.sum()), int(g.sum())
    iou = inter / union if union > 0 else float('nan')
    denom = pred_sum + gt_sum
    dice = 2 * inter / denom if denom > 0 else float('nan')
    return iou, dice


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--strategy", default="A")
    parser.add_argument("--out-dir", default=None, help="Output directory for vis and CSV")
    parser.add_argument("--no-vis", action="store_true")
    parser.add_argument("--img-size", type=int, default=1024)
    args = parser.parse_args()

    split = get_split(strategy=args.strategy)
    samples = split["test"]

    # Default output dir: next to checkpoint
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        ckpt = Path(args.checkpoint)
        run_dir = ckpt.parent
        while run_dir != run_dir.parent:
            if run_dir.name[:4].isdigit() and "-" in run_dir.name:
                break
            run_dir = run_dir.parent
        out_dir = run_dir / "evaluation_7class"
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = out_dir / "vis"
    if not args.no_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.checkpoint)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # CSV fieldnames
    csv_fields = ["sample_id", "species", "microscope", "experiment"]
    for name in RAW_NAMES:
        csv_fields += [f"{name}_IoU", f"{name}_Dice"]
    csv_fields += ["raw_mean_IoU", "raw_mean_Dice"]
    for name in BIO_NAMES:
        csv_fields += [f"{name}_IoU", f"{name}_Dice"]
    csv_fields += ["bio_mean_IoU", "bio_mean_Dice"]

    rows = []

    for sample in tqdm(samples, desc="Evaluating"):
        img = load_sample_normalized(sample)
        h, w = img.shape[:2]
        img_uint8 = to_uint8(img)
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)

        results = model(img_bgr, imgsz=args.img_size, verbose=False, retina_masks=True)[0]
        if results.masks is not None:
            masks = results.masks.data.cpu().numpy().astype(np.uint8)
            labels = results.boxes.cls.cpu().numpy().astype(np.int32)
        else:
            masks = np.zeros((0, h, w), dtype=np.uint8)
            labels = np.zeros(0, dtype=np.int32)

        gt_data = load_sample_annotations(sample, h, w, raw_classes=True)
        pred_filled = get_filled_classes(masks, labels, h, w)
        gt_filled = get_filled_classes(gt_data["masks"], gt_data["labels"], h, w)
        pred_bio = derive_bio(pred_filled)
        gt_bio = derive_bio(gt_filled)

        row = {
            "sample_id": sample.uid,
            "species": sample.species,
            "microscope": sample.microscope,
            "experiment": sample.experiment,
        }

        # 6 raw class metrics
        raw_ious = []
        for cls_id in range(6):
            name = ANNOTATED_CLASSES[cls_id]
            iou, dice = iou_dice(gt_filled[cls_id], pred_filled[cls_id])
            row[f"{name}_IoU"] = round(iou, 4) if not np.isnan(iou) else ""
            row[f"{name}_Dice"] = round(dice, 4) if not np.isnan(dice) else ""
            if not np.isnan(iou):
                raw_ious.append(iou)
        row["raw_mean_IoU"] = round(np.mean(raw_ious), 4) if raw_ious else ""
        row["raw_mean_Dice"] = ""  # filled later

        # 7 bio class metrics
        bio_ious = []
        bio_dices = []
        for name in BIO_NAMES:
            iou, dice = iou_dice(gt_bio[name], pred_bio[name])
            row[f"{name}_IoU"] = round(iou, 4) if not np.isnan(iou) else ""
            row[f"{name}_Dice"] = round(dice, 4) if not np.isnan(dice) else ""
            if not np.isnan(iou):
                bio_ious.append(iou)
            if not np.isnan(dice):
                bio_dices.append(dice)

        # Compute raw mean dice
        raw_dices = []
        for cls_id in range(6):
            name = ANNOTATED_CLASSES[cls_id]
            v = row.get(f"{name}_Dice")
            if v != "" and v is not None:
                raw_dices.append(float(v))
        row["raw_mean_Dice"] = round(np.mean(raw_dices), 4) if raw_dices else ""
        row["bio_mean_IoU"] = round(np.mean(bio_ious), 4) if bio_ious else ""
        row["bio_mean_Dice"] = round(np.mean(bio_dices), 4) if bio_dices else ""

        rows.append(row)

        # Visualization
        if not args.no_vis:
            scale = min(1.0, 300 / max(h, w))
            ch, cw = int(h * scale), int(w * scale)

            gt_panels = []
            pred_panels = []
            for name in BIO_NAMES:
                color = BIO_COLORS[name]

                def overlay(base, mask):
                    vis = base.copy()
                    ov = vis.copy()
                    ov[mask > 0] = color
                    vis = cv2.addWeighted(vis, 0.5, ov, 0.5, 0)
                    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(vis, cnts, -1, color, 2)
                    return cv2.resize(vis, (cw, ch))

                gt_s = overlay(img_uint8, gt_bio[name])
                pred_s = overlay(img_uint8, pred_bio[name])

                iou_val = float(row[f"{name}_IoU"]) if row[f"{name}_IoU"] != "" else float('nan')
                cv2.putText(gt_s, name, (5, 15), font, 0.4, color, 1)
                iou_color = (100, 255, 100) if (not np.isnan(iou_val) and iou_val > 0.5) else (100, 100, 255)
                iou_str = f"IoU={iou_val:.3f}" if not np.isnan(iou_val) else "IoU=N/A"
                cv2.putText(pred_s, iou_str, (5, 15), font, 0.4, iou_color, 1)

                gt_panels.append(gt_s)
                pred_panels.append(pred_s)

            # mIoU panel
            mean_panel = cv2.resize(img_uint8, (cw, ch))
            bio_miou = row["bio_mean_IoU"]
            cv2.putText(mean_panel, f"mIoU={bio_miou:.3f}" if bio_miou != "" else "mIoU=N/A",
                        (5, ch // 2), font, 0.5, (255, 255, 255), 1)
            gt_panels.append(mean_panel)
            pred_panels.append(np.zeros((ch, cw, 3), dtype=np.uint8))

            # Assemble grid
            lbl_w = 35
            gt_lbl = np.zeros((ch, lbl_w, 3), dtype=np.uint8)
            pred_lbl = np.zeros((ch, lbl_w, 3), dtype=np.uint8)
            cv2.putText(gt_lbl, "GT", (2, ch // 2), font, 0.4, (255, 255, 255), 1)
            cv2.putText(pred_lbl, "Pred", (2, ch // 2), font, 0.35, (255, 255, 255), 1)

            top = np.concatenate([gt_lbl] + gt_panels, axis=1)
            bot = np.concatenate([pred_lbl] + pred_panels, axis=1)
            div = np.full((2, top.shape[1], 3), 40, dtype=np.uint8)
            title = np.zeros((25, top.shape[1], 3), dtype=np.uint8)
            cv2.putText(title, sample.uid, (5, 18), font, 0.5, (255, 255, 255), 1)

            grid = np.concatenate([title, top, div, bot], axis=0)
            cv2.imwrite(str(vis_dir / f"{sample.uid}.png"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

    # Save CSV
    rows.sort(key=lambda r: float(r["bio_mean_IoU"]) if r["bio_mean_IoU"] != "" else 0)
    csv_path = out_dir / "per_sample_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nPer-sample CSV saved to {csv_path} (sorted worst-first)")

    # Print summary
    def agg(key):
        vals = [float(r[key]) for r in rows if r[key] != "" and r[key] is not None]
        return np.mean(vals) if vals else 0
    print(f"\n6 Raw Classes — Mean IoU: {agg('raw_mean_IoU'):.4f}  Mean Dice: {agg('raw_mean_Dice'):.4f}")
    print(f"7 Bio Classes — Mean IoU: {agg('bio_mean_IoU'):.4f}  Mean Dice: {agg('bio_mean_Dice'):.4f}")
    print(f"\nSaved {len(rows)} samples to {out_dir}")


if __name__ == "__main__":
    main()
