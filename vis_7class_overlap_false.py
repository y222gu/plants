"""Generate 7-class derived visualizations for overlap_mask=False model."""
import sys, cv2, numpy as np
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.splits import get_split
from src.annotation_utils import load_sample_annotations
from src.preprocessing import load_sample_normalized, to_uint8

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
    epi = sub(filled[0], filled[4])
    exo = sub(filled[4], filled[5])
    cortex = sub(filled[5], filled[2])  # cortex includes aerenchyma
    aer = filled[1]
    endo = sub(filled[2], filled[3])
    vasc = filled[3]
    return {"Whole Root": filled[0], "Epidermis": epi, "Exodermis": exo,
            "Cortex": cortex, "Aerenchyma": aer, "Endodermis": endo, "Vascular": vasc}

def main():
    ckpt = "lambda_output/runs/yolo/yolo26m-seg/2026-04-02_002/weights/best.pt"
    out_dir = Path("output/vis_overlap_false_7_target_classes")
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(ckpt)
    split = get_split()
    test_samples = split["test"]
    font = cv2.FONT_HERSHEY_SIMPLEX

    for sample in tqdm(test_samples, desc="Generating vis"):
        img = load_sample_normalized(sample)
        h, w = img.shape[:2]
        img_uint8 = to_uint8(img)
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)

        # GT
        gt_data = load_sample_annotations(sample, h, w, raw_classes=True)
        gt_filled = get_filled_classes(gt_data["masks"], gt_data["labels"], h, w)
        gt_bio = derive_bio(gt_filled)

        # Pred
        results = model(img_bgr, imgsz=1024, verbose=False, retina_masks=True)[0]
        if results.masks is not None:
            pred_masks = results.masks.data.cpu().numpy().astype(np.uint8)
            pred_labels = results.boxes.cls.cpu().numpy().astype(np.int32)
        else:
            pred_masks = np.zeros((0, h, w), dtype=np.uint8)
            pred_labels = np.zeros(0, dtype=np.int32)
        pred_filled = get_filled_classes(pred_masks, pred_labels, h, w)
        pred_bio = derive_bio(pred_filled)

        # Build grid: 2 rows x 7 cols (6 bio classes + mean IoU panel)
        cell_size = max(h, w)
        # Scale to reasonable size
        scale = min(1.0, 300 / cell_size)
        ch = int(h * scale)
        cw = int(w * scale)

        ious = []
        gt_row = []
        pred_row = []

        for name in BIO_NAMES:
            color = BIO_COLORS[name]
            gt_m = gt_bio[name]
            pred_m = pred_bio[name]

            # IoU
            gb, pb = gt_m.astype(bool), pred_m.astype(bool)
            union = np.logical_or(gb, pb).sum()
            iou = np.logical_and(gb, pb).sum() / max(union, 1) if union > 0 else float('nan')
            ious.append(iou)

            def make_vis(base, mask):
                vis = base.copy()
                ov = vis.copy()
                ov[mask > 0] = color
                vis = cv2.addWeighted(vis, 0.5, ov, 0.5, 0)
                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, cnts, -1, color, 2)
                return cv2.resize(vis, (cw, ch))

            gt_vis = make_vis(img_uint8, gt_m)
            pred_vis = make_vis(img_uint8, pred_m)

            # Label
            cv2.putText(gt_vis, name, (5, 15), font, 0.4, color, 1)
            iou_color = (100, 255, 100) if (not np.isnan(iou) and iou > 0.5) else (100, 100, 255)
            iou_str = f"IoU={iou:.3f}" if not np.isnan(iou) else "IoU=N/A"
            cv2.putText(pred_vis, iou_str, (5, 15), font, 0.4, iou_color, 1)

            gt_row.append(gt_vis)
            pred_row.append(pred_vis)

        # Mean IoU panel
        valid_ious = [x for x in ious if not np.isnan(x)]
        mean_iou = np.mean(valid_ious) if valid_ious else 0.0
        mean_panel_gt = cv2.resize(img_uint8, (cw, ch))
        mean_panel_pred = np.zeros((ch, cw, 3), dtype=np.uint8)
        cv2.putText(mean_panel_gt, "mIoU=%.3f" % mean_iou, (5, ch // 2), font, 0.5, (255, 255, 255), 1)

        gt_row.append(mean_panel_gt)
        pred_row.append(mean_panel_pred)

        # Assemble
        div_h = np.full((2, cw * len(gt_row), 3), 40, dtype=np.uint8)
        top = np.concatenate(gt_row, axis=1)
        bot = np.concatenate(pred_row, axis=1)

        # Row labels
        lbl_w = 35
        gt_lbl = np.zeros((ch, lbl_w, 3), dtype=np.uint8)
        pred_lbl = np.zeros((ch, lbl_w, 3), dtype=np.uint8)
        cv2.putText(gt_lbl, "GT", (2, ch // 2), font, 0.4, (255, 255, 255), 1)
        cv2.putText(pred_lbl, "Pred", (2, ch // 2), font, 0.35, (255, 255, 255), 1)

        top = np.concatenate([gt_lbl, top], axis=1)
        bot = np.concatenate([pred_lbl, bot], axis=1)
        div_h = np.full((2, top.shape[1], 3), 40, dtype=np.uint8)

        # Title
        title = np.zeros((25, top.shape[1], 3), dtype=np.uint8)
        cv2.putText(title, sample.uid, (5, 18), font, 0.5, (255, 255, 255), 1)

        grid = np.concatenate([title, top, div_h, bot], axis=0)
        cv2.imwrite(str(out_dir / f"{sample.uid}.png"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

    print(f"Done: {len(test_samples)} images saved to {out_dir}")

if __name__ == "__main__":
    main()
