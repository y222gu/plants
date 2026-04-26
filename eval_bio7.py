"""Evaluate any model on native classes + 7 bio classes.

Outputs two folders inside the run directory:
    eval_native/   — per-sample CSV, box plots, vis using model's native classes
    eval_bio7/     — per-sample CSV, box plots, vis using shared 7 bio classes

Usage:
    python eval_bio7.py --model-key yolo_overlap_false --checkpoint path/to/best.pt
    python eval_bio7.py --model-key yolo_overlap_true --checkpoint path/to/best.pt --no-vis
    python eval_bio7.py --model-key yolo_overlap_false --checkpoint path/to/best.pt --strategy B-mono
    python eval_bio7.py --model-key microsam --checkpoint path/to/run_dir --no-vis
"""
import csv, cv2, numpy as np, argparse
from pathlib import Path
from tqdm import tqdm

from src.splits import get_split
from src.preprocessing import load_sample_normalized, to_uint8
from src.model_classes import (
    MODEL_REGISTRY, BIO_7_NAMES, BIO_7_COLORS_RGB, BIO_7_PUB_COLORS,
    fill_contours, merge_classes, get_filled_classes, get_raw_classes,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def semantic_to_raw_filled(sem_mask):
    """Convert 7-class semantic mask to 6 raw annotation filled polygons.

    Reconstructs the original filled polygon representation from the
    mutually exclusive semantic regions. Ring subtraction can recover
    the semantic rings from these filled polygons.

    Semantic → Raw mapping:
        Raw 0 (Whole Root):      all non-background (sem >= 1)
        Raw 1 (Aerenchyma):      sem == 2
        Raw 2 (Outer Endo):      sem 3 (endo ring) + sem 4 (vascular) → filled
        Raw 3 (Inner Endo):      sem 4 (vascular) → filled
        Raw 4 (Outer Exo):       sem 5 + 6 + 2 + 3 + 4 → filled (everything inside outer exo)
        Raw 5 (Inner Exo):       sem 6 + 2 + 3 + 4 → filled (everything inside inner exo)
    """
    raw = {}
    raw[0] = (sem_mask >= 1).astype(np.uint8)
    raw[1] = (sem_mask == 2).astype(np.uint8)
    raw[2] = np.isin(sem_mask, [3, 4]).astype(np.uint8)
    raw[3] = (sem_mask == 4).astype(np.uint8)
    raw[4] = np.isin(sem_mask, [2, 3, 4, 5, 6]).astype(np.uint8)
    raw[5] = np.isin(sem_mask, [2, 3, 4, 6]).astype(np.uint8)
    return raw


def mask_to_yolo_polygons(mask, class_id, h, w, min_area=50):
    """Convert a binary mask to YOLO polygon lines.
    Returns list of strings: 'class_id x1 y1 x2 y2 ...' (normalized coords)."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        pts = cnt.squeeze()
        if pts.ndim != 2 or len(pts) < 3:
            continue
        coords = []
        for x, y in pts:
            coords.append(f"{x/w:.6f}")
            coords.append(f"{y/h:.6f}")
        lines.append(f"{class_id} " + " ".join(coords))
    return lines


def iou_dice(gt_mask, pred_mask):
    g, p = gt_mask.astype(bool), pred_mask.astype(bool)
    inter = int(np.logical_and(g, p).sum())
    union = int(np.logical_or(g, p).sum())
    pred_sum, gt_sum = int(p.sum()), int(g.sum())
    if union == 0:
        # Both empty → exclude from mean (nnU-Net convention)
        return float("nan"), float("nan")
    iou = inter / union
    denom = pred_sum + gt_sum
    dice = 2 * inter / denom if denom > 0 else 0.0
    return iou, dice


def run_yolo_inference(model, sample, img_size):
    """YOLO inference → (masks, labels, h, w)."""
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
    return masks, labels, h, w


def run_unet_multilabel_inference(model, sample, img_size):
    """U-Net++ multilabel inference → dict {0..5: (H,W) binary mask}.
    Each channel is thresholded at 0.5 to get a filled binary mask."""
    import torch
    img = load_sample_normalized(sample)
    h, w = img.shape[:2]
    img_resized = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float()
    if torch.cuda.is_available():
        tensor = tensor.cuda()

    with torch.no_grad():
        logits = model(tensor)

    probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()  # (6, img_size, img_size)
    filled = {}
    for c in range(probs.shape[0]):
        prob_full = cv2.resize(probs[c], (w, h), interpolation=cv2.INTER_LINEAR)
        filled[c] = (prob_full > 0.5).astype(np.uint8)
    return filled, h, w


def run_unet_semantic_inference(model, sample, img_size):
    """U-Net++ semantic inference → (H,W) argmax semantic mask with 7 labels."""
    import torch
    img = load_sample_normalized(sample)
    h, w = img.shape[:2]
    img_resized = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float()
    if torch.cuda.is_available():
        tensor = tensor.cuda()

    with torch.no_grad():
        logits = model(tensor)

    sem_mask = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int32)
    # Resize to original
    sem_mask = cv2.resize(sem_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return sem_mask, h, w


def run_microsam_inference(models, sample, img_size):
    """micro-SAM inference with 6 per-class models → dict {0..5: (H,W) binary mask}.

    Each model runs automatic instance segmentation (AIS) via the UNETR decoder.
    No prompts needed at inference.

    Args:
        models: dict {class_id: (predictor, segmenter)} from get_predictor_and_segmenter()
        sample: SampleRecord
        img_size: target image size (1024)

    Returns:
        filled: dict {0..5: (H,W) uint8 binary mask}, h, w (original dimensions)
    """
    from micro_sam.automatic_segmentation import automatic_instance_segmentation

    img = load_sample_normalized(sample)
    h, w = img.shape[:2]
    img_resized = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    img_uint8 = (np.clip(img_resized, 0, 1) * 255).astype(np.uint8)

    filled = {}
    for cls_id in range(6):
        if cls_id not in models:
            filled[cls_id] = np.zeros((h, w), dtype=np.uint8)
            continue
        predictor, segmenter = models[cls_id]

        if cls_id == 1:
            # Aerenchyma: use full AIS (watershed) for multi-instance
            pred_instances = automatic_instance_segmentation(
                predictor=predictor, segmenter=segmenter,
                input_path=img_uint8, ndim=2,
            )
            pred_binary = (pred_instances > 0).astype(np.uint8)
        else:
            # Single-instance classes (0,2-5): threshold UNETR foreground
            # AIS watershed over-segments large structures
            segmenter.initialize(img_uint8, ndim=2)
            pred_binary = (segmenter._foreground > 0.015).astype(np.uint8)

        if pred_binary.shape != (h, w):
            pred_binary = cv2.resize(pred_binary, (w, h),
                                     interpolation=cv2.INTER_NEAREST)
        filled[cls_id] = pred_binary

    return filled, h, w


def get_pred_native_yolo(masks, labels, h, w, model_key):
    """Convert raw YOLO inference output to model's native representation.
    Uses merge_classes (no fill_contours) for fair IoU comparison across models."""
    if model_key == "yolo_overlap_true":
        return get_raw_classes(masks, labels, h, w)
    else:
        return merge_classes(masks, labels, h, w)


def make_vis_grid(img_uint8, gt_dict, pred_dict, row_metrics, class_names,
                  class_colors, sample_uid, mean_key, font):
    """Create a 2-row visualization grid: GT on top, Pred on bottom.

    Args:
        gt_dict: {name: (H,W) mask} for GT
        pred_dict: {name: (H,W) mask} for predictions
        row_metrics: dict with f"{name}_IoU" keys
        class_names: list of class name strings
        class_colors: dict {name: (R,G,B)} or dict {int: (R,G,B)}
        sample_uid: string for title
        mean_key: key in row_metrics for mean IoU (e.g. "native_mean_IoU" or "bio_mean_IoU")
    """
    h, w = img_uint8.shape[:2]
    scale = min(1.0, 300 / max(h, w))
    ch, cw = int(h * scale), int(w * scale)

    gt_panels = []
    pred_panels = []

    for name in class_names:
        # Look up color by name or index
        if isinstance(name, str) and name in class_colors:
            color = class_colors[name]
        elif isinstance(name, int) and name in class_colors:
            color = class_colors[name]
        else:
            color = (200, 200, 200)

        gt_m = gt_dict.get(name, np.zeros((h, w), dtype=np.uint8))
        pred_m = pred_dict.get(name, np.zeros((h, w), dtype=np.uint8))

        def overlay(base, mask):
            vis = base.copy()
            ov = vis.copy()
            ov[mask > 0] = color
            vis = cv2.addWeighted(vis, 0.5, ov, 0.5, 0)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, color, 2)
            return cv2.resize(vis, (cw, ch))

        gt_s = overlay(img_uint8, gt_m)
        pred_s = overlay(img_uint8, pred_m)

        # Display name (truncate if too long)
        display_name = name if isinstance(name, str) else str(name)
        if len(display_name) > 20:
            display_name = display_name[:18] + ".."
        cv2.putText(gt_s, display_name, (5, 15), font, 0.35, color, 1)

        # IoU on pred panel
        iou_key = f"{name}_IoU" if isinstance(name, str) else f"native_{name}_IoU"
        iou_val = row_metrics.get(iou_key, "")
        if iou_val != "" and iou_val is not None:
            iou_f = float(iou_val)
            iou_color = (100, 255, 100) if iou_f > 0.5 else (100, 100, 255)
            cv2.putText(pred_s, f"IoU={iou_f:.3f}", (5, 15), font, 0.35, iou_color, 1)
        else:
            cv2.putText(pred_s, "IoU=N/A", (5, 15), font, 0.35, (100, 100, 255), 1)

        gt_panels.append(gt_s)
        pred_panels.append(pred_s)

    # mIoU panel
    mean_panel = cv2.resize(img_uint8, (cw, ch))
    miou = row_metrics.get(mean_key, "")
    miou_str = f"mIoU={float(miou):.3f}" if miou != "" and miou is not None else "mIoU=N/A"
    cv2.putText(mean_panel, miou_str, (5, ch // 2), font, 0.5, (255, 255, 255), 1)
    gt_panels.append(mean_panel)
    pred_panels.append(np.zeros((ch, cw, 3), dtype=np.uint8))

    # Assemble
    lbl_w = 35
    gt_lbl = np.zeros((ch, lbl_w, 3), dtype=np.uint8)
    pred_lbl = np.zeros((ch, lbl_w, 3), dtype=np.uint8)
    cv2.putText(gt_lbl, "GT", (2, ch // 2), font, 0.4, (255, 255, 255), 1)
    cv2.putText(pred_lbl, "Pred", (2, ch // 2), font, 0.35, (255, 255, 255), 1)

    top = np.concatenate([gt_lbl] + gt_panels, axis=1)
    bot = np.concatenate([pred_lbl] + pred_panels, axis=1)
    div = np.full((2, top.shape[1], 3), 40, dtype=np.uint8)
    title = np.zeros((25, top.shape[1], 3), dtype=np.uint8)
    cv2.putText(title, sample_uid, (5, 18), font, 0.5, (255, 255, 255), 1)

    return np.concatenate([title, top, div, bot], axis=0)


def save_boxplots(csv_path, out_dir, class_names, pub_colors, model_name):
    """Generate box plots from per-sample CSV."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.patches import Patch

    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 8, "axes.titlesize": 10,
        "axes.labelsize": 9, "figure.dpi": 300, "savefig.dpi": 300,
        "axes.spines.top": False, "axes.spines.right": False,
    })

    df = pd.read_csv(csv_path)
    colors = [pub_colors.get(n, "#999999") for n in class_names]

    def make_boxplot(df, metric_suffix, ylabel, title, out_path, group_by=None):
        groups = {"Overall": df}
        if group_by:
            for val in sorted(df[group_by].unique()):
                groups[val] = df[df[group_by] == val]
        n_groups, n_classes = len(groups), len(class_names)
        fig, ax = plt.subplots(figsize=(max(7, n_groups * 1.5), 4))
        w = 0.7 / n_classes
        positions, data, box_colors = [], [], []
        for gi, (gname, gdf) in enumerate(groups.items()):
            for ci, cls in enumerate(class_names):
                vals = pd.to_numeric(gdf[f"{cls}_{metric_suffix}"], errors="coerce").dropna().values
                data.append(vals if len(vals) > 0 else [0.0])
                positions.append(gi + (ci - n_classes / 2 + 0.5) * w)
                box_colors.append(colors[ci])
        bp = ax.boxplot(data, positions=positions, widths=w * 0.8, patch_artist=True,
                        whis=[5, 95], showfliers=False,
                        medianprops=dict(color="black", linewidth=1.0),
                        whiskerprops=dict(color="#444", linewidth=0.6),
                        capprops=dict(color="#444", linewidth=0.6))
        for patch, c in zip(bp["boxes"], box_colors):
            patch.set_facecolor("none"); patch.set_edgecolor(c)
            patch.set_linewidth(1.2)
        # Scatter individual data points on top
        for i, (pos, vals, c) in enumerate(zip(positions, data, box_colors)):
            jitter = np.random.default_rng(42).uniform(-w * 0.3, w * 0.3, size=len(vals))
            ax.scatter(pos + jitter, vals, c=c, s=6, alpha=0.4, edgecolors="none", zorder=3)
        ax.set_xticks(range(n_groups))
        ax.set_xticklabels([f"{g}\n(n={len(groups[g])})" for g in groups])
        ax.set_ylabel(ylabel); ax.set_ylim(0, 1.08); ax.set_title(title)
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="y", which="major", linewidth=0.3, alpha=0.5)
        handles = [Patch(facecolor="none", edgecolor=colors[i], linewidth=1.2, label=class_names[i])
                   for i in range(n_classes)]
        ax.legend(handles=handles, loc="lower left", ncol=min(4, n_classes),
                  frameon=False, fontsize=6)
        fig.tight_layout(); fig.savefig(out_path); plt.close(fig)
        print(f"  Saved {out_path}")

    for metric, ylabel in [("IoU", "IoU"), ("Dice", "Dice")]:
        make_boxplot(df, metric, ylabel, f"{model_name}: {metric} by Species",
                     str(out_dir / f"boxplot_{metric.lower()}_species.png"), "species")
        make_boxplot(df, metric, ylabel, f"{model_name}: {metric} by Microscope",
                     str(out_dir / f"boxplot_{metric.lower()}_microscope.png"), "microscope")
        make_boxplot(df, metric, ylabel, f"{model_name}: {metric} Overall",
                     str(out_dir / f"boxplot_{metric.lower()}_overall.png"))


# ── Main ─────────────────────────────────────────────────────────────────────

def load_predictions_from_txt(pred_path, h, w, model_key):
    """Load saved YOLO polygon predictions from a .txt file.

    All models save predictions in raw 6-class filled polygon format
    (class IDs 0-5). Returns the native prediction format for the model.

    For semantic models, the saved raw 6-class polygons are converted
    back to a 7-class semantic mask via paint order.
    """
    from src.annotation_utils import parse_yolo_annotations, polygons_to_raw_semantic_mask
    anns = parse_yolo_annotations(pred_path, w, h)
    if not anns:
        if model_key == "unet_semantic":
            return np.zeros((h, w), dtype=np.int32)
        else:
            return {i: np.zeros((h, w), dtype=np.uint8) for i in range(6)}

    if model_key == "unet_semantic":
        # Reconstruct 7-class semantic mask from raw 6-class polygons
        return polygons_to_raw_semantic_mask(anns, h, w)
    else:
        masks = np.array([cv2.fillPoly(np.zeros((h, w), dtype=np.uint8),
                                        [a["polygon"].astype(np.int32)], 1) for a in anns])
        labels = np.array([a["class_id"] for a in anns], dtype=np.int32)
        return get_filled_classes(masks, labels, h, w)


def main():
    parser = argparse.ArgumentParser(description="Evaluate on native + 7 bio classes")
    parser.add_argument("--model-key", required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--checkpoint", default=None,
                        help="Model checkpoint (not needed with --from-predictions)")
    parser.add_argument("--from-predictions", default=None, type=Path,
                        help="Path to predictions/ dir with .txt files (skip inference)")
    parser.add_argument("--strategy", default="A")
    parser.add_argument("--out-dir", default=None, help="Override base output dir")
    parser.add_argument("--no-vis", action="store_true")
    parser.add_argument("--save-predictions", action="store_true",
                        help="Save predicted masks as YOLO polygon .txt files")
    parser.add_argument("--img-size", type=int, default=1024)
    args = parser.parse_args()

    if not args.from_predictions and not args.checkpoint:
        parser.error("Either --checkpoint or --from-predictions is required")

    cfg = MODEL_REGISTRY[args.model_key]
    split = get_split(strategy=args.strategy)
    samples = split["test"]

    # Resolve run directory
    if args.out_dir:
        base_dir = Path(args.out_dir)
    elif args.from_predictions:
        # predictions/ dir is inside the run dir
        base_dir = args.from_predictions.parent
    else:
        ckpt = Path(args.checkpoint)
        run_dir = ckpt.parent
        while run_dir != run_dir.parent:
            if run_dir.name[:4].isdigit() and "-" in run_dir.name:
                break
            run_dir = run_dir.parent
        base_dir = run_dir

    native_dir = base_dir / "eval_native"
    bio7_dir = base_dir / "eval_bio7"
    native_dir.mkdir(parents=True, exist_ok=True)
    bio7_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_vis:
        (native_dir / "vis").mkdir(exist_ok=True)
        (bio7_dir / "vis").mkdir(exist_ok=True)
    if args.save_predictions:
        pred_dir = base_dir / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nModel: {cfg.name} ({args.model_key})")
    print(f"Native classes: {list(cfg.native_classes.values())}")
    if args.from_predictions:
        print(f"Predictions from: {args.from_predictions}")
    else:
        print(f"Checkpoint: {args.checkpoint}")
    print(f"Test set: {args.strategy} ({len(samples)} samples)")
    print(f"Output: {base_dir}")

    # Load model (skip if using saved predictions)
    model = None
    if not args.from_predictions:
        if args.model_key.startswith("yolo"):
            from ultralytics import YOLO
            model = YOLO(args.checkpoint)
        elif args.model_key == "unet_multilabel":
            import torch
            from train.train_unet_binary import MultilabelSegModule
            model = MultilabelSegModule.load_from_checkpoint(args.checkpoint)
            model.eval()
            if torch.cuda.is_available():
                model.cuda()
        elif args.model_key == "unet_semantic":
            import torch
            from train.train_unet_semantic import SemanticSegModule
            model = SemanticSegModule.load_from_checkpoint(args.checkpoint)
            model.eval()
            if torch.cuda.is_available():
                model.cuda()
        elif args.model_key == "sam_semantic":
            import torch
            from train.train_sam_semantic import SAMSemanticModule
            model = SAMSemanticModule.load_from_checkpoint(args.checkpoint)
            model.eval()
            if torch.cuda.is_available():
                model.cuda()
        elif args.model_key == "sam_unetpp":
            import torch
            from train.train_sam_unetpp import SAMUNetPPModule
            model = SAMUNetPPModule.load_from_checkpoint(args.checkpoint)
            model.eval()
            if torch.cuda.is_available():
                model.cuda()
        elif args.model_key == "timm_semantic":
            import torch
            from train.train_timm_semantic import TimmSemanticModule
            model = TimmSemanticModule.load_from_checkpoint(args.checkpoint)
            model.eval()
            if torch.cuda.is_available():
                model.cuda()
        elif args.model_key == "microsam":
            import torch
            from micro_sam.automatic_segmentation import get_predictor_and_segmenter
            from src.config import ANNOTATED_CLASSES
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # --checkpoint points to the run base dir containing per-class subdirs
            run_base = Path(args.checkpoint)
            model = {}  # dict {class_id: (predictor, segmenter)}
            for cls_id in range(6):
                cls_name = ANNOTATED_CLASSES[cls_id]
                # Find the exported model for this class
                cls_dir = run_base / f"vit_b_lm_class{cls_id}_A"
                if not cls_dir.exists():
                    print(f"  WARNING: no model dir for class {cls_id} ({cls_name}), skipping")
                    continue
                # Find the latest run subfolder
                runs = sorted(cls_dir.glob("20*"))
                if not runs:
                    print(f"  WARNING: no runs in {cls_dir}, skipping class {cls_id}")
                    continue
                exported = runs[-1] / "exported_model.pt"
                if not exported.exists():
                    print(f"  WARNING: no exported_model.pt in {runs[-1]}, skipping class {cls_id}")
                    continue
                print(f"  Loading class {cls_id} ({cls_name}): {exported}")
                predictor, segmenter = get_predictor_and_segmenter(
                    model_type="vit_b_lm", checkpoint=str(exported), device=device,
                    segmentation_mode="ais",
                )
                model[cls_id] = (predictor, segmenter)
            print(f"  Loaded {len(model)}/6 class models")
        else:
            raise ValueError(f"Inference not implemented for {args.model_key}")

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Native class info
    native_ids = sorted(cfg.native_classes.keys())
    native_names = [cfg.native_classes[i] for i in native_ids]
    native_color_by_name = {cfg.native_classes[i]: cfg.native_colors_rgb[i] for i in native_ids}

    # Pub colors for native classes (use same palette order)
    wong_colors = ["#0072B2", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#D55E00", "#CC79A7", "#000000"]
    native_pub_colors = {name: wong_colors[i % len(wong_colors)] for i, name in enumerate(native_names)}

    # CSV fieldnames
    native_csv_fields = ["sample_id", "species", "microscope", "experiment"]
    for name in native_names:
        native_csv_fields += [f"{name}_IoU", f"{name}_Dice"]
    native_csv_fields += ["mean_IoU", "mean_Dice"]

    bio7_csv_fields = ["sample_id", "species", "microscope", "experiment"]
    for name in BIO_7_NAMES:
        bio7_csv_fields += [f"{name}_IoU", f"{name}_Dice"]
    bio7_csv_fields += ["mean_IoU", "mean_Dice"]

    native_rows = []
    bio7_rows = []

    for sample in tqdm(samples, desc="Evaluating"):
        img = load_sample_normalized(sample)
        h, w = img.shape[:2]
        img_uint8 = to_uint8(img)

        # Load predictions: from saved .txt files or run inference
        if args.from_predictions:
            pred_file = args.from_predictions / f"{sample.uid}.txt"
            if not pred_file.exists():
                print(f"  WARNING: no prediction file for {sample.uid}, skipping")
                continue
            pred_native = load_predictions_from_txt(pred_file, h, w, args.model_key)
        elif args.model_key.startswith("yolo"):
            masks, labels, h, w = run_yolo_inference(model, sample, args.img_size)
            pred_native = get_pred_native_yolo(masks, labels, h, w, args.model_key)
        elif args.model_key == "unet_multilabel":
            pred_native, h, w = run_unet_multilabel_inference(model, sample, args.img_size)
        elif args.model_key in ("unet_semantic", "sam_semantic", "sam_unetpp", "timm_semantic"):
            pred_native, h, w = run_unet_semantic_inference(model, sample, args.img_size)
        elif args.model_key == "microsam":
            pred_native, h, w = run_microsam_inference(model, sample, args.img_size)
        else:
            raise ValueError(f"Inference not implemented for {args.model_key}")
        gt_native = cfg.load_gt(sample, h, w)

        pred_bio = cfg.to_bio7(pred_native, h, w)
        gt_bio = cfg.to_bio7(gt_native, h, w)

        base_row = {
            "sample_id": sample.uid,
            "species": sample.species,
            "microscope": sample.microscope,
            "experiment": sample.experiment,
        }

        # ── Native metrics ──
        native_row = dict(base_row)
        native_ious, native_dices = [], []
        gt_native_named = {}
        pred_native_named = {}
        for cls_id in native_ids:
            name = cfg.native_classes[cls_id]
            # For semantic models, native is (H,W) int32 → extract per-class binary mask
            if isinstance(gt_native, np.ndarray) and gt_native.ndim == 2:
                gt_cls = (gt_native == cls_id).astype(np.uint8)
                pred_cls = (pred_native == cls_id).astype(np.uint8)
            else:
                gt_cls = gt_native[cls_id]
                pred_cls = pred_native[cls_id]
            iou, dice = iou_dice(gt_cls, pred_cls)
            native_row[f"{name}_IoU"] = round(iou, 4) if not np.isnan(iou) else ""
            native_row[f"{name}_Dice"] = round(dice, 4) if not np.isnan(dice) else ""
            if not np.isnan(iou): native_ious.append(iou)
            if not np.isnan(dice): native_dices.append(dice)
            gt_native_named[name] = gt_cls
            pred_native_named[name] = pred_cls
        native_row["mean_IoU"] = round(np.mean(native_ious), 4) if native_ious else ""
        native_row["mean_Dice"] = round(np.mean(native_dices), 4) if native_dices else ""
        native_rows.append(native_row)

        # ── Save predictions as YOLO polygons (raw 6-class format) ──
        # For YOLO: apply fill_contours before saving (not applied during IoU)
        # For semantic: convert 7-class argmax to 6 raw filled polygons
        # For multilabel: already in 6-class format, save as-is
        if args.save_predictions:
            lines = []
            if isinstance(pred_native, np.ndarray) and pred_native.ndim == 2:
                # Semantic model: convert to raw 6-class filled polygons
                raw_filled = semantic_to_raw_filled(pred_native)
                for cls_id in range(6):
                    lines.extend(mask_to_yolo_polygons(
                        raw_filled[cls_id], cls_id, h, w))
            else:
                # YOLO/multilabel/micro-SAM: save as-is
                for cls_id in sorted(pred_native.keys()):
                    lines.extend(mask_to_yolo_polygons(
                        pred_native[cls_id], cls_id, h, w))
            with open(pred_dir / f"{sample.uid}.txt", "w") as f:
                f.write("\n".join(lines))

        # ── Bio-7 metrics ──
        bio7_row = dict(base_row)
        bio_ious, bio_dices = [], []
        for name in BIO_7_NAMES:
            iou, dice = iou_dice(gt_bio[name], pred_bio[name])
            bio7_row[f"{name}_IoU"] = round(iou, 4) if not np.isnan(iou) else ""
            bio7_row[f"{name}_Dice"] = round(dice, 4) if not np.isnan(dice) else ""
            if not np.isnan(iou): bio_ious.append(iou)
            if not np.isnan(dice): bio_dices.append(dice)
        bio7_row["mean_IoU"] = round(np.mean(bio_ious), 4) if bio_ious else ""
        bio7_row["mean_Dice"] = round(np.mean(bio_dices), 4) if bio_dices else ""
        bio7_rows.append(bio7_row)

        # ── Visualizations ──
        if not args.no_vis:
            # Native vis
            grid_native = make_vis_grid(
                img_uint8, gt_native_named, pred_native_named, native_row,
                native_names, native_color_by_name, sample.uid, "mean_IoU", font)
            cv2.imwrite(str(native_dir / "vis" / f"{sample.uid}.png"),
                        cv2.cvtColor(grid_native, cv2.COLOR_RGB2BGR))

            # Bio-7 vis
            bio7_color_by_name = {BIO_7_NAMES[i]: BIO_7_COLORS_RGB[i] for i in range(7)}
            grid_bio7 = make_vis_grid(
                img_uint8, pred_bio, pred_bio, bio7_row,
                BIO_7_NAMES, bio7_color_by_name, sample.uid, "mean_IoU", font)
            # Fix: use GT for top row
            grid_bio7 = make_vis_grid(
                img_uint8, gt_bio, pred_bio, bio7_row,
                BIO_7_NAMES, bio7_color_by_name, sample.uid, "mean_IoU", font)
            cv2.imwrite(str(bio7_dir / "vis" / f"{sample.uid}.png"),
                        cv2.cvtColor(grid_bio7, cv2.COLOR_RGB2BGR))

    # ── Save CSVs ──
    native_rows.sort(key=lambda r: float(r["mean_IoU"]) if r["mean_IoU"] != "" else 0)
    native_csv = native_dir / "per_sample_metrics.csv"
    with open(native_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=native_csv_fields)
        writer.writeheader()
        writer.writerows(native_rows)
    print(f"\nNative CSV saved to {native_csv}")

    bio7_rows.sort(key=lambda r: float(r["mean_IoU"]) if r["mean_IoU"] != "" else 0)
    bio7_csv = bio7_dir / "per_sample_metrics.csv"
    with open(bio7_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=bio7_csv_fields)
        writer.writeheader()
        writer.writerows(bio7_rows)
    print(f"Bio-7 CSV saved to {bio7_csv}")

    # ── Print summaries ──
    def agg(rows, key):
        vals = [float(r[key]) for r in rows if r[key] != "" and r[key] is not None]
        return np.mean(vals) if vals else 0
    print(f"\nNative — Mean IoU: {agg(native_rows, 'mean_IoU'):.4f}  "
          f"Mean Dice: {agg(native_rows, 'mean_Dice'):.4f}")
    print(f"Bio-7  — Mean IoU: {agg(bio7_rows, 'mean_IoU'):.4f}  "
          f"Mean Dice: {agg(bio7_rows, 'mean_Dice'):.4f}")

    # ── Box plots ──
    print("\nGenerating native class box plots...")
    save_boxplots(native_csv, native_dir, native_names, native_pub_colors,
                  f"{cfg.name} Native")

    print("Generating bio-7 box plots...")
    save_boxplots(bio7_csv, bio7_dir, BIO_7_NAMES, BIO_7_PUB_COLORS,
                  f"{cfg.name} Bio-7")

    print(f"\nAll saved to {base_dir}")


if __name__ == "__main__":
    main()
