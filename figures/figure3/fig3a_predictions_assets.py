# Figure 3a

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

from src.config import SampleRecord
from src.preprocessing import load_sample_normalized
from src.augmentation import get_val_transform
from src.annotation_utils import (
    parse_yolo_annotations, polygons_to_raw_semantic_mask,
)
from src.model_classes import unet_semantic_to_bio7, BIO_7_NAMES
from eval_bio7 import iou_dice
from train.train_timm_semantic import TimmSemanticModule


DEFAULT_OUT = HERE / "fig3a_assets"
DATA_DIR = REPO / "data"
TRAIN_IMG_SIZE = 1024

MODELS = [
    {
        "tag": "A",
        "ckpt": REPO / "output/runs/timm/dpt_meta_facebook_dinov3-vits16-pretrain-lvd1689m_equalw_drop_shuf_dfcel_semantic7c_A/2026-04-22_001/checkpoints/best-epoch=117-val_loss=0.2941.ckpt",
    },
    {
        "tag": "Bmono",
        "ckpt": REPO / "output/runs/timm/dpt_meta_facebook_dinov3-vits16-pretrain-lvd1689m_equalw_drop_shuf_dfcel_semantic7c_B-mono/2026-04-22_001/checkpoints/best-epoch=150-val_loss=0.3734.ckpt",
    },
    {
        "tag": "Bdico",
        "ckpt": REPO / "output/runs/timm/dpt_meta_facebook_dinov3-vits16-pretrain-lvd1689m_equalw_drop_shuf_dfcel_semantic7c_B-dico/2026-04-22_001/checkpoints/best-epoch=163-val_loss=0.2318.ckpt",
    },
]

BIO7_PALETTE_HEX = {
    "Whole Root": "#2a9d8f",
    "Epidermis":  "#0a9396",
    "Exodermis":  "#f4a261",
    "Cortex":     "#94d2bd",
    "Endodermis": "#f6e48e",
    "Vascular":   "#e76f61",
    "Aerenchyma": "#264653",
}
PAINT_ORDER = ["Whole Root", "Epidermis", "Exodermis", "Cortex",
               "Endodermis", "Vascular", "Aerenchyma"]


def hex_to_rgb01(h):
    h = h.lstrip("#")
    return (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255)


def pick_device(arg):
    if arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_png(arr01, path):
    arr = (np.clip(arr01, 0, 1) * 255).round().astype(np.uint8)
    Image.fromarray(arr).save(path)


def composite_paper(img_norm):
    tritc, fitc, dapi = img_norm[..., 0], img_norm[..., 1], img_norm[..., 2]
    comp = np.zeros_like(img_norm)
    comp[..., 1] += dapi
    comp[..., 2] += dapi
    comp[..., 0] += fitc
    comp[..., 1] += fitc
    comp[..., 0] += tritc
    return np.clip(comp, 0, 1)


def parse_uid(stem):
    parts = stem.split("_")
    species, scope, exp = parts[:3]
    sample = "_".join(parts[3:])
    return species, scope, exp, sample


def find_sample(species, scope, exp, sample):
    for split in ("train", "val", "test", "oneshot"):
        d = DATA_DIR / split / "image" / species / scope / exp / sample
        if d.is_dir():
            ann = DATA_DIR / split / "annotation" / f"{species}_{scope}_{exp}_{sample}.txt"
            return SampleRecord(
                species=species, microscope=scope, experiment=exp,
                sample_name=sample, image_dir=d, annotation_path=ann,
            )
    raise FileNotFoundError(f"{species}/{scope}/{exp}/{sample} not in data/")


def render_bio7(sem_mask_7):
    h, w = sem_mask_7.shape
    bio = unet_semantic_to_bio7(sem_mask_7, h, w)
    out = np.zeros((h, w, 3), dtype=np.float32)
    for cls_name in PAINT_ORDER:
        rgb = hex_to_rgb01(BIO7_PALETTE_HEX[cls_name])
        m = bio[cls_name].astype(bool)
        out[m] = rgb
    return out


def square_stretch_pipeline(img, val_transform):
    img_r = cv2.resize(img, (TRAIN_IMG_SIZE, TRAIN_IMG_SIZE),
                       interpolation=cv2.INTER_LINEAR)
    if val_transform is not None:
        img_r = val_transform(image=img_r,
                              mask=np.zeros(img_r.shape[:2], dtype=np.int32))["image"]
    return img_r


def predict_argmax(module, img_tensor):
    with torch.inference_mode():
        logits = module(img_tensor).float()
    return logits.argmax(dim=1)[0].cpu().numpy().astype(np.int32)


def load_gt_7class(rec):
    if not rec.annotation_path.exists():
        return np.zeros((TRAIN_IMG_SIZE, TRAIN_IMG_SIZE), dtype=np.int32)
    img = load_sample_normalized(rec)
    ho, wo = img.shape[:2]
    anns = parse_yolo_annotations(rec.annotation_path, wo, ho)
    sem_full = polygons_to_raw_semantic_mask(anns, ho, wo)
    return cv2.resize(sem_full.astype(np.uint8),
                      (TRAIN_IMG_SIZE, TRAIN_IMG_SIZE),
                      interpolation=cv2.INTER_NEAREST).astype(np.int32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--dtype", default="fp16", choices=["fp32", "fp16", "bf16"])
    args = p.parse_args()

    device = pick_device(args.device)
    dtype = {"fp32": torch.float32, "fp16": torch.float16,
             "bf16": torch.bfloat16}[args.dtype]
    args.out.mkdir(parents=True, exist_ok=True)

    uids = sorted(p.name for p in args.out.iterdir() if p.is_dir())
    if not uids:
        print(f"no UIDs in {args.out}")
        return
    print(f"Samples ({len(uids)}): {uids}")

    val_transform = get_val_transform(TRAIN_IMG_SIZE)

    sample_data = {}
    gt_per_sample = {}
    for uid in uids:
        sp, sc, ex, sm = parse_uid(uid)
        rec = find_sample(sp, sc, ex, sm)
        img_norm = load_sample_normalized(rec)
        img_sq = square_stretch_pipeline(img_norm, val_transform)
        comp = composite_paper(img_sq)
        out_dir = args.out / uid
        out_dir.mkdir(parents=True, exist_ok=True)
        save_png(comp, out_dir / "original.png")
        gt_7 = load_gt_7class(rec)
        gt_per_sample[uid] = gt_7
        save_png(render_bio7(gt_7), out_dir / "gt.png")
        x = torch.from_numpy(img_sq).permute(2, 0, 1).unsqueeze(0).to(device).to(dtype)
        sample_data[uid] = (rec, x, out_dir)
        print(f"  prepared {uid}")

    for cfg in MODELS:
        tag = cfg["tag"]
        print(f"\nLoading model {tag}: {cfg['ckpt'].name}")
        module = TimmSemanticModule.load_from_checkpoint(
            str(cfg["ckpt"]), map_location="cpu", strict=False,
        )
        module.eval().to(device)
        if dtype != torch.float32:
            module.to(dtype)
        for uid, (_, x, out_dir) in sample_data.items():
            pred_7 = predict_argmax(module, x)
            save_png(render_bio7(pred_7), out_dir / f"pred_{tag}.png")
            h, w = pred_7.shape
            pred_bio = unet_semantic_to_bio7(pred_7, h, w)
            gt_bio = unet_semantic_to_bio7(gt_per_sample[uid], h, w)
            ious = [float(i) for i in (iou_dice(gt_bio[c], pred_bio[c])[0] for c in BIO_7_NAMES) if not np.isnan(i)]
            miou = float(np.mean(ious)) if ious else float("nan")
            print(f"    {uid} → pred_{tag}.png  mIoU={miou:.3f}")
        del module
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"All assets saved → {args.out}")


if __name__ == "__main__":
    main()
