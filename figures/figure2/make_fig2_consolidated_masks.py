# Figure 2a

import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
sys.path.insert(0, str(PROJECT))

from src.config import SampleRecord
from src.annotation_utils import parse_yolo_annotations, polygons_to_raw_binary_masks
from src.model_classes import fill_contours, yolo_overlap_false_to_bio7
from src.preprocessing import load_sample_normalized

SAMPLE_DIR = HERE / "fig2a_samples"

COLOR_MAP = {
    "Epidermis":  "#0a9396",
    "Exodermis":  "#f4a261",
    "Cortex":     "#94d2bd",
    "Endodermis": "#f6e48e",
    "Vascular":   "#e76f61",
    "Aerenchyma": "#264653",
}
PAINT_ORDER = ["Epidermis", "Exodermis", "Cortex", "Endodermis", "Vascular", "Aerenchyma"]


def hex_to_rgb(h):
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def make_local_sample(uid):
    image_dir = SAMPLE_DIR / uid
    dapi_tifs = list(image_dir.glob("*_DAPI.tif"))
    if not dapi_tifs:
        raise FileNotFoundError(f"No *_DAPI.tif in {image_dir}")
    sample_name = dapi_tifs[0].stem[: -len("_DAPI")]
    return SampleRecord(
        species="", microscope="", experiment="",
        sample_name=sample_name,
        image_dir=image_dir,
        annotation_path=image_dir / "gt.txt",
    )


def render_composite(sample):
    img = load_sample_normalized(sample)
    tritc, fitc, dapi = img[..., 0], img[..., 1], img[..., 2]
    h, w = dapi.shape
    comp = np.zeros((h, w, 3), dtype=np.float32)
    comp[..., 1] += dapi; comp[..., 2] += dapi
    comp[..., 0] += fitc; comp[..., 1] += fitc
    comp[..., 0] += tritc
    return (np.clip(comp, 0, 1) * 255).astype(np.uint8)


def render_bio7_image(bio7):
    h, w = next(iter(bio7.values())).shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for cls in PAINT_ORDER:
        if cls not in bio7:
            continue
        mask = bio7[cls].astype(bool)
        img[mask] = hex_to_rgb(COLOR_MAP[cls])
    return img


def load_polygons_to_bio7(txt_path, h, w):
    anns = parse_yolo_annotations(txt_path, w, h)
    raw = polygons_to_raw_binary_masks(anns, h, w)
    raw = {k: fill_contours(v) for k, v in raw.items()}
    return yolo_overlap_false_to_bio7(raw, h, w)


def main():
    uids = sorted(p.name for p in SAMPLE_DIR.iterdir() if p.is_dir())
    if not uids:
        print(f"no UIDs in {SAMPLE_DIR}")
        return

    print(f"Rendering composite + GT + Pred masks for {len(uids)} samples ...")
    for uid in uids:
        sample = make_local_sample(uid)
        image_dir = SAMPLE_DIR / uid

        comp = render_composite(sample)
        h, w = comp.shape[:2]
        Image.fromarray(comp).save(SAMPLE_DIR / f"{uid}_image.png")

        gt_txt = image_dir / "gt.txt"
        gt_bio7 = load_polygons_to_bio7(gt_txt, h, w)
        Image.fromarray(render_bio7_image(gt_bio7)).save(SAMPLE_DIR / f"{uid}_gt.png")

        pred_txt = image_dir / "pred.txt"
        if pred_txt.exists():
            pred_bio7 = load_polygons_to_bio7(pred_txt, h, w)
            Image.fromarray(render_bio7_image(pred_bio7)).save(SAMPLE_DIR / f"{uid}_pred.png")
        else:
            print(f"  [warn] {uid}: no pred.txt")

        print(f"  {uid}: rendered ({w}×{h})")


if __name__ == "__main__":
    main()
