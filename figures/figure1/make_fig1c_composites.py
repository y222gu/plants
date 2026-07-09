# Figure 1c

import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
sys.path.insert(0, str(PROJECT))

from src.config import SampleRecord
from src.preprocessing import load_sample_normalized


SAMPLE_DIR = HERE / "fig1c_samples"


def make_composite(dapi, fitc, tritc):
    h, w = dapi.shape
    comp = np.zeros((h, w, 3), dtype=np.float32)
    comp[..., 1] += dapi; comp[..., 2] += dapi
    comp[..., 0] += fitc; comp[..., 1] += fitc
    comp[..., 0] += tritc
    return np.clip(comp, 0, 1)


def make_local_sample(uid: str) -> SampleRecord:
    image_dir = SAMPLE_DIR / uid
    dapi_tifs = list(image_dir.glob("*_DAPI.tif"))
    if not dapi_tifs:
        raise FileNotFoundError(f"No *_DAPI.tif in {image_dir}")
    sample_name = dapi_tifs[0].stem[: -len("_DAPI")]
    return SampleRecord(
        species="", microscope="", experiment="",
        sample_name=sample_name,
        image_dir=image_dir,
        annotation_path=Path("/dev/null"),
    )


def main():
    uids = sorted(p.name for p in SAMPLE_DIR.iterdir() if p.is_dir())
    print(f"Found {len(uids)} samples")
    for uid in uids:
        img = load_sample_normalized(make_local_sample(uid))
        tritc, fitc, dapi = img[..., 0], img[..., 1], img[..., 2]
        comp = make_composite(dapi, fitc, tritc)
        rgb = (np.clip(comp, 0, 1) * 255).astype(np.uint8)
        out_png = SAMPLE_DIR / f"{uid}.png"
        Image.fromarray(rgb).save(out_png)
        print(f"  wrote {out_png.name} ({rgb.shape[0]}x{rgb.shape[1]})")


if __name__ == "__main__":
    main()
