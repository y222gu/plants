"""Replace `pred.png` for the 2 science_art samples that are in the test split
with renderings of the model's saved YOLO-polygon predictions, then rebuild
the 5x6 montage. Train samples keep their CPU-fp32 inference predictions."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "figures_for_paper" / "figure3"))

from src.annotation_utils import (                                       # noqa: E402
    parse_yolo_annotations, polygons_to_raw_semantic_mask,
)
from generate_dense_features import save_png                             # noqa: E402
from figures_for_paper.figure3.fig3a_predictions_assets import (         # noqa: E402
    render_bio7,
)
from scripts.render_science_art_pred_pca import (                        # noqa: E402
    INPUT_SIZE, OUT_ROOT, build_montage, collect_uids,
)

RUN_DIR = REPO / "output/runs/timm/dpt_meta_facebook_dinov3-vits16-pretrain-lvd1689m_equalw_drop_shuf_dfcel_semantic7c_A/2026-04-22_001"
PRED_DIR = RUN_DIR / "eval/test/predictions"

TEST_UIDS = [
    "Millet_Olympus_Exp4_1-2cm__75",
    "Tomato_Olympus_Exp5_Solanum_MajkenSlide1_14",
]


def render_saved_prediction(uid: str, size: int = INPUT_SIZE):
    """Saved YOLO polygon .txt → 7-class semantic mask → Bio-7 colored PNG."""
    txt = PRED_DIR / f"{uid}.txt"
    anns = parse_yolo_annotations(txt, size, size)
    sem = polygons_to_raw_semantic_mask(anns, size, size)
    return render_bio7(sem)


def main():
    for uid in TEST_UIDS:
        out = OUT_ROOT / uid / "pred.png"
        rgb = render_saved_prediction(uid)
        save_png(rgb, out)
        print(f"  wrote saved-prediction render → {out}")

    uids = collect_uids()
    print(f"\nRebuilding montage with {len(uids)} samples …")
    build_montage(uids)


if __name__ == "__main__":
    main()
