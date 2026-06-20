"""Build a montage of just the brightened `original.png` panels (no mask,
no PCA). Tomato samples come first, others shuffled after — same ordering
rule as the main montage."""
from __future__ import annotations

import random
import sys
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.render_science_art_pred_pca import (                       # noqa: E402
    OUT_ROOT, collect_uids, ART_DIR,
)

CELL = 800
GAP = 8
BG = (0, 0, 0)

# Pick grid by sample count: 10→2×5, 12→3×4 (3 rows × 4 cols)
GRIDS = {10: (2, 5), 12: (3, 4)}


def main():
    uids = collect_uids()
    n = len(uids)
    if n not in GRIDS:
        raise SystemExit(f"no grid defined for {n} samples (have {sorted(GRIDS)})")
    rows, cols = GRIDS[n]
    out_path = ART_DIR / f"science_art_originals_{rows}x{cols}.png"

    tomato = [u for u in uids if u.lower().startswith("tomato")]
    others = [u for u in uids if not u.lower().startswith("tomato")]
    random.shuffle(tomato); random.shuffle(others)
    ordered = tomato + others

    W = cols * CELL + (cols - 1) * GAP
    H = rows * CELL + (rows - 1) * GAP
    canvas = Image.new("RGB", (W, H), BG)

    for i, uid in enumerate(ordered):
        row, col = divmod(i, cols)
        im = Image.open(OUT_ROOT / uid / "original.png").convert("RGB")
        im = im.resize((CELL, CELL), Image.LANCZOS)
        canvas.paste(im, (col * (CELL + GAP), row * (CELL + GAP)))
        print(f"  [{row},{col}] {uid}")

    canvas.save(out_path)
    print(f"\nWrote {out_path} ({W}x{H})")


if __name__ == "__main__":
    main()
