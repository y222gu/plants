"""Tile the 12 PNGs in /Users/yifeigu/Documents/Siobhan_Lab/science_art/
into a 3-row × 4-column montage with a small black gap."""
import random
from pathlib import Path

from PIL import Image

SRC = Path("/Users/yifeigu/Documents/Siobhan_Lab/science_art")
OUT = SRC / "science_art_montage_3x4.png"

CELL = 800       # px per panel
GAP = 8          # px black gap between panels
ROWS, COLS = 3, 4
BG = (0, 0, 0)


def main():
    files = sorted(
        p for p in SRC.iterdir()
        if p.suffix.lower() == ".png" and p.is_file() and "montage" not in p.name.lower()
    )
    if len(files) != ROWS * COLS:
        print(f"warning: found {len(files)} PNGs, expected {ROWS*COLS} — using first {ROWS*COLS}")
    files = files[: ROWS * COLS]
    tomato = [p for p in files if p.name.lower().startswith("tomato")]
    others = [p for p in files if not p.name.lower().startswith("tomato")]
    random.shuffle(tomato)
    random.shuffle(others)
    files = tomato[:COLS] + others   # tomato → row 1, rest fill rows 2–3

    W = COLS * CELL + (COLS - 1) * GAP
    H = ROWS * CELL + (ROWS - 1) * GAP
    canvas = Image.new("RGB", (W, H), BG)

    for i, f in enumerate(files):
        r, c = divmod(i, COLS)
        im = Image.open(f).convert("RGB")
        im = im.resize((CELL, CELL), Image.LANCZOS)
        x = c * (CELL + GAP)
        y = r * (CELL + GAP)
        canvas.paste(im, (x, y))
        print(f"  [{r},{c}] {f.name}")

    canvas.save(OUT)
    print(f"\nWrote {OUT} ({W}x{H})")


if __name__ == "__main__":
    main()
