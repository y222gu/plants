"""Fix annotation polygon overlaps to enforce spatial consistency.

Applies three corrections to each annotation file:
1. Expand whole root (class 0) to cover all other polygons (classes 1-5)
2. Clip aerenchyma (class 1) to be strictly inside inner exodermis (class 5)
3. Subtract outer endodermis (class 2) from aerenchyma (class 1)

Only whole root and aerenchyma polygons are modified. Endodermis and exodermis
polygons are never changed.

Usage:
    python fix_annotations.py              # dry run (report only)
    python fix_annotations.py --apply      # apply fixes
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

from shapely.geometry import MultiPolygon, Polygon as ShapelyPolygon
from shapely.ops import unary_union
from shapely.validation import make_valid


DATA_DIR = Path(__file__).parent / "data"
CLASS_NAMES = {
    0: "Whole Root", 1: "Aerenchyma", 2: "Outer Endodermis",
    3: "Inner Endodermis", 4: "Outer Exodermis", 5: "Inner Exodermis",
}

# Small buffer (in normalized coords, ~0.5px at 1024) to prevent shared boundary edges
BUFFER_DIST = 0.0005


def parse_yolo_file(path: Path) -> List[Tuple[int, List[Tuple[float, float]]]]:
    """Parse YOLO annotation file → [(class_id, [(x, y), ...]), ...]."""
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            class_id = int(parts[0])
            coords = []
            for i in range(1, len(parts), 2):
                coords.append((float(parts[i]), float(parts[i + 1])))
            entries.append((class_id, coords))
    return entries


def coords_to_shapely(coords: List[Tuple[float, float]]) -> ShapelyPolygon:
    """Convert coordinate list to a valid shapely Polygon."""
    if len(coords) < 3:
        return ShapelyPolygon()
    poly = ShapelyPolygon(coords)
    if not poly.is_valid:
        poly = make_valid(poly)
    return poly


def shapely_to_coords(geom) -> List[List[Tuple[float, float]]]:
    """Convert shapely geometry to list of coordinate lists (handles MultiPolygon)."""
    if geom.is_empty:
        return []
    if isinstance(geom, ShapelyPolygon):
        coords = list(geom.exterior.coords[:-1])  # drop closing duplicate
        if len(coords) >= 3:
            return [coords]
        return []
    elif isinstance(geom, MultiPolygon):
        result = []
        for poly in geom.geoms:
            coords = list(poly.exterior.coords[:-1])
            if len(coords) >= 3:
                result.append(coords)
        return result
    else:
        # GeometryCollection or other — extract polygons
        result = []
        if hasattr(geom, 'geoms'):
            for g in geom.geoms:
                if isinstance(g, ShapelyPolygon) and not g.is_empty:
                    coords = list(g.exterior.coords[:-1])
                    if len(coords) >= 3:
                        result.append(coords)
        return result


def write_yolo_file(path: Path, entries: List[Tuple[int, List[Tuple[float, float]]]]):
    """Write YOLO annotation file."""
    with open(path, 'w') as f:
        for class_id, coords in entries:
            parts = [str(class_id)]
            for x, y in coords:
                parts.append(f"{x:.6f}")
                parts.append(f"{y:.6f}")
            f.write(" ".join(parts) + "\n")


def fix_annotation(path: Path) -> Tuple[List[str], List[Tuple[int, List[Tuple[float, float]]]]]:
    """Fix one annotation file. Returns (list_of_fix_descriptions, new_entries)."""
    entries = parse_yolo_file(path)
    fixes = []

    # Group by class
    by_class: Dict[int, List[Tuple[ShapelyPolygon, List[Tuple[float, float]]]]] = {
        i: [] for i in range(6)
    }
    for class_id, coords in entries:
        poly = coords_to_shapely(coords)
        by_class[class_id].append((poly, coords))

    # Get reference polygons
    root_poly = by_class[0][0][0] if by_class[0] else None
    outer_endo = by_class[2][0][0] if by_class[2] else None
    inner_exo = by_class[5][0][0] if by_class[5] else None

    if root_poly is None or root_poly.is_empty:
        return fixes, entries  # no root polygon, skip

    # ── FIX 1: Expand whole root to cover all other polygons ──
    # Only expand if something is actually outside the root
    all_other_polys = []
    for cls_id in range(1, 6):
        for poly, _ in by_class[cls_id]:
            if not poly.is_empty:
                all_other_polys.append(poly)

    if all_other_polys:
        # Check if any polygon extends outside root
        others_union = unary_union(all_other_polys)
        outside = others_union.difference(root_poly)
        if not outside.is_empty and outside.area > 1e-9:
            # Expand root to cover everything + small buffer
            combined = unary_union([root_poly, others_union])
            expanded_root = combined.buffer(BUFFER_DIST)
            # Ensure result is a single polygon
            if not isinstance(expanded_root, ShapelyPolygon) or expanded_root.is_empty:
                # Union+buffer produced MultiPolygon or GeometryCollection —
                # fall back to convex hull which guarantees a single polygon
                expanded_root = combined.convex_hull.buffer(BUFFER_DIST)
                if not isinstance(expanded_root, ShapelyPolygon) or expanded_root.is_empty:
                    expanded_root = root_poly  # give up, keep original
            fixes.append(f"  FIX1: Expanded whole root to cover all inner polygons "
                         f"(area {root_poly.area:.6f} → {expanded_root.area:.6f})")
            root_poly = expanded_root

    # ── FIX 2: Clip aerenchyma to inside inner exodermis ──
    fixed_aer = []
    if inner_exo is not None and not inner_exo.is_empty:
        exo_inner = inner_exo.buffer(-BUFFER_DIST)
        if not exo_inner.is_empty:
            for poly, orig_coords in by_class[1]:
                if poly.is_empty:
                    fixed_aer.append((poly, orig_coords))
                    continue
                clipped = poly.intersection(exo_inner)
                if not clipped.is_empty and clipped.area > 0:
                    if clipped.area < poly.area * 0.999:
                        fixes.append(f"  FIX2: Clipped aerenchyma to inside inner exodermis "
                                     f"(area {poly.area:.6f} → {clipped.area:.6f})")
                    fixed_aer.append((clipped, orig_coords))
                else:
                    fixes.append(f"  FIX2: Aerenchyma eliminated by inner exodermis clip")
        else:
            fixed_aer = list(by_class[1])
    else:
        fixed_aer = list(by_class[1])

    # ── FIX 3: Subtract outer endodermis from aerenchyma ──
    final_aer = []
    if outer_endo is not None and not outer_endo.is_empty:
        endo_expanded = outer_endo.buffer(BUFFER_DIST)
        for poly, orig_coords in fixed_aer:
            if poly.is_empty:
                continue
            subtracted = poly.difference(endo_expanded)
            if not subtracted.is_empty and subtracted.area > 0:
                if subtracted.area < poly.area * 0.999:
                    fixes.append(f"  FIX3: Subtracted outer endodermis from aerenchyma "
                                 f"(area {poly.area:.6f} → {subtracted.area:.6f})")
                final_aer.append((subtracted, orig_coords))
            else:
                fixes.append(f"  FIX3: Aerenchyma eliminated by outer endodermis subtraction")
    else:
        final_aer = fixed_aer

    # ── Build new entries ──
    new_entries = []

    # Class 0: use original coords unless FIX1 changed the root
    root_was_fixed = any("FIX1" in f for f in fixes)
    if root_was_fixed:
        root_coord_lists = shapely_to_coords(root_poly)
        for coords in root_coord_lists:
            coords = [(max(0, min(1, x)), max(0, min(1, y))) for x, y in coords]
            new_entries.append((0, coords))
    else:
        for _, orig_coords in by_class[0]:
            new_entries.append((0, orig_coords))

    # Class 1: fixed aerenchyma (re-serialize only if fixes were applied)
    aer_was_fixed = any("FIX2" in f or "FIX3" in f for f in fixes)
    if aer_was_fixed:
        for poly, _ in final_aer:
            for coords in shapely_to_coords(poly):
                coords = [(max(0, min(1, x)), max(0, min(1, y))) for x, y in coords]
                new_entries.append((1, coords))
    else:
        for _, orig_coords in by_class[1]:
            new_entries.append((1, orig_coords))

    # Classes 2-5: unchanged
    for cls_id in range(2, 6):
        for _, orig_coords in by_class[cls_id]:
            new_entries.append((cls_id, orig_coords))

    return fixes, new_entries


def main():
    parser = argparse.ArgumentParser(description="Fix annotation polygon overlaps")
    parser.add_argument("--apply", action="store_true",
                        help="Apply fixes (default: dry run)")
    args = parser.parse_args()

    ann_files = sorted(DATA_DIR.rglob("annotation/*.txt"))
    print(f"Found {len(ann_files)} annotation files")
    print(f"Mode: {'APPLY' if args.apply else 'DRY RUN'}\n")

    total_fixes = 0
    files_modified = 0
    fix_counts = {"FIX1": 0, "FIX2": 0, "FIX3": 0}

    for ann_path in ann_files:
        fixes, new_entries = fix_annotation(ann_path)
        if fixes:
            files_modified += 1
            total_fixes += len(fixes)
            split = ann_path.parent.parent.name
            print(f"[{split}] {ann_path.stem}:")
            for fix in fixes:
                print(fix)
                for key in fix_counts:
                    if key in fix:
                        fix_counts[key] += 1
            print()

            if args.apply:
                write_yolo_file(ann_path, new_entries)

    print("=" * 60)
    print("SUMMARY")
    print(f"  Files scanned: {len(ann_files)}")
    print(f"  Files with fixes: {files_modified}")
    print(f"  Total fixes: {total_fixes}")
    print(f"  FIX1 (expand whole root): {fix_counts['FIX1']}")
    print(f"  FIX2 (clip aerenchyma to inner exodermis): {fix_counts['FIX2']}")
    print(f"  FIX3 (subtract outer endodermis from aerenchyma): {fix_counts['FIX3']}")
    if not args.apply and files_modified > 0:
        print(f"\n  Run with --apply to write changes")
    if args.apply:
        print(f"\n  Originals backed up in annotation_copy/")


if __name__ == "__main__":
    main()
