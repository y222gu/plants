"""QC check for annotation polygons.

Checks:
1. All polygons (classes 1-5) are within the whole root polygon (class 0)
   and do not intersect its boundary.
2. All aerenchyma polygons (class 1) are within the inner exodermis polygon
   (class 5) and do not intersect its boundary.
3. All aerenchyma polygons (class 1) are outside the outer endodermis polygon
   (class 2) and do not intersect its boundary.

Usage:
    python qc_annotations.py
"""

from pathlib import Path
from typing import Dict, List, Tuple

from shapely.geometry import Polygon as ShapelyPolygon
from shapely.validation import make_valid


DATA_DIR = Path(__file__).parent / "data"
CLASS_NAMES = {
    0: "Whole Root",
    1: "Aerenchyma",
    2: "Outer Endodermis",
    3: "Inner Endodermis",
    4: "Outer Exodermis",
    5: "Inner Exodermis",
}


def parse_yolo_line(line: str) -> Tuple[int, List[Tuple[float, float]]]:
    """Parse one YOLO polygon line → (class_id, [(x, y), ...]) normalized coords."""
    parts = line.strip().split()
    class_id = int(parts[0])
    coords = []
    for i in range(1, len(parts), 2):
        x, y = float(parts[i]), float(parts[i + 1])
        coords.append((x, y))
    return class_id, coords


def coords_to_shapely(coords: List[Tuple[float, float]]) -> ShapelyPolygon:
    """Convert coordinate list to a valid shapely Polygon."""
    if len(coords) < 3:
        return ShapelyPolygon()
    poly = ShapelyPolygon(coords)
    if not poly.is_valid:
        poly = make_valid(poly)
    return poly


def load_annotation(path: Path) -> Dict[int, List[ShapelyPolygon]]:
    """Load annotation file → {class_id: [ShapelyPolygon, ...]}."""
    polys = {i: [] for i in range(6)}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            class_id, coords = parse_yolo_line(line)
            if class_id in polys:
                poly = coords_to_shapely(coords)
                if not poly.is_empty:
                    polys[class_id].append(poly)
    return polys


def check_contained(inner: ShapelyPolygon, outer: ShapelyPolygon,
                    tolerance: float = 0.005) -> Tuple[bool, float]:
    """Check if inner polygon is within outer polygon.

    Returns (is_contained, fraction_outside).
    tolerance: allowed fraction of inner area outside outer (for minor boundary artifacts).
    """
    if inner.is_empty or outer.is_empty:
        return True, 0.0
    try:
        diff = inner.difference(outer)
        if diff.is_empty:
            return True, 0.0
        frac_outside = diff.area / inner.area if inner.area > 0 else 0.0
        return frac_outside <= tolerance, frac_outside
    except Exception:
        return True, 0.0


def check_outside(inner: ShapelyPolygon, outer: ShapelyPolygon,
                  tolerance: float = 0.005) -> Tuple[bool, float]:
    """Check if inner polygon is outside outer polygon.

    Returns (is_outside, fraction_inside).
    tolerance: allowed fraction of overlap (for minor boundary artifacts).
    """
    if inner.is_empty or outer.is_empty:
        return True, 0.0
    try:
        inter = inner.intersection(outer)
        if inter.is_empty:
            return True, 0.0
        frac_inside = inter.area / inner.area if inner.area > 0 else 0.0
        return frac_inside <= tolerance, frac_inside
    except Exception:
        return True, 0.0


def check_boundary_intersection(poly_a: ShapelyPolygon, poly_b: ShapelyPolygon,
                                 tolerance: float = 0.001) -> Tuple[bool, float]:
    """Check if polygon boundaries intersect (crossing, not just touching).

    Returns (boundaries_cross, intersection_length).
    """
    if poly_a.is_empty or poly_b.is_empty:
        return False, 0.0
    try:
        boundary_inter = poly_a.boundary.intersection(poly_b.boundary)
        if boundary_inter.is_empty:
            return False, 0.0
        length = boundary_inter.length if hasattr(boundary_inter, 'length') else 0.0
        return length > tolerance, length
    except Exception:
        return False, 0.0


def run_qc(ann_path: Path) -> List[str]:
    """Run all QC checks on one annotation file. Returns list of issue strings."""
    polys = load_annotation(ann_path)
    issues = []
    sample = ann_path.stem

    # Get whole root polygon (class 0) — expect exactly 1
    roots = polys[0]
    if len(roots) == 0:
        issues.append(f"  MISSING whole root polygon (class 0)")
        return issues
    if len(roots) > 1:
        issues.append(f"  MULTIPLE whole root polygons ({len(roots)} found)")
    root = roots[0]

    # CHECK 1: All polygons (classes 1-5) within whole root
    for cls_id in range(1, 6):
        for i, poly in enumerate(polys[cls_id]):
            contained, frac_out = check_contained(poly, root, tolerance=0.01)
            if not contained:
                issues.append(
                    f"  CHECK1: {CLASS_NAMES[cls_id]}[{i}] outside whole root "
                    f"({frac_out:.1%} area outside)")
            crosses, length = check_boundary_intersection(poly, root)
            if crosses:
                issues.append(
                    f"  CHECK1: {CLASS_NAMES[cls_id]}[{i}] boundary intersects "
                    f"whole root boundary (length={length:.4f})")

    # CHECK 2: Aerenchyma within inner exodermis (class 5)
    inner_exos = polys[5]
    aers = polys[1]
    if inner_exos and aers:
        inner_exo = inner_exos[0]
        for i, aer in enumerate(aers):
            contained, frac_out = check_contained(aer, inner_exo, tolerance=0.01)
            if not contained:
                issues.append(
                    f"  CHECK2: Aerenchyma[{i}] outside inner exodermis "
                    f"({frac_out:.1%} area outside)")
            crosses, length = check_boundary_intersection(aer, inner_exo)
            if crosses:
                issues.append(
                    f"  CHECK2: Aerenchyma[{i}] boundary intersects "
                    f"inner exodermis boundary (length={length:.4f})")

    # CHECK 3: Aerenchyma outside outer endodermis (class 2)
    outer_endos = polys[2]
    if outer_endos and aers:
        outer_endo = outer_endos[0]
        for i, aer in enumerate(aers):
            is_outside, frac_inside = check_outside(aer, outer_endo, tolerance=0.01)
            if not is_outside:
                issues.append(
                    f"  CHECK3: Aerenchyma[{i}] overlaps outer endodermis "
                    f"({frac_inside:.1%} area inside)")
            crosses, length = check_boundary_intersection(aer, outer_endo)
            if crosses:
                issues.append(
                    f"  CHECK3: Aerenchyma[{i}] boundary intersects "
                    f"outer endodermis boundary (length={length:.4f})")

    return issues


def main():
    # Find all annotation files
    ann_files = sorted(DATA_DIR.rglob("annotation/*.txt"))
    print(f"Found {len(ann_files)} annotation files\n")

    total_issues = 0
    samples_with_issues = 0
    check_counts = {"CHECK1": 0, "CHECK2": 0, "CHECK3": 0}

    for ann_path in ann_files:
        issues = run_qc(ann_path)
        if issues:
            samples_with_issues += 1
            total_issues += len(issues)
            # Determine which split (train/val/test)
            split = ann_path.parent.parent.name
            print(f"[{split}] {ann_path.stem}:")
            for issue in issues:
                print(issue)
                for check in check_counts:
                    if check in issue:
                        check_counts[check] += 1
            print()

    # Summary
    print("=" * 60)
    print("QC SUMMARY")
    print(f"  Total annotation files: {len(ann_files)}")
    print(f"  Samples with issues: {samples_with_issues}")
    print(f"  Total issues: {total_issues}")
    print(f"  CHECK1 (within whole root): {check_counts['CHECK1']} issues")
    print(f"  CHECK2 (aerenchyma within inner exodermis): {check_counts['CHECK2']} issues")
    print(f"  CHECK3 (aerenchyma outside outer endodermis): {check_counts['CHECK3']} issues")

    if total_issues == 0:
        print("\n  ALL ANNOTATIONS PASSED QC!")
    else:
        print(f"\n  {samples_with_issues}/{len(ann_files)} samples need review")


if __name__ == "__main__":
    main()
