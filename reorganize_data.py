"""Reorganize Rice/C10 and Rice/Zeiss experiments for better splitting.

Rice/C10: Split single Exp2 into Exp1/Exp2/Exp3 by genotype (PSY9/PSY17/WT).
Rice/Zeiss: Split single Exp1 into Exp1/Exp2/Exp3 (28:4:3 random split).

Usage:
    python reorganize_data.py --dry-run   # Preview moves without executing
    python reorganize_data.py             # Execute moves (asks for confirmation)
"""

import argparse
import json
import random
import shutil
from pathlib import Path

from src.config import ANNOTATION_DIR, IMAGE_DIR, OUTPUT_DIR


def get_rice_c10_mapping():
    """Map Rice/C10/Exp2 samples to new experiments by genotype."""
    mapping = {}
    exp2_dir = IMAGE_DIR / "Rice" / "C10" / "Exp2"
    if not exp2_dir.exists():
        return mapping

    for sample_dir in sorted(exp2_dir.iterdir()):
        if not sample_dir.is_dir():
            continue
        name = sample_dir.name
        if name.startswith("PSY9"):
            new_exp = "Exp1"
        elif name.startswith("PSY17"):
            new_exp = "Exp2"
        elif name.startswith("WT"):
            new_exp = "Exp3"
        else:
            print(f"WARNING: Unknown genotype for {name}, skipping")
            continue
        mapping[name] = new_exp
    return mapping


def get_rice_zeiss_mapping(seed=42):
    """Map Rice/Zeiss/Exp1 samples to Exp1/Exp2/Exp3 (28:4:3)."""
    mapping = {}
    exp1_dir = IMAGE_DIR / "Rice" / "Zeiss" / "Exp1"
    if not exp1_dir.exists():
        return mapping

    samples = sorted([d.name for d in exp1_dir.iterdir() if d.is_dir()])
    rng = random.Random(seed)
    rng.shuffle(samples)

    # 28 train, 4 val, 3 test
    for i, name in enumerate(samples):
        if i < 28:
            mapping[name] = "Exp1"
        elif i < 32:
            mapping[name] = "Exp2"
        else:
            mapping[name] = "Exp3"
    return mapping


def plan_moves(rice_c10_map, rice_zeiss_map):
    """Generate list of (action, src, dst) tuples."""
    moves = []

    # Rice/C10
    for sample_name, new_exp in sorted(rice_c10_map.items()):
        # Image directory
        src_img = IMAGE_DIR / "Rice" / "C10" / "Exp2" / sample_name
        dst_img = IMAGE_DIR / "Rice" / "C10" / new_exp / sample_name
        if src_img != dst_img:
            moves.append(("move_dir", str(src_img), str(dst_img)))

        # Annotation file
        src_ann = ANNOTATION_DIR / f"Rice_C10_Exp2_{sample_name}.txt"
        dst_ann = ANNOTATION_DIR / f"Rice_C10_{new_exp}_{sample_name}.txt"
        if src_ann != dst_ann:
            moves.append(("rename_file", str(src_ann), str(dst_ann)))

    # Rice/Zeiss
    for sample_name, new_exp in sorted(rice_zeiss_map.items()):
        src_img = IMAGE_DIR / "Rice" / "Zeiss" / "Exp1" / sample_name
        dst_img = IMAGE_DIR / "Rice" / "Zeiss" / new_exp / sample_name
        if src_img != dst_img:
            moves.append(("move_dir", str(src_img), str(dst_img)))

        src_ann = ANNOTATION_DIR / f"Rice_Zeiss_Exp1_{sample_name}.txt"
        dst_ann = ANNOTATION_DIR / f"Rice_Zeiss_{new_exp}_{sample_name}.txt"
        if src_ann != dst_ann:
            moves.append(("rename_file", str(src_ann), str(dst_ann)))

    return moves


def print_plan(moves, rice_c10_map, rice_zeiss_map):
    """Print reorganization plan."""
    # Rice/C10 summary
    c10_exps = {}
    for name, exp in rice_c10_map.items():
        c10_exps.setdefault(exp, []).append(name)

    print("=" * 70)
    print("RICE/C10 REORGANIZATION")
    print("=" * 70)
    print(f"Splitting Exp2 ({len(rice_c10_map)} samples) by genotype:")
    for exp in sorted(c10_exps):
        samples = c10_exps[exp]
        print(f"  {exp}: {len(samples)} samples ({samples[0]} ... {samples[-1]})")

    # Rice/Zeiss summary
    zeiss_exps = {}
    for name, exp in rice_zeiss_map.items():
        zeiss_exps.setdefault(exp, []).append(name)

    print()
    print("=" * 70)
    print("RICE/ZEISS REORGANIZATION")
    print("=" * 70)
    print(f"Splitting Exp1 ({len(rice_zeiss_map)} samples) randomly:")
    for exp in sorted(zeiss_exps):
        samples = zeiss_exps[exp]
        print(f"  {exp}: {len(samples)} samples ({samples[0]} ... {samples[-1]})")

    # Detailed moves
    print()
    print("=" * 70)
    print(f"TOTAL OPERATIONS: {len(moves)}")
    print("=" * 70)
    for action, src, dst in moves:
        tag = "DIR " if action == "move_dir" else "FILE"
        # Show relative paths for readability
        src_rel = Path(src).relative_to(Path(src).parents[4]) if action == "move_dir" else Path(src).name
        dst_rel = Path(dst).relative_to(Path(dst).parents[4]) if action == "move_dir" else Path(dst).name
        print(f"  [{tag}] {src_rel} -> {dst_rel}")


def execute_moves(moves):
    """Execute the planned moves."""
    for i, (action, src, dst) in enumerate(moves):
        src_path = Path(src)
        dst_path = Path(dst)

        if not src_path.exists():
            print(f"  SKIP (missing): {src}")
            continue

        dst_path.parent.mkdir(parents=True, exist_ok=True)

        if action == "move_dir":
            shutil.move(str(src_path), str(dst_path))
        elif action == "rename_file":
            src_path.rename(dst_path)

        if (i + 1) % 20 == 0:
            print(f"  Completed {i + 1}/{len(moves)} operations...")

    print(f"  Completed all {len(moves)} operations.")


def cleanup_empty_dirs():
    """Remove empty experiment directories after moves."""
    removed = []
    for species in ["Rice"]:
        for micro in ["C10", "Zeiss"]:
            base = IMAGE_DIR / species / micro
            if not base.exists():
                continue
            for exp_dir in base.iterdir():
                if exp_dir.is_dir() and not any(exp_dir.iterdir()):
                    exp_dir.rmdir()
                    removed.append(str(exp_dir))
    if removed:
        print(f"Removed {len(removed)} empty directories:")
        for d in removed:
            print(f"  {d}")


def delete_cached_splits():
    """Delete cached split JSONs since experiment names changed."""
    split_dir = OUTPUT_DIR / "splits"
    if not split_dir.exists():
        return
    deleted = []
    for f in split_dir.glob("*.json"):
        f.unlink()
        deleted.append(f.name)
    if deleted:
        print(f"Deleted {len(deleted)} cached split files: {', '.join(deleted)}")


def verify_samples():
    """Re-discover all samples and verify total count."""
    from src.dataset import SampleRegistry
    registry = SampleRegistry()
    print(f"\nVerification: {len(registry)} samples discovered")
    print(registry.summary())

    # Verify Rice/C10 experiments
    c10_samples = registry.filter(species=["Rice"], microscopes=["C10"])
    c10_exps = set(s.experiment for s in c10_samples)
    print(f"\nRice/C10: {len(c10_samples)} samples in {len(c10_exps)} experiments: {sorted(c10_exps)}")

    # Verify Rice/Zeiss experiments
    zeiss_samples = registry.filter(species=["Rice"], microscopes=["Zeiss"])
    zeiss_exps = set(s.experiment for s in zeiss_samples)
    print(f"Rice/Zeiss: {len(zeiss_samples)} samples in {len(zeiss_exps)} experiments: {sorted(zeiss_exps)}")

    assert len(registry) == 1308, f"Expected 1308 samples, got {len(registry)}"
    print("\nAll 1308 samples accounted for.")


def main():
    parser = argparse.ArgumentParser(description="Reorganize Rice experiments for splitting")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show planned moves without executing")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for Zeiss split")
    args = parser.parse_args()

    rice_c10_map = get_rice_c10_mapping()
    rice_zeiss_map = get_rice_zeiss_mapping(seed=args.seed)
    moves = plan_moves(rice_c10_map, rice_zeiss_map)

    print_plan(moves, rice_c10_map, rice_zeiss_map)

    if args.dry_run:
        print("\n[DRY RUN] No changes made.")
        return

    # Ask for confirmation
    print()
    response = input("Proceed with reorganization? (yes/no): ").strip().lower()
    if response != "yes":
        print("Aborted.")
        return

    print("\nExecuting moves...")
    execute_moves(moves)

    print("\nCleaning up empty directories...")
    cleanup_empty_dirs()

    print("\nDeleting cached splits...")
    delete_cached_splits()

    print("\nVerifying...")
    verify_samples()

    print("\nReorganization complete!")


if __name__ == "__main__":
    main()