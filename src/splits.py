"""Dataset splitting: load train/val/test from physical directories."""

from collections import defaultdict
from typing import Dict, List, Optional

from .config import TRAIN_DIR, VAL_DIR, TEST_DIR, SampleRecord
from .dataset import SampleRegistry


def get_split(
    strategy: str = "A",  # noqa: ARG001
    registry: Optional["SampleRegistry"] = None,  # noqa: ARG001
    seed: int = 42,  # noqa: ARG001
    species: Optional[str] = None,
    cache: bool = True,  # noqa: ARG001
    **_kwargs,
) -> Dict[str, List[SampleRecord]]:
    """Load dataset split from physical train/val/test directories.

    Data is pre-split into data/train/, data/val/, data/test/ subdirectories.
    This function creates a SampleRegistry for each and returns the combined
    split dict.

    The ``strategy``, ``registry``, ``seed``, and ``cache`` parameters are
    accepted for backward compatibility but ignored.

    Args:
        species: If provided, filter to only this species.

    Returns:
        Dict with "train", "val", "test" lists of SampleRecord.
    """
    split: Dict[str, List[SampleRecord]] = {}
    for subset, data_dir in [("train", TRAIN_DIR), ("val", VAL_DIR), ("test", TEST_DIR)]:
        reg = SampleRegistry(data_dir=data_dir, require_annotations=True)
        samples = reg.samples
        if species:
            samples = [s for s in samples if s.species == species]
        split[subset] = samples

    return split


def print_split_summary(split: Dict[str, List[SampleRecord]]) -> None:
    """Print summary statistics for a split."""
    for subset, samples in split.items():
        print(f"\n{subset.upper()}: {len(samples)} samples")
        species_counts = defaultdict(int)
        micro_counts = defaultdict(int)
        exp_set = set()
        for s in samples:
            species_counts[s.species] += 1
            micro_counts[s.microscope] += 1
            exp_set.add(s.group_key)
        print(f"  Experiments: {len(exp_set)}")
        for sp, c in sorted(species_counts.items()):
            print(f"  {sp}: {c}")
        for m, c in sorted(micro_counts.items()):
            print(f"  {m}: {c}")
