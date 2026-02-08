"""Experiment-level dataset splitting strategies."""

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import OUTPUT_DIR, SampleRecord
from .dataset import SampleRegistry


def _experiments_to_samples(
    registry: SampleRegistry,
    experiments: List[str],
) -> List[SampleRecord]:
    """Get all samples belonging to the given experiment keys."""
    exp_set = set(experiments)
    return [s for s in registry.samples if s.group_key in exp_set]


def strategy1_standard(
    registry: SampleRegistry,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, List[SampleRecord]]:
    """Strategy 1: Standard 80/10/10 split, stratified by species x microscope.

    Splits at experiment level, stratified so each (species, microscope) combo
    has proportional representation in train/val/test.
    """
    rng = random.Random(seed)
    sm_groups = registry.get_species_microscope_groups()

    train_exps, val_exps, test_exps = [], [], []

    for sm_key, experiments in sorted(sm_groups.items()):
        # Sort experiments by sample count (descending) then shuffle within
        # size tiers so the largest experiments are more likely to end up in train
        exp_groups = registry.get_experiment_groups()
        exps_with_size = [(e, len(exp_groups[e])) for e in experiments]
        exps_with_size.sort(key=lambda x: -x[1])  # largest first

        n = len(exps_with_size)
        if n <= 1:
            # Single experiment: put in train
            train_exps.extend([e for e, _ in exps_with_size])
            continue

        if n <= 3:
            # Very few experiments: put largest in train, smallest in test/val
            # Sort smallest-first for allocation to test/val
            exps_with_size.sort(key=lambda x: x[1])
            test_exps.append(exps_with_size[0][0])
            if n == 3:
                val_exps.append(exps_with_size[1][0])
                train_exps.append(exps_with_size[2][0])
            else:  # n == 2
                train_exps.append(exps_with_size[1][0])
            continue

        # Normal case: shuffle and split proportionally
        exps = [e for e, _ in exps_with_size]
        rng.shuffle(exps)

        n_test = max(1, round(n * (1 - train_ratio - val_ratio)))
        n_val = max(1, round(n * val_ratio))

        test_exps.extend(exps[:n_test])
        val_exps.extend(exps[n_test:n_test + n_val])
        train_exps.extend(exps[n_test + n_val:])

    return {
        "train": _experiments_to_samples(registry, train_exps),
        "val": _experiments_to_samples(registry, val_exps),
        "test": _experiments_to_samples(registry, test_exps),
    }


def strategy2_generalizability(
    registry: SampleRegistry,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, List[SampleRecord]]:
    """Strategy 2: Generalizability test.

    Train: Sorghum + Rice (C10 + Olympus only)
    Test:  Millet/Olympus (unseen species) + Rice/Zeiss (unseen microscope)
    Val:   10% of train experiments held out
    """
    rng = random.Random(seed)

    # Train pool: Sorghum(all) + Rice(C10, Olympus)
    train_pool = registry.filter(species=["Sorghum"]) + \
                 registry.filter(species=["Rice"], microscopes=["C10", "Olympus"])

    # Test: Millet/Olympus + Rice/Zeiss
    test_samples = registry.filter(species=["Millet"]) + \
                   registry.filter(species=["Rice"], microscopes=["Zeiss"])

    # Split train pool by experiment for val holdout
    train_exps = defaultdict(list)
    for s in train_pool:
        train_exps[s.group_key].append(s)

    exp_keys = sorted(train_exps.keys())
    rng.shuffle(exp_keys)

    n_val = max(1, round(len(exp_keys) * val_ratio))
    val_exp_keys = exp_keys[:n_val]
    train_exp_keys = exp_keys[n_val:]

    train_samples = []
    for k in train_exp_keys:
        train_samples.extend(train_exps[k])
    val_samples = []
    for k in val_exp_keys:
        val_samples.extend(train_exps[k])

    return {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples,
    }


def strategy3_per_species(
    registry: SampleRegistry,
    species: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, List[SampleRecord]]:
    """Strategy 3: Per-species 80/10/10 split.

    Trains a species-specific model. Split at experiment level within species.
    """
    rng = random.Random(seed)
    species_samples = registry.filter(species=[species])

    # Group by experiment
    exp_groups = defaultdict(list)
    for s in species_samples:
        exp_groups[s.group_key].append(s)

    exp_keys = sorted(exp_groups.keys())
    rng.shuffle(exp_keys)

    n = len(exp_keys)
    n_test = max(1, round(n * (1 - train_ratio - val_ratio)))
    n_val = max(1, round(n * val_ratio))

    test_keys = exp_keys[:n_test]
    val_keys = exp_keys[n_test:n_test + n_val]
    train_keys = exp_keys[n_test + n_val:]

    return {
        "train": [s for k in train_keys for s in exp_groups[k]],
        "val": [s for k in val_keys for s in exp_groups[k]],
        "test": [s for k in test_keys for s in exp_groups[k]],
    }


def save_split(
    split: Dict[str, List[SampleRecord]],
    path: Path,
    strategy_name: str = "",
) -> None:
    """Save split to JSON for reproducibility."""
    data = {
        "strategy": strategy_name,
        "counts": {k: len(v) for k, v in split.items()},
        "splits": {},
    }
    for subset, samples in split.items():
        data["splits"][subset] = [s.uid for s in samples]

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_split(
    path: Path,
    registry: SampleRegistry,
) -> Dict[str, List[SampleRecord]]:
    """Load a saved split JSON and resolve to SampleRecords."""
    with open(path) as f:
        data = json.load(f)

    uid_to_sample = {s.uid: s for s in registry.samples}
    result = {}
    for subset, uids in data["splits"].items():
        result[subset] = [uid_to_sample[uid] for uid in uids if uid in uid_to_sample]
    return result


def get_split(
    strategy: str,
    registry: Optional[SampleRegistry] = None,
    seed: int = 42,
    species: Optional[str] = None,
    cache: bool = True,
) -> Dict[str, List[SampleRecord]]:
    """Get or create a dataset split.

    Args:
        strategy: "strategy1", "strategy2", or "strategy3".
        registry: SampleRegistry (created if None).
        seed: Random seed.
        species: Required for strategy3.
        cache: If True, save/load from disk.

    Returns:
        Dict with "train", "val", "test" lists of SampleRecord.
    """
    if registry is None:
        registry = SampleRegistry()

    split_dir = OUTPUT_DIR / "splits"
    suffix = f"_{species}" if species else ""
    split_path = split_dir / f"{strategy}{suffix}_seed{seed}.json"

    if cache and split_path.exists():
        return load_split(split_path, registry)

    if strategy == "strategy1":
        split = strategy1_standard(registry, seed=seed)
    elif strategy == "strategy2":
        split = strategy2_generalizability(registry, seed=seed)
    elif strategy == "strategy3":
        if species is None:
            raise ValueError("species required for strategy3")
        split = strategy3_per_species(registry, species, seed=seed)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    if cache:
        save_split(split, split_path, strategy_name=strategy)

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
