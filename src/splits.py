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
    """Strategy 1: Standard split, stratified by species x microscope.

    Millet/Olympus and Rice/Zeiss: few experiments, split smallest→test, next→val.
    Rice/C10, Rice/Olympus, Sorghum/C10, Sorghum/Olympus: accumulate smallest
    experiments until test/val minimums are met (same distribution as Strategy 2).
    """
    rng = random.Random(seed)

    train_samples = []
    val_samples = []
    test_samples = []

    # Helper: accumulate smallest experiments until >= min_samples
    def _pick_experiments(species, microscope, min_samples, exclude=None):
        combo = registry.filter(species=[species], microscopes=[microscope])
        exps = defaultdict(list)
        for s in combo:
            exps[s.group_key].append(s)
        sorted_keys = sorted(exps.keys(), key=lambda k: len(exps[k]))
        picked_keys = set()
        picked_samples = []
        total = 0
        for k in sorted_keys:
            if exclude and k in exclude:
                continue
            picked_keys.add(k)
            picked_samples.extend(exps[k])
            total += len(exps[k])
            if total >= min_samples:
                break
        return picked_keys, picked_samples

    # --- Exclude Rice/Zeiss entirely (reserved for deployment) ---
    # --- Millet/Olympus: few experiments, use n<=3 logic ---
    exp_groups = registry.get_experiment_groups()
    sm_groups = registry.get_species_microscope_groups()
    millet_oly = sm_groups.get("Millet/Olympus", [])
    exps_with_size = [(e, len(exp_groups[e])) for e in millet_oly]
    exps_with_size.sort(key=lambda x: x[1])  # smallest first

    n = len(exps_with_size)
    if n >= 3:
        test_samples.extend(exp_groups[exps_with_size[0][0]])
        val_samples.extend(exp_groups[exps_with_size[1][0]])
        for e, _ in exps_with_size[2:]:
            train_samples.extend(exp_groups[e])
    elif n == 2:
        test_samples.extend(exp_groups[exps_with_size[0][0]])
        train_samples.extend(exp_groups[exps_with_size[1][0]])
    elif n == 1:
        train_samples.extend(exp_groups[exps_with_size[0][0]])

    # --- Seen combos: hold out smallest experiments for test ---
    test_holdout_keys = set()
    test_min = [
        ("Rice", "C10", 10),
        ("Rice", "Olympus", 20),
        ("Sorghum", "C10", 10),
        ("Sorghum", "Olympus", 20),
        ("Tomato", "C10", 10),
        ("Tomato", "Olympus", 20),
    ]
    for sp, micro, min_n in test_min:
        keys, samples = _pick_experiments(sp, micro, min_n)
        test_holdout_keys |= keys
        test_samples.extend(samples)

    # Val: hold out a few Sorghum/C10 not in test
    val_holdout_keys = set()
    sorg_c10 = registry.filter(species=["Sorghum"], microscopes=["C10"])
    sc10_exps = defaultdict(list)
    for s in sorg_c10:
        sc10_exps[s.group_key].append(s)
    for k in sorted(sc10_exps.keys(), key=lambda k: len(sc10_exps[k])):
        if k not in test_holdout_keys:
            val_holdout_keys.add(k)
            val_samples.extend(sc10_exps[k])
            break

    # Remaining experiments from all seen combos: 10% to val, rest to train
    excluded = test_holdout_keys | val_holdout_keys
    remaining_exps = defaultdict(list)
    for sp, micro in [("Rice", "C10"), ("Rice", "Olympus"),
                       ("Sorghum", "C10"), ("Sorghum", "Olympus"),
                       ("Tomato", "C10"), ("Tomato", "Olympus")]:
        for s in registry.filter(species=[sp], microscopes=[micro]):
            if s.group_key not in excluded:
                remaining_exps[s.group_key].append(s)

    rem_keys = sorted(remaining_exps.keys())
    rng.shuffle(rem_keys)

    n_val = max(1, round(len(rem_keys) * val_ratio))
    for k in rem_keys[:n_val]:
        val_samples.extend(remaining_exps[k])
    for k in rem_keys[n_val:]:
        train_samples.extend(remaining_exps[k])

    return {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples,
    }


def strategy2_generalizability(
    registry: SampleRegistry,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, List[SampleRecord]]:
    """Strategy 2: Generalizability test.

    Train: Sorghum + Rice (C10 + Olympus only), minus held-out experiments
    Test:  Millet/Olympus (unseen species) + Rice/Zeiss (unseen microscope)
           + held-out experiments from each seen combo to meet minimums
    Val:   Sorghum/C10 experiments + 10% of remaining train experiments
    """
    rng = random.Random(seed)

    # Test: Millet/Olympus + Rice/Zeiss (unseen species/microscope)
    test_samples = registry.filter(species=["Millet"]) + \
                   registry.filter(species=["Rice"], microscopes=["Zeiss"])

    # Helper: accumulate smallest experiments until >= min_samples
    def _pick_experiments(species, microscope, min_samples):
        combo = registry.filter(species=[species], microscopes=[microscope])
        exps = defaultdict(list)
        for s in combo:
            exps[s.group_key].append(s)
        # Sort by size (smallest first)
        sorted_keys = sorted(exps.keys(), key=lambda k: len(exps[k]))
        picked_keys = set()
        picked_samples = []
        total = 0
        for k in sorted_keys:
            picked_keys.add(k)
            picked_samples.extend(exps[k])
            total += len(exps[k])
            if total >= min_samples:
                break
        return picked_keys, picked_samples

    # Hold out experiments for test: meet minimums per combo
    test_holdout_keys = set()
    test_min = [
        ("Rice", "C10", 10),
        ("Rice", "Olympus", 20),
        ("Sorghum", "C10", 10),
        ("Sorghum", "Olympus", 20),
    ]
    for sp, micro, min_n in test_min:
        keys, samples = _pick_experiments(sp, micro, min_n)
        test_holdout_keys |= keys
        test_samples.extend(samples)

    # Hold out a few Sorghum/C10 for val
    val_holdout_keys = set()
    val_holdout_samples = []
    sorg_c10 = registry.filter(species=["Sorghum"], microscopes=["C10"])
    sc10_exps = defaultdict(list)
    for s in sorg_c10:
        sc10_exps[s.group_key].append(s)
    # Pick smallest not already in test
    for k in sorted(sc10_exps.keys(), key=lambda k: len(sc10_exps[k])):
        if k not in test_holdout_keys:
            val_holdout_keys.add(k)
            val_holdout_samples.extend(sc10_exps[k])
            break

    # Train pool: Sorghum(all) + Rice(C10, Olympus), minus holdouts
    train_pool = registry.filter(species=["Sorghum"]) + \
                 registry.filter(species=["Rice"], microscopes=["C10", "Olympus"])

    excluded = test_holdout_keys | val_holdout_keys
    train_exps = defaultdict(list)
    for s in train_pool:
        if s.group_key not in excluded:
            train_exps[s.group_key].append(s)

    exp_keys = sorted(train_exps.keys())
    rng.shuffle(exp_keys)

    n_val = max(1, round(len(exp_keys) * val_ratio))
    val_exp_keys = exp_keys[:n_val]
    train_exp_keys = exp_keys[n_val:]

    train_samples = []
    for k in train_exp_keys:
        train_samples.extend(train_exps[k])
    val_samples = val_holdout_samples[:]
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


def _normalize_strategy(strategy: str) -> str:
    """Accept both 'A'/'B'/'C' and legacy 'strategy1'/'strategy2'/'strategy3'."""
    mapping = {
        "A": "A", "B": "B", "C": "C",
        "strategy1": "A", "strategy2": "B", "strategy3": "C",
    }
    normalized = mapping.get(strategy)
    if normalized is None:
        raise ValueError(f"Unknown strategy: {strategy!r}. Use 'A', 'B', or 'C'.")
    return normalized


def get_split(
    strategy: str,
    registry: Optional[SampleRegistry] = None,
    seed: int = 42,
    species: Optional[str] = None,
    cache: bool = True,
) -> Dict[str, List[SampleRecord]]:
    """Get or create a dataset split.

    Args:
        strategy: "A", "B", or "C" (also accepts legacy "strategy1"/"strategy2"/"strategy3").
        registry: SampleRegistry (created if None).
        seed: Random seed.
        species: Required for strategy C.
        cache: If True, save/load from disk.

    Returns:
        Dict with "train", "val", "test" lists of SampleRecord.
    """
    strategy = _normalize_strategy(strategy)

    if registry is None:
        registry = SampleRegistry()

    split_dir = OUTPUT_DIR / "splits"
    suffix = f"_{species}" if species else ""
    split_path = split_dir / f"{strategy}{suffix}_seed{seed}.json"

    if cache and split_path.exists():
        return load_split(split_path, registry)

    if strategy == "A":
        split = strategy1_standard(registry, seed=seed)
    elif strategy == "B":
        split = strategy2_generalizability(registry, seed=seed)
    elif strategy == "C":
        if species is None:
            raise ValueError("species required for strategy C")
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
