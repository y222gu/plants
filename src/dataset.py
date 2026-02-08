"""SampleRegistry: discovers, filters, and groups all annotated samples."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Set

from .config import IMAGE_DIR, ANNOTATION_DIR, SampleRecord


class SampleRegistry:
    """Central registry of all annotated samples.

    Discovers samples by walking IMAGE_DIR, matches with annotation files,
    and provides filtering/grouping for splits and data loading.
    """

    def __init__(self):
        self.samples: List[SampleRecord] = []
        self._discover()

    def _discover(self):
        """Walk IMAGE_DIR to find all samples with matching annotations."""
        annotation_files: Set[str] = set()
        if ANNOTATION_DIR.exists():
            annotation_files = {f.name for f in ANNOTATION_DIR.iterdir() if f.is_file()}

        for root, dirs, files in os.walk(IMAGE_DIR):
            if not dirs and files:  # leaf directory
                tif_files = [f for f in files if f.lower().endswith((".tif", ".tiff"))]
                if not tif_files:
                    continue

                full_path = Path(root)
                rel_path = full_path.relative_to(IMAGE_DIR)
                parts = rel_path.parts  # (Species, Microscope, Experiment, SampleName)

                if len(parts) != 4:
                    continue

                species, microscope, experiment, sample_name = parts
                annotation_name = "_".join(parts) + ".txt"

                if annotation_name not in annotation_files:
                    continue

                self.samples.append(SampleRecord(
                    species=species,
                    microscope=microscope,
                    experiment=experiment,
                    sample_name=sample_name,
                    image_dir=full_path,
                    annotation_path=ANNOTATION_DIR / annotation_name,
                ))

        # Sort for deterministic ordering
        self.samples.sort(key=lambda s: s.uid)

    def filter(
        self,
        species: Optional[List[str]] = None,
        microscopes: Optional[List[str]] = None,
        experiments: Optional[List[str]] = None,
    ) -> List[SampleRecord]:
        """Return samples matching the given filters (None = no filter)."""
        result = self.samples
        if species is not None:
            s = set(species)
            result = [r for r in result if r.species in s]
        if microscopes is not None:
            m = set(microscopes)
            result = [r for r in result if r.microscope in m]
        if experiments is not None:
            e = set(experiments)
            result = [r for r in result if r.experiment in e]
        return result

    def get_experiment_groups(self) -> Dict[str, List[SampleRecord]]:
        """Group samples by experiment key (Species/Microscope/Experiment)."""
        groups: Dict[str, List[SampleRecord]] = {}
        for s in self.samples:
            groups.setdefault(s.group_key, []).append(s)
        return groups

    def get_species_microscope_groups(self) -> Dict[str, List[str]]:
        """Map (Species, Microscope) → list of experiment keys."""
        groups: Dict[str, List[str]] = {}
        for s in self.samples:
            sm_key = f"{s.species}/{s.microscope}"
            exp_key = s.group_key
            if sm_key not in groups:
                groups[sm_key] = []
            if exp_key not in groups[sm_key]:
                groups[sm_key].append(exp_key)
        return groups

    def summary(self) -> str:
        """Print dataset summary statistics."""
        lines = [f"Total samples: {len(self.samples)}"]
        # Per species
        species_counts: Dict[str, int] = {}
        for s in self.samples:
            species_counts[s.species] = species_counts.get(s.species, 0) + 1
        for sp, cnt in sorted(species_counts.items()):
            lines.append(f"  {sp}: {cnt}")
        # Per microscope
        micro_counts: Dict[str, int] = {}
        for s in self.samples:
            micro_counts[s.microscope] = micro_counts.get(s.microscope, 0) + 1
        for m, cnt in sorted(micro_counts.items()):
            lines.append(f"  {m}: {cnt}")
        # Experiments
        exp_groups = self.get_experiment_groups()
        lines.append(f"Total experiments: {len(exp_groups)}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> SampleRecord:
        return self.samples[idx]
