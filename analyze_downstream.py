"""Downstream analysis entry point.

Run on ground truth annotations or model predictions to compute
aerenchyma ratios, channel intensities, and instance counts.

Usage:
    python analyze_downstream.py --source gt --strategy strategy1
    python analyze_downstream.py --source predictions --pred-dir output/yolo/predictions
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.annotation_utils import load_sample_annotations
from src.config import OUTPUT_DIR
from src.dataset import SampleRegistry
from src.downstream import analyze_sample
from src.preprocessing import load_sample_normalized
from src.splits import get_split


def main():
    parser = argparse.ArgumentParser(description="Downstream analysis")
    parser.add_argument("--source", choices=["gt", "predictions"], default="gt",
                        help="Analyze ground truth or model predictions")
    parser.add_argument("--strategy", default="strategy1",
                        help="Split strategy (for selecting samples)")
    parser.add_argument("--subset", default="test",
                        help="Which split subset to analyze")
    parser.add_argument("--pred-dir", type=str, default=None,
                        help="Directory with prediction NPZ files (for --source predictions)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    registry = SampleRegistry()
    split = get_split(args.strategy, registry, seed=args.seed)
    samples = split[args.subset]
    print(f"Analyzing {len(samples)} samples from {args.subset} set ({args.strategy})")

    results = []
    for sample in samples:
        img = load_sample_normalized(sample)
        h, w = img.shape[:2]

        if args.source == "gt":
            ann = load_sample_annotations(sample, h, w)
            masks = ann["masks"]
            labels = ann["labels"]
        else:
            pred_path = Path(args.pred_dir) / f"{sample.uid}.npz"
            if not pred_path.exists():
                print(f"  Skipping {sample.uid}: no prediction file")
                continue
            data = np.load(pred_path)
            masks = data["masks"]
            labels = data["labels"]

        result = analyze_sample(img, masks, labels, sample_id=sample.uid)
        result["species"] = sample.species
        result["microscope"] = sample.microscope
        result["experiment"] = sample.experiment

        # Flatten intensity dicts
        for region in ["endodermis_intensity", "vascular_intensity"]:
            for ch, val in result[region].items():
                result[f"{region}_{ch}"] = val
            del result[region]

        results.append(result)

    df = pd.DataFrame(results)

    # Save
    if args.output:
        out_path = Path(args.output)
    else:
        out_dir = OUTPUT_DIR / "downstream"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{args.source}_{args.strategy}_{args.subset}.csv"

    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"  Samples analyzed: {len(df)}")
    print(f"  Aerenchyma ratio: {df['aerenchyma_ratio'].mean():.4f} +/- {df['aerenchyma_ratio'].std():.4f}")
    print(f"  Aerenchyma count: {df['aerenchyma_count'].mean():.1f} +/- {df['aerenchyma_count'].std():.1f}")

    # Per species
    print(f"\nPer Species:")
    for sp, group in df.groupby("species"):
        print(f"  {sp}: ratio={group['aerenchyma_ratio'].mean():.4f}, "
              f"count={group['aerenchyma_count'].mean():.1f}")


if __name__ == "__main__":
    main()
