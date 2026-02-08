"""Run all training jobs sequentially.

Order:
  1. Cellpose v3 - Strategy 2
  2. Cellpose v3 - Strategy 1
  3. SAM - Strategy 2
  4. SAM - Strategy 1
  5. Cellpose v2 - Strategy 2
  6. Cellpose v2 - Strategy 1
"""

import os
import subprocess
import sys
import time

# Force PyTorch to use the NVIDIA GPU (not integrated AMD)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

JOBS = [
#    ("Cellpose v3 - Strategy 2", [sys.executable, "train_cellpose.py", "--version", "3", "--strategy", "strategy2", "--all-classes"]),
#     ("Cellpose v3 - Strategy 1", [sys.executable, "train_cellpose.py", "--version", "3", "--strategy", "strategy1", "--all-classes"]),
    ("SAM - Strategy 2",         [sys.executable, "train_sam.py", "--strategy", "strategy2"]),
    ("SAM - Strategy 1",         [sys.executable, "train_sam.py", "--strategy", "strategy1"]),
    ("Cellpose v2 - Strategy 2", [sys.executable, "train_cellpose.py", "--version", "2", "--strategy", "strategy2", "--all-classes"]),
    ("Cellpose v2 - Strategy 1", [sys.executable, "train_cellpose.py", "--version", "2", "--strategy", "strategy1", "--all-classes"]),
]


def main():
    total = len(JOBS)
    failed = []

    for i, (name, cmd) in enumerate(JOBS, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{total}] {name}")
        print(f"{'='*60}\n")

        start = time.time()
        result = subprocess.run(cmd)
        elapsed = time.time() - start

        if result.returncode != 0:
            print(f"\nERROR: {name} failed with exit code {result.returncode} "
                  f"(after {elapsed:.0f}s)")
            failed.append(name)
            continue

        print(f"\nCompleted {name} in {elapsed:.0f}s")

    print(f"\n{'='*60}")
    if failed:
        print(f"Finished with {len(failed)} failure(s):")
        for f in failed:
            print(f"  - {f}")
    else:
        print("All training runs completed successfully!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
