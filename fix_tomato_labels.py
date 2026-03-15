"""Fix tomato annotation class indices.

Simultaneous remapping to avoid double-mapping:
  0 -> 3  (was showing as Whole Root, should be Inner Endodermis)
  1 -> 5  (was showing as Aerenchyma, should be Inner Exodermis)
  2 -> 2  (Outer Endodermis, unchanged)
  3 -> 4  (was showing as Inner Endodermis, should be Outer Exodermis)
  4 -> 0  (was showing as class 4, should be Whole Root)
"""

import glob
import os

ANNOTATION_DIR = os.path.join("data", "annotation")
REMAP = {0: 3, 1: 5, 3: 4, 4: 0}  # 2 stays the same

files = sorted(glob.glob(os.path.join(ANNOTATION_DIR, "Tomato_*.txt")))
print(f"Found {len(files)} tomato annotation files")

for fpath in files:
    with open(fpath, "r") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if not parts:
            new_lines.append(line)
            continue
        old_cls = int(parts[0])
        new_cls = REMAP.get(old_cls, old_cls)
        parts[0] = str(new_cls)
        new_lines.append(" ".join(parts) + "\n")

    with open(fpath, "w") as f:
        f.writelines(new_lines)

print("Remapping complete. Verifying...")

# Verify class distribution
from collections import Counter
counter = Counter()
for fpath in files:
    with open(fpath, "r") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                counter[int(parts[0])] += 1

print("Class distribution after fix:")
for cls in sorted(counter):
    print(f"  Class {cls}: {counter[cls]} polygons")
