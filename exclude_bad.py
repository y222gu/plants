import os
import shutil

base_dir = r"C:\Users\Yifei\Documents\plants"
data_dir = os.path.join(base_dir, "data")
image_dir = os.path.join(data_dir, "image")
annotation_dir = os.path.join(data_dir, "annotation")
bad_dir = os.path.join(base_dir, "preview", "bad")
excluded_dir = os.path.join(base_dir, "data_excluded")

# Build mapping: flattened name -> relative path for all leaf image directories.
# A leaf directory has no subdirectories (only files).
# Flattened name = path components joined by "_", e.g.
#   Sorghum/Olympus/21/21_2_01 -> Sorghum_Olympus_21_21_2_01
leaf_map = {}
for dirpath, dirnames, filenames in os.walk(image_dir):
    if not dirnames and filenames:  # leaf directory
        rel_path = os.path.relpath(dirpath, image_dir)
        parts = rel_path.split(os.sep)
        flat_name = "_".join(parts)
        leaf_map[flat_name] = rel_path

# Process each bad preview
moved = 0
errors = []

for filename in sorted(os.listdir(bad_dir)):
    if not filename.endswith("_preview.png"):
        continue
    base_name = filename[: -len("_preview.png")]

    if base_name not in leaf_map:
        errors.append(f"No image directory found for: {base_name}")
        continue

    rel_path = leaf_map[base_name]
    src_image = os.path.join(image_dir, rel_path)
    src_annot = os.path.join(annotation_dir, base_name + ".txt")

    dst_image = os.path.join(excluded_dir, "image", rel_path)
    dst_annot = os.path.join(excluded_dir, "annotation", base_name + ".txt")

    # Move image directory
    if os.path.isdir(src_image):
        os.makedirs(os.path.dirname(dst_image), exist_ok=True)
        shutil.move(src_image, dst_image)
    else:
        errors.append(f"Image dir missing: {src_image}")

    # Move annotation file
    if os.path.isfile(src_annot):
        os.makedirs(os.path.dirname(dst_annot), exist_ok=True)
        shutil.move(src_annot, dst_annot)
    else:
        errors.append(f"Annotation file missing: {src_annot}")

    moved += 1
    print(f"Moved: {base_name}")

print(f"\nTotal moved: {moved}")
if errors:
    print(f"\nErrors ({len(errors)}):")
    for e in errors:
        print(f"  {e}")
