#!/bin/bash
# ============================================================================
# One-time HPC setup script — run on login node
# Usage: bash slurm/setup_hpc.sh
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_NAME="plants"

echo "============================================"
echo "HPC Setup for Plant Root Segmentation"
echo "Project: $PROJECT_DIR"
echo "============================================"

# ── 1. Create conda environment ─────────────────────────────────────────────
echo ""
echo "[1/5] Creating conda environment '$ENV_NAME'..."

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "  Environment '$ENV_NAME' already exists."
    echo "  To recreate: conda env remove -n $ENV_NAME && bash $0"
else
    # Prefer mamba for faster solves
    if command -v mamba &>/dev/null; then
        echo "  Using mamba..."
        mamba env create -f "$PROJECT_DIR/environment.yml"
    else
        echo "  Using conda (mamba not found — this may be slow)..."
        conda env create -f "$PROJECT_DIR/environment.yml"
    fi
    echo "  Environment created."
fi

# Activate for remaining setup steps
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
echo "  Activated: $(python --version), torch=$(python -c 'import torch; print(torch.__version__)')"

# ── 2. Verify imports ────────────────────────────────────────────────────────
echo ""
echo "[2/5] Verifying Python imports..."

python -c "
import sys
packages = [
    ('torch', 'torch'),
    ('torchvision', 'torchvision'),
    ('ultralytics', 'ultralytics'),
    ('segmentation_models_pytorch', 'segmentation-models-pytorch'),
    ('pytorch_lightning', 'pytorch-lightning'),
    ('segment_anything', 'segment-anything'),
    ('cellpose', 'cellpose'),
    ('albumentations', 'albumentations'),
    ('cv2', 'opencv'),
    ('tifffile', 'tifffile'),
    ('pycocotools', 'pycocotools'),
    ('scipy', 'scipy'),
    ('pandas', 'pandas'),
    ('skimage', 'scikit-image'),
    ('matplotlib', 'matplotlib'),
    ('tqdm', 'tqdm'),
    ('yaml', 'pyyaml'),
    ('numpy', 'numpy'),
    ('PIL', 'pillow'),
]
failed = []
for module, name in packages:
    try:
        __import__(module)
        print(f'  OK: {name}')
    except ImportError as e:
        print(f'  FAIL: {name} ({e})')
        failed.append(name)

if failed:
    print(f'\nERROR: {len(failed)} package(s) failed to import: {failed}')
    sys.exit(1)
else:
    print(f'\nAll {len(packages)} packages imported successfully.')
"

# Check CUDA availability
python -c "
import torch
if torch.cuda.is_available():
    print(f'  CUDA: {torch.version.cuda}, GPUs: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'    GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('  WARNING: CUDA not available (expected on login node)')
"

# ── 3. Download SAM checkpoint ───────────────────────────────────────────────
echo ""
echo "[3/5] Downloading SAM checkpoint..."

SAM_DIR="$PROJECT_DIR/output/checkpoints"
SAM_FILE="$SAM_DIR/sam_vit_b_01ec64.pth"
mkdir -p "$SAM_DIR"

if [ -f "$SAM_FILE" ]; then
    echo "  SAM checkpoint already exists at $SAM_FILE"
else
    echo "  Downloading sam_vit_b_01ec64.pth (~375 MB)..."
    wget -q --show-progress -O "$SAM_FILE" \
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    echo "  Downloaded to $SAM_FILE"
fi

# ── 4. Pre-cache pretrained weights ─────────────────────────────────────────
echo ""
echo "[4/5] Pre-caching pretrained model weights (compute nodes may lack internet)..."

python -c "
print('  Caching YOLO11m-seg weights...')
from ultralytics import YOLO
model = YOLO('yolo11m-seg')
del model
print('    OK')

print('  Caching ResNet34 (ImageNet) weights...')
import segmentation_models_pytorch as smp
model = smp.UnetPlusPlus(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=4)
del model
print('    OK')

print('  Caching Cellpose cyto3 weights...')
from cellpose import models
model = models.CellposeModel(gpu=False, model_type='cyto3')
del model
print('    OK')

print('  All pretrained weights cached.')
"

# ── 5. Create directories and verify data ────────────────────────────────────
echo ""
echo "[5/5] Setting up directories and verifying data..."

mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$PROJECT_DIR/output/runs"
echo "  Created logs/ and output/runs/ directories"

# Check data directory
if [ -d "$PROJECT_DIR/data/image" ] && [ -d "$PROJECT_DIR/data/annotation" ]; then
    N_IMAGES=$(find "$PROJECT_DIR/data/image" -name "*.tif" | wc -l)
    N_ANNOTATIONS=$(find "$PROJECT_DIR/data/annotation" -name "*.txt" | wc -l)
    echo "  Data found: $N_IMAGES TIF images, $N_ANNOTATIONS annotations"
else
    echo "  WARNING: data/ directory not found or incomplete."
    echo "  Expected: data/image/ and data/annotation/"
    echo "  Create a symlink: ln -s /path/to/data $PROJECT_DIR/data"
fi

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Verify data: ls $PROJECT_DIR/data/image/"
echo "  2. Edit SLURM script: vim $PROJECT_DIR/slurm/run_benchmark.sh"
echo "     - Check partition name, module loads, email"
echo "  3. Submit: sbatch $PROJECT_DIR/slurm/run_benchmark.sh"
echo "============================================"
