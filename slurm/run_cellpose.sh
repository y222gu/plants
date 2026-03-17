#!/bin/bash
# ============================================================================
# SLURM job script: Cellpose v3 — submit 5 per-class jobs + 1 evaluation job
#
# Usage:
#   bash slurm/run_cellpose.sh          # submits 5 training + 1 eval job
#   squeue -u $USER                     # monitor all jobs
#   tail -f logs/cellpose_class0_*.out  # follow one class
# ============================================================================

set -euo pipefail

CLASS_NAMES=("Whole_Root" "Aerenchyma" "Endodermis" "Vascular" "Exodermis")
JOB_IDS=()

for CLASS_ID in 0 1 2 3 4; do
    CLASS_NAME="${CLASS_NAMES[$CLASS_ID]}"
    echo "Submitting Cellpose class ${CLASS_ID} (${CLASS_NAME})..."

    JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=cp-${CLASS_NAME}
#SBATCH --partition=gpu-qi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100_80:1
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=logs/cellpose_class${CLASS_ID}_%j.out
#SBATCH --error=logs/cellpose_class${CLASS_ID}_%j.err

set -euo pipefail

module load conda3/4.13.0
eval "\$(conda shell.bash hook)"
conda activate plants

PROJECT_DIR="\$HOME/plants"
cd "\$PROJECT_DIR"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

echo "============================================"
echo "Job ID: \$SLURM_JOB_ID"
echo "Node:   \$(hostname)"
echo "Date:   \$(date)"
echo "Run:    Cellpose v3 — class ${CLASS_ID} (${CLASS_NAME})"
echo "============================================"

python -c "
import torch
print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)')
"

mkdir -p logs

stdbuf -oL python train/train_cellpose.py \
    --strategy A \
    --version 3 \
    --class-id ${CLASS_ID} \
    --num-classes 5 \
    --img-size 512 \
    --batch-size 16 \
    --epochs 150 \
    --lr 0.1 \
    2>&1 | tee "logs/cellpose_class${CLASS_ID}_\${SLURM_JOB_ID}.log"

echo ""
echo "Cellpose class ${CLASS_ID} (${CLASS_NAME}) complete: \$(date)"
EOF
)

    JOB_IDS+=("$JOB_ID")
    echo "  Submitted job ${JOB_ID}"
done

# Build dependency string: afterok:id1:id2:id3:id4:id5
DEP_STR=$(IFS=:; echo "${JOB_IDS[*]}")

echo ""
echo "Submitting Cellpose evaluation job (depends on all 5 training jobs)..."

EVAL_JOB_ID=$(sbatch --parsable --dependency=afterok:${DEP_STR} <<'EVALEOF'
#!/bin/bash
#SBATCH --job-name=cp-eval
#SBATCH --partition=gpu-qi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100_80:1
#SBATCH --mem=80G
#SBATCH --time=4:00:00
#SBATCH --output=logs/cellpose_eval_%j.out
#SBATCH --error=logs/cellpose_eval_%j.err

set -euo pipefail

module load conda3/4.13.0
eval "$(conda shell.bash hook)"
conda activate plants

PROJECT_DIR="$HOME/plants"
cd "$PROJECT_DIR"

export PYTHONUNBUFFERED=1

echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "Date:   $(date)"
echo "Run:    Cellpose evaluation (all 5 classes)"
echo "============================================"

stdbuf -oL python evaluate.py \
    --model cellpose \
    --strategy A \
    --num-classes 5 \
    --checkpoint output/runs/cellpose/ \
    --no-vis \
    2>&1 | tee "logs/cellpose_eval_${SLURM_JOB_ID}.log"

echo ""
echo "Cellpose evaluation complete: $(date)"
EVALEOF
)

echo "  Submitted eval job ${EVAL_JOB_ID} (depends on ${DEP_STR})"
echo ""
echo "All 6 jobs submitted (5 training + 1 evaluation). Use 'squeue -u \$USER' to monitor."
