#!/bin/bash
# ============================================================================
# Convenience script: Submit all 5 benchmark runs as separate SLURM jobs
#
# Usage:
#   bash slurm/run_benchmark.sh              # Submit all 5 real runs
#   bash slurm/run_benchmark.sh --test       # Submit all 5 test runs (2 epochs)
# ============================================================================

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs

if [ "${1:-}" = "--test" ]; then
    echo "Submitting 5 TEST runs (2 epochs each, 1 GPU each)..."
    echo ""
    for script in slurm/run_{1_yolo,2_unet_4c,3_unet_5c,sam,cellpose}_test.sh; do
        if [ -f "$script" ]; then
            JOB=$(sbatch "$script" | awk '{print $4}')
            echo "  Submitted $(basename $script): Job $JOB"
        else
            echo "  WARNING: $script not found, skipping"
        fi
    done
else
    echo "Submitting 5 REAL runs (1 GPU each)..."
    echo ""
    JOB1=$(sbatch slurm/run_1_yolo.sh | awk '{print $4}')
    echo "  Run 1 (YOLO):        Job $JOB1"
    JOB2=$(sbatch slurm/run_2_unet_4c.sh | awk '{print $4}')
    echo "  Run 2 (U-Net++ 4c):  Job $JOB2"
    JOB3=$(sbatch slurm/run_3_unet_5c.sh | awk '{print $4}')
    echo "  Run 3 (U-Net++ 5c):  Job $JOB3"
    JOB4=$(sbatch slurm/run_sam.sh | awk '{print $4}')
    echo "  Run 4 (SAM):         Job $JOB4"
    JOB5=$(sbatch slurm/run_cellpose.sh | awk '{print $4}')
    echo "  Run 5 (Cellpose):    Job $JOB5"
fi

echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f logs/<model>_<JOBID>.out"
