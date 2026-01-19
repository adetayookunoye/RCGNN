#!/bin/bash
#SBATCH --job-name=rcgnn_lo
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/rcgnn_v2_low_acyclic_%j.out
#SBATCH --error=logs/rcgnn_v2_low_acyclic_%j.err

echo "======================================="
echo "Job ID: $SLURM_JOB_ID"
echo "LOW ACYCLICITY: gamma=0.01 (10x lower)"
echo "Start time: $(date)"
echo "======================================="

cd ~/rcgnn
module load PyTorch/2.1.2-foss-2023a
export PYTHONUNBUFFERED=1

python -u scripts/train_rcgnn_v2.py \
    --epochs 50 \
    --batch_size 64 \
    --lr 0.001 \
    --lambda_sparse 0.0001 \
    --gamma_acyclic 0.01 \
    --lambda_disentangle 0.1 \
    --lambda_recon 1.0 \
    --data_dir data/interim/uci_air \
    --seed 42

echo "======================================="
echo "End time: $(date)"
echo "======================================="
