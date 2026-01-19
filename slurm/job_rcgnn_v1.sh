#!/bin/bash
#SBATCH --job-name=rcgnn_v1
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/rcgnn_v1_%j.out

# RC-GNN v1 Training on UCI Air Data
# Uses: TriLatentEncoder + StructureLearner (original architecture)

# Load CUDA-enabled PyTorch module (NOT the CPU-only version!)
module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

cd /home/aoo29179/rcgnn

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

echo "=========================================="
echo "RC-GNN v1 Training"
echo "=========================================="
echo "Start time: $(date)"

# Verify Python is available
which python
python --version

nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

# =================================================================
# GRADIENT FIX RUN: A_base initialized near 0, tau=1.0
# =================================================================
# Root cause found: sigmoid(-3/0.25) = sigmoid(-12) ≈ 6e-6
# → Gradient vanishes in saturation zone!
#
# Fix applied:
#   - A_base initialized with randn*0.1 (near 0, not -3.0)
#   - tau starts at 1.0 (sigmoid(0/1) = 0.5 has gradient 0.25)
#   - tau anneals to 0.5 during training
#   - Warmup: λ_sparse=0, λ_acyclic=0 for first 50 epochs
# =================================================================
python scripts/train_rcgnn_v1.py \
    --data_dir data/interim/uci_air \
    --epochs 100 \
    --lr 0.001 \
    --batch_size 32 \
    --latent_dim 32 \
    --hidden_dim 64 \
    --lambda_sparse 0.01 \
    --lambda_acyclic 0.1 \
    --acyclic_warmup 50 \
    --lambda_gate 0.1 \
    --lambda_entropy 0.001 \
    --gate_lr_factor 0.1 \
    --device cuda \
    --seed 42

echo ""
echo "=========================================="
echo "RC-GNN v1 Training Complete"
echo "=========================================="
echo "End time: $(date)"
