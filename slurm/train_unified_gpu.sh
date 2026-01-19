#!/bin/bash
#SBATCH --job-name=rcgnn_unified
#SBATCH --partition=gpu_p
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/rcgnn_unified_%j.out
#SBATCH --error=logs/rcgnn_unified_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aoo29179@uga.edu

echo "=========================================="
echo "RC-GNN Unified Training (GPU)"
echo "=========================================="
echo "Start time: $(date)"
echo "Job ID: $SLURM_JOB_ID"

# Load modules
module load Python/3.11.3-GCCcore-12.3.0

# Environment
cd /home/aoo29179/rcgnn
source ~/.bashrc

# Check environment
which python
python --version
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo ""
echo "======================================================"
echo "              UNIFIED TRAINING                        "
echo "======================================================"
echo "Device: cuda"
echo "Data: data/interim/uci_air"
echo "Epochs: 100"
echo "Key features:"
echo "  - 3-stage training: discovery → pruning → refinement"
echo "  - Publication-quality fixes"
echo "  - Gradient stability"
echo "  - Comprehensive metrics (TopK-F1, Best-F1, AUC-F1)"
echo "======================================================"

# Run Unified training
python scripts/train_rcgnn_unified.py \
    --data_dir data/interim/uci_air \
    --output_dir artifacts/unified \
    --epochs 100 \
    --lr 5e-4 \
    --batch_size 32 \
    --latent_dim 32 \
    --hidden_dim 64 \
    --lambda_recon 100.0 \
    --lambda_sparse 1e-4 \
    --target_edges 13 \
    --patience 20 \
    --device cuda

echo ""
echo "=========================================="
echo "Unified Training Complete"
echo "=========================================="
echo "End time: $(date)"
