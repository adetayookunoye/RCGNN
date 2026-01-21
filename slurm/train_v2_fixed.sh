#!/bin/bash
#SBATCH --job-name=rcgnn_v2
#SBATCH --partition=gpu_p
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/rcgnn_v2_%j.out
#SBATCH --error=logs/rcgnn_v2_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aoo29179@uga.edu

echo "=========================================="
echo "RC-GNN V2 Training (Fixed Schedule)"
echo "=========================================="
echo "Start time: $(date)"
echo "Job ID: $SLURM_JOB_ID"

# Load modules
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

# Environment
cd /scratch/aoo29179/rcgnn
source ~/.bashrc

# Check environment
which python
python --version
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo ""
echo "======================================================"
echo "              V2 TRAINING (Fixed Schedule)            "
echo "======================================================"
echo "CRITICAL FIXES APPLIED:"
echo "  1. Temperature annealing: 2.0 → 0.5"
echo "  2. Gradual sparsity ramp (not cliff)"
echo "  3. Proper health check (sparse = healthy)"
echo "  4. Using regimes_3 for invariance identification"
echo "  5. Multi-K evaluation (K=13,20,30)"
echo "======================================================"

# Run V2 training with MULTI-REGIME dataset (Step 4 fix)
python scripts/train_rcgnn_unified.py \
    --data_dir data/interim/uci_air_c/regimes_3 \
    --output_dir artifacts/v2_regimes3 \
    --epochs 150 \
    --lr 5e-4 \
    --batch_size 32 \
    --latent_dim 32 \
    --hidden_dim 64 \
    --lambda_recon 50.0 \
    --lambda_sparse 0.01 \
    --target_edges 13 \
    --patience 30 \
    --device cuda

echo ""
echo "=========================================="
echo "V2 Training Complete"
echo "=========================================="
echo "End time: $(date)"
