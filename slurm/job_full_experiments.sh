#!/bin/bash
#SBATCH --job-name=full_experiments
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:A100:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/full_experiments_%j.out

# COMPLETE EXPERIMENT SUITE (4x GPU DDP)
# Runs: Ablation Study → Stability Analysis → Corruption Sweep
# Total estimated time: 2-3 hours with 4 GPUs

# Load CUDA-enabled PyTorch module (NOT conda - no env exists)
module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

cd /home/aoo29179/rcgnn

# Force unbuffered Python output to see progress in real-time
export PYTHONUNBUFFERED=1

# DDP settings
export MASTER_ADDR=localhost
export MASTER_PORT=12355
NUM_GPUS=4

# Verify Python and GPUs are available
which python
python --version
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
nvidia-smi --query-gpu=name,memory.total --format=csv

echo "=========================================="
echo "FULL EXPERIMENT SUITE (4x GPU DDP)"
echo "=========================================="
echo "Start time: $(date)"
echo ""
echo "Experiments:"
echo "  1. Ablation Study (3 configs)"
echo "  2. Stability Analysis (5 seeds)"
echo "  3. Corruption Sweep (5 levels × 5 seeds × 2 methods)"
echo ""
echo "=========================================="
echo ""

# ==========================================
# EXPERIMENT 1: ABLATION STUDY
# ==========================================
echo "=========================================="
echo "EXPERIMENT 1: ABLATION STUDY"
echo "=========================================="
date
echo ""

DATA_DIR="data/interim/uci_air"
EPOCHS=40

# Config 1: Full model
echo "----------------------------------------"
echo "Config 1: Full RC-GNN model"
date
torchrun --nproc_per_node=${NUM_GPUS} scripts/train_rcgnn_v2.py \
    --data_dir ${DATA_DIR} \
    --epochs ${EPOCHS} \
    --lambda_disentangle 0.1 \
    --lambda_sparse 0.0001 \
    --seed 42 \
    --ddp \
    --output_suffix "_ablation_full"
echo ""

# Config 2: Without disentanglement
echo "----------------------------------------"
echo "Config 2: Without disentanglement"
date
torchrun --nproc_per_node=${NUM_GPUS} scripts/train_rcgnn_v2.py \
    --data_dir ${DATA_DIR} \
    --epochs ${EPOCHS} \
    --lambda_disentangle 0.0 \
    --lambda_sparse 0.0001 \
    --seed 42 \
    --ddp \
    --output_suffix "_ablation_no_disent"
echo ""

# Config 3: Without sparsity
echo "----------------------------------------"
echo "Config 3: Without sparsity penalty"
date
torchrun --nproc_per_node=${NUM_GPUS} scripts/train_rcgnn_v2.py \
    --data_dir ${DATA_DIR} \
    --epochs ${EPOCHS} \
    --lambda_disentangle 0.1 \
    --lambda_sparse 0.0 \
    --seed 42 \
    --ddp \
    --output_suffix "_ablation_no_sparse"
echo ""

echo "✓ Ablation study complete"
echo ""

# ==========================================
# EXPERIMENT 2: STABILITY ANALYSIS
# ==========================================
echo "=========================================="
echo "EXPERIMENT 2: STABILITY ANALYSIS"
echo "=========================================="
date
echo ""

SEEDS=(1 2 3 4 5)

for seed in "${SEEDS[@]}"; do
    echo "----------------------------------------"
    echo "Training with seed=${seed}"
    date
    torchrun --nproc_per_node=${NUM_GPUS} scripts/train_rcgnn_v2.py \
        --data_dir ${DATA_DIR} \
        --epochs ${EPOCHS} \
        --lambda_disentangle 0.1 \
        --lambda_sparse 0.0001 \
        --seed ${seed} \
        --ddp \
        --output_suffix "_stability_seed${seed}"
    echo ""
done

echo "✓ Stability analysis complete"
echo ""

# ==========================================
# EXPERIMENT 3: CORRUPTION SWEEP
# ==========================================
echo "=========================================="
echo "EXPERIMENT 3: CORRUPTION SWEEP"
echo "=========================================="
date
echo ""

# Step 1: Generate corrupted datasets
echo "Step 1: Generating corrupted datasets..."
python scripts/generate_corrupted_datasets.py
echo ""

# Corruption levels and seeds
CORRUPTION_RATES=(0.0 0.10 0.20 0.30 0.40)
SEEDS_CORRUPTION=(1 2 3 4 5)

# Step 2: Train RC-GNN at each corruption level
echo "Step 2: Training RC-GNN across corruption levels..."
for corr in "${CORRUPTION_RATES[@]}"; do
    for seed in "${SEEDS_CORRUPTION[@]}"; do
        echo "----------------------------------------"
        echo "RC-GNN: corruption=${corr}, seed=${seed}"
        date
        torchrun --nproc_per_node=${NUM_GPUS} scripts/train_corruption_sweep.py \
            --corruption ${corr} \
            --seed ${seed} \
            --epochs 40 \
            --ddp
        echo ""
    done
done

# Step 3: Train NOTEARS at each corruption level (CPU-based, no DDP needed)
echo "Step 3: Training NOTEARS baseline..."
for corr in "${CORRUPTION_RATES[@]}"; do
    for seed in "${SEEDS_CORRUPTION[@]}"; do
        echo "----------------------------------------"
        echo "NOTEARS: corruption=${corr}, seed=${seed}"
        date
        python scripts/train_notears_sweep.py \
            --corruption ${corr} \
            --seed ${seed}
        echo ""
    done
done

echo "✓ Corruption sweep complete"
echo ""

# ==========================================
# GENERATE ALL ANALYSIS REPORTS
# ==========================================
echo "=========================================="
echo "GENERATING ANALYSIS REPORTS"
echo "=========================================="
date
echo ""

# Analyze corruption sweep
echo "Analyzing corruption sweep results..."
python scripts/analyze_corruption_sweep.py
echo ""

echo "✓ All analysis complete"
echo ""

# ==========================================
# SUMMARY
# ==========================================
echo "=========================================="
echo "FULL EXPERIMENT SUITE COMPLETE"
echo "=========================================="
echo "End time: $(date)"
echo ""
echo "Results saved to:"
echo "  - Ablation: artifacts/checkpoints/*_ablation_*.pt"
echo "  - Stability: artifacts/checkpoints/*_stability_*.pt"
echo "  - Corruption: artifacts/corruption_sweep/"
echo ""
echo "Next steps:"
echo "  1. Review corruption sweep: cat artifacts/corruption_sweep/summary.json"
echo "  2. View degradation curve: open artifacts/corruption_sweep/degradation_curve.png"
echo "  3. Compare ablation checkpoints manually"
echo "  4. Compute stability statistics from seed runs"
echo ""
echo "=========================================="
