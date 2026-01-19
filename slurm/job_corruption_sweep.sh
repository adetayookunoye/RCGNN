#!/bin/bash
#SBATCH --job-name=corruption_sweep
#SBATCH --partition=scavenge_p
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=corruption_sweep_%j.out

# PROPER CORRUPTION SWEEP: Train at each level with 5 seeds
# This will take ~20-30 hours total

module load Miniconda3/24.1.2-0
source activate /home/adetayo/.conda/envs/rcgnn

cd /home/adetayo/Documents/CSCI\ Forms/Adetayo\ Research/Robust\ Causal\ Graph\ Neural\ Networks\ under\ Compound\ Sensor\ Corruptions/rcgnn

echo "=========================================="
echo "CORRUPTION SWEEP: Training at all levels"
echo "=========================================="
date
echo ""

# Step 1: Generate corrupted datasets
echo "Step 1: Generating corrupted datasets..."
python scripts/generate_corrupted_datasets.py
echo ""

# Corruption levels and seeds
CORRUPTION_RATES=(0.0 0.10 0.20 0.30 0.40)
SEEDS=(1 2 3 4 5)

# Step 2: Train RC-GNN at each corruption level
echo "Step 2: Training RC-GNN..."
for corr in "${CORRUPTION_RATES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo "----------------------------------------"
        echo "RC-GNN: corruption=${corr}, seed=${seed}"
        date
        python scripts/train_corruption_sweep.py \
            --corruption ${corr} \
            --seed ${seed} \
            --epochs 40 \
            --device cpu
        echo ""
    done
done

# Step 3: Train NOTEARS at each corruption level
echo "Step 3: Training NOTEARS baseline..."
for corr in "${CORRUPTION_RATES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo "----------------------------------------"
        echo "NOTEARS: corruption=${corr}, seed=${seed}"
        date
        python scripts/train_notears_sweep.py \
            --corruption ${corr} \
            --seed ${seed}
        echo ""
    done
done

echo "=========================================="
echo "CORRUPTION SWEEP COMPLETE"
echo "=========================================="
date
echo ""
echo "Results saved to: artifacts/corruption_sweep/"
echo ""
echo "Next: Run analysis script to compute statistics"
