#!/bin/bash
#SBATCH --job-name=orient_posthoc
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/orient_posthoc_%j.out
#SBATCH --error=logs/orient_posthoc_%j.err

# Post-hoc skeleton orientation for V9.2.11 RC-GNN output
# Decouples skeleton learning (what RC-GNN does well) from orientation

cd /scratch/aoo29179/rcgnn
source activate rcgnn 2>/dev/null || conda activate rcgnn

echo "=== Post-hoc Orientation Pipeline ==="
echo "Job:  $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo ""

mkdir -p logs

# --- Run on h1_easy seed_0 (primary benchmark) ---
echo "=== h1_easy seed_0 ==="
python scripts/orient_skeleton.py \
  --artifacts-dir artifacts/table2a/h1_easy/seed_0 \
  --data-dir data/interim/table2a/h1_easy/seed_0 \
  --output artifacts/table2a/h1_easy/seed_0/orientation_results.json

echo ""
echo "=== Complete ==="
echo "Results: artifacts/table2a/h1_easy/seed_0/orientation_results.json"
cat artifacts/table2a/h1_easy/seed_0/orientation_results.json
