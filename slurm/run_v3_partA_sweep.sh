#!/bin/bash
#SBATCH --job-name=v3_partA_sweep
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/v3_partA_sweep_%j.out
#SBATCH --error=logs/v3_partA_sweep_%j.err

# RC-GNN V3 with Part A improvements sweep
# Part A: λ_inv=10.0, TopK-F1 early stopping, patience=15
#
# This sweep tests the hypothesis that stronger invariance penalty
# helps identify causal (not just correlated) edges, especially
# on multi-regime datasets.

echo "=============================================="
echo "RC-GNN V3 Part A Sweep"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo ""
echo "Part A improvements:"
echo "  - λ_inv = 10.0 (100x stronger invariance)"
echo "  - Early stopping on TopK-F1 (causal metric)"
echo "  - Patience = 15 epochs"
echo "=============================================="

# Load modules
module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.0.0

cd ~/rcgnn

# Activate environment
source ~/.venv/rcgnn/bin/activate

# Datasets to test (prioritize multi-regime)
DATASETS=(
    "extreme"      # 5 regimes - should benefit most
    "severe"       # 3 regimes
    "moderate"     # 2 regimes
    "mild"         # 1 regime
    "clean_full"   # 1 regime (baseline)
    "clean_real"   # 1 regime
    "mcar_20"      # MCAR corruption
    "mcar_30"
    "mcar_40"
    "mar_light"    # MAR corruption
    "mar_moderate"
    "mar_heavy"
    "mnar_light"   # MNAR corruption
    "mnar_moderate"
    "mnar_heavy"
    "combined_light"
    "combined_moderate"
    "combined_heavy"
    "temporal_drift_slow"
    "temporal_drift_fast"
)

echo ""
echo "Testing ${#DATASETS[@]} datasets with Part A improvements..."
echo ""

for ds in "${DATASETS[@]}"; do
    data_dir="data/interim/uci_air_c/${ds}"
    out_dir="artifacts/v3_partA/${ds}"
    
    if [ ! -d "$data_dir" ]; then
        echo "SKIP: $ds (directory not found)"
        continue
    fi
    
    echo "=============================================="
    echo "=== Testing: $ds ==="
    echo "=============================================="
    
    python scripts/train_rcgnn_v3.py \
        --data_dir "$data_dir" \
        --output_dir "$out_dir" \
        --epochs 50 \
        --batch_size 64 \
        --lr 1e-3 \
        --lambda_inv 10.0 \
        --early_stop_metric topk_f1 \
        --early_stop_patience 15 \
        --device cuda
    
    echo ""
done

echo "=============================================="
echo "Part A Sweep Complete!"
echo "Finished: $(date)"
echo "=============================================="

# Summary of results
echo ""
echo "=== RESULTS SUMMARY ==="
for ds in "${DATASETS[@]}"; do
    metrics_file="artifacts/v3_partA/${ds}/final_metrics.json"
    if [ -f "$metrics_file" ]; then
        topk=$(python -c "import json; m=json.load(open('$metrics_file')); print(f\"{m['topk_f1']:.4f}\")")
        best_ep=$(python -c "import json; m=json.load(open('$metrics_file')); print(m['best_epoch'])")
        echo "$ds: TopK-F1=$topk (epoch $best_ep)"
    fi
done
