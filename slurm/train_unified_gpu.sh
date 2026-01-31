#!/bin/bash
#SBATCH --job-name=rcgnn_12ds
#SBATCH --partition=gpu_p
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=32G
#SBATCH --time=36:00:00
#SBATCH --output=logs/rcgnn_unified_%j.out
#SBATCH --error=logs/rcgnn_unified_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aoo29179@uga.edu

echo "=========================================="
echo "RC-GNN Unified Training V9.0 (Full 12-Dataset Benchmark)"
echo "=========================================="
echo "V9.0: Training on all 12 curated corruption datasets"
echo "      Comprehensive robustness evaluation"
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
echo "        UNIFIED TRAINING V9.0 (Full Benchmark)        "
echo "======================================================"
echo "V9.0 PARETO CHECKPOINTS (all subject to budget guard):"
echo "  - best_topk_sparse: max TopK-F1 in budget window"
echo "  - best_skel_sparse: max Skeleton-F1 in budget window"
echo "  - best_score: max composite score"
echo "Composite: topk + 0.5*skel + 0.5*dir_bonus - 0.5*budget_pen"
echo "Guard: past DISC, edge_sum ∈ [6.5, 19.5], @0.2 > 0"
echo "======================================================"

# ============================================================================
# ALL 12 CURATED DATASETS
# ============================================================================
# Category         | Dataset              | Missing% | Regimes | Notes
# -----------------|----------------------|----------|---------|---------------
# BASELINES        | clean_full           | 0%       | 1       | Perfect data
#                  | clean_real           | 14.7%    | 1       | Natural UCI
# -----------------|----------------------|----------|---------|---------------
# TRAINED (V8)     | compound_full        | 25%      | 3       | All corruptions
#                  | compound_mnar_bias   | 25%      | 1       | MNAR + bias
#                  | extreme              | 40%      | 5       | Max corruption
#                  | mcar_40              | 40%      | 1       | High MCAR
# -----------------|----------------------|----------|---------|---------------
# NEW ABLATIONS    | mcar_20              | 20%      | 1       | Low MCAR
#                  | mnar_structural      | 20%      | 1       | Hardest MNAR
#                  | moderate             | 20%      | 2       | MAR corruption
#                  | noise_0.5            | 0%       | 1       | Noise only
#                  | regimes_5            | 0%       | 5       | Multi-regime
#                  | sensor_failure       | 25%      | 2       | Realistic
# ============================================================================

DATASETS=(
    # Baselines (train on clean to establish upper bound)
    "clean_full"
    "clean_real"
    # Previously trained (re-run for consistency)
    "compound_full"
    "compound_mnar_bias"
    "extreme"
    "mcar_40"
    # New ablation studies
    "mcar_20"
    "mnar_structural"
    "moderate"
    "noise_0.5"
    "regimes_5"
    "sensor_failure"
)

for CORRUPTION in "${DATASETS[@]}"; do
    echo ""
    echo "======================================================"
    echo "  RUNNING DATASET: ${CORRUPTION}"
    echo "======================================================"
    echo "Start: $(date)"
    
    python scripts/train_rcgnn_unified.py \
        --data_dir data/interim/uci_air_c/${CORRUPTION} \
        --output_dir artifacts/unified_v9_${CORRUPTION} \
        --epochs 150 \
        --lr 5e-4 \
        --batch_size 32 \
        --latent_dim 32 \
        --hidden_dim 64 \
        --lambda_recon 1.0 \
        --lambda_causal 1.0 \
        --target_edges 13 \
        --patience 40 \
        --device cuda
    
    echo "Finished ${CORRUPTION}: $(date)"
    echo ""
done

echo ""
echo "======================================================"
echo "  ALL 12 DATASETS COMPLETE"
echo "======================================================"

echo ""
echo "=========================================="
echo "Unified Training V9.0 Complete"
echo "=========================================="
echo "End time: $(date)"

echo ""
echo "======================================================"
echo "       PHASE 2: BASELINE COMPARISON (Auto-Start)"
echo "======================================================"
echo "Running comprehensive evaluation on trained artifacts"
echo "Start time: $(date)"
echo ""

# Run comprehensive evaluation with all baselines
# Capture output to both console and text file
python scripts/comprehensive_evaluation.py \
    --artifacts-dir artifacts \
    --data-dir data/interim \
    --output artifacts/evaluation_report_final.json \
    2>&1 | tee artifacts/COMPREHENSIVE_EVALUATION.txt

echo ""
echo "======================================================"
echo "       COMPREHENSIVE EVALUATION COMPLETE"
echo "======================================================"
echo "Output: artifacts/evaluation_report_final.json"
echo "End time: $(date)"

echo ""
echo "======================================================"
echo "       ALL BASELINES ANALYSIS COMPLETE"
echo "======================================================"
echo "Summary: artifacts/baseline_comparison_summary.csv"
echo "Details: artifacts/unified_v9_*/baselines/"
echo "End time: $(date)"
echo ""
echo "=========================================="
echo "FULL PIPELINE COMPLETE"
echo "  ✓ Training: 12 datasets"
echo "  ✓ Baselines: 6 methods × 12 datasets"
echo "  ✓ Summary: CSV report generated"
echo "=========================================="
