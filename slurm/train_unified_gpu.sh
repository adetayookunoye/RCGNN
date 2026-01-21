#!/bin/bash
#SBATCH --job-name=rcgnn_4ds
#SBATCH --partition=gpu_p
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/rcgnn_unified_%j.out
#SBATCH --error=logs/rcgnn_unified_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aoo29179@uga.edu

echo "=========================================="
echo "RC-GNN Unified Training V8.3 (Relaxed Confidence)"
echo "=========================================="
echo "V8.3: Lowered confidence threshold 0.2→0.1"
echo "      Fixes soft graph rejection issue"
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
echo "        UNIFIED TRAINING V8.2 (Pareto Checkpointing)    "
echo "======================================================"
echo "V8.2 PARETO CHECKPOINTS (all subject to budget guard):"
echo "  - best_topk_sparse: max TopK-F1 in budget window"
echo "  - best_skel_sparse: max Skeleton-F1 in budget window"
echo "  - best_score: max composite score"
echo "Composite: topk + 0.5*skel + 0.5*dir_bonus - 0.5*budget_pen"
echo "Guard: past DISC, edge_sum ∈ [6.5, 19.5], @0.2 > 0"
echo "======================================================"

# Datasets to run (in priority order by signal gap)
# Dataset              Miss%  Env   Gap     CorrF1  TP/13  Rating
# compound_mnar_bias   25.3%   1   0.1533  0.2222   3     BEST
# compound_full        25.0%   3   0.0614  0.3704   5     GOOD
# extreme              40.0%   5   0.0140  0.3704   5     WEAK
# mcar_40              40.0%   1   0.0052  0.1538   2     WEAK

DATASETS=("compound_mnar_bias" "compound_full" "extreme" "mcar_40")

for CORRUPTION in "${DATASETS[@]}"; do
    echo ""
    echo "======================================================"
    echo "  RUNNING DATASET: ${CORRUPTION}"
    echo "======================================================"
    echo "Start: $(date)"
    
    python scripts/train_rcgnn_unified.py \
        --data_dir data/interim/uci_air_c/${CORRUPTION} \
        --output_dir artifacts/unified_v8_${CORRUPTION} \
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
echo "  ALL 4 DATASETS COMPLETE"
echo "======================================================"

echo ""
echo "=========================================="
echo "Unified Training V4 Complete"
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
echo "Details: artifacts/unified_v8_*/baselines/"
echo "End time: $(date)"
echo ""
echo "=========================================="
echo "FULL PIPELINE COMPLETE"
echo "  ✓ Training: 4 datasets"
echo "  ✓ Baselines: 6 methods × 4 datasets"
echo "  ✓ Summary: CSV report generated"
echo "=========================================="
