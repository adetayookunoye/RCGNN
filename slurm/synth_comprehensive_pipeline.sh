#!/bin/bash
#SBATCH --job-name=rcgnn_synth_full
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --time=5-00:00:00
#SBATCH --output=logs/synth_comprehensive_%j.out
#SBATCH --error=logs/synth_comprehensive_%j.err
#SBATCH --mail-type=END,FAIL

# ============================================================================
# COMPREHENSIVE SYNTHETIC SEM BENCHMARK PIPELINE
# ============================================================================
# Addresses ALL 5 reviewer concerns for causal validity:
#   1. Multiple seeds (5 seeds, mean ± std)
#   2. Harder graphs (scale-free with hubs, d=13/20/50)
#   3. Stronger corruption (40% MNAR, high noise, high bias)
#   4. Mechanism diversity (linear and MLP)
#   5. K-robustness (report F1 for K ≠ true edges)
#
# Total configurations: 8 × 5 seeds = 40 training runs
# Estimated time: ~20 hours (300 epochs × 40 models)
# ============================================================================

echo "======================================================"
echo " COMPREHENSIVE SYNTHETIC SEM BENCHMARK"
echo " Job ID: $SLURM_JOB_ID"
echo " Start time: $(date)"
echo "======================================================"

cd /scratch/aoo29179/rcgnn

module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
source ~/.bashrc

which python
python --version

mkdir -p logs artifacts

# Configuration
BASE_DIR="data/interim/synth_comprehensive"
ARTIFACTS_DIR="artifacts/synth_comprehensive"

# ============================================================================
# STAGE 1: GENERATE ALL DATASETS
# ============================================================================
echo ""
echo "======================================================"
echo " STAGE 1: Generate All Synthetic SEM Datasets"
echo " (8 configurations × 5 seeds = 40 datasets)"
echo "======================================================"

python scripts/synthetic_sem_comprehensive.py \
    --mode generate \
    --base_dir $BASE_DIR

if [ $? -ne 0 ]; then
    echo "[FAIL] Data generation failed!"
    exit 1
fi

echo "[OK] All datasets generated"

# ============================================================================
# STAGE 2: TRAIN RC-GNN ON ALL DATASETS
# ============================================================================
echo ""
echo "======================================================"
echo " STAGE 2: Train RC-GNN on All Datasets"
echo " (40 models × 300 epochs)"
echo "======================================================"

# Train each dataset one by one
for dataset_dir in $BASE_DIR/*/; do
    dataset_name=$(basename $dataset_dir)
    output_dir="$ARTIFACTS_DIR/$dataset_name"
    
    echo ""
    echo "------------------------------------------------------"
    echo " Training: $dataset_name"
    echo " $(date)"
    echo "------------------------------------------------------"
    
    # Extract seed from dataset name
    SEED=$(echo $dataset_name | grep -oP 'seed\K\d+')
    
    python scripts/train_rcgnn_unified.py \
        --data_dir "$dataset_dir" \
        --output_dir "$output_dir" \
        --epochs 300 \
        --seed ${SEED:-42}
    
    if [ $? -ne 0 ]; then
        echo "[WARN] Training failed for $dataset_name, continuing..."
    else
        echo "[OK] Trained $dataset_name"
    fi
done

# ============================================================================
# STAGE 3: EVALUATE WITH K-ROBUSTNESS
# ============================================================================
echo ""
echo "======================================================"
echo " STAGE 3: Evaluate with K-Robustness Analysis"
echo " K values: 0.5×, 0.75×, 1×, 1.25×, 1.5×, 2×"
echo "======================================================"

python scripts/synthetic_sem_comprehensive.py \
    --mode evaluate \
    --base_dir $BASE_DIR \
    --artifacts_dir $ARTIFACTS_DIR

# ============================================================================
# STAGE 4: GENERATE SUMMARY TABLES
# ============================================================================
echo ""
echo "======================================================"
echo " STAGE 4: Generate Summary Tables"
echo "======================================================"

python scripts/synthetic_sem_comprehensive.py \
    --mode summary \
    --artifacts_dir $ARTIFACTS_DIR

# ============================================================================
# DONE
# ============================================================================
echo ""
echo "======================================================"
echo " COMPREHENSIVE BENCHMARK COMPLETE"
echo " End time: $(date)"
echo "======================================================"
echo ""
echo "Results saved to:"
echo "  $ARTIFACTS_DIR/k_robustness_results.json"
echo ""
echo "Key tables for paper:"
echo "  1. Skeleton F1 across K values (K-robustness)"
echo "  2. Directed F1 at K=true_edges (main results)"
echo "  3. All metrics with mean ± std across 5 seeds"
