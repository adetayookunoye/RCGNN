#!/bin/bash
#SBATCH --job-name=table2_datagen
#SBATCH --partition=batch_p
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=logs/table2_datagen_%j.out
#SBATCH --error=logs/table2_datagen_%j.err
#SBATCH --mail-type=END,FAIL

# ============================================================================
# TABLE 2 DATASET GENERATION (Table 2A + Table 2B)
# ============================================================================
# Purpose: Generate all datasets required for Table 2 in SPIE 2026 paper
#
# Table 2A - Hypothesis Benchmarks (H1/H2/H3):
#   h1_easy, h1_medium, h1_hard, h2_multi_env, h2_stability, h3_policy
#   (6 benchmarks × 5 seeds = 30 datasets)
#
# Table 2B - SEM Benchmark Grid:
#   er_d13_lin, er_d13_mlp, er_d20_lin, er_d20_mlp, er_d50_mlp,
#   sf_d13_mlp, sf_d20_mlp, sf_d13_hard
#   (8 configs × 5 seeds = 40 datasets)
#
# Total: 70 datasets
#
# Output structure:
#   data/interim/table2a/{benchmark}/seed_{seed}/
#   data/interim/sem_table2/{config}/seed_{seed}/
# ============================================================================

echo "======================================================"
echo " Table 2 Dataset Generation"
echo " Job ID: $SLURM_JOB_ID"
echo " Start time: $(date)"
echo "======================================================"

# ============================================================================
# SETUP
# ============================================================================

# Change to project directory (adjust for your cluster)
cd /scratch/aoo29179/rcgnn || cd $SLURM_SUBMIT_DIR || exit 1

# Load modules (adjust for your cluster)
module load Python/3.11.3-GCCcore-12.3.0 2>/dev/null || true
source ~/.bashrc 2>/dev/null || true

# Verify Python
echo ""
echo "Python environment:"
which python
python --version

# Create directories
mkdir -p logs
mkdir -p data/interim/table2a
mkdir -p data/interim/sem_table2

# ============================================================================
# TABLE 2A: H1/H2/H3 HYPOTHESIS BENCHMARKS (6 benchmarks × 5 seeds = 30 datasets)
# ============================================================================

echo ""
echo "======================================================"
echo " TABLE 2A: Hypothesis Benchmarks (H1/H2/H3)"
echo " 6 benchmarks × 5 seeds = 30 datasets"
echo "======================================================"

TABLE2A_BENCHMARKS="h1_easy h1_medium h1_hard h2_multi_env h2_stability h3_policy"
TABLE2A_SEEDS="0 1 2 3 4"

for benchmark in $TABLE2A_BENCHMARKS; do
    for seed in $TABLE2A_SEEDS; do
        echo ""
        echo "[Table 2A] Generating: $benchmark seed=$seed"
        echo "------------------------------------------------------"
        
        python scripts/synth_corruption_benchmark.py \
            --benchmark $benchmark \
            --seed $seed \
            --output data/interim/table2a/${benchmark}/seed_${seed}
        
        if [ $? -eq 0 ]; then
            echo "[OK] $benchmark seed=$seed generated successfully"
        else
            echo "[FAIL] $benchmark seed=$seed generation failed!"
        fi
    done
done

echo ""
echo "[Table 2A] Complete: 6 benchmarks × 5 seeds = 30 datasets generated"

# ============================================================================
# TABLE 2B: SEM BENCHMARK GRID (8 configs × 5 seeds = 40 datasets)
# ============================================================================

echo ""
echo "======================================================"
echo " TABLE 2B: SEM Benchmark Grid"
echo "======================================================"

TABLE2B_CONFIGS="er_d13_lin er_d13_mlp er_d20_lin er_d20_mlp er_d50_mlp sf_d13_mlp sf_d20_mlp sf_d13_hard"
TABLE2B_SEEDS="0,1,2,3,4"

# Option 1: Generate all at once (faster, uses more memory)
echo ""
echo "[Table 2B] Generating all configs with seeds: $TABLE2B_SEEDS"
echo "------------------------------------------------------"

python scripts/synth_corruption_benchmark.py \
    --table2 \
    --seeds $TABLE2B_SEEDS \
    --output data/interim/sem_table2

if [ $? -eq 0 ]; then
    echo "[OK] All Table 2B datasets generated successfully"
else
    echo "[WARN] Batch generation failed, trying individual configs..."
    
    # Option 2: Generate one config at a time (fallback)
    for cfg in $TABLE2B_CONFIGS; do
        for seed in 0 1 2 3 4; do
            echo ""
            echo "[Table 2B] Generating: $cfg seed=$seed"
            
            python scripts/synth_corruption_benchmark.py \
                --table2 \
                --config $cfg \
                --seeds $seed \
                --output data/interim/sem_table2
            
            if [ $? -ne 0 ]; then
                echo "[FAIL] $cfg seed=$seed failed!"
            fi
        done
    done
fi

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "======================================================"
echo " GENERATION COMPLETE"
echo " End time: $(date)"
echo "======================================================"

echo ""
echo "Table 2A (H1/H2/H3 Hypothesis Benchmarks):"
echo "  Location: data/interim/table2a/"
ls -la data/interim/table2a/ 2>/dev/null || echo "  (directory not found)"

echo ""
echo "Table 2B (SEM Benchmark Grid):"
echo "  Location: data/interim/sem_table2/"
ls -la data/interim/sem_table2/ 2>/dev/null || echo "  (directory not found)"

echo ""
echo "Dataset counts:"
echo "  Table 2A: $(find data/interim/table2a -name 'A_true.npy' 2>/dev/null | wc -l) datasets"
echo "  Table 2B: $(find data/interim/sem_table2 -name 'A_true.npy' 2>/dev/null | wc -l) datasets"
echo "  Expected: 30 (Table 2A) + 40 (Table 2B) = 70 total"

echo ""
echo "Next steps:"
echo "  1. Train RC-GNN on each dataset"
echo "  2. Run baselines (NOTEARS, GOLEM, etc.)"
echo "  3. Evaluate with comprehensive_evaluation.py"
echo ""
echo "Example training command:"
echo "  python scripts/train_rcgnn_unified.py --data_root data/interim/sem_table2/er_d13_lin/seed_0"
echo ""
