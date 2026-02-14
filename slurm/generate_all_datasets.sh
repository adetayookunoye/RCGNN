#!/bin/bash
#SBATCH --job-name=table2_all_data
#SBATCH --partition=batch_p
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=logs/table2_all_data_%j.out
#SBATCH --error=logs/table2_all_data_%j.err
#SBATCH --mail-type=END,FAIL

# ============================================================================
# UNIFIED DATASET GENERATION FOR TABLE 2 (SPIE 2026 PAPER)
# ============================================================================
#
# This script generates ALL datasets required for Table 2:
#
# TABLE 2A - Hypothesis Benchmarks (H1/H2/H3)
#   h1_easy, h1_medium, h1_hard, h2_multi_env, h2_stability, h3_policy
#   6 benchmarks × 5 seeds = 30 datasets
#   Script: synth_corruption_benchmark.py --benchmark
#
# TABLE 2B - SEM Benchmark Grid (classic causal discovery sweep)
#   er_d13_lin, er_d13_mlp, er_d20_lin, er_d20_mlp, er_d50_mlp,
#   sf_d13_mlp, sf_d20_mlp, sf_d13_hard
#   8 configs × 5 seeds = 40 datasets
#   Script: synth_corruption_benchmark.py --table2 --config
#
# TABLE 2C - Causal Validity Ablation (known DAG by construction)
#   compound_sem_medium × 5 seeds = 5 datasets
#   Script: run_synthetic_benchmark.py --benchmark compound_sem_medium
#   Key framing: "Ground-truth DAG is known by construction; results 
#   quantify recovery under MNAR + drift + bias, eliminating concerns 
#   about domain-knowledge graph subjectivity."
#
# TOTAL: 30 + 40 + 5 = 75 datasets
#
# Output structure:
#   data/interim/table2a/{benchmark}/seed_{seed}/
#   data/interim/sem_table2/{config}/seed_{seed}/
#   data/interim/table2c/compound_sem_medium/seed_{seed}/
#
# ============================================================================

echo "========================================================================"
echo " TABLE 2 UNIFIED DATASET GENERATION"
echo " Job ID: ${SLURM_JOB_ID:-local}"
echo " Start time: $(date)"
echo "========================================================================"

# ============================================================================
# SETUP
# ============================================================================

# Change to project directory (adjust for your cluster)
cd /scratch/aoo29179/rcgnn || cd $SLURM_SUBMIT_DIR || cd "$(dirname "$0")/.." || exit 1

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
mkdir -p data/interim/table2c

# Seeds for all experiments
SEEDS="0 1 2 3 4"

# Track statistics
TOTAL_EXPECTED=75
TOTAL_GENERATED=0
FAILED_DATASETS=""

echo ""
echo "Configuration:"
echo "  Seeds: $SEEDS (5 seeds for mean ± std)"
echo "  Table 2A: 6 benchmarks × 5 seeds = 30 datasets"
echo "  Table 2B: 8 configs × 5 seeds = 40 datasets"
echo "  Table 2C: 1 benchmark × 5 seeds = 5 datasets"
echo "  Total expected: $TOTAL_EXPECTED datasets"

# ============================================================================
# TABLE 2A: H1/H2/H3 HYPOTHESIS BENCHMARKS
# ============================================================================
# H1: Structural accuracy under missingness severity
# H2: Stability improvement via invariance loss
# H3: Expert agreement on policy-relevant pathways
# ============================================================================

echo ""
echo "========================================================================"
echo " TABLE 2A: Hypothesis Benchmarks (H1/H2/H3)"
echo " 6 benchmarks × 5 seeds = 30 datasets"
echo "========================================================================"

TABLE2A_BENCHMARKS="h1_easy h1_medium h1_hard h2_multi_env h2_stability h3_policy"

for benchmark in $TABLE2A_BENCHMARKS; do
    for seed in $SEEDS; do
        echo ""
        echo "[Table 2A] Generating: $benchmark seed=$seed"
        echo "------------------------------------------------------------------------"
        
        OUTPUT_DIR="data/interim/table2a/${benchmark}/seed_${seed}"
        
        python scripts/synth_corruption_benchmark.py \
            --benchmark $benchmark \
            --seed $seed \
            --output $OUTPUT_DIR
        
        if [ $? -eq 0 ]; then
            echo "[OK] $benchmark seed=$seed -> $OUTPUT_DIR"
            TOTAL_GENERATED=$((TOTAL_GENERATED + 1))
        else
            echo "[FAIL] $benchmark seed=$seed"
            FAILED_DATASETS="$FAILED_DATASETS table2a/$benchmark/seed_$seed"
        fi
    done
done

echo ""
echo "[Table 2A] Generated: $TOTAL_GENERATED / 30 datasets"

# ============================================================================
# TABLE 2B: SEM BENCHMARK GRID
# ============================================================================
# Classic causal discovery sweep:
#   - Graph families: ER (Erdős-Rényi), SF (Scale-Free)
#   - Dimensions: d=13, 20, 50
#   - Mechanisms: Linear, MLP (nonlinear)
#   - Corruption: Medium (20% MNAR), Hard (40% MNAR)
# ============================================================================

echo ""
echo "========================================================================"
echo " TABLE 2B: SEM Benchmark Grid"
echo " 8 configs × 5 seeds = 40 datasets"
echo "========================================================================"

TABLE2B_CONFIGS="er_d13_lin er_d13_mlp er_d20_lin er_d20_mlp er_d50_mlp sf_d13_mlp sf_d20_mlp sf_d13_hard"
TABLE2B_SEEDS="0,1,2,3,4"

TABLE2B_START=$TOTAL_GENERATED

for cfg in $TABLE2B_CONFIGS; do
    echo ""
    echo "[Table 2B] Generating: $cfg (seeds: $TABLE2B_SEEDS)"
    echo "------------------------------------------------------------------------"
    
    python scripts/synth_corruption_benchmark.py \
        --table2 \
        --config $cfg \
        --seeds $TABLE2B_SEEDS \
        --output data/interim/sem_table2
    
    if [ $? -eq 0 ]; then
        echo "[OK] $cfg × 5 seeds generated"
        TOTAL_GENERATED=$((TOTAL_GENERATED + 5))
    else
        echo "[FAIL] $cfg generation failed"
        FAILED_DATASETS="$FAILED_DATASETS sem_table2/$cfg"
    fi
done

TABLE2B_COUNT=$((TOTAL_GENERATED - TABLE2B_START))
echo ""
echo "[Table 2B] Generated: $TABLE2B_COUNT / 40 datasets"

# ============================================================================
# TABLE 2C: CAUSAL VALIDITY ABLATION
# ============================================================================
# Key framing for paper:
#   "Ground-truth DAG is known BY CONSTRUCTION; results quantify recovery 
#   under MNAR + drift + bias, eliminating concerns about domain-knowledge 
#   graph subjectivity."
#
# Uses compound_sem_medium:
#   - d=13, edges=13 (ER graph)
#   - MLP mechanism (nonlinear)
#   - 5 environments
#   - MNAR 30%, noise=0.4, drift=0.2, bias=1.0
# ============================================================================

echo ""
echo "========================================================================"
echo " TABLE 2C: Causal Validity Ablation"
echo " compound_sem_medium × 5 seeds = 5 datasets"
echo " (Ground-truth DAG known by construction)"
echo "========================================================================"

TABLE2C_BENCHMARK="compound_sem_medium"
TABLE2C_START=$TOTAL_GENERATED

for seed in $SEEDS; do
    echo ""
    echo "[Table 2C] Generating: $TABLE2C_BENCHMARK seed=$seed"
    echo "------------------------------------------------------------------------"
    
    OUTPUT_DIR="data/interim/table2c/${TABLE2C_BENCHMARK}/seed_${seed}"
    
    python scripts/run_synthetic_benchmark.py \
        --benchmark $TABLE2C_BENCHMARK \
        --seed $seed \
        --output $OUTPUT_DIR
    
    if [ $? -eq 0 ]; then
        echo "[OK] $TABLE2C_BENCHMARK seed=$seed -> $OUTPUT_DIR"
        TOTAL_GENERATED=$((TOTAL_GENERATED + 1))
    else
        echo "[FAIL] $TABLE2C_BENCHMARK seed=$seed"
        FAILED_DATASETS="$FAILED_DATASETS table2c/$TABLE2C_BENCHMARK/seed_$seed"
    fi
done

TABLE2C_COUNT=$((TOTAL_GENERATED - TABLE2C_START))
echo ""
echo "[Table 2C] Generated: $TABLE2C_COUNT / 5 datasets"

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "========================================================================"
echo " DATASET GENERATION COMPLETE"
echo " End time: $(date)"
echo "========================================================================"

echo ""
echo "Dataset Summary:"
echo "  Table 2A (H1/H2/H3): $(find data/interim/table2a -name 'A_true.npy' 2>/dev/null | wc -l) / 30 datasets"
echo "  Table 2B (SEM grid): $(find data/interim/sem_table2 -name 'A_true.npy' 2>/dev/null | wc -l) / 40 datasets"
echo "  Table 2C (Validity): $(find data/interim/table2c -name 'A_true.npy' 2>/dev/null | wc -l) / 5 datasets"
echo "  ---------------------------------------------------------"
echo "  Total generated: $TOTAL_GENERATED / $TOTAL_EXPECTED"

if [ -n "$FAILED_DATASETS" ]; then
    echo ""
    echo "FAILED DATASETS:"
    for failed in $FAILED_DATASETS; do
        echo "  - $failed"
    done
fi

echo ""
echo "Output locations:"
echo "  Table 2A: data/interim/table2a/{benchmark}/seed_{seed}/"
echo "  Table 2B: data/interim/sem_table2/{config}/seed_{seed}/"
echo "  Table 2C: data/interim/table2c/compound_sem_medium/seed_{seed}/"

echo ""
echo "========================================================================"
echo " NEXT STEPS"
echo "========================================================================"
echo ""
echo "1. Train RC-GNN on each dataset (use separate SLURM array job)"
echo "2. Run baselines (NOTEARS, GOLEM, etc.) on each dataset"
echo "3. Evaluate with comprehensive_evaluation.py"
echo ""
echo "Example training command:"
echo "  python scripts/train_rcgnn_unified.py \\"
echo "      --data_root data/interim/table2a/h1_easy/seed_0 \\"
echo "      --output_dir artifacts/table2a/h1_easy/seed_0"
echo ""
echo "Paper framing for Table 2C:"
echo '  "Ground-truth DAG is known by construction; results quantify'
echo '   recovery under MNAR + drift + bias, eliminating concerns about'
echo '   domain-knowledge graph subjectivity."'
echo ""
