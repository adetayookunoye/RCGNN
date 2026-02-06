#!/bin/bash
#SBATCH --job-name=rcgnn_table2_driver
#SBATCH --partition=batch_p
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=logs/table2_driver_%j.out
#SBATCH --error=logs/table2_driver_%j.err
#SBATCH --mail-type=END,FAIL

# ============================================================================
# TABLE 2 DRIVER SCRIPT
# ============================================================================
# One-command execution of the entire Table 2 pipeline:
#   1. Generates tasks.tsv (75 tasks)
#   2. Submits GPU array job (parallel training)
#   3. Submits aggregation job (waits for array to complete)
#
# Usage:
#   cd /scratch/aoo29179/rcgnn
#   mkdir -p logs
#   sbatch slurm/table2_driver.sh
#
# Outputs:
#   artifacts/table2/tasks.tsv           - Task manifest
#   artifacts/table2/table2_all_runs.csv - All results
#   artifacts/table2/table2_summary_meanstd.csv - Mean ± std
#   artifacts/table2/table2A.tex, table2B.tex, table2C.tex - LaTeX tables
# ============================================================================

set -euo pipefail

REPO="/scratch/aoo29179/rcgnn"
SEEDS="0,1,2,3,4"
TASKS_FILE="artifacts/table2/tasks.tsv"
LOG_DIR="logs"

cd "$REPO"
mkdir -p "$LOG_DIR" artifacts/table2

echo "======================================================"
echo " TABLE 2 DRIVER"
echo " Job ID: $SLURM_JOB_ID"
echo " Repo: $REPO"
echo " Seeds: $SEEDS"
echo " Start time: $(date)"
echo "======================================================"

# Load modules (keep consistent with your cluster env)
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
source ~/.bashrc

echo ""
echo "Python environment:"
which python
python --version

# ============================================================================
# STEP A: Generate tasks list
# ============================================================================
echo ""
echo "== A) Generate tasks list =="
python scripts/generate_table2_tasks.py --seeds "$SEEDS" --out "$TASKS_FILE"

LINES=$(wc -l < "$TASKS_FILE" | tr -d ' ')
echo "[OK] tasks.tsv lines = $LINES (expect 76 = header + 75 tasks)"

# Compute number of tasks (subtract header)
NUM_TASKS=$((LINES - 1))
if [[ "$NUM_TASKS" -le 0 ]]; then
    echo "[FAIL] No tasks found in $TASKS_FILE"
    exit 1
fi
echo "[OK] NUM_TASKS = $NUM_TASKS"

# ============================================================================
# STEP B: Submit GPU array job
# ============================================================================
echo ""
echo "== B) Submit GPU array job =="
ARRAY_JOB_ID=$(sbatch --parsable --array=1-"$NUM_TASKS" slurm/table2_array.sh "$TASKS_FILE")
echo "[OK] Submitted array job: $ARRAY_JOB_ID"

# ============================================================================
# STEP C: Submit aggregation job (depends on array completion)
# ============================================================================
echo ""
echo "== C) Submit aggregation job (depends on array) =="

# Use afterany so aggregation runs even if some tasks fail
# Change to afterok if you want aggregation only on full success
AGG_JOB_ID=$(sbatch --parsable --dependency=afterany:"$ARRAY_JOB_ID" << 'EOF'
#!/bin/bash
#SBATCH --job-name=rcgnn_table2_agg
#SBATCH --partition=batch_p
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/table2_agg_%j.out
#SBATCH --error=logs/table2_agg_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

REPO="/scratch/aoo29179/rcgnn"
cd "$REPO"

module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
source ~/.bashrc

mkdir -p logs artifacts/table2

echo "======================================================"
echo " TABLE 2 AGGREGATION"
echo " Job ID: $SLURM_JOB_ID"
echo " Repo: $REPO"
echo " Start time: $(date)"
echo "======================================================"

python scripts/aggregate_table2.py --artifacts_root artifacts --out_dir artifacts/table2

echo ""
echo "======================================================"
echo " AGGREGATION COMPLETE"
echo " End time: $(date)"
echo "======================================================"
echo ""
echo "Outputs:"
ls -la artifacts/table2/

echo ""
echo "Quick preview of summary:"
head -20 artifacts/table2/table2_summary_meanstd.csv 2>/dev/null || echo "(no summary yet)"
EOF
)

echo "[OK] Submitted aggregation job: $AGG_JOB_ID"

# ============================================================================
# SUMMARY
# ============================================================================
echo ""
echo "======================================================"
echo " TABLE 2 PIPELINE SUBMITTED"
echo "======================================================"
echo ""
echo " Driver job:      $SLURM_JOB_ID (this job, finishing now)"
echo " Array job:       $ARRAY_JOB_ID (1-$NUM_TASKS tasks)"
echo " Aggregation job: $AGG_JOB_ID (afterany:$ARRAY_JOB_ID)"
echo ""
echo " Monitor progress:"
echo "   squeue -u \$USER"
echo "   tail -f logs/table2_${ARRAY_JOB_ID}_*.out"
echo ""
echo " Expected outputs after completion:"
echo "   artifacts/table2/table2_all_runs.csv"
echo "   artifacts/table2/table2_summary_meanstd.csv"
echo "   artifacts/table2/table2A.tex"
echo "   artifacts/table2/table2B.tex"
echo "   artifacts/table2/table2C.tex"
echo ""
echo "======================================================"
