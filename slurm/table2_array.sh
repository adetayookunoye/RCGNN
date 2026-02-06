#!/bin/bash
#SBATCH --job-name=rcgnn_table2
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=logs/table2_%A_%a.out
#SBATCH --error=logs/table2_%A_%a.err
#SBATCH --mail-type=END,FAIL

# ============================================================================
# TABLE 2 SLURM ARRAY JOB
# ============================================================================
# Runs all 75 Table 2 tasks as a job array.
#
# Usage:
#   # First generate the task manifest:
#   python scripts/generate_table2_tasks.py
#
#   # Then submit the array job:
#   sbatch --array=1-75 slurm/table2_array.sh
#
#   # Or run a subset:
#   sbatch --array=1-30 slurm/table2_array.sh   # Table 2A only
#   sbatch --array=31-70 slurm/table2_array.sh  # Table 2B only
#   sbatch --array=71-75 slurm/table2_array.sh  # Table 2C only
#
# Task breakdown:
#   Tasks 1-30:  Table 2A (H1/H2/H3 hypothesis benchmarks)
#   Tasks 31-70: Table 2B (SEM benchmark grid)
#   Tasks 71-75: Table 2C (Causal validity ablation)
#
# Each task:
#   1. Generates dataset (if not exists)
#   2. Trains RC-GNN
#   3. Runs comprehensive evaluation
#   4. Writes run_meta.json and evaluation.json
# ============================================================================

set -euo pipefail

echo "======================================================"
echo " RC-GNN TABLE 2 EXPERIMENT"
echo " Job ID: ${SLURM_JOB_ID}"
echo " Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo " Start time: $(date)"
echo "======================================================"

# ============================================================================
# SETUP
# ============================================================================

# Change to project directory
cd /scratch/aoo29179/rcgnn

# Load modules
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
source ~/.bashrc

# Verify environment
echo ""
echo "Environment:"
which python
python --version
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "(no GPU info)"

# Create directories
mkdir -p logs artifacts/table2

# ============================================================================
# PARSE TASK FROM TSV
# ============================================================================

TASKS_FILE="${1:-artifacts/table2/tasks.tsv}"

if [[ ! -f "${TASKS_FILE}" ]]; then
    echo "[FAIL] Tasks file not found: ${TASKS_FILE}"
    echo "  Run: python scripts/generate_table2_tasks.py"
    exit 1
fi

# SLURM_ARRAY_TASK_ID is 1-indexed, TSV has header on line 1
# So task 1 -> line 2, task 2 -> line 3, etc.
LINE_NUM=$((SLURM_ARRAY_TASK_ID + 1))
LINE=$(sed -n "${LINE_NUM}p" "${TASKS_FILE}" || true)

if [[ -z "${LINE}" ]]; then
    echo "[FAIL] Could not read line ${LINE_NUM} from ${TASKS_FILE}"
    echo "  Total lines in file: $(wc -l < "${TASKS_FILE}")"
    exit 1
fi

# Parse TSV columns: table, config, seed, data_dir, artifact_dir
IFS=$'\t' read -r table config seed data_dir artifact_dir <<< "${LINE}"

echo ""
echo "Task Configuration:"
echo "  Table:        ${table}"
echo "  Config:       ${config}"
echo "  Seed:         ${seed}"
echo "  Data dir:     ${data_dir}"
echo "  Artifact dir: ${artifact_dir}"
echo ""

# ============================================================================
# RUN TASK
# ============================================================================

python scripts/run_one_table2_task.py \
    --table "${table}" \
    --config "${config}" \
    --seed "${seed}" \
    --data_dir "${data_dir}" \
    --artifact_dir "${artifact_dir}"

RC=$?

echo ""
echo "======================================================"
echo " TASK COMPLETE"
echo " Exit code: ${RC}"
echo " End time: $(date)"
echo "======================================================"

exit ${RC}
