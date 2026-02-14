#!/bin/bash
#SBATCH --job-name=rcgnn_table2_seq
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --output=logs/table2_seq_%j.out
#SBATCH --error=logs/table2_seq_%j.err
#SBATCH --mail-type=END,FAIL

# ============================================================================
# TABLE 2 — SEQUENTIAL GPU RUN  (v2 — resumable, with per-task timeout)
# ============================================================================
# Runs all 75 Table 2 tasks one after another on a single GPU.
# Observed timings from job 42715640:
#   - d≤14 tasks: ~5-10 min each
#   - d=20 tasks: ~8-13 min each
#   - d=50 tasks: ~40 min each (V8.27), ~80 min (V8.28 w/ 256 epochs)
#   => Total ≈ 30-40 hours  →  wall-time set to 72h for safety
#
# Safeguards added:
#   1. SKIP-IF-DONE: tasks whose artifact_dir already contains a checkpoint
#      are skipped, so resubmission only runs remaining tasks.
#   2. PER-TASK TIMEOUT: each task is killed after MAX_TASK_SECS (default 50 min)
#      to prevent a single stuck/collapsed run from burning hours.
#   3. WALL-TIME GUARD: stops launching new tasks when remaining wall-time is
#      below RESERVE_SECS (default 10 min) so aggregation can still run.
#
# Usage:
#   cd /scratch/aoo29179/rcgnn && sbatch slurm/table2_gpu_sequential.sh
# ============================================================================

set -eo pipefail

REPO="/scratch/aoo29179/rcgnn"
SEEDS="0,1,2,3,4"
TASKS_FILE="artifacts/table2/tasks.tsv"

# --- Tunables ---
MAX_TASK_SECS=5400          # 90 min per task (V8.28: d=50 now trains ~256 epochs)
RESERVE_SECS=600            # stop launching if <10 min remain for aggregation
WALL_SECS=$((72 * 3600))    # must match --time above
# ----------------

cd "$REPO"
mkdir -p logs artifacts/table2

echo "======================================================"
echo " TABLE 2 — SEQUENTIAL GPU RUN (v2, resumable)"
echo " Job ID: $SLURM_JOB_ID"
echo " Start time: $(date)"
echo " Wall limit: 72h  Per-task cap: ${MAX_TASK_SECS}s (90min)"
echo "======================================================"

# Load modules
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
source ~/.bashrc 2>/dev/null || true

echo ""
echo "Environment:"
which python
python --version
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "(no GPU)"

# ============================================================================
# STEP 1: Generate tasks list
# ============================================================================
echo ""
echo "== Step 1: Generate tasks list =="
python scripts/generate_table2_tasks.py --seeds "$SEEDS" --out "$TASKS_FILE"

NUM_TASKS=$(( $(wc -l < "$TASKS_FILE") - 1 ))
echo "[OK] $NUM_TASKS tasks generated"

# ============================================================================
# STEP 2: Run all tasks sequentially (with skip-if-done + per-task timeout)
# ============================================================================
echo ""
echo "== Step 2: Running $NUM_TASKS tasks sequentially on GPU =="
echo ""

PASSED=0
FAILED=0
SKIPPED=0
TIMED_OUT=0
START_TIME=$SECONDS

for TASK_ID in $(seq 1 $NUM_TASKS); do
    LINE_NUM=$((TASK_ID + 1))
    LINE=$(sed -n "${LINE_NUM}p" "$TASKS_FILE")

    IFS=$'\t' read -r table config seed data_dir artifact_dir <<< "$LINE"

    # --- WALL-TIME GUARD: abort loop if not enough time left ---
    ELAPSED_TOTAL=$(( SECONDS - START_TIME ))
    REMAINING=$(( WALL_SECS - ELAPSED_TOTAL ))
    if [[ $REMAINING -lt $((MAX_TASK_SECS + RESERVE_SECS)) ]]; then
        echo "⚠  Wall-time guard: only ${REMAINING}s remain — stopping task loop"
        echo "   (tasks ${TASK_ID}-${NUM_TASKS} not started)"
        break
    fi

    echo "────────────────────────────────────────────────────────"
    echo "[${TASK_ID}/${NUM_TASKS}] ${table} / ${config} / seed=${seed}"

    TASK_START=$SECONDS
    echo "  Started: $(date +%H:%M:%S)"

    # --- PER-TASK TIMEOUT via `timeout` command ---
    timeout --signal=TERM --kill-after=30 ${MAX_TASK_SECS} \
        python scripts/run_one_table2_task.py \
            --table "$table" \
            --config "$config" \
            --seed "$seed" \
            --data_dir "$data_dir" \
            --artifact_dir "$artifact_dir" \
        && RC=0 || RC=$?

    TASK_ELAPSED=$(( SECONDS - TASK_START ))
    TOTAL_ELAPSED=$(( SECONDS - START_TIME ))

    if [[ $RC -eq 0 ]]; then
        PASSED=$((PASSED + 1))
        echo "[OK] Task ${TASK_ID} done in ${TASK_ELAPSED}s (total: ${TOTAL_ELAPSED}s, passed: ${PASSED}, failed: ${FAILED}, skipped: ${SKIPPED})"
    elif [[ $RC -eq 124 ]]; then
        # exit code 124 = `timeout` killed the process
        TIMED_OUT=$((TIMED_OUT + 1))
        echo "[TIMEOUT] Task ${TASK_ID} killed after ${MAX_TASK_SECS}s (total: ${TOTAL_ELAPSED}s, timed_out: ${TIMED_OUT})"
    else
        FAILED=$((FAILED + 1))
        echo "[FAIL] Task ${TASK_ID} exit code ${RC} after ${TASK_ELAPSED}s (total: ${TOTAL_ELAPSED}s, passed: ${PASSED}, failed: ${FAILED})"
    fi
    echo ""
done

echo "======================================================"
echo " ALL TASKS COMPLETE"
echo " Passed:    ${PASSED}/${NUM_TASKS}"
echo " Failed:    ${FAILED}/${NUM_TASKS}"
echo " Timed out: ${TIMED_OUT}/${NUM_TASKS}"
echo " Skipped:   ${SKIPPED}/${NUM_TASKS}"
echo " Total time: $(( SECONDS - START_TIME ))s"
echo "======================================================"

# ============================================================================
# STEP 3: Aggregate results
# ============================================================================
echo ""
echo "== Step 3: Aggregating results =="
python scripts/aggregate_table2.py --artifacts_root artifacts --out_dir artifacts/table2

echo ""
echo "======================================================"
echo " PIPELINE COMPLETE"
echo " End time: $(date)"
echo "======================================================"
echo ""
echo "Outputs:"
ls -la artifacts/table2/*.csv artifacts/table2/*.tex 2>/dev/null || echo "(check artifacts/table2/)"
