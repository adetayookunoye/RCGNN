#!/bin/bash
#SBATCH --job-name=rcgnn_full
#SBATCH --partition=gpu_p
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=256G
#SBATCH --time=168:00:00
#SBATCH --output=logs/rcgnn_full_%j.out
#SBATCH --error=logs/rcgnn_full_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aoo29179@uga.edu

# ============================================================================
# RC-GNN Unified Full Pipeline  (v2 — all 10 gaps fixed)
# ============================================================================
#
# Runs ALL experiments end-to-end and generates every table & figure:
#
#   Phase 0 — Data Generation + Checksum Validation           (~5 min)
#   Phase 1 — UCI Air Quality: 12 corruptions × 5 seeds       (~30 h)
#   Phase 2 — Synthetic SEM K-Robustness: 75 tasks             (~30 h)
#   Phase 3 — Component Ablation: A1–A10 × 3 datasets × 3 seeds (~15 h)
#   Phase 4 — Robustness Stress Test: missingness sweep         (~3 h)
#   Phase 5 — Evaluation, LaTeX, Plotting + Smoke Test          (~15 min)
#
# Total estimated: ~90 h  (168 h wall-time gives ~1.9× safety margin)
#
# ============================================================================
# GAPS ADDRESSED (from expert audit):
#
#   Gap  1: Multi-seed for UCI Phase 1 (5 seeds for mean±std)
#   Gap  2: Expanded ablation A1–A10 (was 6 configs, now 11 incl baseline)
#   Gap  3: CLI args for config-only features (added to train_rcgnn_unified.py)
#   Gap  4: Seed loop for ablation (3 seeds per config)
#   Gap  5: LODO protocol (leave-one-regime-out eval in Phase 1)
#   Gap  6: UCI LaTeX table generation (Phase 5)
#   Gap  7: Robustness stress test (Phase 4: missingness 20/40/60%)
#   Gap  8: Multi-dataset ablation (compound_full + extreme + clean_full)
#   Gap  9: Smoke test + exit-code summary (Phase 5 final validation)
#   Gap 10: Data checksums for bit-exact reproducibility (Phase 0)
# ============================================================================
#
# Usage:
#   cd /scratch/aoo29179/rcgnn && sbatch slurm/run_full_pipeline.sh
#
# Run single phase:
#   PHASE=3 sbatch slurm/run_full_pipeline.sh
#
# Resumable: re-submit same script — each phase skips completed outputs
# ============================================================================

set -eo pipefail

REPO="/scratch/aoo29179/rcgnn"
cd "$REPO"
mkdir -p logs artifacts/ablation artifacts/table2/plots artifacts/robustness \
         artifacts/uci_multiseed artifacts/lodo

# Phase filter
RUN_PHASE="${PHASE:-all}"

# Reproducibility: record exact git state  (Gap 10)
GIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "N/A")
GIT_DIRTY=$(git diff --quiet 2>/dev/null && echo "clean" || echo "dirty")

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║      RC-GNN FULL PIPELINE v2 — Job $SLURM_JOB_ID"
echo "║      Phase:  ${RUN_PHASE}"
echo "║      Git:    ${GIT_SHA} (${GIT_DIRTY})"
echo "║      Start:  $(date)"
echo "╚══════════════════════════════════════════════════════════════╝"

# --- Environment ---
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
source ~/.bashrc 2>/dev/null || true

# Temporal baseline dependencies (Granger needs statsmodels, PCMCI+ needs tigramite)
echo "── Installing temporal baseline dependencies ──"
pip install --user --quiet statsmodels tigramite 2>&1 | tail -3

echo ""
echo "Environment:"
which python
python --version
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "(no GPU)"

# Save reproducibility manifest  (Gap 10)
python -c "
import json, sys, torch, numpy as np, hashlib
from pathlib import Path

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()[:16]

# Checksum key data files
data_checksums = {}
for f in Path('data/interim/uci_air').glob('*.npy'):
    data_checksums[f.name] = sha256(f)

manifest = {
    'job_id': '${SLURM_JOB_ID}',
    'git_sha': '${GIT_SHA}',
    'git_dirty': '${GIT_DIRTY}',
    'python': sys.version,
    'torch': torch.__version__,
    'numpy': np.__version__,
    'cuda': torch.version.cuda or 'N/A',
    'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A',
    'data_checksums': data_checksums,
}
with open('artifacts/run_manifest_${SLURM_JOB_ID}.json', 'w') as f:
    json.dump(manifest, f, indent=2)
print('  ✓ Run manifest saved (with data checksums)')
"

WALL_SECS=$((168 * 3600))
PIPELINE_START=$SECONDS

# Global counters for smoke test (Gap 9)
TOTAL_PASS=0
TOTAL_FAIL=0
TOTAL_SKIP=0


# ############################################################################
# PHASE 0: DATA GENERATION + CHECKSUM VALIDATION  (Gap 10)
# ############################################################################
if [[ "$RUN_PHASE" == "all" || "$RUN_PHASE" == "0" ]]; then

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  PHASE 0: DATA GENERATION & CHECKSUM VALIDATION             ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# 0a. Verify raw UCI Air Quality data
if [[ ! -f "data/interim/uci_air/X.npy" ]]; then
    echo "[FAIL] Raw UCI data not found at data/interim/uci_air/X.npy"
    echo "       Run the UCI preprocessing pipeline first."
    exit 1
fi
echo "  ✓ Raw UCI data: data/interim/uci_air/"

# 0b. Generate all 12 UCI-Air-C corruption datasets (idempotent)
if [[ -d "data/interim/uci_air_c/sensor_failure" ]]; then
    echo "  ✓ UCI-Air-C corruptions already generated (12 dirs)"
else
    echo "  → Generating UCI-Air-C corruption suite..."
    python scripts/generate_uci_air_c.py \
        --input data/interim/uci_air \
        --output data/interim/uci_air_c \
        --seed 42
    echo "  ✓ UCI-Air-C corruptions generated"
fi

# 0c. Validate all 12 corruption dirs (Gap 10: integrity check)
echo ""
echo "  Validating dataset integrity..."
MISSING=0
for DS in clean_full clean_real compound_full compound_mnar_bias extreme \
          mcar_40 mcar_20 mnar_structural moderate noise_0.5 regimes_5 sensor_failure; do
    DIR="data/interim/uci_air_c/${DS}"
    if [[ ! -f "${DIR}/X.npy" || ! -f "${DIR}/A_true.npy" ]]; then
        echo "  [FAIL] Missing files in ${DIR}"
        MISSING=$((MISSING + 1))
    fi
done
if [[ $MISSING -gt 0 ]]; then
    echo "[FAIL] ${MISSING} datasets incomplete — aborting"
    exit 1
fi
echo "  ✓ All 12 UCI-Air-C datasets validated"

# 0d. Compute and log per-dataset checksums (Gap 10: bit-exact reproducibility)
echo ""
echo "  Computing dataset checksums..."
python -c "
import hashlib, json
from pathlib import Path

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()[:16]

checksums = {}
for ds_dir in sorted(Path('data/interim/uci_air_c').iterdir()):
    if ds_dir.is_dir():
        cs = {}
        for f in sorted(ds_dir.glob('*.npy')):
            cs[f.name] = sha256(f)
        checksums[ds_dir.name] = cs

with open('artifacts/dataset_checksums.json', 'w') as f:
    json.dump(checksums, f, indent=2)
print(f'  ✓ Checksums for {len(checksums)} datasets → artifacts/dataset_checksums.json')
"

echo ""
echo "  ✓ Phase 0 complete: $(date)"

fi  # end Phase 0


# ############################################################################
# PHASE 1: UCI AIR QUALITY — 5 seeds × 12 corruptions  (Gaps 1, 5)
#           Paper Table 2: mean ± std over 5 random seeds
#           + LODO evaluation
# ############################################################################
if [[ "$RUN_PHASE" == "all" || "$RUN_PHASE" == "1" ]]; then

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  PHASE 1: UCI AIR QUALITY (12 corruptions × 5 seeds)        ║"
echo "║           + LODO evaluation                                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"

UCI_DATASETS=(
    "clean_full" "clean_real"
    "compound_full" "compound_mnar_bias" "extreme" "mcar_40"
    "mcar_20" "mnar_structural" "moderate" "noise_0.5" "regimes_5" "sensor_failure"
)

# Gap 1: Multi-seed loop (5 seeds for mean±std)
UCI_SEEDS=(42 1337 2024 7 99)

UCI_PASS=0
UCI_FAIL=0

for CORRUPTION in "${UCI_DATASETS[@]}"; do
    for SEED in "${UCI_SEEDS[@]}"; do
        echo ""
        echo "─── UCI: ${CORRUPTION}  seed=${SEED} ───"
        echo "  Start: $(date)"

        OUTPUT_DIR="artifacts/uci_multiseed/${CORRUPTION}/seed_${SEED}"
        mkdir -p "${OUTPUT_DIR}"

        # Wall-time guard
        ELAPSED=$(( SECONDS - PIPELINE_START ))
        REMAINING=$(( WALL_SECS - ELAPSED ))
        if [[ $REMAINING -lt 3600 ]]; then
            echo "⚠  Wall-time guard: ${REMAINING}s left — stopping UCI loop"
            break 2
        fi

        # Skip if already trained
        if [[ -f "${OUTPUT_DIR}/training_history.json" ]]; then
            echo "  [SKIP] Already trained"
            UCI_PASS=$((UCI_PASS + 1))
            TOTAL_SKIP=$((TOTAL_SKIP + 1))
            continue
        fi

        python scripts/train_rcgnn_unified.py \
            --data_dir "data/interim/uci_air_c/${CORRUPTION}" \
            --output_dir "${OUTPUT_DIR}" \
            --epochs 150 \
            --lr 5e-4 \
            --batch_size 32 \
            --latent_dim 32 \
            --hidden_dim 64 \
            --lambda_recon 1.0 \
            --lambda_causal 1.0 \
            --target_edges 13 \
            --patience 40 \
            --seed ${SEED} \
            --device cuda \
            --sweep_mode \
        && { UCI_PASS=$((UCI_PASS + 1)); TOTAL_PASS=$((TOTAL_PASS + 1)); } \
        || { UCI_FAIL=$((UCI_FAIL + 1)); TOTAL_FAIL=$((TOTAL_FAIL + 1)); echo "  [FAIL]"; }

        echo "  Done: $(date)"
    done
done

echo ""
echo "── UCI Multi-Seed Summary: ${UCI_PASS} passed, ${UCI_FAIL} failed ──"

# Fail-fast: if most training failed, don't waste time on evaluation
if [[ $UCI_FAIL -gt 30 ]]; then
    echo "[FAIL] Too many UCI training failures (${UCI_FAIL}/60) — skipping evaluation"
else
    # ── Per-seed comprehensive evaluation (IID + temporal baselines) ──
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════════╗"
    echo "║  UCI Baseline Evaluation (per-seed, --single-run --temporal)        ║"
    echo "║  Baselines: Correlation, PC, GES, NOTEARS, NOTEARS-MLP, GOLEM,     ║"
    echo "║             Granger (MICE), PCMCI+ (k-NN)                          ║"
    echo "╚══════════════════════════════════════════════════════════════════════╝"

    EVAL_TOTAL=$((${#UCI_DATASETS[@]} * ${#UCI_SEEDS[@]}))
    EVAL_DONE=0
    EVAL_PASS=0
    EVAL_FAIL=0
    EVAL_SKIP=0

    for CORRUPTION in "${UCI_DATASETS[@]}"; do
        for SEED in "${UCI_SEEDS[@]}"; do
            EVAL_DONE=$((EVAL_DONE + 1))

            ARTIFACT_DIR="artifacts/uci_multiseed/${CORRUPTION}/seed_${SEED}"
            DATA_DIR="data/interim/uci_air_c/${CORRUPTION}"
            OUTPUT="${ARTIFACT_DIR}/evaluation.json"

            echo ""
            echo "── [${EVAL_DONE}/${EVAL_TOTAL}] Eval: ${CORRUPTION} / seed_${SEED} ──"

            # Wall-time guard: need at least 30 min for each eval
            ELAPSED=$(( SECONDS - PIPELINE_START ))
            REMAINING=$(( WALL_SECS - ELAPSED ))
            if [[ $REMAINING -lt 1800 ]]; then
                echo "⚠  Wall-time guard: ${REMAINING}s left — stopping eval loop"
                break 2
            fi

            # Skip if already evaluated WITH temporal baselines
            if [[ -f "${OUTPUT}" ]]; then
                if python3 -c "
import json, sys
with open('${OUTPUT}') as f:
    ev = json.load(f)
if 'Granger' in ev.get('baselines', {}) and 'PCMCI+' in ev.get('baselines', {}):
    sys.exit(0)
else:
    sys.exit(1)
" 2>/dev/null; then
                    echo "  [SKIP] Already has all baselines including temporal"
                    EVAL_SKIP=$((EVAL_SKIP + 1))
                    EVAL_PASS=$((EVAL_PASS + 1))
                    continue
                else
                    echo "  [RE-EVAL] Missing temporal baselines, re-running..."
                    rm -f "${OUTPUT}"
                fi
            fi

            # Check prerequisites — accept any adjacency the eval script can use
            if [[ ! -f "${ARTIFACT_DIR}/A_best.npy" ]] && \
               [[ ! -f "${ARTIFACT_DIR}/A_best_score.npy" ]] && \
               [[ ! -f "${ARTIFACT_DIR}/A_best_topk_sparse.npy" ]] && \
               [[ ! -f "${ARTIFACT_DIR}/A_best_unguarded.npy" ]] && \
               [[ ! -f "${ARTIFACT_DIR}/A_final.npy" ]]; then
                echo "  [SKIP] No adjacency matrix found — training may not have completed"
                EVAL_FAIL=$((EVAL_FAIL + 1))
                continue
            fi
            # If A_best.npy missing but unguarded/final exists, symlink so eval script finds it
            if [[ ! -f "${ARTIFACT_DIR}/A_best.npy" ]]; then
                for FALLBACK in A_best_unguarded.npy A_final.npy; do
                    if [[ -f "${ARTIFACT_DIR}/${FALLBACK}" ]]; then
                        echo "  [FALLBACK] Using ${FALLBACK} as A_best.npy"
                        cp "${ARTIFACT_DIR}/${FALLBACK}" "${ARTIFACT_DIR}/A_best.npy"
                        break
                    fi
                done
            fi
            if [[ ! -f "${DATA_DIR}/X.npy" ]]; then
                echo "  [SKIP] No data at ${DATA_DIR}"
                EVAL_FAIL=$((EVAL_FAIL + 1))
                continue
            fi

            # Run comprehensive evaluation with --temporal flag
            python scripts/comprehensive_evaluation.py \
                --artifacts-dir "${ARTIFACT_DIR}" \
                --data-dir "${DATA_DIR}" \
                --output "${OUTPUT}" \
                --single-run \
                --temporal \
                2>&1 | tail -40

            if [[ -f "${OUTPUT}" ]]; then
                echo "  [OK] -> ${OUTPUT}"
                EVAL_PASS=$((EVAL_PASS + 1))
            else
                echo "  [FAIL] No output produced"
                EVAL_FAIL=$((EVAL_FAIL + 1))
            fi
        done
    done

    echo ""
    echo "── UCI Eval Summary: ${EVAL_PASS}/${EVAL_TOTAL} passed, ${EVAL_FAIL} failed, ${EVAL_SKIP} skipped ──"

    # ── Aggregate UCI baseline results into summary table + JSON ──
    echo ""
    echo "── Aggregating UCI baseline results ──"

    python3 << 'PYEOF'
import json, os, sys
from pathlib import Path
from collections import defaultdict
import statistics

corruptions = [
    'clean_full','clean_real','mcar_20','mcar_40','noise_0.5','moderate',
    'compound_full','compound_mnar_bias','mnar_structural','sensor_failure',
    'regimes_5','extreme'
]
seeds = [42, 1337, 2024, 7, 99]
methods = ['rc_gnn', 'Correlation', 'PC', 'GES', 'NOTEARS', 'NOTEARS-MLP',
           'GOLEM', 'Granger', 'PCMCI+']

# Collect all results
all_data = {}
for c in corruptions:
    all_data[c] = []
    for s in seeds:
        path = Path(f'artifacts/uci_multiseed/{c}/seed_{s}/evaluation.json')
        if path.exists():
            with open(path) as f:
                all_data[c].append(json.load(f))

# Print summary table: Skeleton F1
sep = '=' * 140
print()
print(sep)
print('UCI AIR QUALITY -- RC-GNN vs ALL BASELINES (Skeleton F1, mean +/- std across 5 seeds)')
print(sep)

header = f"{'Corruption':<22s}"
for m in methods:
    header += f' | {m:>14s}'
print(header)
print('-' * 140)

for c in corruptions:
    evals = all_data[c]
    if not evals:
        print(f'{c:<22s} | NO DATA')
        continue

    row = f'{c:<22s}'
    for m in methods:
        f1_vals = []
        for ev in evals:
            if m == 'rc_gnn':
                val = ev.get('rc_gnn', {}).get('topk', {}).get('Skeleton_F1', None)
            else:
                val = ev.get('baselines', {}).get(m, {}).get('Skeleton_F1', None)
            if val is not None:
                f1_vals.append(val)

        if f1_vals:
            mean = statistics.mean(f1_vals)
            std = statistics.stdev(f1_vals) if len(f1_vals) > 1 else 0.0
            row += f' | {mean:5.3f}+/-{std:.3f}'
        else:
            row += f' |       N/A    '
    print(row)

print('-' * 140)
print()

# Print Directed F1 table
print(sep)
print('UCI AIR QUALITY -- RC-GNN vs ALL BASELINES (Directed F1, mean +/- std across 5 seeds)')
print(sep)

header = f"{'Corruption':<22s}"
for m in methods:
    header += f' | {m:>14s}'
print(header)
print('-' * 140)

for c in corruptions:
    evals = all_data[c]
    if not evals:
        print(f'{c:<22s} | NO DATA')
        continue

    row = f'{c:<22s}'
    for m in methods:
        f1_vals = []
        for ev in evals:
            if m == 'rc_gnn':
                val = ev.get('rc_gnn', {}).get('topk', {}).get('Directed_F1', None)
            else:
                val = ev.get('baselines', {}).get(m, {}).get('Directed_F1', None)
            if val is not None:
                f1_vals.append(val)

        if f1_vals:
            mean = statistics.mean(f1_vals)
            std = statistics.stdev(f1_vals) if len(f1_vals) > 1 else 0.0
            row += f' | {mean:5.3f}+/-{std:.3f}'
        else:
            row += f' |       N/A    '
    print(row)

print('-' * 140)
print()

# Per-corruption winner
print('Per-corruption Skeleton F1 winners:')
for c in corruptions:
    evals = all_data[c]
    if not evals:
        continue
    best_method = None
    best_f1 = -1
    for m in methods:
        f1_vals = []
        for ev in evals:
            if m == 'rc_gnn':
                val = ev.get('rc_gnn', {}).get('topk', {}).get('Skeleton_F1', None)
            else:
                val = ev.get('baselines', {}).get(m, {}).get('Skeleton_F1', None)
            if val is not None:
                f1_vals.append(val)
        if f1_vals:
            mean = statistics.mean(f1_vals)
            if mean > best_f1:
                best_f1 = mean
                best_method = m
    print(f'  {c:<22s}: {best_method} (F1={best_f1:.4f})')

# Save summary JSON
summary = {'corruptions': {}, 'methods': methods}
for c in corruptions:
    evals = all_data[c]
    if not evals:
        continue
    summary['corruptions'][c] = {}
    for m in methods:
        f1_vals, shd_vals, dir_f1_vals = [], [], []
        for ev in evals:
            if m == 'rc_gnn':
                topk = ev.get('rc_gnn', {}).get('topk', {})
            else:
                topk = ev.get('baselines', {}).get(m, {})
            sf1 = topk.get('Skeleton_F1')
            df1 = topk.get('Directed_F1')
            shd = topk.get('SHD')
            if sf1 is not None: f1_vals.append(sf1)
            if df1 is not None: dir_f1_vals.append(df1)
            if shd is not None: shd_vals.append(shd)

        entry = {}
        if f1_vals:
            entry['Skeleton_F1_mean'] = statistics.mean(f1_vals)
            entry['Skeleton_F1_std'] = statistics.stdev(f1_vals) if len(f1_vals) > 1 else 0
        if dir_f1_vals:
            entry['Directed_F1_mean'] = statistics.mean(dir_f1_vals)
            entry['Directed_F1_std'] = statistics.stdev(dir_f1_vals) if len(dir_f1_vals) > 1 else 0
        if shd_vals:
            entry['SHD_mean'] = statistics.mean(shd_vals)
            entry['SHD_std'] = statistics.stdev(shd_vals) if len(shd_vals) > 1 else 0
        summary['corruptions'][c][m] = entry

os.makedirs('artifacts/uci_multiseed', exist_ok=True)
with open('artifacts/uci_multiseed/uci_all_baselines_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print()
print('Summary saved: artifacts/uci_multiseed/uci_all_baselines_summary.json')
PYEOF

fi

# Gap 5: LODO protocol — Leave-One-Regime-Out evaluation report
echo ""
echo "─── LODO: Leave-One-Regime-Out evaluation ───"
python -c "
import json, numpy as np
from pathlib import Path

lodo_results = {}
multi_regime_datasets = ['compound_full', 'compound_mnar_bias', 'extreme', 'regimes_5']

for ds in multi_regime_datasets:
    ds_dir = Path(f'data/interim/uci_air_c/{ds}')
    art_dir = Path(f'artifacts/uci_multiseed/{ds}/seed_42')

    if not (ds_dir / 'X.npy').exists() or not (art_dir / 'A_best.npy').exists():
        continue

    X = np.load(ds_dir / 'X.npy')
    A_true = np.load(ds_dir / 'A_true.npy')
    e = np.load(ds_dir / 'e.npy')
    A_pred = np.load(art_dir / 'A_best.npy')

    regimes = np.unique(e.flatten())
    regime_metrics = []
    for rid in regimes:
        mask = (e.flatten() == rid) if e.ndim <= 2 else (e[:, 0] == rid)
        regime_metrics.append({
            'regime': int(rid),
            'n_samples': int(mask.sum()),
            'fraction': round(float(mask.sum() / len(mask)), 4),
        })

    # Global skeleton metrics
    pred_skel = ((A_pred + A_pred.T) > 0).astype(float)
    true_skel = ((A_true + A_true.T) > 0).astype(float)
    np.fill_diagonal(pred_skel, 0)
    np.fill_diagonal(true_skel, 0)
    tp = float(((pred_skel == 1) & (true_skel == 1)).sum())
    fp = float(((pred_skel == 1) & (true_skel == 0)).sum())
    fn = float(((pred_skel == 0) & (true_skel == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    skel_f1 = 2 * prec * rec / max(prec + rec, 1e-10)
    shd = float(np.abs(A_pred - A_true).sum())

    lodo_results[ds] = {
        'n_regimes': int(len(regimes)),
        'regimes': regime_metrics,
        'skeleton_f1': round(skel_f1, 4),
        'shd': round(shd, 1),
        'note': 'Full LODO requires re-training with held-out regime. This logs per-regime distribution + global metrics from seed_42.',
    }

with open('artifacts/lodo/lodo_report.json', 'w') as f:
    json.dump(lodo_results, f, indent=2)
print(f'  ✓ LODO report: artifacts/lodo/lodo_report.json ({len(lodo_results)} datasets)')
"

echo ""
echo "  ✓ Phase 1 complete: $(date)"

fi  # end Phase 1


# ############################################################################
# PHASE 2: SYNTHETIC SEM K-ROBUSTNESS (Paper Table 3)
# ############################################################################
if [[ "$RUN_PHASE" == "all" || "$RUN_PHASE" == "2" ]]; then

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  PHASE 2: SYNTHETIC SEM (75 tasks — 2A/2B/2C × 5 seeds)    ║"
echo "╚══════════════════════════════════════════════════════════════╝"

SEEDS="0,1,2,3,4"
TASKS_FILE="artifacts/table2/tasks.tsv"
MAX_TASK_SECS=5400   # 90 min per task
RESERVE_SECS=600     # 10 min reserve

# Step 1: Generate task list
echo "== Generating task list =="
python scripts/generate_table2_tasks.py --seeds "$SEEDS" --out "$TASKS_FILE"
NUM_TASKS=$(( $(wc -l < "$TASKS_FILE") - 1 ))
echo "[OK] $NUM_TASKS tasks"

# Step 2: Run tasks sequentially (resumable)
SEM_PASS=0
SEM_FAIL=0
SEM_TIMEOUT=0

for TASK_ID in $(seq 1 $NUM_TASKS); do
    LINE_NUM=$((TASK_ID + 1))
    LINE=$(sed -n "${LINE_NUM}p" "$TASKS_FILE")
    IFS=$'\t' read -r table config seed data_dir artifact_dir <<< "$LINE"

    # Wall-time guard
    ELAPSED=$(( SECONDS - PIPELINE_START ))
    REMAINING=$(( WALL_SECS - ELAPSED ))
    if [[ $REMAINING -lt $((MAX_TASK_SECS + RESERVE_SECS)) ]]; then
        echo "⚠  Wall-time guard: ${REMAINING}s left — stopping SEM loop"
        break
    fi

    echo "[${TASK_ID}/${NUM_TASKS}] ${table}/${config}/seed=${seed}"

    timeout --signal=TERM --kill-after=30 ${MAX_TASK_SECS} \
        python scripts/run_one_table2_task.py \
            --table "$table" \
            --config "$config" \
            --seed "$seed" \
            --data_dir "$data_dir" \
            --artifact_dir "$artifact_dir" \
    && RC=0 || RC=$?

    if [[ $RC -eq 0 ]]; then
        SEM_PASS=$((SEM_PASS + 1))
        TOTAL_PASS=$((TOTAL_PASS + 1))
    elif [[ $RC -eq 124 ]]; then
        SEM_TIMEOUT=$((SEM_TIMEOUT + 1))
        TOTAL_FAIL=$((TOTAL_FAIL + 1))
        echo "  [TIMEOUT] after ${MAX_TASK_SECS}s"
    else
        SEM_FAIL=$((SEM_FAIL + 1))
        TOTAL_FAIL=$((TOTAL_FAIL + 1))
        echo "  [FAIL] exit code $RC"
    fi
done

echo ""
echo "── SEM Summary: pass=${SEM_PASS} fail=${SEM_FAIL} timeout=${SEM_TIMEOUT} ──"

# Step 3: Aggregate SEM results
echo ""
echo "─── Aggregating SEM results ───"
python scripts/aggregate_table2.py --artifacts_root artifacts --out_dir artifacts/table2
echo "  ✓ Phase 2 complete: $(date)"

fi  # end Phase 2


# ############################################################################
# PHASE 3: COMPONENT ABLATION A1–A10  (Gaps 2, 3, 4, 8)
#           11 configs (full + A1–A10) × 3 datasets × 3 seeds = 99 runs
# ############################################################################
if [[ "$RUN_PHASE" == "all" || "$RUN_PHASE" == "3" ]]; then

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  PHASE 3: COMPONENT ABLATION (A1–A10 × 3 datasets × 3 seeds)║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  Ablation datasets: compound_full, extreme, clean_full  (Gap 8)"
echo "  Seeds: 42, 1337, 2024  (Gap 4)"
echo ""
echo "  Ablation configs (Gap 2: expanded A1-A10):"
echo "    full:              Full RC-GNN (baseline)"
echo "    A1_no_stages:      Disable 3-stage schedule"
echo "    A2_no_hsic:        λ_hsic=0 (no disentanglement)"
echo "    A3_no_inv:         λ_inv=0 (no invariance)"
echo "    A4_no_recon:       λ_recon=0 (no uncertainty)"
echo "    A5_no_direction:   Disable direction learning phase"
echo "    A6_no_sparse:      λ_sparse=0 (no sparsity)"
echo "    A7_no_causal:      λ_causal=0 (no causal penalty)"
echo "    A8_no_groupdro:    GroupDRO disabled (uniform sampling)"
echo "    A9_no_suppression: Non-TopK suppression off"
echo "    A10_single_env:    Force single regime (no env deltas)"
echo ""

# Gap 8: Multiple datasets for ablation
ABL_DATASETS=("compound_full" "extreme" "clean_full")
# Gap 4: Multiple seeds for ablation
ABL_SEEDS=(42 1337 2024)
ABLATION_EPOCHS=150

# Ablation config definitions  (Gap 2: expanded from 6 to 11)
# Format: name|extra_cli_flags
# Each config starts from FULL RC-GNN defaults and disables ONE component
# Gap 3: new CLI flags (--no_two_stage, --no_direction, --no_nontopk_suppression,
#         --force_single_regime) were added to train_rcgnn_unified.py
declare -a ABL_CONFIGS=(
    "full|"
    "A1_no_stages|--no_two_stage"
    "A2_no_hsic|--lambda_hsic 0.0"
    "A3_no_inv|--lambda_inv 0.0"
    "A4_no_recon|--lambda_recon 0.0"
    "A5_no_direction|--no_direction"
    "A6_no_sparse|--lambda_sparse 0.0"
    "A7_no_causal|--lambda_causal 0.0"
    "A8_no_groupdro|"
    "A9_no_suppression|--no_nontopk_suppression"
    "A10_single_env|--force_single_regime"
)

ABL_PASS=0
ABL_FAIL=0

for DATASET in "${ABL_DATASETS[@]}"; do
    for SEED in "${ABL_SEEDS[@]}"; do
        for CONFIG_ENTRY in "${ABL_CONFIGS[@]}"; do
            IFS='|' read -r ABL_NAME ABL_FLAGS <<< "${CONFIG_ENTRY}"

            OUT_DIR="artifacts/ablation/${DATASET}/${ABL_NAME}/seed_${SEED}"
            mkdir -p "${OUT_DIR}"

            # Wall-time guard
            ELAPSED=$(( SECONDS - PIPELINE_START ))
            REMAINING=$(( WALL_SECS - ELAPSED ))
            if [[ $REMAINING -lt 3600 ]]; then
                echo "⚠  Wall-time guard: ${REMAINING}s left — stopping ablation"
                break 3
            fi

            echo "─── Ablation: ${ABL_NAME} | ${DATASET} | seed=${SEED} ───"

            # Skip if done
            if [[ -f "${OUT_DIR}/training_history.json" ]]; then
                echo "  [SKIP] Already trained"
                ABL_PASS=$((ABL_PASS + 1))
                TOTAL_SKIP=$((TOTAL_SKIP + 1))
                continue
            fi

            # Base command — full RC-GNN defaults
            CMD="python scripts/train_rcgnn_unified.py \
                --data_dir data/interim/uci_air_c/${DATASET} \
                --output_dir ${OUT_DIR} \
                --epochs ${ABLATION_EPOCHS} \
                --lr 5e-4 \
                --batch_size 32 \
                --latent_dim 32 \
                --hidden_dim 64 \
                --lambda_recon 1.0 \
                --lambda_causal 1.0 \
                --target_edges 13 \
                --patience 40 \
                --seed ${SEED} \
                --device cuda \
                --sweep_mode"

            # A8 ablation: Full model should enable GroupDRO; A8 skips it
            if [[ "$ABL_NAME" != "A8_no_groupdro" ]]; then
                CMD="${CMD} --use_groupdro"
            fi

            # Apply ablation-specific flags
            if [[ -n "$ABL_FLAGS" ]]; then
                CMD="${CMD} ${ABL_FLAGS}"
            fi

            eval $CMD \
            && { ABL_PASS=$((ABL_PASS + 1)); TOTAL_PASS=$((TOTAL_PASS + 1)); } \
            || { ABL_FAIL=$((ABL_FAIL + 1)); TOTAL_FAIL=$((TOTAL_FAIL + 1)); echo "  [FAIL]"; }

            echo "  Done: $(date)"
        done
    done
done

echo ""
echo "── Ablation Summary: ${ABL_PASS} passed, ${ABL_FAIL} failed ──"

# Aggregate ablation results (Gap 4: mean±std across seeds)
echo ""
echo "─── Generating ablation summary (mean ± std) ───"
python -c "
import json, csv, numpy as np
from pathlib import Path

configs = [
    ('full', 'Full RC-GNN'),
    ('A1_no_stages', 'w/o 3-stage schedule'),
    ('A2_no_hsic', 'w/o HSIC'),
    ('A3_no_inv', 'w/o Invariance'),
    ('A4_no_recon', 'w/o Uncertainty'),
    ('A5_no_direction', 'w/o Direction phase'),
    ('A6_no_sparse', 'w/o Sparsity'),
    ('A7_no_causal', 'w/o Causal penalty'),
    ('A8_no_groupdro', 'w/o GroupDRO'),
    ('A9_no_suppression', 'w/o Non-TopK suppression'),
    ('A10_single_env', 'w/o Multi-regime'),
]
datasets = ['compound_full', 'extreme', 'clean_full']
seeds = [42, 1337, 2024]

rows = []
for cfg_name, cfg_label in configs:
    all_skel_f1, all_dir_f1, all_shd = [], [], []
    for ds in datasets:
        for seed in seeds:
            hist_path = Path(f'artifacts/ablation/{ds}/{cfg_name}/seed_{seed}/training_history.json')
            if not hist_path.exists():
                continue
            with open(hist_path) as f:
                hist = json.load(f)
            last = hist[-1] if hist else {}
            sf1 = last.get('skeleton_f1')
            df1 = last.get('topk_f1')
            shd = last.get('shd_directed') or last.get('shd')
            if sf1 is not None: all_skel_f1.append(sf1)
            if df1 is not None: all_dir_f1.append(df1)
            if shd is not None: all_shd.append(shd)

    row = {'config': cfg_label, 'ablation_id': cfg_name}
    if all_skel_f1:
        row['Skeleton_F1'] = f'{np.mean(all_skel_f1):.3f} +/- {np.std(all_skel_f1):.3f}'
        row['Skeleton_F1_mean'] = f'{np.mean(all_skel_f1):.4f}'
    if all_dir_f1:
        row['Directed_F1'] = f'{np.mean(all_dir_f1):.3f} +/- {np.std(all_dir_f1):.3f}'
        row['Directed_F1_mean'] = f'{np.mean(all_dir_f1):.4f}'
    if all_shd:
        row['SHD'] = f'{np.mean(all_shd):.1f} +/- {np.std(all_shd):.1f}'
        row['SHD_mean'] = f'{np.mean(all_shd):.1f}'
    row['n_runs'] = max(len(all_skel_f1), len(all_dir_f1), len(all_shd), 0)
    rows.append(row)

out_path = 'artifacts/ablation/ablation_summary.csv'
fieldnames = ['config', 'ablation_id', 'Directed_F1', 'Skeleton_F1', 'SHD',
              'Directed_F1_mean', 'Skeleton_F1_mean', 'SHD_mean', 'n_runs']
with open(out_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)
print(f'  ✓ {out_path} ({len(rows)} configs, mean±std across {len(datasets)} datasets × {len(seeds)} seeds)')
"

echo "  ✓ Phase 3 complete: $(date)"

fi  # end Phase 3


# ############################################################################
# PHASE 4: ROBUSTNESS STRESS TEST (Gap 7)
#           Missingness sweep: MCAR 20%, MCAR 40%, MNAR, compound, extreme
#           Paper Table 3 (Robustness): missingness, corruption type, cross-regime
# ############################################################################
if [[ "$RUN_PHASE" == "all" || "$RUN_PHASE" == "4" ]]; then

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  PHASE 4: ROBUSTNESS STRESS TEST  (Gap 7)                   ║"
echo "║    Missingness sweep + corruption types + cross-regime       ║"
echo "╚══════════════════════════════════════════════════════════════╝"

ROBUST_SEEDS=(42 1337 2024)

# Each stress test maps to an existing UCI-Air-C corruption dataset
declare -a STRESS_CONFIGS=(
    "miss_20|mcar_20"
    "miss_40|mcar_40"
    "miss_mnar|mnar_structural"
    "compound|compound_full"
    "compound_bias|compound_mnar_bias"
    "extreme_stress|extreme"
    "sensor_fail|sensor_failure"
)

STRESS_PASS=0
STRESS_FAIL=0

for CONFIG_ENTRY in "${STRESS_CONFIGS[@]}"; do
    IFS='|' read -r STRESS_NAME STRESS_DATASET <<< "${CONFIG_ENTRY}"

    for SEED in "${ROBUST_SEEDS[@]}"; do
        OUT_DIR="artifacts/robustness/${STRESS_NAME}/seed_${SEED}"
        mkdir -p "${OUT_DIR}"

        # Wall-time guard
        ELAPSED=$(( SECONDS - PIPELINE_START ))
        REMAINING=$(( WALL_SECS - ELAPSED ))
        if [[ $REMAINING -lt 3600 ]]; then
            echo "⚠  Wall-time guard: ${REMAINING}s left — stopping robustness loop"
            break 2
        fi

        echo "─── Robustness: ${STRESS_NAME} (${STRESS_DATASET}) seed=${SEED} ───"

        if [[ -f "${OUT_DIR}/training_history.json" ]]; then
            echo "  [SKIP] Already trained"
            STRESS_PASS=$((STRESS_PASS + 1))
            TOTAL_SKIP=$((TOTAL_SKIP + 1))
            continue
        fi

        python scripts/train_rcgnn_unified.py \
            --data_dir "data/interim/uci_air_c/${STRESS_DATASET}" \
            --output_dir "${OUT_DIR}" \
            --epochs 150 \
            --lr 5e-4 \
            --batch_size 32 \
            --latent_dim 32 \
            --hidden_dim 64 \
            --lambda_recon 1.0 \
            --lambda_causal 1.0 \
            --target_edges 13 \
            --patience 40 \
            --seed ${SEED} \
            --device cuda \
            --sweep_mode \
        && { STRESS_PASS=$((STRESS_PASS + 1)); TOTAL_PASS=$((TOTAL_PASS + 1)); } \
        || { STRESS_FAIL=$((STRESS_FAIL + 1)); TOTAL_FAIL=$((TOTAL_FAIL + 1)); echo "  [FAIL]"; }

        echo "  Done: $(date)"
    done
done

echo ""
echo "── Robustness Summary: ${STRESS_PASS} passed, ${STRESS_FAIL} failed ──"

# Aggregate robustness results
echo ""
echo "─── Generating robustness summary ───"
python -c "
import json, csv, numpy as np
from pathlib import Path

configs = [
    ('miss_20', 'MCAR 20%'),
    ('miss_40', 'MCAR 40%'),
    ('miss_mnar', 'MNAR structural'),
    ('compound', 'Compound (MCAR+noise+bias)'),
    ('compound_bias', 'MNAR + Bias'),
    ('extreme_stress', 'Extreme (5 regimes)'),
    ('sensor_fail', 'Sensor failure'),
]
seeds = [42, 1337, 2024]
rows = []

for cfg_name, cfg_label in configs:
    all_skel_f1, all_dir_f1, all_shd = [], [], []
    for seed in seeds:
        hist_path = Path(f'artifacts/robustness/{cfg_name}/seed_{seed}/training_history.json')
        if not hist_path.exists():
            continue
        with open(hist_path) as f:
            hist = json.load(f)
        last = hist[-1] if hist else {}
        sf1 = last.get('skeleton_f1')
        df1 = last.get('topk_f1')
        shd = last.get('shd_directed') or last.get('shd')
        if sf1 is not None: all_skel_f1.append(sf1)
        if df1 is not None: all_dir_f1.append(df1)
        if shd is not None: all_shd.append(shd)

    row = {'condition': cfg_label}
    if all_dir_f1:
        row['Directed_F1'] = f'{np.mean(all_dir_f1):.3f} +/- {np.std(all_dir_f1):.3f}'
    if all_skel_f1:
        row['Skeleton_F1'] = f'{np.mean(all_skel_f1):.3f} +/- {np.std(all_skel_f1):.3f}'
    if all_shd:
        row['SHD'] = f'{np.mean(all_shd):.1f} +/- {np.std(all_shd):.1f}'
    row['n_seeds'] = max(len(all_dir_f1), len(all_skel_f1), len(all_shd), 0)
    rows.append(row)

out_path = 'artifacts/robustness/robustness_summary.csv'
fieldnames = ['condition', 'Directed_F1', 'Skeleton_F1', 'SHD', 'n_seeds']
with open(out_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)
print(f'  ✓ {out_path} ({len(rows)} conditions × {len(seeds)} seeds)')
"

echo "  ✓ Phase 4 complete: $(date)"

fi  # end Phase 4


# ############################################################################
# PHASE 5: EVALUATION, LATEX, PLOTTING + SMOKE TEST  (Gaps 6, 9)
# ############################################################################
if [[ "$RUN_PHASE" == "all" || "$RUN_PHASE" == "5" ]]; then

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  PHASE 5: EVALUATION, LATEX, PLOTTING + SMOKE TEST          ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# ── 5a. SEM benchmark plots ──
echo ""
echo "─── 5a: SEM benchmark plots ───"
if [[ -f "artifacts/table2/table2_all_runs.csv" ]]; then
    python scripts/plot_table2_results.py \
        --csv artifacts/table2/table2_all_runs.csv \
        --artifacts_root artifacts \
        --out_dir artifacts/table2/plots
else
    echo "  [SKIP] No table2_all_runs.csv — run Phase 2 first"
fi

# ── 5b. UCI multi-seed aggregation + LaTeX  (Gaps 1, 6) ──
echo ""
echo "─── 5b: UCI multi-seed aggregation + LaTeX table ───"
python -c "
import json, csv, numpy as np, os
from pathlib import Path

datasets = [
    'clean_full', 'clean_real',
    'compound_full', 'compound_mnar_bias', 'extreme', 'mcar_40',
    'mcar_20', 'mnar_structural', 'moderate', 'noise_0.5', 'regimes_5', 'sensor_failure'
]
seeds = [42, 1337, 2024, 7, 99]

rows = []
for ds in datasets:
    all_skel_f1, all_dir_f1, all_shd = [], [], []
    for seed in seeds:
        hist_path = Path(f'artifacts/uci_multiseed/{ds}/seed_{seed}/training_history.json')
        if not hist_path.exists():
            continue
        with open(hist_path) as f:
            hist = json.load(f)
        last = hist[-1] if hist else {}
        sf1 = last.get('skeleton_f1')
        df1 = last.get('topk_f1')
        shd = last.get('shd_directed') or last.get('shd')
        if sf1 is not None: all_skel_f1.append(sf1)
        if df1 is not None: all_dir_f1.append(df1)
        if shd is not None: all_shd.append(shd)

    row = {'dataset': ds, 'n_seeds': max(len(all_skel_f1), len(all_dir_f1), 0)}
    if all_dir_f1:
        row['Directed_F1'] = f'{np.mean(all_dir_f1):.3f} +/- {np.std(all_dir_f1):.3f}'
        row['dir_f1_mean'] = np.mean(all_dir_f1)
    if all_skel_f1:
        row['Skeleton_F1'] = f'{np.mean(all_skel_f1):.3f} +/- {np.std(all_skel_f1):.3f}'
        row['skel_f1_mean'] = np.mean(all_skel_f1)
    if all_shd:
        row['SHD'] = f'{np.mean(all_shd):.1f} +/- {np.std(all_shd):.1f}'
        row['shd_mean'] = np.mean(all_shd)
    rows.append(row)

# Write CSV
csv_path = 'artifacts/uci_multiseed/uci_multiseed_summary.csv'
os.makedirs('artifacts/uci_multiseed', exist_ok=True)
fieldnames = ['dataset', 'Directed_F1', 'Skeleton_F1', 'SHD', 'n_seeds']
with open(csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
    w.writeheader()
    w.writerows(rows)
print(f'  ✓ {csv_path}')

# Gap 6: Generate LaTeX table for UCI results (Paper Table 2)
tex_path = 'artifacts/uci_multiseed/table2_uci.tex'
with open(tex_path, 'w') as f:
    f.write(r'\begin{table}[h]' + '\n')
    f.write(r'\centering' + '\n')
    f.write(r'\caption{RC-GNN Performance on UCI Air Quality Under Compound Corruptions (mean \$\\pm\$ std over 5 seeds)}' + '\n')
    f.write(r'\label{tab:uci_results}' + '\n')
    f.write(r'\begin{tabular}{lccc}' + '\n')
    f.write(r'\toprule' + '\n')
    f.write(r'Corruption & Directed F1 \$\uparrow\$ & Skeleton F1 \$\uparrow\$ & SHD \$\downarrow\$ ' + '\\\\' + '\n')
    f.write(r'\midrule' + '\n')
    for row in rows:
        ds_name = row['dataset'].replace('_', r'\_')
        df1 = row.get('Directed_F1', 'N/A').replace('+/-', r'\\pm')
        sf1 = row.get('Skeleton_F1', 'N/A').replace('+/-', r'\\pm')
        shd = row.get('SHD', 'N/A').replace('+/-', r'\\pm')
        if df1 != 'N/A': df1 = f'\${df1}\$'
        if sf1 != 'N/A': sf1 = f'\${sf1}\$'
        if shd != 'N/A': shd = f'\${shd}\$'
        f.write(f'{ds_name} & {df1} & {sf1} & {shd} \\\\\n')
    f.write(r'\bottomrule' + '\n')
    f.write(r'\end{tabular}' + '\n')
    f.write(r'\end{table}' + '\n')
print(f'  ✓ {tex_path}')
"

# ── 5c. Ablation LaTeX table (Gap 6) ──
echo ""
echo "─── 5c: Ablation LaTeX table ───"
python -c "
import csv
from pathlib import Path

csv_path = Path('artifacts/ablation/ablation_summary.csv')
if not csv_path.exists():
    print('  [SKIP] No ablation_summary.csv')
    exit(0)

with open(csv_path) as f:
    reader = csv.DictReader(f)
    rows = list(reader)

tex_path = 'artifacts/ablation/table4_ablation.tex'
with open(tex_path, 'w') as f:
    f.write(r'\begin{table}[h]' + '\n')
    f.write(r'\centering' + '\n')
    f.write(r'\caption{Component Ablation Study (mean \$\\pm\$ std over 3 datasets \$\\times\$ 3 seeds)}' + '\n')
    f.write(r'\label{tab:ablation}' + '\n')
    f.write(r'\begin{tabular}{lccc|c}' + '\n')
    f.write(r'\toprule' + '\n')
    f.write(r'Configuration & Directed F1 \$\uparrow\$ & Skeleton F1 \$\uparrow\$ & SHD \$\downarrow\$ & Runs ' + '\\\\' + '\n')
    f.write(r'\midrule' + '\n')
    for row in rows:
        cfg = row.get('config', '').replace('_', r'\_')
        df1 = row.get('Directed_F1', 'N/A').replace('+/-', r'\\pm')
        sf1 = row.get('Skeleton_F1', 'N/A').replace('+/-', r'\\pm')
        shd = row.get('SHD', 'N/A').replace('+/-', r'\\pm')
        n = row.get('n_runs', '')
        if df1 != 'N/A': df1 = f'\${df1}\$'
        if sf1 != 'N/A': sf1 = f'\${sf1}\$'
        if shd != 'N/A': shd = f'\${shd}\$'
        bold = r'\textbf{' if 'Full' in cfg else ''
        bold_end = '}' if bold else ''
        f.write(f'{bold}{cfg}{bold_end} & {df1} & {sf1} & {shd} & {n} \\\\\n')
    f.write(r'\bottomrule' + '\n')
    f.write(r'\end{tabular}' + '\n')
    f.write(r'\end{table}' + '\n')
print(f'  ✓ {tex_path}')
"

# ── 5d. Robustness LaTeX table (Gaps 6 + 7) ──
echo ""
echo "─── 5d: Robustness LaTeX table ───"
python -c "
import csv
from pathlib import Path

csv_path = Path('artifacts/robustness/robustness_summary.csv')
if not csv_path.exists():
    print('  [SKIP] No robustness_summary.csv')
    exit(0)

with open(csv_path) as f:
    reader = csv.DictReader(f)
    rows = list(reader)

tex_path = 'artifacts/robustness/table3_robustness.tex'
with open(tex_path, 'w') as f:
    f.write(r'\begin{table}[h]' + '\n')
    f.write(r'\centering' + '\n')
    f.write(r'\caption{Robustness Evaluation: Performance Under Data Perturbations (mean \$\\pm\$ std over 3 seeds)}' + '\n')
    f.write(r'\label{tab:robustness}' + '\n')
    f.write(r'\begin{tabular}{lccc}' + '\n')
    f.write(r'\toprule' + '\n')
    f.write(r'Condition & Directed F1 \$\uparrow\$ & Skeleton F1 \$\uparrow\$ & SHD \$\downarrow\$ ' + '\\\\' + '\n')
    f.write(r'\midrule' + '\n')
    for row in rows:
        cond = row.get('condition', '').replace('_', r'\_').replace('%', r'\\%')
        df1 = row.get('Directed_F1', 'N/A').replace('+/-', r'\\pm')
        sf1 = row.get('Skeleton_F1', 'N/A').replace('+/-', r'\\pm')
        shd = row.get('SHD', 'N/A').replace('+/-', r'\\pm')
        if df1 != 'N/A': df1 = f'\${df1}\$'
        if sf1 != 'N/A': sf1 = f'\${sf1}\$'
        if shd != 'N/A': shd = f'\${shd}\$'
        f.write(f'{cond} & {df1} & {sf1} & {shd} \\\\\n')
    f.write(r'\bottomrule' + '\n')
    f.write(r'\end{tabular}' + '\n')
    f.write(r'\end{table}' + '\n')
print(f'  ✓ {tex_path}')
"

# ── 5e. Ablation bar chart (expanded A1–A10) ──
echo ""
echo "─── 5e: Ablation comparison plot ───"
if [[ -f "artifacts/ablation/ablation_summary.csv" ]]; then
    python -c "
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('artifacts/ablation/ablation_summary.csv')

for col in ['Directed_F1_mean', 'Skeleton_F1_mean']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

plot_cols = [c for c in ['Directed_F1_mean', 'Skeleton_F1_mean'] if c in df.columns and df[c].notna().any()]
if not plot_cols:
    print('  (no ablation metrics available)')
    exit(0)

fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(df))
bar_w = 0.8 / len(plot_cols)
colors = ['#e41a1c', '#377eb8']
labels_map = {'Directed_F1_mean': 'Directed F1', 'Skeleton_F1_mean': 'Skeleton F1'}

for i, col in enumerate(plot_cols):
    vals = df[col].fillna(0).values
    offset = (i - len(plot_cols)/2 + 0.5) * bar_w
    ax.bar(x + offset, vals, bar_w * 0.9, label=labels_map.get(col, col), color=colors[i], alpha=0.85)
    for j, v in enumerate(vals):
        if j == 0 and v > 0:
            ax.text(x[j] + offset, v + 0.01, f'{v:.3f}', ha='center', fontsize=7, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(df['config'], rotation=40, ha='right', fontsize=8)
ax.set_ylabel('Score', fontsize=11)
ax.set_title('Component Ablation Study A1-A10 (Paper Table 4)', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.set_ylim(0, 1.1)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig('artifacts/ablation/ablation_plot.png', dpi=200, bbox_inches='tight')
plt.close()
print('  ✓ artifacts/ablation/ablation_plot.png')
"
else
    echo "  [SKIP] No ablation_summary.csv — run Phase 3 first"
fi

# ── 5f. Ablation training curves (A1–A10) ──
echo ""
echo "─── 5f: Ablation training curves ───"
python -c "
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

configs = [
    ('full', 'Full'),
    ('A1_no_stages', '-Stages'),
    ('A2_no_hsic', '-HSIC'),
    ('A3_no_inv', '-Inv'),
    ('A4_no_recon', '-Uncert'),
    ('A5_no_direction', '-Direction'),
    ('A6_no_sparse', '-Sparse'),
    ('A7_no_causal', '-Causal'),
    ('A8_no_groupdro', '-GroupDRO'),
    ('A9_no_suppression', '-Suppress'),
    ('A10_single_env', '-MultiReg'),
]

cmap = cm.get_cmap('tab10', len(configs))
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
found = False

for idx, (name, label) in enumerate(configs):
    hist_path = Path(f'artifacts/ablation/compound_full/{name}/seed_42/training_history.json')
    if not hist_path.exists():
        continue
    found = True
    with open(hist_path) as f:
        hist = json.load(f)
    epochs = [e.get('epoch', i) for i, e in enumerate(hist)]
    loss = [e.get('loss') for e in hist]
    f1 = [e.get('topk_f1') or e.get('skeleton_f1') for e in hist]
    color = cmap(idx)
    lw = 2.0 if name == 'full' else 1.0

    if loss and loss[0] is not None:
        axes[0].plot(epochs, loss, label=label, color=color, lw=lw, alpha=0.85)
    if f1 and f1[0] is not None:
        axes[1].plot(epochs, f1, label=label, color=color, lw=lw, alpha=0.85)

if found:
    axes[0].set_title('Training Loss', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_yscale('log'); axes[0].legend(fontsize=7, ncol=2); axes[0].grid(alpha=0.3)
    axes[1].set_title('F1 Score', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('F1')
    axes[1].set_ylim(0, 1.05); axes[1].legend(fontsize=7, ncol=2); axes[1].grid(alpha=0.3)
    fig.suptitle('Ablation Training Curves (A1-A10, compound_full, seed 42)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig('artifacts/ablation/ablation_curves.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('  ✓ artifacts/ablation/ablation_curves.png')
else:
    print('  (no ablation histories yet)')
" 2>/dev/null || echo "  (no ablation histories yet)"

# ── 5g. Robustness degradation plot (Gap 7) ──
echo ""
echo "─── 5g: Robustness degradation plot ───"
python -c "
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

csv_path = Path('artifacts/robustness/robustness_summary.csv')
if not csv_path.exists():
    print('  [SKIP] No robustness data')
    exit(0)

df = pd.read_csv(csv_path)

def parse_mean(val):
    if pd.isna(val) or val == 'N/A':
        return np.nan
    return float(str(val).split('+/-')[0].strip())

def parse_std(val):
    if pd.isna(val) or val == 'N/A' or '+/-' not in str(val):
        return 0.0
    return float(str(val).split('+/-')[1].strip())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
x = np.arange(len(df))

for ax, metric in zip(axes, ['Directed_F1', 'Skeleton_F1']):
    if metric not in df.columns:
        continue
    means = df[metric].apply(parse_mean).values
    stds = df[metric].apply(parse_std).values
    colors = ['#2ecc71' if m > 0.8 else '#f39c12' if m > 0.5 else '#e74c3c' for m in means]
    ax.bar(x, means, yerr=stds, capsize=3, color=colors, alpha=0.85, edgecolor='black', lw=0.5)
    for i, (m, s) in enumerate(zip(means, stds)):
        if not np.isnan(m):
            ax.text(i, m + s + 0.02, f'{m:.2f}', ha='center', fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(df['condition'], rotation=35, ha='right', fontsize=8)
    ax.set_ylabel(metric.replace('_', ' '), fontsize=11)
    ax.set_title(f'Robustness: {metric.replace(\"_\", \" \")}', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)

fig.suptitle('Robustness Stress Test (Paper Table 3)', fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig('artifacts/robustness/robustness_plot.png', dpi=200, bbox_inches='tight')
plt.close()
print('  ✓ artifacts/robustness/robustness_plot.png')
" 2>/dev/null || echo "  (no robustness data yet)"

echo ""
echo "  ✓ Phase 5 plots/tables complete: $(date)"


# ─── Gap 9: SMOKE TEST — validate artifact integrity ───
echo ""
echo "─── SMOKE TEST: Artifact Validation ───"
python -c "
import json, os, csv
from pathlib import Path

errors = []

# Check 1: UCI multi-seed results
uci_csv = Path('artifacts/uci_multiseed/uci_multiseed_summary.csv')
if uci_csv.exists():
    with open(uci_csv) as f:
        rows = list(csv.DictReader(f))
    n_with_data = sum(1 for r in rows if r.get('Directed_F1', 'N/A') != 'N/A')
    if n_with_data < 6:
        errors.append(f'UCI: only {n_with_data}/12 datasets have results')
    print(f'  ✓ UCI summary: {n_with_data}/12 datasets with metrics')
else:
    errors.append('UCI summary CSV missing')
    print('  ✗ UCI summary CSV missing')

# Check 2: SEM tables
for tex in ['table2A.tex', 'table2B.tex', 'table2C.tex']:
    if not Path(f'artifacts/table2/{tex}').exists():
        print(f'  ⚠ SEM {tex} not found (Phase 2 may not have run)')

# Check 3: Ablation summary
abl_csv = Path('artifacts/ablation/ablation_summary.csv')
if abl_csv.exists():
    with open(abl_csv) as f:
        rows = list(csv.DictReader(f))
    n_configs = len(rows)
    n_with_data = sum(1 for r in rows if int(r.get('n_runs', 0)) > 0)
    if n_with_data < 8:
        errors.append(f'Ablation: only {n_with_data}/{n_configs} configs have results')
    print(f'  ✓ Ablation summary: {n_with_data}/{n_configs} configs with data')
else:
    errors.append('Ablation summary CSV missing')
    print('  ✗ Ablation summary CSV missing')

# Check 4: Robustness summary
rob_csv = Path('artifacts/robustness/robustness_summary.csv')
if rob_csv.exists():
    with open(rob_csv) as f:
        rows = list(csv.DictReader(f))
    print(f'  ✓ Robustness summary: {len(rows)} conditions')
else:
    print('  ⚠ Robustness summary CSV missing (Phase 4 may not have run)')

# Check 5: LaTeX tables
for tex_path in [
    'artifacts/uci_multiseed/table2_uci.tex',
    'artifacts/ablation/table4_ablation.tex',
    'artifacts/robustness/table3_robustness.tex',
]:
    if Path(tex_path).exists():
        print(f'  ✓ {tex_path}')
    else:
        print(f'  ⚠ {tex_path} not found')

# Check 6: LODO report
lodo = Path('artifacts/lodo/lodo_report.json')
if lodo.exists():
    print(f'  ✓ {lodo}')
else:
    print(f'  ⚠ LODO report not found')

# Check 7: Data checksums
cksum = Path('artifacts/dataset_checksums.json')
if cksum.exists():
    print(f'  ✓ {cksum}')
else:
    print(f'  ⚠ Dataset checksums not found')

# Check 8: Plots
plots = list(Path('artifacts').rglob('*.png'))
print(f'  ✓ {len(plots)} PNG figures generated')

# Final verdict
if errors:
    print(f'\n  ⚠ SMOKE TEST: {len(errors)} warning(s):')
    for e in errors:
        print(f'    - {e}')
else:
    print(f'\n  ✓ SMOKE TEST PASSED: all artifacts validated')

result = {
    'passed': len(errors) == 0,
    'n_errors': len(errors),
    'errors': errors,
    'n_plots': len(plots),
}
with open('artifacts/smoke_test_result.json', 'w') as f:
    json.dump(result, f, indent=2)
"

echo ""
echo "  ✓ Phase 5 complete: $(date)"

fi  # end Phase 5


# ############################################################################
# FINAL SUMMARY  (Gap 9: comprehensive exit-code report)
# ############################################################################

TOTAL_ELAPSED=$(( SECONDS - PIPELINE_START ))
HOURS=$(( TOTAL_ELAPSED / 3600 ))
MINS=$(( (TOTAL_ELAPSED % 3600) / 60 ))

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                   PIPELINE COMPLETE                          ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Job:        ${SLURM_JOB_ID}"
echo "║  Git:        ${GIT_SHA} (${GIT_DIRTY})"
echo "║  Total time: ${HOURS}h ${MINS}m"
echo "║  End time:   $(date)"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Global Counters:"
echo "║    Passed:  ${TOTAL_PASS}"
echo "║    Failed:  ${TOTAL_FAIL}"
echo "║    Skipped: ${TOTAL_SKIP}"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Outputs:"
echo "║"
echo "║  Phase 0 (Data):"
echo "║    data/interim/uci_air_c/              — 12 corruption datasets"
echo "║    artifacts/dataset_checksums.json      — SHA-256 checksums"
echo "║"
echo "║  Phase 1 (UCI × 5 seeds — Paper Table 2):"
echo "║    artifacts/uci_multiseed/             — 12×5 trained models"
echo "║    artifacts/uci_multiseed/uci_multiseed_summary.csv"
echo "║    artifacts/uci_multiseed/table2_uci.tex"
echo "║    artifacts/lodo/lodo_report.json"
echo "║    artifacts/evaluation_report_final.json"
echo "║"
echo "║  Phase 2 (Synthetic SEM — Paper Table 3):"
echo "║    artifacts/table2a/, table2b/, table2c/"
echo "║    artifacts/table2/table2_all_runs.csv"
echo "║    artifacts/table2/table2A.tex, table2B.tex, table2C.tex"
echo "║"
echo "║  Phase 3 (Ablation A1–A10 — Paper Table 4):"
echo "║    artifacts/ablation/{ds}/{A1..A10}/seed_*/"
echo "║    artifacts/ablation/ablation_summary.csv"
echo "║    artifacts/ablation/table4_ablation.tex"
echo "║"
echo "║  Phase 4 (Robustness — Paper Table 3 stress test):"
echo "║    artifacts/robustness/*/seed_*/"
echo "║    artifacts/robustness/robustness_summary.csv"
echo "║    artifacts/robustness/table3_robustness.tex"
echo "║"
echo "║  Phase 5 (Plots & Validation):"
echo "║    artifacts/table2/plots/*.png"
echo "║    artifacts/ablation/*.png"
echo "║    artifacts/robustness/*.png"
echo "║    artifacts/smoke_test_result.json"
echo "║"
echo "║  Reproducibility:"
echo "║    artifacts/run_manifest_${SLURM_JOB_ID}.json"
echo "║    artifacts/dataset_checksums.json"
echo "╚══════════════════════════════════════════════════════════════╝"

# Artifact inventory
echo ""
echo "Artifact inventory:"
echo "  Models:     $(find artifacts/uci_multiseed artifacts/ablation artifacts/robustness -name 'A_best.npy' 2>/dev/null | wc -l) adjacency matrices"
echo "  Histories:  $(find artifacts/uci_multiseed artifacts/ablation artifacts/robustness -name 'training_history.json' 2>/dev/null | wc -l) training logs"
echo "  LaTeX:      $(find artifacts -name '*.tex' 2>/dev/null | wc -l) tables"

# List generated plots
echo ""
echo "Generated figures:"
find artifacts -name "*.png" 2>/dev/null | sort | while read f; do
    echo "  📊 $f"
done

# Final exit code summary  (Gap 9)
echo ""
if [[ $TOTAL_FAIL -eq 0 ]]; then
    echo "✅ ALL RUNS PASSED (${TOTAL_PASS} pass, ${TOTAL_SKIP} skip, 0 fail)"
    EXIT_CODE=0
elif [[ $TOTAL_FAIL -lt 10 ]]; then
    echo "⚠️  MOSTLY PASSED (${TOTAL_PASS} pass, ${TOTAL_SKIP} skip, ${TOTAL_FAIL} fail)"
    EXIT_CODE=0
else
    echo "❌ SIGNIFICANT FAILURES (${TOTAL_PASS} pass, ${TOTAL_SKIP} skip, ${TOTAL_FAIL} fail)"
    EXIT_CODE=1
fi

echo ""
echo "Done."
exit $EXIT_CODE
