#!/bin/bash
#SBATCH --job-name=uci_eval
#SBATCH --partition=gpu_p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/uci_eval_%j.out
#SBATCH --error=logs/uci_eval_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aoo29179@uga.edu

# =============================================================================
# UCI Air Quality Baseline Evaluation
# Runs comprehensive_evaluation.py --single-run --temporal against all 60
# UCI artifacts (12 corruptions × 5 seeds) to compute ALL baselines:
#   IID: Correlation, PC, GES, NOTEARS, NOTEARS-MLP, GOLEM
#   Temporal: Granger (MICE imputation), PCMCI+ (k-NN imputation)
# =============================================================================

set -euo pipefail

cd /scratch/aoo29179/rcgnn

# Load environment
module load Python/3.11.3-GCCcore-12.3.0
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

# Install required packages for temporal baselines
echo "── Installing temporal baseline dependencies ──"
pip install --user --quiet statsmodels tigramite 2>&1 | tail -3
echo ""

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  UCI Air Quality — Full Baseline Evaluation                        ║"
echo "║  12 configs × 5 seeds = 60 runs                                    ║"
echo "║  Baselines: Correlation, PC, GES, NOTEARS, NOTEARS-MLP, GOLEM,    ║"
echo "║             Granger (MICE), PCMCI+ (k-NN)                         ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo "Start: $(date)"
echo "Node:  $(hostname)"
echo ""

CORRUPTIONS=(
    clean_full
    clean_real
    mcar_20
    mcar_40
    noise_0.5
    moderate
    compound_full
    compound_mnar_bias
    mnar_structural
    sensor_failure
    regimes_5
    extreme
)

SEEDS=(42 1337 2024 7 99)

TOTAL=$((${#CORRUPTIONS[@]} * ${#SEEDS[@]}))
DONE=0
PASS=0
FAIL=0
SKIP=0

for CORRUPTION in "${CORRUPTIONS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        DONE=$((DONE + 1))

        ARTIFACT_DIR="artifacts/uci_multiseed/${CORRUPTION}/seed_${SEED}"
        DATA_DIR="data/interim/uci_air_c/${CORRUPTION}"
        OUTPUT="${ARTIFACT_DIR}/evaluation.json"

        echo ""
        echo "══ [${DONE}/${TOTAL}] ${CORRUPTION} / seed_${SEED} ══"

        # Skip if already evaluated WITH temporal baselines
        if [[ -f "${OUTPUT}" ]]; then
            # Check if temporal baselines are already present
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
                SKIP=$((SKIP + 1))
                PASS=$((PASS + 1))
                continue
            else
                echo "  [RE-EVAL] Missing temporal baselines, re-running..."
                rm -f "${OUTPUT}"
            fi
        fi

        # Check prerequisites
        if [[ ! -f "${ARTIFACT_DIR}/A_best.npy" ]]; then
            echo "  [SKIP] No A_best.npy — training may not have completed"
            FAIL=$((FAIL + 1))
            continue
        fi
        if [[ ! -f "${DATA_DIR}/X.npy" ]]; then
            echo "  [SKIP] No data at ${DATA_DIR}"
            FAIL=$((FAIL + 1))
            continue
        fi

        # Run comprehensive evaluation with --temporal flag
        echo "  Running: comprehensive_evaluation.py --single-run --temporal"
        python scripts/comprehensive_evaluation.py \
            --artifacts-dir "${ARTIFACT_DIR}" \
            --data-dir "${DATA_DIR}" \
            --output "${OUTPUT}" \
            --single-run \
            --temporal \
            2>&1 | tail -40

        if [[ -f "${OUTPUT}" ]]; then
            echo "  [OK] → ${OUTPUT}"
            PASS=$((PASS + 1))
        else
            echo "  [FAIL] No output produced"
            FAIL=$((FAIL + 1))
        fi
    done
done

echo ""
echo "══════════════════════════════════════════════════════════════════════"
echo "  UCI Baseline Evaluation Complete"
echo "  Passed: ${PASS}/${TOTAL}  Failed: ${FAIL}/${TOTAL}  Skipped: ${SKIP}"
echo "  End: $(date)"
echo "══════════════════════════════════════════════════════════════════════"

# ============================================================================
# Aggregate results into summary table
# ============================================================================
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
print('UCI AIR QUALITY — RC-GNN vs ALL BASELINES (Skeleton F1, mean +/- std across 5 seeds)')
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
                bl = ev.get('baselines', {}).get(m, {})
                val = bl.get('Skeleton_F1', None)
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
print('UCI AIR QUALITY — RC-GNN vs ALL BASELINES (Directed F1, mean +/- std across 5 seeds)')
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
                bl = ev.get('baselines', {}).get(m, {})
                val = bl.get('Directed_F1', None)
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

echo ""
echo "Done: $(date)"
