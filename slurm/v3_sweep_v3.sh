#!/bin/bash
#SBATCH --job-name=v3_sweep
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/v3_sweep_%j.out
#SBATCH --error=logs/v3_sweep_%j.err

echo "=========================================="
echo "V3 Corruption Sweep (THREE-STAGE BUDGET + SCHEDULED SPARSITY)"
echo "Start: $(date)"
echo "=========================================="

module load Python/3.11.3-GCCcore-12.3.0
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

cd /home/aoo29179/rcgnn
export OMP_NUM_THREADS=64
export MKL_NUM_THREADS=64
export NUMEXPR_NUM_THREADS=64
export PYTHONUNBUFFERED=1

# Optimize PyTorch for CPU
export TORCH_NUM_THREADS=64

python -c "import torch; print(f'torch={torch.__version__}, threads={torch.get_num_threads()}')"

# Results file - proper CSV with all metrics (Best-F1 is PRIMARY)
RESULTS="artifacts/v3_sweep_results_v4.csv"
echo "corruption,best_f1,best_threshold,shd_best,auc_f1,topk_f1,topk_shd,shd_03,a_max" > $RESULTS

# All corruption variants (FIXED directory names)
# clean_full = no missingness (imputed), clean_real = natural missingness (14.68%)
CORRUPTIONS="clean_full clean_real mild moderate severe extreme mcar_20 mcar_30 mcar_40 mnar_self mnar_threshold mnar_structural noise_0.1 noise_0.3 noise_0.5 bias_additive regimes_3 regimes_5 compound_full sensor_failure"

for C in $CORRUPTIONS; do
    echo ""
    echo "=== Testing: $C ==="
    DATA="data/interim/uci_air_c/$C"
    
    if [ ! -d "$DATA" ]; then
        echo "SKIP: $DATA not found"
        continue
    fi
    
    # Run training (batch_size=128 for faster CPU, num_workers=8)
    python scripts/train_rcgnn_v3.py \
        --data_dir $DATA \
        --epochs 50 \
        --batch_size 128 \
        --device cpu \
        --seed 42 \
        --output_dir artifacts/v3_sweep/$C
    
    # Read metrics from JSON (more reliable than parsing stdout)
    METRICS_FILE="artifacts/v3_sweep/$C/final_metrics.json"
    if [ -f "$METRICS_FILE" ]; then
        # Parse JSON with python - Best-F1 is PRIMARY
        python3 -c "
import json
with open('$METRICS_FILE') as f:
    m = json.load(f)
print(f'$C,{m.get(\"best_f1\",0):.4f},{m.get(\"best_threshold\",0.3):.2f},{m.get(\"shd_best\",999)},{m.get(\"auc_f1\",0):.4f},{m.get(\"topk_f1\",0):.4f},{m.get(\"topk_shd\",999)},{m.get(\"shd_03\",999)},{m.get(\"a_max\",0):.4f}')
" >> $RESULTS
        # Simple echo of result
        BEST_F1=$(python3 -c "import json; m=json.load(open('$METRICS_FILE')); print(f\"{m.get('best_f1',0):.4f}\")")
        echo "[$C] Best-F1=$BEST_F1"
    else
        echo "$C,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR" >> $RESULTS
        echo "[$C] ERROR: No metrics file"
    fi
done

echo ""
echo "=========================================="
echo "COMPLETE: $(date)"
echo "=========================================="
echo ""
echo "=== FINAL RESULTS ==="
cat $RESULTS
echo ""
echo "=== SUMMARY TABLE (Best-F1 = PRIMARY METRIC) ==="
python3 << 'PYEOF'
import csv
with open('artifacts/v3_sweep_results_v4.csv') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

header = f"{'Corruption':<20} {'Best-F1':>8} {'SHD@best':>9} {'AUC-F1':>8} {'TopK-F1':>8} {'SHD@0.3':>7}"
print(header)
print('-' * 70)
for r in rows:
    try:
        line = f"{r['corruption']:<20} {float(r['best_f1']):>8.4f} {int(r['shd_best']):>9} {float(r['auc_f1']):>8.4f} {float(r['topk_f1']):>8.4f} {int(r['shd_03']):>7}"
        print(line)
    except:
        print(f"{r['corruption']:<20} ERROR")
PYEOF
