#!/bin/bash
#SBATCH --job-name=v3_sweep
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=logs/v3_sweep_%j.out
#SBATCH --error=logs/v3_sweep_%j.err

echo "=========================================="
echo "V3 Corruption Sweep"
echo "Start: $(date)"
echo "=========================================="

module load Python/3.11.3-GCCcore-12.3.0
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

cd /home/aoo29179/rcgnn
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

python -c "import torch; print(f'torch={torch.__version__}')"

# Results file
RESULTS="artifacts/v3_sweep_results.csv"
echo "corruption,best_f1,best_threshold,topk_f1,shd_0.3,final_loss" > $RESULTS

# Run each corruption
for C in clean mild moderate severe extreme mcar_20 mcar_40 mnar_structural compound_full sensor_failure; do
    echo ""
    echo "=== Testing: $C ==="
    DATA="data/interim/uci_air_c/$C"
    
    if [ ! -d "$DATA" ]; then
        echo "SKIP: $DATA not found"
        continue
    fi
    
    # Run 30 epochs (takes ~20 min per corruption)
    OUTPUT=$(python scripts/train_rcgnn_v3.py \
        --data_dir $DATA \
        --epochs 30 \
        --batch_size 64 \
        --device cpu \
        --seed 42 \
        --output_dir artifacts/v3_sweep/$C 2>&1)
    
    echo "$OUTPUT" | tail -5
    
    # Extract metrics from output
    BEST_F1=$(echo "$OUTPUT" | grep "Best:" | tail -1 | grep -oP "F1=\K[0-9.]+")
    BEST_T=$(echo "$OUTPUT" | grep "Best:" | tail -1 | grep -oP "@t=\K[0-9.]+")
    TOPK=$(echo "$OUTPUT" | grep "TopK-F1" | tail -1 | grep -oP "TopK-F1=\K[0-9.]+")
    SHD=$(echo "$OUTPUT" | grep "@0.3:" | tail -1 | grep -oP "SHD=\K[0-9]+")
    LOSS=$(echo "$OUTPUT" | grep "loss=" | tail -1 | grep -oP "loss=\K[0-9.]+")
    
    echo "$C,$BEST_F1,$BEST_T,$TOPK,$SHD,$LOSS" >> $RESULTS
    echo "[$C] Best F1=$BEST_F1 @ t=$BEST_T"
done

echo ""
echo "=========================================="
echo "COMPLETE: $(date)"
echo "=========================================="
echo ""
echo "=== FINAL RESULTS ==="
cat $RESULTS
