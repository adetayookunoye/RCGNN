#!/bin/bash
#SBATCH --job-name=v3_sweep
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/v3_sweep_%j.out
#SBATCH --error=logs/v3_sweep_%j.err

echo "=========================================="
echo "V3 Corruption Sweep"
echo "Start: $(date)"
echo "=========================================="

# Use the same environment as GPU jobs (known to work)
module load Python/3.11.3-GCCcore-12.3.0
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

cd /home/aoo29179/rcgnn
export OMP_NUM_THREADS=8

# Verify
python -c "import torch, numpy; print(f'OK: torch={torch.__version__}')"

# Corruptions to test
CORRUPTIONS="clean mild moderate severe extreme mcar_20 mcar_40 mnar_structural compound_full sensor_failure"

echo ""
echo "=== RESULTS ==="

for C in $CORRUPTIONS; do
    echo ""
    echo "--- $C ---"
    DATA="data/interim/uci_air_c/$C"
    
    if [ ! -d "$DATA" ]; then
        echo "SKIP: not found"
        continue
    fi
    
    # Run training (30 epochs for speed)
    python scripts/train_rcgnn_v3.py \
        --data_dir $DATA \
        --epochs 30 \
        --batch_size 64 \
        --device cpu \
        --seed 42 \
        --output_dir artifacts/v3_sweep/$C \
        2>&1 | tail -10
done

echo ""
echo "=========================================="
echo "COMPLETE: $(date)"
echo "=========================================="
