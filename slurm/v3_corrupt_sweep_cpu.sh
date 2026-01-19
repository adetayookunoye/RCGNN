#!/bin/bash
#SBATCH --job-name=v3_corrupt_sweep
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/v3_corrupt_sweep_%j.out
#SBATCH --error=logs/v3_corrupt_sweep_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aoo29179@uga.edu

echo "=========================================="
echo "V3 Corruption Sweep (8 CPU)"
echo "=========================================="
echo "Start time: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "CPUs: $SLURM_CPUS_PER_TASK"

# Load modules
module load Python/3.11.3-GCCcore-12.3.0
module load SciPy-bundle/2023.07-gfbf-2023a

# Environment
cd /home/aoo29179/rcgnn
export OMP_NUM_THREADS=8

# Install/verify dependencies
pip install --user torch --quiet 2>/dev/null

# Check Python
which python
python --version
python -c "import numpy; import torch; print(f'numpy={numpy.__version__}, torch={torch.__version__}')"

# Define corruption levels to test (key ones for paper)
CORRUPTIONS=(
    "clean"
    "mild"
    "moderate" 
    "severe"
    "extreme"
    "mcar_20"
    "mcar_40"
    "mnar_structural"
    "compound_full"
    "sensor_failure"
)

# Output file for results
RESULTS_FILE="artifacts/v3_corruption_sweep_results.txt"
echo "=== V3 Corruption Sweep Results ===" > $RESULTS_FILE
echo "Date: $(date)" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE

# Run V3 on each corruption level
for CORRUPT in "${CORRUPTIONS[@]}"; do
    echo ""
    echo "======================================================"
    echo "Testing: $CORRUPT"
    echo "======================================================"
    
    DATA_DIR="data/interim/uci_air_c/$CORRUPT"
    
    if [ ! -d "$DATA_DIR" ]; then
        echo "SKIP: $DATA_DIR not found"
        continue
    fi
    
    echo "[$CORRUPT] Starting at $(date)"
    
    # Run V3 with 50 epochs (enough for convergence)
    python scripts/train_rcgnn_v3.py \
        --data_dir $DATA_DIR \
        --epochs 50 \
        --batch_size 64 \
        --lr 1e-3 \
        --latent_dim 32 \
        --hidden_dim 64 \
        --lambda_recon 1.0 \
        --lambda_miss 0.1 \
        --lambda_hsic 0.1 \
        --lambda_acyclic 0.1 \
        --lambda_sparse 0.01 \
        --warmup_epochs 5 \
        --ramp_epochs 10 \
        --device cpu \
        --seed 42 \
        --output_dir artifacts/v3_sweep/$CORRUPT \
        2>&1 | tee -a logs/v3_${CORRUPT}.log
    
    # Extract final metrics
    FINAL_LINE=$(tail -5 logs/v3_${CORRUPT}.log | grep -E "Best|Final|F1")
    echo "[$CORRUPT] $FINAL_LINE" >> $RESULTS_FILE
    
    echo "[$CORRUPT] Completed at $(date)"
done

echo ""
echo "=========================================="
echo "SWEEP COMPLETE"
echo "=========================================="
echo "End time: $(date)"
echo ""
echo "=== SUMMARY ==="
cat $RESULTS_FILE
