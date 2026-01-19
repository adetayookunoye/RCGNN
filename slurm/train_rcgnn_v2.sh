#!/bin/bash
#SBATCH --job-name=rcgnn_v2              # Job name
#SBATCH --partition=batch                 # Partition (queue) name
#SBATCH --ntasks=1                        # Run a single task
#SBATCH --cpus-per-task=8                 # Number of CPU cores per task
#SBATCH --mem=32G                         # Memory per node
#SBATCH --time=04:00:00                   # Time limit hrs:min:sec
#SBATCH --output=logs/rcgnn_v2_%j.out     # Standard output log
#SBATCH --error=logs/rcgnn_v2_%j.err      # Standard error log
#SBATCH --mail-type=BEGIN,END,FAIL        # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=adetayo@uga.edu       # Where to send mail

# Print job info
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start time: $(date)"
echo "========================================"

# Change to project directory
cd $SLURM_SUBMIT_DIR
# Or use absolute path:
# cd /home/adetayo/rcgnn

# Load required modules (adjust based on Sapelo's available modules)
module load PyTorch/2.1.2-foss-2023a

# Activate virtual environment if you have one
# source ~/.venv/rcgnn/bin/activate
# Or conda:
# source ~/.bashrc
# conda activate rcgnn

# Set number of threads for PyTorch
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Create logs directory if it doesn't exist
mkdir -p logs

# Unbuffered Python output
export PYTHONUNBUFFERED=1

# Run training
echo "Starting RC-GNN v2 training..."
python -u scripts/train_rcgnn_v2.py \
    --epochs 100 \
    --batch_size 64 \
    --lr 0.001 \
    --lambda_sparse 0.01 \
    --gamma_acyclic 1.0 \
    --lambda_disentangle 0.1 \
    --lambda_recon 1.0 \
    --data_dir data/interim/uci_air \
    --seed 42

echo "========================================"
echo "End time: $(date)"
echo "========================================"
