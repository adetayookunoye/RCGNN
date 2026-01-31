#!/bin/bash
#SBATCH --job-name=rcgnn_eval
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/rcgnn_eval_%j.out
#SBATCH --error=logs/rcgnn_eval_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aoo29179@uga.edu

echo "=========================================="
echo "RC-GNN Comprehensive Evaluation"
echo "=========================================="
echo "Evaluating all 12 datasets with corrected threshold (0.5)"
echo "Start time: $(date)"
echo "Job ID: $SLURM_JOB_ID"

# Load modules (same as training script)
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

# Environment
cd /scratch/aoo29179/rcgnn
source ~/.bashrc

# Check environment
which python
python --version
echo "Checking numpy..."
python -c "import numpy; print(f'numpy version: {numpy.__version__}')"

echo ""
echo "======================================================"
echo "        COMPREHENSIVE EVALUATION (Threshold 0.5)      "
echo "======================================================"

# Create logs directory if it doesn't exist
mkdir -p logs

# Run comprehensive evaluation
python scripts/comprehensive_evaluation.py \
    --artifacts-dir artifacts \
    --data-dir data/interim \
    --output artifacts/evaluation_report_corrected.json

echo ""
echo "======================================================"
echo "EVALUATION COMPLETE"
echo "======================================================"
echo "Output: artifacts/evaluation_report_corrected.json"
echo "End time: $(date)"
