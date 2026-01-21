#!/bin/bash
#SBATCH --job-name=rcgnn_eval
#SBATCH --partition=gpu_p
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/eval_comprehensive_%j.out
#SBATCH --error=logs/eval_comprehensive_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aoo29179@uga.edu

echo "=========================================="
echo "RC-GNN Comprehensive Evaluation (GPU)"
echo "=========================================="
echo "Start time: $(date)"
echo "Job ID: $SLURM_JOB_ID"

# Load modules
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

# Environment
cd /scratch/aoo29179/rcgnn
source ~/.bashrc

# Check environment
which python
python --version
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo ""
echo "======================================================"
echo "  COMPREHENSIVE EVALUATION: RC-GNN vs ALL BASELINES  "
echo "======================================================"
echo "Baselines: RC-GNN, Correlation, NOTears-Lite, NOTEARS, Granger, PCMCI+, PC"
echo "Corruptions: 4 datasets (extreme, compound_full, compound_mnar_bias, mcar_40)"
echo "Metrics: SHD, F1 (skeleton + directed), Disentanglement, Invariance, Domain"
echo "======================================================"
echo ""

# Run comprehensive evaluation
python scripts/comprehensive_evaluation.py \
  --artifacts-dir artifacts \
  --data-dir data/interim \
  --output artifacts/evaluation_report_gpu_$(date +%s).json \
  2>&1 | tee logs/eval_comprehensive_${SLURM_JOB_ID}.log

EXIT_CODE=$?

echo ""
echo "======================================================"
echo "Evaluation completed with exit code: $EXIT_CODE"
echo "Output saved to: artifacts/evaluation_report_gpu_*.json"
echo "Log saved to: logs/eval_comprehensive_${SLURM_JOB_ID}.log"
echo "End time: $(date)"
echo "======================================================"

exit $EXIT_CODE
