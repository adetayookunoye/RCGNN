#!/bin/bash
#SBATCH --job-name=rcgnn_synth_sem
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/synth_sem_%j.out
#SBATCH --error=logs/synth_sem_%j.err
#SBATCH --mail-type=END,FAIL

# ============================================================================
# SYNTHETIC SEM BENCHMARK PIPELINE
# ============================================================================
# Purpose: Run complete pipeline for synthetic SEM with KNOWN TRUE DAG
# This addresses reviewer concern about domain-knowledge-based evaluation
#
# Stages:
#   1. Generate synthetic SEM data (d=13, 13 edges, MNAR+bias+noise)
#   2. Train RC-GNN on synthetic data
#   3. Evaluate against baselines using KNOWN TRUE DAG
#
# Output: artifacts/synth_sem_*/
# ============================================================================

echo "======================================================"
echo " RC-GNN Synthetic SEM Pipeline"
echo " Job ID: $SLURM_JOB_ID"
echo " Start time: $(date)"
echo "======================================================"

# Setup
cd /scratch/aoo29179/rcgnn

# Load modules (same as working eval script)
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
source ~/.bashrc

# Verify Python
which python
python --version

mkdir -p logs artifacts

# Configuration
SEED=42
BENCHMARK="compound_sem_medium"  # Options: compound_sem_easy, compound_sem_medium, compound_sem_hard
DATA_ROOT="data/interim/synth_${BENCHMARK}"
ARTIFACT_DIR="artifacts/synth_${BENCHMARK}"

echo ""
echo "Configuration:"
echo "  Benchmark: $BENCHMARK"
echo "  Data root: $DATA_ROOT"
echo "  Artifact dir: $ARTIFACT_DIR"
echo "  Seed: $SEED"

# ============================================================================
# STAGE 1: GENERATE SYNTHETIC SEM DATA
# ============================================================================
echo ""
echo "======================================================"
echo " STAGE 1: Generate Synthetic SEM Data"
echo "======================================================"

python scripts/run_synthetic_benchmark.py \
    --benchmark $BENCHMARK \
    --seed $SEED

if [ $? -ne 0 ]; then
    echo "[FAIL] Synthetic data generation failed!"
    exit 1
fi

echo "[OK] Synthetic data generated at $DATA_ROOT"
echo ""
echo "Generated files:"
ls -la $DATA_ROOT/

# Show true DAG info
python -c "
import numpy as np
A = np.load('${DATA_ROOT}/A_true.npy')
print(f'TRUE DAG (known by construction):')
print(f'  Nodes: {A.shape[0]}')
print(f'  Edges: {int(A.sum())}')
print(f'  Density: {A.sum() / (A.shape[0] * (A.shape[0]-1)):.3f}')
"

# ============================================================================
# STAGE 2: TRAIN RC-GNN ON SYNTHETIC DATA
# ============================================================================
echo ""
echo "======================================================"
echo " STAGE 2: Train RC-GNN on Synthetic SEM Data"
echo "======================================================"

mkdir -p $ARTIFACT_DIR

# Create temporary configs for synthetic data
cat > configs/synth_data.yaml << EOF
paths:
  root: "$DATA_ROOT"
  output: "$ARTIFACT_DIR"
EOF

cat > configs/synth_train.yaml << EOF
device: "cuda"
seed: $SEED
epochs: 300
batch_size: 32
learning_rate: 0.001
weight_decay: 0.0001
patience: 50
log_interval: 10
checkpoint_interval: 50
gradient_clip: 1.0
EOF

# Use correct CLI arguments for train_rcgnn_unified.py
python scripts/train_rcgnn_unified.py \
    --data_dir $DATA_ROOT \
    --output_dir $ARTIFACT_DIR \
    --epochs 300 \
    --batch_size 32 \
    --lr 0.001 \
    --weight_decay 0.0001 \
    --patience 50 \
    --grad_clip 1.0 \
    --target_edges 13 \
    --seed $SEED \
    --device cuda

if [ $? -ne 0 ]; then
    echo "[FAIL] Training failed!"
    exit 1
fi

echo "[OK] Training complete"
echo ""
echo "Training artifacts:"
ls -la $ARTIFACT_DIR/

# Copy ground truth for evaluation
cp $DATA_ROOT/A_true.npy $ARTIFACT_DIR/

# ============================================================================
# STAGE 3: EVALUATE AGAINST BASELINES
# ============================================================================
echo ""
echo "======================================================"
echo " STAGE 3: Evaluate Against Baselines (Known True DAG)"
echo "======================================================"

# Run comprehensive evaluation
python scripts/comprehensive_evaluation.py \
    --artifacts_dir $ARTIFACT_DIR \
    --data_root $DATA_ROOT \
    --output_file $ARTIFACT_DIR/evaluation_synth_sem.json

if [ $? -ne 0 ]; then
    echo "[WARN] Evaluation had issues, checking output..."
fi

echo ""
echo "======================================================"
echo " RESULTS SUMMARY"
echo "======================================================"

# Print summary from evaluation
python -c "
import json
import os

eval_file = '${ARTIFACT_DIR}/evaluation_synth_sem.json'
if os.path.exists(eval_file):
    with open(eval_file) as f:
        results = json.load(f)
    
    print('SYNTHETIC SEM BENCHMARK RESULTS')
    print('=' * 60)
    print('Ground truth: KNOWN BY CONSTRUCTION (not domain knowledge!)')
    print()
    
    if 'baseline_comparison' in results:
        print('Method Comparison:')
        print('-' * 60)
        for entry in results['baseline_comparison'][:7]:
            method = entry.get('Method', 'Unknown')
            shd = entry.get('SHD', 'N/A')
            skel_f1 = entry.get('Skeleton_F1', 'N/A')
            dir_f1 = entry.get('Directed_F1', 'N/A')
            if isinstance(skel_f1, float):
                print(f'{method:20s} | SHD={shd:3d} | Skel-F1={skel_f1:.3f} | Dir-F1={dir_f1:.3f}')
            else:
                print(f'{method:20s} | SHD={shd} | Skel-F1={skel_f1} | Dir-F1={dir_f1}')
else:
    print('Evaluation file not found. Check logs for errors.')
"

echo ""
echo "======================================================"
echo " OUTPUT FILES"
echo "======================================================"
echo ""
echo "Synthetic data: $DATA_ROOT/"
echo "Trained model:  $ARTIFACT_DIR/"
echo "Evaluation:     $ARTIFACT_DIR/evaluation_synth_sem.json"
echo ""
echo "To download results:"
echo "  rsync -avz sapelo2:$ARTIFACT_DIR/ local_artifacts/synth_${BENCHMARK}/"
echo ""
echo "======================================================"
echo " Pipeline completed with exit code: $?"
echo " End time: $(date)"
echo "======================================================"
