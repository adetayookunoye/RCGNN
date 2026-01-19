#!/bin/bash
#SBATCH --job-name=rcgnn_ablation
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/ablation_%j.out

echo "======================================="
echo "RC-GNN Ablation + Corruption Experiments"
echo "Start time: $(date)"
echo "======================================="

cd ~/rcgnn
module load Python/3.11.3-GCCcore-12.3.0
module load PyTorch/2.1.2-foss-2023a

EPOCHS=50
DATA_DIR="data/interim/uci_air"

echo ""
echo "============================================="
echo "EXPERIMENT 1: ABLATION STUDY"
echo "============================================="

# Run 1: Full RC-GNN
echo ""
echo "--- RC-GNN (full) ---"
python scripts/train_rcgnn_v2.py \
    --data_dir $DATA_DIR \
    --epochs $EPOCHS \
    --lr 0.001 \
    --lambda_disentangle 0.1 \
    --lambda_sparse 0.0001 \
    --gamma_acyclic 0.01 \
    --save_prefix rcgnn_full \
    --device cpu 2>&1 | grep -E "Epoch|Val:|Best|Final"

# Run 2: Without disentangle
echo ""
echo "--- w/o disentangle ---"
python scripts/train_rcgnn_v2.py \
    --data_dir $DATA_DIR \
    --epochs $EPOCHS \
    --lr 0.001 \
    --lambda_disentangle 0.0 \
    --lambda_sparse 0.0001 \
    --gamma_acyclic 0.01 \
    --save_prefix rcgnn_no_disentangle \
    --device cpu 2>&1 | grep -E "Epoch|Val:|Best|Final"

# Run 3: Without sparsity (proxy for invariance)
echo ""
echo "--- w/o sparsity regularization ---"
python scripts/train_rcgnn_v2.py \
    --data_dir $DATA_DIR \
    --epochs $EPOCHS \
    --lr 0.001 \
    --lambda_disentangle 0.1 \
    --lambda_sparse 0.0 \
    --gamma_acyclic 0.01 \
    --save_prefix rcgnn_no_sparse \
    --device cpu 2>&1 | grep -E "Epoch|Val:|Best|Final"

echo ""
echo "============================================="
echo "EXPERIMENT 2: CORRUPTION SWEEP"
echo "============================================="
echo "(Using eval script with different corruption levels)"

# Run the corruption sweep Python script
python -c "
import numpy as np
import torch
import sys
sys.path.insert(0, 'src')
from rc_gnn import RCGNN_V2
from sklearn.metrics import average_precision_score

def apply_corruption(X, M, target_rate, seed=42):
    np.random.seed(seed)
    current = (M == 0).mean()
    if target_rate <= current:
        return X.copy(), M.copy()
    additional = (target_rate - current) / (1 - current)
    X_c, M_c = X.copy(), M.copy()
    observed = (M == 1)
    extra_mask = np.random.rand(*M.shape) < additional
    M_c[observed & extra_mask] = 0
    X_c[M_c == 0] = 0
    return X_c, M_c

def compute_metrics(A_pred, A_true):
    pred = A_pred.flatten()
    true = A_true.flatten().astype(int)
    auprc = average_precision_score(true, pred)
    k = int(A_true.sum())
    topk_idx = np.argsort(pred)[-k:]
    tp = true[topk_idx].sum()
    topk_f1 = 2 * (tp/k) * (tp/k) / (tp/k + tp/k + 1e-8) if tp > 0 else 0
    return auprc, topk_f1

# Load data
X = np.load('data/interim/uci_air/X.npy')
M = np.load('data/interim/uci_air/M.npy')
A_true = np.load('data/interim/uci_air/A_true.npy')

print('Corruption Rate | AUPRC | Top-k F1')
print('-' * 40)

for rate in [0.0, 0.10, 0.20, 0.30, 0.40]:
    X_c, M_c = apply_corruption(X, M, rate)
    actual = (M_c == 0).mean()
    
    # Load best model and evaluate
    try:
        ckpt = torch.load('artifacts/checkpoints/rcgnn_v2_best.pt', map_location='cpu')
        model = RCGNN_V2(d=X.shape[-1], input_dim=1, latent_dim=32, hidden_dim=64, n_envs=1)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        with torch.no_grad():
            A_pred = model.get_adjacency_matrix().numpy()
        auprc, topk_f1 = compute_metrics(A_pred, A_true)
        print(f'{rate*100:>5.0f}% ({actual*100:.1f}%) |  {auprc:.4f} | {topk_f1:.4f}')
    except Exception as e:
        print(f'{rate*100:>5.0f}% | Error: {e}')
"

echo ""
echo "============================================="
echo "EXPERIMENT 3: STABILITY (5 seeds)"
echo "============================================="

# Run 5 seeds and collect results
for SEED in 1 2 3 4 5; do
    echo ""
    echo "--- Seed $SEED ---"
    python scripts/train_rcgnn_v2.py \
        --data_dir $DATA_DIR \
        --epochs 30 \
        --lr 0.001 \
        --lambda_disentangle 0.1 \
        --lambda_sparse 0.0001 \
        --gamma_acyclic 0.01 \
        --seed $SEED \
        --save_prefix rcgnn_seed${SEED} \
        --device cpu 2>&1 | grep -E "Best F1|Final"
done

echo ""
echo "======================================="
echo "All experiments complete!"
echo "End time: $(date)"
echo "======================================="
