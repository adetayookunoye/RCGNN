import pytest
import torch
import numpy as np
import os
from src.models.disentanglement import MINELoss, InfoNCELoss, compute_disentanglement_metrics
from src.models.rcgnn import RCGNN
from src.training.loop import train_epoch
from src.dataio.synth import build_synth

@pytest.fixture
def synthetic_batch():
    """Generate synthetic batch for testing."""
    root = build_synth("data/interim/test_synth", d=10, T=128, regimes=2, Tw=32, Ts=16)
    X = np.load(os.path.join(root, "X.npy"))
    M = np.load(os.path.join(root, "M.npy"))
    e = np.load(os.path.join(root, "e.npy"))
    S = np.load(os.path.join(root, "S.npy"))
    return {
        'X': torch.tensor(X, dtype=torch.float32),
        'M': torch.tensor(M, dtype=torch.float32),
        'e': torch.tensor(e, dtype=torch.long),
        'S': torch.tensor(S, dtype=torch.float32)
    }

def test_mine_loss_decreases():
    """Test that MINE loss decreases during training."""
    x = torch.randn(100, 32)
    y = x + 0.1 * torch.randn(100, 32)  # Correlated with x
    z = torch.randn(100, 32)  # Independent
    
    mine = MINELoss(input_dim=32)
    optimizer = torch.optim.Adam(mine.parameters())
    
    initial_mi = []
    final_mi = []
    
    # Train for a few steps
    for _ in range(5):
        mi_xy, _ = mine(x, y)
        mi_xz, _ = mine(x, z)
        initial_mi.append((mi_xy.item(), mi_xz.item()))
        
    for _ in range(50):
        optimizer.zero_grad()
        mi_xy, reg = mine(x, y)
        loss = -mi_xy + reg  # Maximize MI
        loss.backward()
        optimizer.step()
        
    for _ in range(5):
        mi_xy, _ = mine(x, y)
        mi_xz, _ = mine(x, z)
        final_mi.append((mi_xy.item(), mi_xz.item()))
        
    # MI should be higher for correlated variables after training
    assert sum(f[0] > f[1] for f in final_mi) > sum(i[0] > i[1] for i in initial_mi)

def test_infonce_accuracy_improves():
    """Test that InfoNCE positive pair accuracy improves."""
    x = torch.randn(100, 32)
    y = x + 0.1 * torch.randn(100, 32)
    
    infonce = InfoNCELoss(input_dim=32)
    optimizer = torch.optim.Adam(infonce.parameters())
    
    initial_accs = []
    final_accs = []
    
    # Get initial accuracies
    for _ in range(5):
        _, acc = infonce(x, y)
        initial_accs.append(acc.item())
        
    # Train
    for _ in range(50):
        optimizer.zero_grad()
        loss, _ = infonce(x, y)
        loss.backward()
        optimizer.step()
        
    # Get final accuracies
    for _ in range(5):
        _, acc = infonce(x, y)
        final_accs.append(acc.item())
        
    assert sum(final_accs) / len(final_accs) > sum(initial_accs) / len(initial_accs)

def test_disentanglement_improves(synthetic_batch):
    """Test that latent spaces become more disentangled during training."""
    model = RCGNN(
        d=synthetic_batch['X'].shape[-1],
        cfg={
            "imputer": {"d_model": 32, "n_layers": 2},
            "encoders": {
                "z_s": {"d_hidden": 64},
                "z_n": {"d_hidden": 64},
                "z_b": {"d_hidden": 16}
            },
            "structure": {
                "gnn_hidden": 64,
                "sparsify": {"method": "topk", "k": 3},
                "temp_start": 5.0, "temp_end": 0.5
            },
            "mechanisms": {"node_mlp": [64, 64]},
            "loss": {
                "lambda_dis": 0.1,
                "lambda_acy": 1.0,
                "lambda_sparse": 1.0
            },
            'disentangle': {
                'method': 'mine',
                'hidden_dim': 64,
                'lambda_mi': 0.1
            }
        }
    )
    optimizer = torch.optim.Adam(model.parameters())
    
    # Track disentanglement metrics
    initial_metrics = []
    for _ in range(3):
        outputs = model(synthetic_batch)
        metrics = compute_disentanglement_metrics(
            outputs['z_s'],
            outputs['z_n'],
            outputs['z_b']
        )
        initial_metrics.append(metrics)
    
    # Train for several steps
    for _ in range(20):
        train_epoch(model, [synthetic_batch], optimizer)
        
    # Check final disentanglement
    final_metrics = []
    for _ in range(3):
        outputs = model(synthetic_batch)
        metrics = compute_disentanglement_metrics(
            outputs['z_s'],
            outputs['z_n'],
            outputs['z_b']
        )
        final_metrics.append(metrics)
    
    # Average correlations should decrease
    def avg_correlation(metrics_list):
        corrs = []
        for m in metrics_list:
            corrs.extend([m['corr_s_n'], m['corr_s_b'], m['corr_n_b']])
        return sum(corrs) / len(corrs)
        
    assert avg_correlation(final_metrics) < avg_correlation(initial_metrics)
    
    # HSIC independence should decrease
    def avg_hsic(metrics_list):
        hsics = []
        for m in metrics_list:
            hsics.extend([m['hsic_s_n'], m['hsic_s_b'], m['hsic_n_b']])
        return sum(hsics) / len(hsics)
            
    assert avg_hsic(final_metrics) < avg_hsic(initial_metrics)