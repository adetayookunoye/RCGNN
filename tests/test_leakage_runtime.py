
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
sys.path.append(os.getcwd())
try:
    from scripts.train_rcgnn_unified import train_epoch, DEFAULT_CONFIG
except ImportError:
    # Try importing directly if scripts in path
    sys.path.append(os.path.join(os.getcwd(), 'scripts'))
    from train_rcgnn_unified import train_epoch, DEFAULT_CONFIG

from src.models.rcgnn import RCGNN

def test_no_leakage_when_oracle_false():
    """
    Verify that providing A_true does NOT affect the loss when oracle_direction_supervision is False.
    This ensures that A_true is completely ignored by the loss function, preventing leakage.
    """
    # Setup dummy data
    B, T, d = 4, 10, 5
    X = torch.randn(B, T, d)
    M = torch.ones_like(X)
    e = torch.zeros(B, dtype=torch.long)
    dataset = TensorDataset(X, M, e)
    loader = DataLoader(dataset, batch_size=B)
    
    # Setup dummy model
    model = RCGNN(d=d, latent_dim=4, hidden_dim=8, n_regimes=1)
    
    # Setup config
    config = DEFAULT_CONFIG.copy()
    config.update({
        "epochs": 10,
        "oracle_direction_supervision": False # KEY: Should disable leakage
    })
    
    device = torch.device("cpu")
    optimizer = torch.optim.Adam(model.parameters())
    
    # Case 1: A_true is None
    torch.manual_seed(42)
    # Re-init model to ensure identical start
    model = RCGNN(d=d, latent_dim=4, hidden_dim=8, n_regimes=1)
    
    metrics_none, _ = train_epoch(
        model, loader, optimizer, device,
        epoch=1, total_epochs=10, config=config,
        A_true=None
    )
    
    # Case 2: A_true is provided but oracle=False
    A_true = torch.zeros(d, d).float()
    A_true[0, 1] = 1.0 # Deterministic A_true, do not use RNG
    
    torch.manual_seed(42)
    model = RCGNN(d=d, latent_dim=4, hidden_dim=8, n_regimes=1) # Re-init
    
    metrics_with, _ = train_epoch(
        model, loader, optimizer, device,
        epoch=1, total_epochs=10, config=config,
        A_true=A_true
    )
    
    # Assertions
    # Losses should be IDENTICAL
    assert "loss" in metrics_none
    assert "loss" in metrics_with
    
    diff = abs(metrics_none["loss"] - metrics_with["loss"])
    assert diff < 1e-6, f"Leakage detected! Loss changed when A_true was provided (Oracle=False). Diff: {diff}"
    print("SUCCESS: Loss is identical with/without A_true when Oracle=False.")

def test_leakage_when_oracle_true():
    """
    Verify that providing A_true AND setting oracle=True DOES affects the loss (sanity check).
    We want to make sure the flag actually enables the supervision logic (so we can debug if needed).
    """
    # Setup dummy data
    B, T, d = 4, 10, 5
    X = torch.randn(B, T, d)
    M = torch.ones_like(X)
    e = torch.zeros(B, dtype=torch.long)
    dataset = TensorDataset(X, M, e)
    loader = DataLoader(dataset, batch_size=B)
    
    config = DEFAULT_CONFIG.copy()
    config.update({
        "epochs": 10,
        "oracle_direction_supervision": True, # KEY: Should ENABLE supervision
        "use_direction_margin": True,        # Ensure some A_true logic uses it
        "lambda_direction_margin": 10.0      # Make it obvious
    })
    
    device = torch.device("cpu")
    optimizer = torch.optim.Adam(torch.nn.Linear(1,1).parameters()) # Dummy
    
    # Case 1: A_true provided + Oracle=True
    torch.manual_seed(42)
    model = RCGNN(d=d, latent_dim=4, hidden_dim=8, n_regimes=1)
    
    A_true = torch.randint(0, 2, (d, d)).float()
    # Ensure A_true actually has edges so margin loss triggers
    A_true.fill_(0)
    A_true[0, 1] = 1.0 
    
    metrics_with, _ = train_epoch(
        model, loader, optimizer, device,
        epoch=1, total_epochs=10, config=config,
        A_true=A_true
    )
    
    # Case 2: A_true Not provided (but Oracle=True asked) - should fallback to no-supervision
    torch.manual_seed(42)
    model = RCGNN(d=d, latent_dim=4, hidden_dim=8, n_regimes=1)
    
    metrics_without, _ = train_epoch(
        model, loader, optimizer, device,
        epoch=1, total_epochs=10, config=config,
        A_true=None
    )
    
    # Assertions
    # Losses should be DIFFERENT (because A_true supervision adds loss)
    diff = abs(metrics_with["loss"] - metrics_without["loss"])
    assert diff > 1e-4, f"Oracle mode failed to add supervision usage! Loss identical. Diff: {diff}"
    print("SUCCESS: Oracle=True adds extra loss (as expected for debugging mode).")

if __name__ == "__main__":
    test_no_leakage_when_oracle_false()
    test_leakage_when_oracle_true()
