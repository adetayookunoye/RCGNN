"""Data loaders for RC-GNN."""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class SynthDataset(Dataset):
    """Synthetic dataset for causal structure learning with missingness."""
    
    def __init__(self, X, M, e, S, A_true=None):
        """
        Initialize synthetic dataset.
        
        Args:
            X: Data matrix [N, T, d]
            M: Missingness mask [N, T, d]
            e: Environment labels [N]
            S: Spike/shift labels [N, 1]
            A_true: Ground truth adjacency [d, d] (optional)
        """
        self.X = torch.FloatTensor(X)
        self.M = torch.FloatTensor(M)
        self.e = torch.LongTensor(e)
        self.S = torch.FloatTensor(S) if S is not None else None
        self.A_true = torch.FloatTensor(A_true) if A_true is not None else None
        self.N = X.shape[0]
        
    def __len__(self):
        return self.N
    
    def __getitem__(self, idx):
        sample = {
            "X": self.X[idx],
            "M": self.M[idx],
            "e": self.e[idx],
        }
        if self.S is not None:
            sample["S"] = self.S[idx]
        if self.A_true is not None:
            sample["A_true"] = self.A_true
        return sample


def load_synth(root, split="train", seed=1337):
    """
    Load synthetic dataset.
    
    Args:
        root: Root directory containing X.npy, M.npy, e.npy, S.npy, A_true.npy
        split: "train", "val", or "test" - specifies split based on regime
        seed: Random seed for reproducibility
        
    Returns:
        SynthDataset instance
    """
    # Set seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    root = Path(root)
    
    # Check if pre-split files exist (from synth_bench.py)
    X_split_file = root / f"X_{split}.npy"
    if X_split_file.exists():
        # Use pre-split files
        X = np.load(X_split_file)
        M = np.load(root / f"M_{split}.npy")
        e = np.load(root / f"e_{split}.npy")
        S = np.load(root / f"S_{split}.npy")
        
        A_true = None
        if (root / "A_true.npy").exists():
            A_true = np.load(root / "A_true.npy")
        
        return SynthDataset(X, M, e, S, A_true)
    
    # Otherwise, load full arrays and split
    X = np.load(root / "X.npy")  # [N, T, d]
    M = np.load(root / "M.npy")  # [N, T, d]
    e = np.load(root / "e.npy")  # [N]
    S = np.load(root / "S.npy")  # [N, 1]
    
    A_true = None
    if (root / "A_true.npy").exists():
        A_true = np.load(root / "A_true.npy")
    
    N = X.shape[0]
    n_regimes = len(np.unique(e))
    
    # Split by regime
    regimes = np.unique(e)
    np.random.shuffle(regimes)
    
    if n_regimes == 1:
        # Single regime: use temporal split
        split_idx = int(0.7 * N)
        val_split_idx = int(0.85 * N)
        
        if split == "train":
            indices = np.arange(0, split_idx)
        elif split == "val":
            indices = np.arange(split_idx, val_split_idx)
        else:  # test
            indices = np.arange(val_split_idx, N)
    else:
        # Multiple regimes: split by regime
        train_regimes = regimes[:max(1, int(0.7 * n_regimes))]
        val_regimes = regimes[int(0.7 * n_regimes):int(0.85 * n_regimes)]
        test_regimes = regimes[int(0.85 * n_regimes):]
        
        if split == "train":
            indices = np.where(np.isin(e, train_regimes))[0]
        elif split == "val":
            indices = np.where(np.isin(e, val_regimes))[0]
        else:  # test
            indices = np.where(np.isin(e, test_regimes))[0]
    
    X_split = X[indices]
    M_split = M[indices]
    e_split = e[indices]
    S_split = S[indices] if S is not None else None
    
    return SynthDataset(X_split, M_split, e_split, S_split, A_true)


def get_dataloaders(root, batch_size=8, seed=1337, num_workers=0):
    """
    Get train, val, test dataloaders.
    
    Args:
        root: Root directory containing dataset files
        batch_size: Batch size
        seed: Random seed
        num_workers: Number of workers for data loading
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_ds = load_synth(root, split="train", seed=seed)
    val_ds = load_synth(root, split="val", seed=seed)
    test_ds = load_synth(root, split="test", seed=seed)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader
