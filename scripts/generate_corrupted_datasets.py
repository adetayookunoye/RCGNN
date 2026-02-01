#!/usr/bin/env python3
"""
Generate corrupted versions of UCI Air dataset at different missingness levels.
Creates: data/interim/uci_air_{corruption_rate}/ for each level.
"""
import numpy as np
import os
import json
from pathlib import Path

def apply_corruption(X, M, target_rate, seed=42):
    """Add missingness to reach target rate."""
    np.random.seed(seed)
    current = (M == 0).mean()
    
    if target_rate <= current:
        return X.copy(), M.copy()
    
    # Calculate how much additional missingness needed
    additional = (target_rate - current) / (1 - current)
    
    X_c = X.copy()
    M_c = M.copy()
    
    # Apply additional missing mask only to currently observed entries
    observed = (M == 1)
    extra_mask = np.random.rand(*M.shape) < additional
    
    M_c[observed & extra_mask] = 0
    X_c[M_c == 0] = 0
    
    actual_rate = (M_c == 0).mean()
    return X_c, M_c, actual_rate

def main():
    base_dir = Path("data/interim/uci_air")
    
    # Load original data
    X_orig = np.load(base_dir / "X.npy")
    M_orig = np.load(base_dir / "M.npy")
    A_true = np.load(base_dir / "A_true.npy")
    e = np.load(base_dir / "e.npy")
    S = np.load(base_dir / "S.npy")
    
    print("=" * 60)
    print("GENERATING CORRUPTED DATASETS")
    print("=" * 60)
    print(f"Original data: {X_orig.shape}")
    print(f"Original missing rate: {(M_orig == 0).mean():.2%}")
    print()
    
    corruption_rates = [0.0, 0.10, 0.20, 0.30, 0.40]
    
    for rate in corruption_rates:
        # Create directory
        output_dir = Path(f"data/interim/uci_air_{int(rate*100):02d}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Apply corruption with fixed seed for reproducibility
        X_corrupt, M_corrupt, actual_rate = apply_corruption(X_orig, M_orig, rate, seed=42)
        
        # Save corrupted data
        np.save(output_dir / "X.npy", X_corrupt)
        np.save(output_dir / "M.npy", M_corrupt)
        np.save(output_dir / "A_true.npy", A_true)
        np.save(output_dir / "e.npy", e)
        np.save(output_dir / "S.npy", S)
        
        # Save metadata
        meta = {
            "target_missing_rate": rate,
            "actual_missing_rate": float(actual_rate),
            "N": X_corrupt.shape[0],
            "T": X_corrupt.shape[1],
            "d": X_corrupt.shape[2],
            "n_edges": int(A_true.sum())
        }
        
        with open(output_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        
        print(f"[OK] Created {output_dir}")
        print(f" Target: {rate:.0%}, Actual: {actual_rate:.1%}")
        print(f" X: {X_corrupt.shape}, Missing: {(M_corrupt == 0).sum()} entries")
        print()
    
    print("=" * 60)
    print("DATASET GENERATION COMPLETE")
    print("=" * 60)
    print(f"Created {len(corruption_rates)} corrupted datasets")
    print()
    print("Available datasets:")
    for rate in corruption_rates:
        dir_name = f"uci_air_{int(rate*100):02d}"
        print(f" - data/interim/{dir_name}/ ({rate:.0%} missing)")

if __name__ == "__main__":
    main()
