#!/usr/bin/env python3
"""
Run synthetic SEM benchmark with KNOWN TRUE DAG for causal validity claims.

This addresses reviewer concern: "Domain knowledge graphs may encode plausible 
but not identifiable causal relations."

With synthetic SEM:
- True DAG is known BY CONSTRUCTION (not domain knowledge)
- We inject MNAR + bias + noise corruptions
- Recovery metrics are truly measuring causal structure learning

Usage:
    python scripts/run_synthetic_benchmark.py --benchmark compound_sem
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# IID verification utility (reviewer defense)
# ============================================================================

def compute_sample_autocorr(X, lag, var_idx=0):
    """
    Compute autocorrelation ACROSS SAMPLES (N dimension) for a given variable.
    
    This verifies that samples are IID: if samples are drawn independently,
    consecutive samples should have autocorr ≈ 0.
    
    Args:
        X: array of shape [N, T, d] or [N, d]
        lag: lag for autocorrelation
        var_idx: which variable to check
        
    Returns:
        float: autocorrelation coefficient
    """
    # Get mean value per sample for the target variable
    if X.ndim == 3:
        x = X[:, :, var_idx].mean(axis=1)  # [N] - mean over T for each sample
    else:
        x = X[:, var_idx]  # [N]
    
    if lag >= len(x):
        return 0.0
    
    x1 = x[:-lag]
    x2 = x[lag:]
    
    num = np.mean((x1 - x1.mean()) * (x2 - x2.mean()))
    den = np.var(x1)
    
    return float(num / (den + 1e-8))


# ============================================================================
# Inline DAG and corruption functions (avoid torch import)
# ============================================================================

def generate_er_dag(d, num_edges, seed=None):
    """Generate Erdős-Rényi random DAG."""
    if seed is not None:
        np.random.seed(seed)
    
    A = np.zeros((d, d))
    edges_added = 0
    
    while edges_added < num_edges:
        i, j = np.random.randint(0, d, 2)
        if i < j and A[i, j] == 0:  # Only lower triangular → DAG
            A[i, j] = 1.0
            edges_added += 1
    
    return A


def generate_scale_free_dag(d, attachment=2, seed=None):
    """Generate scale-free DAG using Barabási-Albert model."""
    if seed is not None:
        np.random.seed(seed)
    
    A = np.zeros((d, d))
    degrees = np.ones(d)
    
    for new_node in range(attachment, d):
        probs = degrees[:new_node] / degrees[:new_node].sum()
        targets = np.random.choice(new_node, size=min(attachment, new_node), replace=False, p=probs)
        for target in targets:
            A[target, new_node] = 1.0
            degrees[new_node] += 1
            degrees[target] += 1
    
    return A


def generate_data_from_dag(A_true, T, mechanism="linear", noise_scale=0.1, seed=None):
    """
    Generate time series from DAG with FIXED structural equation weights.
    
    IMPORTANT: Weights are sampled ONCE per variable (stationary SEM),
    not regenerated at each time step. This ensures textbook SEM correctness.
    """
    if seed is not None:
        np.random.seed(seed)
    
    d = A_true.shape[0]
    X = np.zeros((T, d))
    
    # Topological sort
    in_degree = A_true.sum(axis=0).copy()
    order = []
    remaining = set(range(d))
    
    while remaining:
        roots = [i for i in remaining if in_degree[i] == 0]
        if not roots:
            roots = list(remaining)
        order.extend(roots)
        for i in roots:
            remaining.remove(i)
            for j in range(d):
                if A_true[i, j] > 0:
                    in_degree[j] -= 1
    
    # =========================================================================
    # CRITICAL FIX: Sample SEM weights ONCE per variable (stationary SEM)
    # =========================================================================
    W_dict = {}  # Linear weights: W_dict[j] = weight matrix for node j
    MLP_dict = {}  # MLP weights: MLP_dict[j] = (W1, b1, W2) for node j
    
    for j in range(d):
        parents = np.where(A_true[:, j] > 0)[0]
        if len(parents) > 0:
            if mechanism == "linear":
                # Fixed linear coefficients for X_j = sum(W_ij * X_pa(j)) + noise
                W_dict[j] = np.random.randn(len(parents), 1) * 0.5
            elif mechanism == "mlp":
                # Fixed MLP weights for nonlinear SEM
                hidden_dim = 10
                W1 = np.random.randn(len(parents), hidden_dim) * 0.5
                b1 = np.random.randn(hidden_dim) * 0.1
                W2 = np.random.randn(hidden_dim, 1) * 0.5
                MLP_dict[j] = (W1, b1, W2)
    
    # Generate data with FIXED weights
    for t in range(T):
        for j in order:
            parents = np.where(A_true[:, j] > 0)[0]
            if len(parents) == 0:
                # Root node: just noise
                X[t, j] = np.random.randn() * noise_scale
            else:
                X_parents = X[t, parents].reshape(1, -1)
                if mechanism == "linear":
                    # X_j = W^T @ X_pa(j) + noise (FIXED W)
                    X[t, j] = (X_parents @ W_dict[j] + np.random.randn() * noise_scale)[0, 0]
                elif mechanism == "mlp":
                    # X_j = f_MLP(X_pa(j)) + noise (FIXED MLP weights)
                    W1, b1, W2 = MLP_dict[j]
                    h = np.tanh(X_parents @ W1 + b1)
                    X[t, j] = (h @ W2 + np.random.randn() * noise_scale)[0, 0]
    return X


def apply_mcar(X, missing_rate=0.2, seed=None):
    if seed is not None:
        np.random.seed(seed)
    return (np.random.rand(*X.shape) > missing_rate).astype(np.float32)


def apply_mar(X, missing_rate=0.2, seed=None):
    if seed is not None:
        np.random.seed(seed)
    T, d = X.shape
    M = np.ones_like(X)
    for j in range(1, d):
        prob = 1 / (1 + np.exp(-2.0 * np.abs(X[:, j-1]) + missing_rate*5))
        M[:, j] = (np.random.rand(T) > prob).astype(np.float32)
    return M


def apply_mnar(X, missing_rate=0.2, seed=None):
    """
    Apply MNAR (Missing Not At Random) with CALIBRATED missingness rate.
    
    MNAR mechanism: extreme values (high |X|) are more likely to be missing.
    Uses quantile-based thresholding to guarantee EXACT target rate.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Step 1: Compute MNAR propensity score (higher |X| -> higher missing prob)
    propensity = np.abs(X)
    
    # Step 2: Add noise for stochasticity
    noise = np.random.randn(*X.shape) * 0.3 * propensity.std()
    propensity_noisy = propensity + noise
    
    # Step 3: Quantile-based threshold to achieve EXACT target missing rate
    threshold = np.quantile(propensity_noisy, 1 - missing_rate)
    
    # Step 4: Apply threshold (M=1 observed, M=0 missing)
    M = (propensity_noisy <= threshold).astype(np.float32)
    return M


def add_heteroscedastic_noise(X, noise_scale=0.5, seed=None):
    """Add heteroscedastic noise to ALL values (correct semantics)."""
    if seed is not None:
        np.random.seed(seed)
    noise_std = noise_scale * (1 + np.abs(X))
    noise = np.random.randn(*X.shape) * noise_std
    return X + noise


def add_drift(X, drift_magnitude=0.2, seed=None):
    """Add sensor drift to ALL values (correct semantics)."""
    if seed is not None:
        np.random.seed(seed)
    T, d = X.shape
    drift = np.zeros((T, d))
    drift[0] = np.random.randn(d) * drift_magnitude * 0.1
    for t in range(1, T):
        drift[t] = 0.9 * drift[t-1] + np.random.randn(d) * drift_magnitude * 0.1
    return X + drift


def add_bias(X, bias_scale, d, seed=None):
    """Add systematic bias to ALL values (correct semantics)."""
    if seed is not None:
        np.random.seed(seed)
    bias = np.random.randn(d) * bias_scale
    return X + bias


def apply_missingness(X, M):
    """Apply missingness mask: set missing values to 0."""
    return X * M


# ============================================================================
# Compound Corruption SEM Benchmarks (for causal validity claims)
# ============================================================================

SEM_BENCHMARKS = {
    "compound_sem_easy": {
        "description": "Compound SEM - Easy: Known DAG with compound corruptions",
        "graph_type": "er",
        "d": 13,  # Match UCI Air Quality dimensions
        "edges": 13,  # Similar sparsity to UCI
        "mechanism": "linear",
        "n_envs": 3,
        "samples_per_env": 1000,
        "T_per_sample": 50,
        "corruption": {
            "missing_type": "mnar",  # MNAR - hardest missingness
            "missing_rate": 0.20,
            "noise_scale": 0.3,
            "drift_magnitude": 0.15,
            "bias": 0.5,
        }
    },
    
    "compound_sem_medium": {
        "description": "Compound SEM - Medium: MNAR + bias + noise, MLP mechanism",
        "graph_type": "er",
        "d": 13,
        "edges": 13,
        "mechanism": "mlp",  # Nonlinear
        "n_envs": 5,
        "samples_per_env": 1000,
        "T_per_sample": 50,
        "corruption": {
            "missing_type": "mnar",
            "missing_rate": 0.30,
            "noise_scale": 0.4,
            "drift_magnitude": 0.20,
            "bias": 1.0,
        }
    },
    
    "compound_sem_hard": {
        "description": "Compound SEM - Hard: 40% MNAR + high bias + high noise",
        "graph_type": "sf",  # Scale-free (more realistic)
        "d": 13,
        "edges": None,
        "mechanism": "mlp",
        "n_envs": 5,
        "samples_per_env": 1000,
        "T_per_sample": 50,
        "corruption": {
            "missing_type": "mnar",
            "missing_rate": 0.40,
            "noise_scale": 0.5,
            "drift_magnitude": 0.30,
            "bias": 1.5,
        }
    },
}


def generate_compound_sem_benchmark(config, output_dir, seed=42):
    """Generate compound corruption SEM benchmark with KNOWN TRUE DAG."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f" SYNTHETIC SEM BENCHMARK: {config['description']}")
    print(f"{'='*70}")
    print(f" • Graph: {config['graph_type'].upper()}, d={config['d']}")
    print(f" • Mechanism: {config['mechanism']}")
    print(f" • Environments: {config['n_envs']}")
    print(f" • Corruption: {config['corruption']['missing_type']} @ {config['corruption']['missing_rate']*100:.0f}%")
    print(f" • Noise: {config['corruption']['noise_scale']}, Drift: {config['corruption']['drift_magnitude']}")
    
    np.random.seed(seed)
    
    # 1. Generate TRUE DAG (known by construction!)
    if config['graph_type'] == 'er':
        A_true = generate_er_dag(config['d'], config['edges'], seed=seed)
    else:
        A_true = generate_scale_free_dag(config['d'], attachment=2, seed=seed)
    
    true_edges = int(A_true.sum())
    print(f"\n[OK] Generated TRUE DAG: {true_edges} edges (KNOWN BY CONSTRUCTION)")
    
    # 2. Generate clean data from SEM
    all_X = []
    all_S = []
    all_M = []
    all_e = []
    
    for env_id in range(config['n_envs']):
        env_seed = seed + env_id * 1000
        
        X_samples = []
        S_samples = []
        M_samples = []
        
        for sample_id in range(config['samples_per_env']):
            sample_seed = env_seed + sample_id
            
            # Generate clean data from true DAG
            S = generate_data_from_dag(
                A_true, 
                config['T_per_sample'], 
                mechanism=config['mechanism'],
                noise_scale=0.1,  # Clean signal noise
                seed=sample_seed
            )
            
            # Apply compound corruptions
            corr = config['corruption']
            
            # 2a. Missingness mask
            if corr['missing_type'] == 'mcar':
                M = apply_mcar(S, corr['missing_rate'], seed=sample_seed)
            elif corr['missing_type'] == 'mar':
                M = apply_mar(S, corr['missing_rate'], seed=sample_seed)
            else:  # mnar
                M = apply_mnar(S, corr['missing_rate'], seed=sample_seed)
            
            # 2b. Add heteroscedastic noise to ALL values
            X = add_heteroscedastic_noise(S, corr['noise_scale'], seed=sample_seed)
            
            # 2c. Add sensor drift to ALL values
            X = add_drift(X, corr['drift_magnitude'], seed=sample_seed)
            
            # 2d. Add systematic bias to ALL values
            X = add_bias(X, corr['bias'], config['d'], seed=sample_seed)
            
            # 2e. Apply missingness - set missing values to 0 (CORRECT SEMANTICS)
            X = apply_missingness(X, M)
            
            X_samples.append(X)
            S_samples.append(S)
            M_samples.append(M)
        
        all_X.extend(X_samples)
        all_S.extend(S_samples)
        all_M.extend(M_samples)
        all_e.extend([env_id] * config['samples_per_env'])
        
        print(f"   Env {env_id}: {config['samples_per_env']} samples, T={config['T_per_sample']}")
    
    # Stack and save
    X = np.stack(all_X, axis=0)  # [N, T, d]
    S = np.stack(all_S, axis=0)
    M = np.stack(all_M, axis=0)
    e = np.array(all_e)
    
    # Compute corruption statistics
    missing_rate = 1 - M.mean()
    
    print(f"\n[OK] Dataset shape: X={X.shape}")
    print(f"[OK] Actual missing rate: {missing_rate*100:.1f}%")
    print(f"[OK] Environment distribution: {np.bincount(e)}")
    
    # ==================================================================
    # IID VERIFICATION: Autocorrelation sanity check (reviewer defense)
    # ==================================================================
    # Check that SAMPLES are IID (no temporal dependence across N dimension)
    # This is different from within-sample correlation (expected from DAG)
    print("\n[CHECK] IID verification: sample-to-sample autocorrelation")
    
    lags = [1, 5, 10]
    n_vars_to_check = min(3, X.shape[2])  # Check first few variables
    
    all_pass = True
    for var_idx in range(n_vars_to_check):
        for lag in lags:
            ac = compute_sample_autocorr(X, lag, var_idx)
            if abs(ac) >= 0.1:
                all_pass = False
            status = "✓" if abs(ac) < 0.1 else "⚠"
            if var_idx == 0:  # Only print for first variable
                print(f"  Var 0, Lag {lag:2d}: autocorr ≈ {ac:+.4f}  {status}")
    
    if all_pass:
        print("  → All autocorr ≈ 0: samples are IID (not time-series).")
    else:
        print("  → Some autocorr > 0.1: check data generation.")
    
    # Also record in meta for provenance
    iid_check = {
        "var0_lag1": compute_sample_autocorr(X, 1, 0),
        "var0_lag5": compute_sample_autocorr(X, 5, 0),
        "var0_lag10": compute_sample_autocorr(X, 10, 0),
        "iid_verified": all_pass,
    }
    
    # Save
    np.save(os.path.join(output_dir, "X.npy"), X.astype(np.float32))
    np.save(os.path.join(output_dir, "S.npy"), S.astype(np.float32))
    np.save(os.path.join(output_dir, "M.npy"), M.astype(np.float32))
    np.save(os.path.join(output_dir, "e.npy"), e.astype(np.int64))
    np.save(os.path.join(output_dir, "A_true.npy"), A_true.astype(np.float32))
    
    # Get git commit for provenance
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_commit = "unknown"
    
    # Save config with provenance (meta.json)
    meta = {
        "table": "2C",
        "benchmark": config.get("description", "compound_sem"),
        **config,
        "seed": seed,
        "true_edges": true_edges,
        "actual_missing_rate": float(missing_rate),
        "iid_verification": iid_check,  # Reviewer defense: autocorr proof
        "git_commit": git_commit,
        "created_at": datetime.now().isoformat(),
        "note": "Ground-truth DAG known by construction; eliminates domain-knowledge subjectivity.",
    }
    
    with open(os.path.join(output_dir, "meta.json"), 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f"\n[SAVED] Dataset to {output_dir}/")
    print(f"   • X.npy: Corrupted observations")
    print(f"   • S.npy: Clean source signals")
    print(f"   • M.npy: Missingness masks")
    print(f"   • e.npy: Environment labels")
    print(f"   • A_true.npy: TRUE CAUSAL DAG (known by construction!)")
    
    return A_true, X, S, M, e


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic SEM benchmark for causal validity claims"
    )
    parser.add_argument(
        "--benchmark", 
        type=str, 
        default="compound_sem_medium",
        choices=list(SEM_BENCHMARKS.keys()),
        help="Benchmark configuration"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=None,
        help="Output directory (default: data/interim/synth_{benchmark})"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    config = SEM_BENCHMARKS[args.benchmark]
    output_dir = args.output or f"data/interim/synth_{args.benchmark}"
    
    A_true, X, S, M, e = generate_compound_sem_benchmark(config, output_dir, args.seed)
    
    print(f"\n{'='*70}")
    print(" NEXT STEPS:")
    print(f"{'='*70}")
    print(f" 1. Train RC-GNN:")
    print(f"    python scripts/train_rcgnn_unified.py \\")
    print(f"        --data_root {output_dir} \\")
    print(f"        --output_dir artifacts/synth_{args.benchmark}")
    print(f"")
    print(f" 2. Evaluate against baselines:")
    print(f"    python scripts/comprehensive_evaluation.py \\")
    print(f"        --artifacts_dir artifacts/synth_{args.benchmark}")
    print(f"")
    print(f" NOTE: Ground truth A_true.npy is KNOWN BY CONSTRUCTION,")
    print(f"       not derived from domain knowledge!")


if __name__ == "__main__":
    main()
