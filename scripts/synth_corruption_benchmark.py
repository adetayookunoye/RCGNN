#!/usr/bin/env python3
"""
Generate synthetic multi-environment causal discovery benchmarks for hypothesis testing.

Specifically designed for:
- H1: Structural accuracy under missingness (RC-GNN vs baselines)
- H2: Stability improvement via invariance loss
- H3: Expert agreement on policy-relevant pathways

Features:
- Multi-environment data with controlled corruption patterns
- Tunable MCAR/MAR/MNAR missingness per environment
- Heteroscedastic noise and sensor drift
- Ground truth adjacency matrices
- Policy-relevant causal pathways (for H3)

Usage:
    python scripts/synth_corruption_benchmark.py --benchmark h1_easy
    python scripts/synth_corruption_benchmark.py --benchmark h2_multi_env
    python scripts/synth_corruption_benchmark.py --benchmark h3_policy --policy_config configs/policies.yaml
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# Table-2 SEM Benchmark Configs (8 configs for paper)
# ============================================================================

# Corruption levels for Table-2
TABLE2_CORRUPTIONS = {
    "medium": {"missing_type": "mnar", "missing_rate": 0.20, "noise_scale": 0.30, "drift_magnitude": 0.1},
    "hard":   {"missing_type": "mnar", "missing_rate": 0.40, "noise_scale": 0.50, "drift_magnitude": 0.2},
}

# Exact 8 configs from Table-2 in paper
TABLE2_CONFIGS = [
    {"name": "er_d13_lin",  "graph": "er", "d": 13, "edges": 13, "mechanism": "linear", "corruption": "medium"},
    {"name": "er_d13_mlp",  "graph": "er", "d": 13, "edges": 13, "mechanism": "mlp",    "corruption": "medium"},
    {"name": "er_d20_lin",  "graph": "er", "d": 20, "edges": 20, "mechanism": "linear", "corruption": "medium"},
    {"name": "er_d20_mlp",  "graph": "er", "d": 20, "edges": 20, "mechanism": "mlp",    "corruption": "medium"},
    {"name": "er_d50_mlp",  "graph": "er", "d": 50, "edges": 50, "mechanism": "mlp",    "corruption": "medium"},
    {"name": "sf_d13_mlp",  "graph": "sf", "d": 13, "edges": 22, "mechanism": "mlp",    "corruption": "medium"},
    {"name": "sf_d20_mlp",  "graph": "sf", "d": 20, "edges": 36, "mechanism": "mlp",    "corruption": "medium"},
    {"name": "sf_d13_hard", "graph": "sf", "d": 13, "edges": 22, "mechanism": "mlp",    "corruption": "hard"},
]

# ============================================================================
# H1/H2/H3 Hypothesis Testing Benchmarks (separate from Table-2)
# ============================================================================

BENCHMARKS = {
    "h1_easy": {
        "description": "H1 - Easy: Low missingness, few environments",
        "graph_type": "er",
        "d": 15,
        "edges": 30,
        "mechanism": "linear",
        "n_envs": 3,
        "samples_per_env": 500,
        "T_per_sample": 50,
        "corruption_configs": [
            {"missing_type": "mcar", "missing_rate": 0.1, "noise_scale": 0.1, "drift_magnitude": 0.0},
            {"missing_type": "mcar", "missing_rate": 0.15, "noise_scale": 0.15, "drift_magnitude": 0.05},
            {"missing_type": "mcar", "missing_rate": 0.2, "noise_scale": 0.2, "drift_magnitude": 0.1},
        ]
    },
    
    "h1_medium": {
        "description": "H1 - Medium: Moderate missingness, diverse corruption types",
        "graph_type": "er",
        "d": 15,
        "edges": 30,
        "mechanism": "mlp",
        "n_envs": 4,
        "samples_per_env": 600,
        "T_per_sample": 50,
        "corruption_configs": [
            {"missing_type": "mcar", "missing_rate": 0.2, "noise_scale": 0.2, "drift_magnitude": 0.1},
            {"missing_type": "mar", "missing_rate": 0.25, "noise_scale": 0.3, "drift_magnitude": 0.15},
            {"missing_type": "mnar", "missing_rate": 0.3, "noise_scale": 0.4, "drift_magnitude": 0.2},
            {"missing_type": "mcar", "missing_rate": 0.2, "noise_scale": 0.2, "drift_magnitude": 0.1},
        ]
    },
    
    "h1_hard": {
        "description": "H1 - Hard: High missingness (40-60%), all corruption types, MLP",
        "graph_type": "sf",
        "d": 20,
        "edges": None, # Will use scale-free
        "mechanism": "mlp",
        "n_envs": 5,
        "samples_per_env": 700,
        "T_per_sample": 50,
        "corruption_configs": [
            {"missing_type": "mcar", "missing_rate": 0.4, "noise_scale": 0.3, "drift_magnitude": 0.15},
            {"missing_type": "mar", "missing_rate": 0.45, "noise_scale": 0.4, "drift_magnitude": 0.2},
            {"missing_type": "mnar", "missing_rate": 0.5, "noise_scale": 0.5, "drift_magnitude": 0.25},
            {"missing_type": "mcar", "missing_rate": 0.35, "noise_scale": 0.25, "drift_magnitude": 0.1},
            {"missing_type": "mar", "missing_rate": 0.55, "noise_scale": 0.45, "drift_magnitude": 0.3},
        ]
    },
    
    "h2_multi_env": {
        "description": "H2 - Multi-environment: Diverse environments for stability testing",
        "graph_type": "er",
        "d": 20,
        "edges": 40,
        "mechanism": "linear",
        "n_envs": 5,
        "samples_per_env": 400,
        "T_per_sample": 50,
        "corruption_configs": [
            {"missing_type": "mcar", "missing_rate": 0.15, "noise_scale": 0.1, "drift_magnitude": 0.05},
            {"missing_type": "mcar", "missing_rate": 0.25, "noise_scale": 0.2, "drift_magnitude": 0.1},
            {"missing_type": "mcar", "missing_rate": 0.35, "noise_scale": 0.3, "drift_magnitude": 0.15},
            {"missing_type": "mcar", "missing_rate": 0.25, "noise_scale": 0.2, "drift_magnitude": 0.1},
            {"missing_type": "mcar", "missing_rate": 0.20, "noise_scale": 0.15, "drift_magnitude": 0.08},
        ]
    },
    
    "h2_stability": {
        "description": "H2 - Stability: Test invariance loss effectiveness",
        "graph_type": "er",
        "d": 15,
        "edges": 25,
        "mechanism": "mlp",
        "n_envs": 4,
        "samples_per_env": 500,
        "T_per_sample": 50,
        "corruption_configs": [
            {"missing_type": "mar", "missing_rate": 0.2, "noise_scale": 0.2, "drift_magnitude": 0.1},
            {"missing_type": "mar", "missing_rate": 0.35, "noise_scale": 0.35, "drift_magnitude": 0.2},
            {"missing_type": "mnar", "missing_rate": 0.3, "noise_scale": 0.3, "drift_magnitude": 0.15},
            {"missing_type": "mnar", "missing_rate": 0.45, "noise_scale": 0.4, "drift_magnitude": 0.25},
        ]
    },
    
    "h3_policy": {
        "description": "H3 - Policy-relevant: Air quality domain pathways",
        "graph_type": "er",
        "d": 25,
        "edges": 50,
        "mechanism": "mlp",
        "n_envs": 4,
        "samples_per_env": 600,
        "T_per_sample": 50,
        "policy_edges": [(2, 5), (2, 8), (5, 12), (8, 12), (12, 20)], # PM2.5-related pathways
        "corruption_configs": [
            {"missing_type": "mcar", "missing_rate": 0.15, "noise_scale": 0.2, "drift_magnitude": 0.05},
            {"missing_type": "mar", "missing_rate": 0.25, "noise_scale": 0.3, "drift_magnitude": 0.1},
            {"missing_type": "mnar", "missing_rate": 0.3, "noise_scale": 0.4, "drift_magnitude": 0.15},
            {"missing_type": "mcar", "missing_rate": 0.2, "noise_scale": 0.25, "drift_magnitude": 0.08},
        ]
    }
}


# ============================================================================
# Graph Generation
# ============================================================================

def generate_er_dag(d, num_edges, seed=None):
    """Generate Erdős-Rényi random DAG."""
    if seed is not None:
        np.random.seed(seed)
    
    A_true = np.zeros((d, d))
    edges_added = 0
    max_attempts = num_edges * 10
    attempts = 0
    
    while edges_added < num_edges and attempts < max_attempts:
        i, j = np.random.randint(0, d, size=2)
        if i < j and A_true[i, j] == 0:
            A_true[i, j] = 1
            edges_added += 1
        attempts += 1
    
    # Randomly permute to break ordering
    perm = np.random.permutation(d)
    A_true = A_true[perm, :][:, perm]
    
    return A_true


def generate_scale_free_dag(d, attachment=2, seed=None):
    """Generate scale-free DAG using Barabási-Albert model."""
    if seed is not None:
        np.random.seed(seed)
    
    A_true = np.zeros((d, d))
    degrees = np.zeros(d)
    
    # Start with small complete graph
    for i in range(min(attachment, d)):
        for j in range(i+1, min(attachment, d)):
            A_true[i, j] = 1
            degrees[i] += 1
            degrees[j] += 1
    
    # Add remaining nodes with preferential attachment
    for new_node in range(attachment, d):
        probs = degrees[:new_node] / (degrees[:new_node].sum() + 1e-10)
        targets = np.random.choice(new_node, size=min(attachment, new_node), 
                                     replace=False, p=probs)
        for target in targets:
            if new_node > target:
                A_true[target, new_node] = 1
            else:
                A_true[new_node, target] = 1
            degrees[new_node] += 1
            degrees[target] += 1
    
    return A_true


def generate_scale_free_dag_target_edges(d, target_edges, seed=None):
    """
    Generate scale-free DAG with approximately target_edges.
    
    Uses binary search on attachment parameter to find the right edge count.
    """
    rng = np.random.default_rng(seed)
    
    # Binary search for attachment that gives ~target_edges
    best_A = None
    best_diff = float('inf')
    
    for attachment in range(1, min(d, 10)):
        # Generate SF DAG without global seed (use rng instead)
        A = np.zeros((d, d))
        degrees = np.ones(d) * 0.1  # Small initial degree to avoid div-by-zero
        
        # Start with small complete graph
        m0 = min(attachment, d)
        for i in range(m0):
            for j in range(i+1, m0):
                A[i, j] = 1
                degrees[i] += 1
                degrees[j] += 1
        
        # Add remaining nodes with preferential attachment
        for new_node in range(m0, d):
            probs = degrees[:new_node].copy()
            probs = probs / probs.sum()  # Normalize
            k = min(attachment, new_node)
            try:
                targets = rng.choice(new_node, size=k, replace=False, p=probs)
            except ValueError:
                # Fallback: uniform random selection
                targets = rng.choice(new_node, size=k, replace=False)
            for target in targets:
                if new_node > target:
                    A[target, new_node] = 1
                else:
                    A[new_node, target] = 1
                degrees[new_node] += 1
                degrees[target] += 1
        
        n_edges = int(A.sum())
        diff = abs(n_edges - target_edges)
        
        if diff < best_diff:
            best_diff = diff
            best_A = A.copy()
        
        if n_edges >= target_edges:
            break
    
    # Fine-tune: add or remove edges to hit exact target
    n_edges = int(best_A.sum())
    
    if n_edges < target_edges:
        # Add random edges (respecting DAG constraint)
        candidates = [(i, j) for i in range(d) for j in range(i+1, d) if best_A[i, j] == 0]
        rng.shuffle(candidates)
        for i, j in candidates:
            if n_edges >= target_edges:
                break
            best_A[i, j] = 1
            n_edges += 1
    
    elif n_edges > target_edges:
        # Remove random edges
        edges = list(zip(*np.where(best_A > 0)))
        rng.shuffle(edges)
        for i, j in edges:
            if n_edges <= target_edges:
                break
            best_A[i, j] = 0
            n_edges -= 1
    
    # Permute to break any ordering artifacts
    perm = rng.permutation(d)
    best_A = best_A[perm][:, perm]
    
    return best_A


# ============================================================================
# Data Generation (with Corruption)
# ============================================================================

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
                else:
                    raise ValueError(f"Unknown mechanism: {mechanism}")
    
    return X


def apply_mcar(X, missing_rate=0.2, seed=None):
    """Missing Completely At Random."""
    if seed is not None:
        np.random.seed(seed)
    M = (np.random.rand(*X.shape) > missing_rate).astype(np.float32)
    return M


def apply_mar(X, missing_rate=0.2, dependency_strength=2.0, seed=None):
    """Missing At Random (depends on observed features)."""
    if seed is not None:
        np.random.seed(seed)
    
    T, d = X.shape
    M = np.ones_like(X)
    
    for j in range(1, d):
        prob_missing = 1 / (1 + np.exp(-dependency_strength * np.abs(X[:, j-1]) + missing_rate*5))
        M[:, j] = (np.random.rand(T) > prob_missing).astype(np.float32)
    
    return M


def apply_mnar(X, missing_rate=0.2, self_dependency=2.0, seed=None):
    """
    Missing Not At Random (depends on the value itself).
    
    CALIBRATED version: Uses quantile-based thresholding to guarantee EXACT target rate.
    The self_dependency parameter is kept for API compatibility but is no longer used.
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


def add_drift(X, drift_magnitude=0.2, drift_type="ar1", seed=None):
    """Add sensor drift to ALL values (correct semantics)."""
    if seed is not None:
        np.random.seed(seed)
    
    T, d = X.shape
    
    if drift_type == "ar1":
        drift = np.zeros((T, d))
        drift[0] = np.random.randn(d) * drift_magnitude * 0.1
        for t in range(1, T):
            drift[t] = 0.9 * drift[t-1] + np.random.randn(d) * drift_magnitude * 0.1
    else:
        drift = np.zeros((T, d))
    
    return X + drift


def apply_missingness(X, M):
    """Apply missingness mask: set missing values to 0 (correct semantics)."""
    return X * M


# ============================================================================
# Multi-Environment Dataset Generation
# ============================================================================

def generate_multi_env_benchmark(config, output_dir, seed=42):
    """Generate multi-environment benchmark dataset."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n Generating benchmark: {config['description']}")
    print(f" Graph type: {config['graph_type']}, d={config['d']}")
    print(f" Environments: {config['n_envs']}, Mechanism: {config['mechanism']}")
    
    # Generate graph
    if config['graph_type'] == 'er':
        A_true = generate_er_dag(config['d'], config['edges'], seed=seed)
    else:
        A_true = generate_scale_free_dag(config['d'], attachment=2, seed=seed)
    
    print(f"[DONE] Generated DAG with {int(A_true.sum())} edges")
    
    # Generate data per environment
    X_all, M_all, S_all, e_all = [], [], [], []
    
    for env_idx in range(config['n_envs']):
        corruption_config = config['corruption_configs'][env_idx]
        print(f"\n Env {env_idx+1}/{config['n_envs']}: {corruption_config['missing_type'].upper()} "
              f"({corruption_config['missing_rate']*100:.0f}%), "
              f"noise={corruption_config['noise_scale']:.2f}, "
              f"drift={corruption_config['drift_magnitude']:.2f}")
        
        for sample_idx in range(config['samples_per_env']):
            # Generate clean signal
            S = generate_data_from_dag(
                A_true, config['T_per_sample'], mechanism=config['mechanism'],
                noise_scale=0.1, seed=seed + env_idx*1000 + sample_idx
            )
            
            # Apply missingness
            if corruption_config['missing_type'] == 'mcar':
                M = apply_mcar(S, corruption_config['missing_rate'], 
                             seed=seed + env_idx*1000 + sample_idx + 10000)
            elif corruption_config['missing_type'] == 'mar':
                M = apply_mar(S, corruption_config['missing_rate'],
                            seed=seed + env_idx*1000 + sample_idx + 10000)
            else: # mnar
                M = apply_mnar(S, corruption_config['missing_rate'],
                             seed=seed + env_idx*1000 + sample_idx + 10000)
            
            # Add noise to ALL values
            X = add_heteroscedastic_noise(S, corruption_config['noise_scale'],
                                        seed=seed + env_idx*1000 + sample_idx + 20000)
            
            # Add drift to ALL values
            X = add_drift(X, corruption_config['drift_magnitude'], drift_type='ar1',
                        seed=seed + env_idx*1000 + sample_idx + 30000)
            
            # Apply missingness - set missing values to 0 (CORRECT SEMANTICS)
            X = apply_missingness(X, M)
            
            X_all.append(X)
            M_all.append(M)
            S_all.append(S)
            e_all.append(env_idx)
    
    X_all = np.stack(X_all, axis=0)
    M_all = np.stack(M_all, axis=0)
    S_all = np.stack(S_all, axis=0)
    e_all = np.array(e_all, dtype=np.int32)
    
    print(f"\n[DONE] Generated {X_all.shape[0]} samples: shape {X_all.shape}")
    
    # Train/val split
    n_total = X_all.shape[0]
    n_train = int(0.8 * n_total)
    indices = np.random.RandomState(seed).permutation(n_total)
    train_idx, val_idx = indices[:n_train], indices[n_train:]
    
    X_train, M_train, S_train, e_train = X_all[train_idx], M_all[train_idx], S_all[train_idx], e_all[train_idx]
    X_val, M_val, S_val, e_val = X_all[val_idx], M_all[val_idx], S_all[val_idx], e_all[val_idx]
    
    # Save
    np.save(os.path.join(output_dir, "A_true.npy"), A_true)
    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "M_train.npy"), M_train)
    np.save(os.path.join(output_dir, "S_train.npy"), S_train)
    np.save(os.path.join(output_dir, "e_train.npy"), e_train)
    np.save(os.path.join(output_dir, "X_val.npy"), X_val)
    np.save(os.path.join(output_dir, "M_val.npy"), M_val)
    np.save(os.path.join(output_dir, "S_val.npy"), S_val)
    np.save(os.path.join(output_dir, "e_val.npy"), e_val)
    
    # Get git commit for provenance
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_commit = "unknown"
    
    # Metadata with provenance
    metadata = {
        "table": "2A",
        "benchmark": config.get("name", "unknown"),
        "description": config['description'],
        "graph_type": config['graph_type'],
        "d": config['d'],
        "edges": int(A_true.sum()),
        "mechanism": config['mechanism'],
        "n_envs": config['n_envs'],
        "samples_per_env": config['samples_per_env'],
        "T_per_sample": config['T_per_sample'],
        "corruption_configs": config['corruption_configs'],
        "seed": seed,
        "train_samples": int(X_train.shape[0]),
        "val_samples": int(X_val.shape[0]),
        "policy_edges": config.get("policy_edges", None),
        "git_commit": git_commit,
        "created_at": datetime.now().isoformat(),
    }
    
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n[DONE] Benchmark saved to: {output_dir}")
    print(f" Train: {X_train.shape[0]} samples ({X_train.shape})")
    print(f" Val: {X_val.shape[0]} samples ({X_val.shape})")
    if "policy_edges" in config:
        print(f" Policy edges (H3): {config['policy_edges']}")


# ============================================================================
# Table-2 Benchmark Generation (8 configs × 5 seeds = 40 datasets)
# ============================================================================

def generate_table2_all(output_root, seeds, T_samples=1000, n_envs=3, samples_per_env=500, configs=None):
    """
    Generate Table-2 SEM benchmarks.
    
    Output structure:
        {output_root}/{config_name}/seed_{seed}/
            - A_true.npy
            - X.npy (corrupted, with missingness applied)
            - M.npy (binary mask)
            - S.npy (clean signal)
            - e.npy (environment labels)
            - meta.json
    
    Args:
        output_root: Base directory for all datasets
        seeds: List of seeds (e.g., [0,1,2,3,4])
        T_samples: Total samples (will be split by n_envs)
        n_envs: Number of environments
        samples_per_env: Samples per environment
        configs: List of Table-2 configs to generate (default: all TABLE2_CONFIGS)
    """
    import hashlib
    
    # Default to all configs if not specified
    if configs is None:
        configs = TABLE2_CONFIGS
    
    os.makedirs(output_root, exist_ok=True)
    
    total_datasets = len(configs) * len(seeds)
    completed = 0
    
    print("=" * 70)
    print(f"GENERATING TABLE-2 SEM BENCHMARK GRID")
    print(f"  Configs: {len(configs)}")
    print(f"  Seeds: {seeds}")
    print(f"  Total datasets: {total_datasets}")
    print(f"  Output: {output_root}")
    print("=" * 70)
    
    for cfg in configs:
        corr = TABLE2_CORRUPTIONS[cfg["corruption"]]
        
        for seed in seeds:
            completed += 1
            print(f"\n[{completed}/{total_datasets}] {cfg['name']} seed={seed}")
            
            # Set seed
            np.random.seed(seed)
            
            # Generate graph
            if cfg["graph"] == "er":
                A_true = generate_er_dag(cfg["d"], cfg["edges"], seed=seed)
            else:
                A_true = generate_scale_free_dag_target_edges(cfg["d"], cfg["edges"], seed=seed)
            
            actual_edges = int(A_true.sum())
            print(f"  Graph: {cfg['graph'].upper()} d={cfg['d']}, edges={actual_edges} (target={cfg['edges']})")
            
            # Generate multi-environment data
            X_all, M_all, S_all, e_all = [], [], [], []
            
            for env_idx in range(n_envs):
                # Vary corruption slightly per environment for multi-env training
                env_missing_rate = corr["missing_rate"] * (0.8 + 0.4 * env_idx / (n_envs - 1))
                env_noise_scale = corr["noise_scale"] * (0.8 + 0.4 * env_idx / (n_envs - 1))
                
                for sample_idx in range(samples_per_env):
                    sample_seed = seed * 100000 + env_idx * 1000 + sample_idx
                    
                    # Generate clean signal
                    S = generate_data_from_dag(
                        A_true, T=50, mechanism=cfg["mechanism"],
                        noise_scale=0.1, seed=sample_seed
                    )
                    
                    # Apply MNAR missingness
                    M = apply_mnar(S, env_missing_rate, seed=sample_seed + 10000)
                    
                    # Add heteroscedastic noise
                    X = add_heteroscedastic_noise(S, env_noise_scale, seed=sample_seed + 20000)
                    
                    # Add drift
                    X = add_drift(X, corr["drift_magnitude"], seed=sample_seed + 30000)
                    
                    # Apply missingness mask
                    X = apply_missingness(X, M)
                    
                    X_all.append(X)
                    M_all.append(M)
                    S_all.append(S)
                    e_all.append(env_idx)
            
            X_all = np.stack(X_all, axis=0).astype(np.float32)
            M_all = np.stack(M_all, axis=0).astype(np.float32)
            S_all = np.stack(S_all, axis=0).astype(np.float32)
            e_all = np.array(e_all, dtype=np.int32)
            
            # Save to output directory
            ddir = Path(output_root) / cfg["name"] / f"seed_{seed}"
            ddir.mkdir(parents=True, exist_ok=True)
            
            np.save(ddir / "A_true.npy", A_true.astype(np.float32))
            np.save(ddir / "X.npy", X_all)
            np.save(ddir / "M.npy", M_all)
            np.save(ddir / "S.npy", S_all)
            np.save(ddir / "e.npy", e_all)
            
            # Compute actual missing rate
            actual_missing_rate = 1 - M_all.mean()
            
            # Get git commit for provenance
            try:
                git_commit = subprocess.check_output(
                    ["git", "rev-parse", "--short", "HEAD"],
                    stderr=subprocess.DEVNULL
                ).decode().strip()
            except Exception:
                git_commit = "unknown"
            
            # Metadata with provenance
            meta = {
                "table": "2B",
                "config_name": cfg["name"],
                "graph_type": cfg["graph"],
                "d": cfg["d"],
                "target_edges": cfg["edges"],
                "actual_edges": actual_edges,
                "mechanism": cfg["mechanism"],
                "corruption_level": cfg["corruption"],
                "corruption_params": corr,
                "n_envs": n_envs,
                "samples_per_env": samples_per_env,
                "T_per_sample": 50,
                "seed": seed,
                "actual_missing_rate": float(actual_missing_rate),
                "shape": list(X_all.shape),
                "git_commit": git_commit,
                "created_at": datetime.now().isoformat(),
            }
            
            with open(ddir / "meta.json", "w") as f:
                json.dump(meta, f, indent=2)
            
            # Dataset signature for integrity checks
            def sha256_file(path):
                h = hashlib.sha256()
                with open(path, "rb") as f:
                    for chunk in iter(lambda: f.read(1 << 20), b""):
                        h.update(chunk)
                return h.hexdigest()[:16]
            
            sig = {
                "A_true": sha256_file(ddir / "A_true.npy"),
                "X": sha256_file(ddir / "X.npy"),
                "M": sha256_file(ddir / "M.npy"),
            }
            with open(ddir / "signature.json", "w") as f:
                json.dump(sig, f, indent=2)
            
            print(f"  -> {ddir} (missing={actual_missing_rate:.1%})")
    
    print("\n" + "=" * 70)
    print(f"[DONE] Generated {completed} Table-2 datasets")
    print(f"  Output: {output_root}")
    print("=" * 70)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic corruption benchmarks for hypothesis testing"
    )
    
    parser.add_argument(
        "--benchmark", type=str, default=None,
        choices=list(BENCHMARKS.keys()),
        help="H1/H2/H3 benchmark configuration to generate"
    )
    parser.add_argument(
        "--table2", action="store_true",
        help="Generate Table-2 SEM benchmarks (all configs if no --config, else just specified config)"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        choices=[c["name"] for c in TABLE2_CONFIGS],
        help="Specific Table-2 config to generate (use with --table2)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (for single benchmark)"
    )
    parser.add_argument(
        "--seeds", type=str, default="0,1,2,3,4",
        help="Comma-separated seeds for --table2 mode (default: 0,1,2,3,4)"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all available benchmarks"
    )
    
    args = parser.parse_args()
    
    # List mode
    if args.list:
        print("\n=== H1/H2/H3 Hypothesis Benchmarks ===\n")
        for name, config in BENCHMARKS.items():
            print(f"  {name:20s} - {config['description']}")
        
        print("\n=== Table-2 SEM Configs (use --table2) ===\n")
        for cfg in TABLE2_CONFIGS:
            corr = TABLE2_CORRUPTIONS[cfg["corruption"]]
            print(f"  {cfg['name']:15s} - {cfg['graph'].upper()} d={cfg['d']:2d} edges={cfg['edges']:2d} "
                  f"{cfg['mechanism']:6s} {cfg['corruption']:6s} (MNAR {corr['missing_rate']*100:.0f}%)")
        return
    
    # Table-2 mode: generate SEM benchmarks
    if args.table2:
        output_root = args.output or "data/interim/sem_table2"
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        
        # If --config specified, filter to just that config
        if args.config:
            configs_to_run = [c for c in TABLE2_CONFIGS if c["name"] == args.config]
            if not configs_to_run:
                print(f"[ERROR] Unknown Table-2 config: {args.config}")
                print(f"  Available: {', '.join(c['name'] for c in TABLE2_CONFIGS)}")
                return
        else:
            configs_to_run = TABLE2_CONFIGS
        
        generate_table2_all(output_root, seeds, configs=configs_to_run)
        return
    
    # Single benchmark mode
    if args.benchmark is None:
        print("[ERROR] Must specify --benchmark or --table2")
        print("  Use --list to see available benchmarks")
        return
    
    if args.benchmark not in BENCHMARKS:
        print(f"[FAIL] Unknown benchmark: {args.benchmark}")
        print(f" Available: {', '.join(BENCHMARKS.keys())}")
        return
    
    config = BENCHMARKS[args.benchmark].copy()
    config['name'] = args.benchmark
    
    # Set output directory
    if args.output is None:
        output_dir = f"data/interim/synth_corrupted_{args.benchmark}"
    else:
        output_dir = args.output
    
    # Generate benchmark
    generate_multi_env_benchmark(config, output_dir, seed=args.seed)
    
    print("\n" + "="*70)
    print(f"[DONE] Benchmark ready for training!")
    print(f" Next: python scripts/train_rcgnn.py configs/train.yaml --data_root {output_dir}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
