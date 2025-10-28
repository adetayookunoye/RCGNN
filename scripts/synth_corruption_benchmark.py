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
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# Benchmark-Specific Configurations
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
        "edges": None,  # Will use scale-free
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
        "policy_edges": [(2, 5), (2, 8), (5, 12), (8, 12), (12, 20)],  # PM2.5-related pathways
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


# ============================================================================
# Data Generation (with Corruption)
# ============================================================================

def generate_data_from_dag(A_true, T, mechanism="linear", noise_scale=0.1, seed=None):
    """Generate time series from DAG with given mechanism."""
    if seed is not None:
        np.random.seed(seed)
    
    d = A_true.shape[0]
    X = np.zeros((T, d))
    
    # Topological sort
    in_degree = A_true.sum(axis=0)
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
    
    # Generate data
    for t in range(T):
        for j in order:
            parents = np.where(A_true[:, j] > 0)[0]
            
            if len(parents) == 0:
                X[t, j] = np.random.randn() * noise_scale
            else:
                X_parents = X[t, parents].reshape(1, -1)
                
                if mechanism == "linear":
                    W = np.random.randn(len(parents), 1) * 0.5
                    X[t, j] = (X_parents @ W + np.random.randn() * noise_scale)[0, 0]
                elif mechanism == "mlp":
                    W1 = np.random.randn(len(parents), 10) * 0.5
                    b1 = np.zeros(10)
                    W2 = np.random.randn(10, 1) * 0.5
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
    """Missing Not At Random (depends on the value itself)."""
    if seed is not None:
        np.random.seed(seed)
    
    prob_missing = 1 / (1 + np.exp(-self_dependency * np.abs(X) + missing_rate*5))
    M = (np.random.rand(*X.shape) > prob_missing).astype(np.float32)
    
    return M


def add_heteroscedastic_noise(X, M, noise_scale=0.5, seed=None):
    """Add heteroscedastic noise (variance depends on X magnitude)."""
    if seed is not None:
        np.random.seed(seed)
    
    noise_std = noise_scale * (1 + np.abs(X))
    noise = np.random.randn(*X.shape) * noise_std
    X_noisy = X + noise * M
    
    return X_noisy


def add_drift(X, M, drift_magnitude=0.2, drift_type="ar1", seed=None):
    """Add sensor drift/bias over time."""
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
    
    X_drifted = X + drift * M
    return X_drifted


# ============================================================================
# Multi-Environment Dataset Generation
# ============================================================================

def generate_multi_env_benchmark(config, output_dir, seed=42):
    """Generate multi-environment benchmark dataset."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n📊 Generating benchmark: {config['description']}")
    print(f"   Graph type: {config['graph_type']}, d={config['d']}")
    print(f"   Environments: {config['n_envs']}, Mechanism: {config['mechanism']}")
    
    # Generate graph
    if config['graph_type'] == 'er':
        A_true = generate_er_dag(config['d'], config['edges'], seed=seed)
    else:
        A_true = generate_scale_free_dag(config['d'], attachment=2, seed=seed)
    
    print(f"✅ Generated DAG with {int(A_true.sum())} edges")
    
    # Generate data per environment
    X_all, M_all, S_all, e_all = [], [], [], []
    
    for env_idx in range(config['n_envs']):
        corruption_config = config['corruption_configs'][env_idx]
        print(f"\n   Env {env_idx+1}/{config['n_envs']}: {corruption_config['missing_type'].upper()} "
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
            else:  # mnar
                M = apply_mnar(S, corruption_config['missing_rate'],
                             seed=seed + env_idx*1000 + sample_idx + 10000)
            
            # Add noise
            X = add_heteroscedastic_noise(S, M, corruption_config['noise_scale'],
                                        seed=seed + env_idx*1000 + sample_idx + 20000)
            
            # Add drift
            X = add_drift(X, M, corruption_config['drift_magnitude'], drift_type='ar1',
                        seed=seed + env_idx*1000 + sample_idx + 30000)
            
            X_all.append(X)
            M_all.append(M)
            S_all.append(S)
            e_all.append(env_idx)
    
    X_all = np.stack(X_all, axis=0)
    M_all = np.stack(M_all, axis=0)
    S_all = np.stack(S_all, axis=0)
    e_all = np.array(e_all, dtype=np.int32)
    
    print(f"\n✅ Generated {X_all.shape[0]} samples: shape {X_all.shape}")
    
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
    
    # Metadata
    metadata = {
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
        "policy_edges": config.get("policy_edges", None)
    }
    
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Benchmark saved to: {output_dir}")
    print(f"   Train: {X_train.shape[0]} samples ({X_train.shape})")
    print(f"   Val: {X_val.shape[0]} samples ({X_val.shape})")
    if "policy_edges" in config:
        print(f"   Policy edges (H3): {config['policy_edges']}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic corruption benchmarks for hypothesis testing"
    )
    
    parser.add_argument(
        "--benchmark", type=str, default="h1_easy",
        choices=list(BENCHMARKS.keys()),
        help="Benchmark configuration to generate"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory (defaults to data/interim/synth_corrupted_{benchmark_name})"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all available benchmarks"
    )
    
    args = parser.parse_args()
    
    if args.list:
        print("\n📊 Available Benchmarks:\n")
        for name, config in BENCHMARKS.items():
            print(f"  {name:20s} - {config['description']}")
        return
    
    # Get benchmark config
    if args.benchmark not in BENCHMARKS:
        print(f"❌ Unknown benchmark: {args.benchmark}")
        print(f"   Available: {', '.join(BENCHMARKS.keys())}")
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
    print(f"✅ Benchmark ready for training!")
    print(f"   Next: python scripts/train_rcgnn.py configs/train.yaml --data_root {output_dir}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
