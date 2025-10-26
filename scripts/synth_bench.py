#!/usr/bin/env python3
"""
Generate synthetic causal discovery benchmarks with compound corruptions.

This script creates controlled synthetic datasets for testing RC-GNN and baselines:
- Multiple graph types (ER, Scale-Free)
- Various mechanisms (linear, MLP, GP)
- Compound corruptions (MCAR/MAR/MNAR + noise + drift)
- Multiple environments with different corruption levels

Usage:
    python scripts/synth_bench.py --help
    python scripts/synth_bench.py --graph_type er --d 10 --edges 20 --output data/interim/synth_er10
    python scripts/synth_bench.py --sweep  # Run full factorial sweep
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
# Graph Generation
# ============================================================================

def generate_er_dag(d, num_edges, seed=None):
    """Generate Erdős-Rényi random DAG.
    
    Args:
        d: Number of nodes
        num_edges: Number of edges
        seed: Random seed
        
    Returns:
        A_true: Binary adjacency matrix (d, d)
    """
    if seed is not None:
        np.random.seed(seed)
    
    A_true = np.zeros((d, d))
    edges_added = 0
    max_attempts = num_edges * 10  # Prevent infinite loop
    attempts = 0
    
    while edges_added < num_edges and attempts < max_attempts:
        i, j = np.random.randint(0, d, size=2)
        if i < j and A_true[i, j] == 0:  # Lower triangular (DAG constraint)
            A_true[i, j] = 1
            edges_added += 1
        attempts += 1
    
    # Randomly permute rows/cols to break ordering
    perm = np.random.permutation(d)
    A_true = A_true[perm, :][:, perm]
    
    return A_true


def generate_scale_free_dag(d, attachment=2, seed=None):
    """Generate scale-free DAG using Barabási-Albert model.
    
    Args:
        d: Number of nodes
        attachment: Number of edges to attach from new node
        seed: Random seed
        
    Returns:
        A_true: Binary adjacency matrix (d, d)
    """
    if seed is not None:
        np.random.seed(seed)
    
    A_true = np.zeros((d, d))
    degrees = np.zeros(d)
    
    # Start with a small complete graph
    for i in range(min(attachment, d)):
        for j in range(i+1, min(attachment, d)):
            A_true[i, j] = 1
            degrees[i] += 1
            degrees[j] += 1
    
    # Add remaining nodes with preferential attachment
    for new_node in range(attachment, d):
        # Probability proportional to degree
        probs = degrees[:new_node] / (degrees[:new_node].sum() + 1e-10)
        targets = np.random.choice(new_node, size=min(attachment, new_node), 
                                     replace=False, p=probs)
        for target in targets:
            if new_node > target:  # Maintain DAG (lower triangular)
                A_true[target, new_node] = 1
            else:
                A_true[new_node, target] = 1
            degrees[new_node] += 1
            degrees[target] += 1
    
    return A_true


# ============================================================================
# Mechanisms (Data Generation)
# ============================================================================

def linear_mechanism(X_parents, W, noise_scale=0.1):
    """Linear mechanism: X_j = W^T X_pa(j) + noise."""
    return X_parents @ W + np.random.randn(*X_parents.shape[:-1], 1) * noise_scale


def mlp_mechanism(X_parents, hidden_dim=20, noise_scale=0.1, seed=None):
    """Nonlinear MLP mechanism."""
    if seed is not None:
        np.random.seed(seed)
    
    input_dim = X_parents.shape[-1]
    W1 = np.random.randn(input_dim, hidden_dim) * 0.5
    b1 = np.random.randn(hidden_dim) * 0.1
    W2 = np.random.randn(hidden_dim, 1) * 0.5
    b2 = np.random.randn(1) * 0.1
    
    # Forward pass
    h = np.tanh(X_parents @ W1 + b1)
    return h @ W2 + b2 + np.random.randn(*X_parents.shape[:-1], 1) * noise_scale


def generate_data_from_dag(A_true, T, mechanism="linear", noise_scale=0.1, seed=None):
    """Generate time series data from DAG.
    
    Args:
        A_true: Binary adjacency matrix (d, d)
        T: Number of time steps
        mechanism: "linear" or "mlp"
        noise_scale: Noise standard deviation
        seed: Random seed
        
    Returns:
        X: Generated data (T, d)
    """
    if seed is not None:
        np.random.seed(seed)
    
    d = A_true.shape[0]
    X = np.zeros((T, d))
    
    # Topological order (for DAG)
    in_degree = A_true.sum(axis=0)
    order = []
    remaining = set(range(d))
    
    while remaining:
        # Find nodes with no incoming edges from remaining nodes
        roots = [i for i in remaining if in_degree[i] == 0]
        if not roots:  # Cycle detected (shouldn't happen with valid DAG)
            roots = list(remaining)
        
        order.extend(roots)
        for i in roots:
            remaining.remove(i)
            # Reduce in-degree of children
            for j in range(d):
                if A_true[i, j] > 0:
                    in_degree[j] -= 1
    
    # Generate data in topological order
    for t in range(T):
        for j in order:
            parents = np.where(A_true[:, j] > 0)[0]
            
            if len(parents) == 0:
                # Root node: sample from standard normal
                X[t, j] = np.random.randn() * noise_scale
            else:
                # Child node: apply mechanism
                X_parents = X[t, parents].reshape(1, -1)
                
                if mechanism == "linear":
                    W = np.random.randn(len(parents), 1) * 0.5
                    X[t, j] = linear_mechanism(X_parents, W, noise_scale)[0, 0]
                elif mechanism == "mlp":
                    X[t, j] = mlp_mechanism(X_parents, noise_scale=noise_scale, seed=seed+t*d+j)[0, 0]
                else:
                    raise ValueError(f"Unknown mechanism: {mechanism}")
    
    return X


# ============================================================================
# Corruptions
# ============================================================================

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
    
    # Each feature's missingness depends on the previous feature
    for j in range(1, d):
        # Probability of missing increases with abs(X[:, j-1])
        prob_missing = 1 / (1 + np.exp(-dependency_strength * np.abs(X[:, j-1]) + missing_rate*5))
        M[:, j] = (np.random.rand(T) > prob_missing).astype(np.float32)
    
    return M


def apply_mnar(X, missing_rate=0.2, self_dependency=2.0, seed=None):
    """Missing Not At Random (depends on the value itself)."""
    if seed is not None:
        np.random.seed(seed)
    
    # Probability of missing increases with abs(X) itself
    prob_missing = 1 / (1 + np.exp(-self_dependency * np.abs(X) + missing_rate*5))
    M = (np.random.rand(*X.shape) > prob_missing).astype(np.float32)
    
    return M


def add_heteroscedastic_noise(X, M, noise_scale=0.5, seed=None):
    """Add heteroscedastic noise (variance depends on X magnitude)."""
    if seed is not None:
        np.random.seed(seed)
    
    # Noise std proportional to abs(X)
    noise_std = noise_scale * (1 + np.abs(X))
    noise = np.random.randn(*X.shape) * noise_std
    
    # Only add noise to observed values
    X_noisy = X + noise * M
    return X_noisy


def add_drift(X, M, drift_magnitude=0.2, drift_type="linear", seed=None):
    """Add sensor drift/bias over time."""
    if seed is not None:
        np.random.seed(seed)
    
    T, d = X.shape
    
    if drift_type == "linear":
        # Linear drift: bias increases linearly over time
        time_vec = np.linspace(0, 1, T).reshape(T, 1)
        drift = drift_magnitude * time_vec * np.random.randn(1, d)
    elif drift_type == "ar1":
        # AR(1) drift: autocorrelated bias
        drift = np.zeros((T, d))
        drift[0] = np.random.randn(d) * drift_magnitude * 0.1
        for t in range(1, T):
            drift[t] = 0.9 * drift[t-1] + np.random.randn(d) * drift_magnitude * 0.1
    else:
        drift = np.zeros((T, d))
    
    # Apply drift only to observed values
    X_drifted = X + drift * M
    return X_drifted


# ============================================================================
# Multi-Environment Generation
# ============================================================================

def generate_multi_env_dataset(
    A_true, 
    n_envs=3, 
    samples_per_env=500,
    T_per_sample=50,
    mechanism="linear",
    corruption_configs=None,
    seed=None
):
    """Generate multi-environment dataset with different corruptions.
    
    Args:
        A_true: True DAG (d, d)
        n_envs: Number of environments
        samples_per_env: Number of samples per environment
        T_per_sample: Time steps per sample
        mechanism: Data generation mechanism
        corruption_configs: List of dicts with corruption params per env
        seed: Random seed
        
    Returns:
        X: Data (n_envs * samples_per_env, T_per_sample, d)
        M: Masks (same shape)
        S: Underlying signal (same shape)
        e: Environment labels (n_envs * samples_per_env,)
    """
    if seed is not None:
        np.random.seed(seed)
    
    d = A_true.shape[0]
    total_samples = n_envs * samples_per_env
    
    X_all = []
    M_all = []
    S_all = []
    e_all = []
    
    # Default corruption configs if not provided
    if corruption_configs is None:
        corruption_configs = []
        for env_idx in range(n_envs):
            corruption_configs.append({
                "missing_type": ["mcar", "mar", "mnar"][env_idx % 3],
                "missing_rate": 0.2 + env_idx * 0.1,  # 20%, 30%, 40%
                "noise_scale": 0.1 + env_idx * 0.2,   # 0.1, 0.3, 0.5
                "drift_magnitude": 0.0 + env_idx * 0.1  # 0.0, 0.1, 0.2
            })
    
    for env_idx in range(n_envs):
        config = corruption_configs[env_idx]
        
        for sample_idx in range(samples_per_env):
            # Generate clean signal
            S = generate_data_from_dag(
                A_true, T_per_sample, mechanism=mechanism, 
                noise_scale=0.1, seed=seed + env_idx*1000 + sample_idx
            )
            
            # Apply missingness
            if config["missing_type"] == "mcar":
                M = apply_mcar(S, config["missing_rate"], seed=seed + env_idx*1000 + sample_idx + 10000)
            elif config["missing_type"] == "mar":
                M = apply_mar(S, config["missing_rate"], seed=seed + env_idx*1000 + sample_idx + 10000)
            else:  # mnar
                M = apply_mnar(S, config["missing_rate"], seed=seed + env_idx*1000 + sample_idx + 10000)
            
            # Add noise
            X = add_heteroscedastic_noise(S, M, config["noise_scale"], seed=seed + env_idx*1000 + sample_idx + 20000)
            
            # Add drift
            X = add_drift(X, M, config["drift_magnitude"], drift_type="ar1", seed=seed + env_idx*1000 + sample_idx + 30000)
            
            X_all.append(X)
            M_all.append(M)
            S_all.append(S)
            e_all.append(env_idx)
    
    X_all = np.stack(X_all, axis=0)  # (N, T, d)
    M_all = np.stack(M_all, axis=0)
    S_all = np.stack(S_all, axis=0)
    e_all = np.array(e_all, dtype=np.int32)
    
    return X_all, M_all, S_all, e_all


# ============================================================================
# Dataset Saving
# ============================================================================

def save_dataset(output_dir, A_true, X_train, M_train, S_train, e_train, 
                 X_val, M_val, S_val, e_val, metadata):
    """Save dataset in RC-GNN format."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save arrays
    np.save(os.path.join(output_dir, "A_true.npy"), A_true)
    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "M_train.npy"), M_train)
    np.save(os.path.join(output_dir, "S_train.npy"), S_train)
    np.save(os.path.join(output_dir, "e_train.npy"), e_train)
    np.save(os.path.join(output_dir, "X_val.npy"), X_val)
    np.save(os.path.join(output_dir, "M_val.npy"), M_val)
    np.save(os.path.join(output_dir, "S_val.npy"), S_val)
    np.save(os.path.join(output_dir, "e_val.npy"), e_val)
    
    # Save metadata
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Dataset saved to: {output_dir}")
    print(f"   - Train: {X_train.shape[0]} samples")
    print(f"   - Val: {X_val.shape[0]} samples")
    print(f"   - Features: {X_train.shape[-1]}")
    print(f"   - Time steps: {X_train.shape[1]}")
    print(f"   - True edges: {int(A_true.sum())}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic causal discovery benchmarks")
    
    # Graph parameters
    parser.add_argument("--graph_type", type=str, default="er", choices=["er", "sf"],
                        help="Graph type: er (Erdős-Rényi) or sf (Scale-Free)")
    parser.add_argument("--d", type=int, default=10, help="Number of nodes")
    parser.add_argument("--edges", type=int, default=20, help="Number of edges (for ER)")
    parser.add_argument("--attachment", type=int, default=2, help="Attachment parameter (for SF)")
    
    # Data parameters
    parser.add_argument("--mechanism", type=str, default="linear", choices=["linear", "mlp"],
                        help="Data generation mechanism")
    parser.add_argument("--n_envs", type=int, default=3, help="Number of environments")
    parser.add_argument("--samples_per_env", type=int, default=500, help="Samples per environment")
    parser.add_argument("--T", type=int, default=50, help="Time steps per sample")
    
    # Corruption parameters
    parser.add_argument("--missing_type", type=str, default="mcar", choices=["mcar", "mar", "mnar"],
                        help="Missingness type")
    parser.add_argument("--missing_rate", type=float, default=0.2, help="Missing data rate")
    parser.add_argument("--noise_scale", type=float, default=0.1, help="Noise scale")
    parser.add_argument("--drift", type=float, default=0.0, help="Drift magnitude")
    
    # Output
    parser.add_argument("--output", type=str, default="data/interim/synth_test",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Special modes
    parser.add_argument("--sweep", action="store_true", help="Run factorial sweep")
    
    args = parser.parse_args()
    
    if args.sweep:
        print("🔄 Running factorial sweep...")
        run_factorial_sweep()
        return
    
    # Generate graph
    print(f"\n📊 Generating {args.graph_type.upper()} graph (d={args.d})...")
    if args.graph_type == "er":
        A_true = generate_er_dag(args.d, args.edges, seed=args.seed)
    else:  # sf
        A_true = generate_scale_free_dag(args.d, args.attachment, seed=args.seed)
    
    print(f"✅ Graph generated: {int(A_true.sum())} edges")
    
    # Corruption configs (uniform for all envs in single-run mode)
    corruption_configs = [{
        "missing_type": args.missing_type,
        "missing_rate": args.missing_rate,
        "noise_scale": args.noise_scale,
        "drift_magnitude": args.drift
    } for _ in range(args.n_envs)]
    
    # Generate data
    print(f"\n🔧 Generating multi-environment data ({args.n_envs} envs, {args.mechanism} mechanism)...")
    X_all, M_all, S_all, e_all = generate_multi_env_dataset(
        A_true, 
        n_envs=args.n_envs,
        samples_per_env=args.samples_per_env,
        T_per_sample=args.T,
        mechanism=args.mechanism,
        corruption_configs=corruption_configs,
        seed=args.seed
    )
    
    # Train/val split (80/20)
    n_total = X_all.shape[0]
    n_train = int(0.8 * n_total)
    
    indices = np.random.RandomState(args.seed).permutation(n_total)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    X_train, M_train, S_train, e_train = X_all[train_idx], M_all[train_idx], S_all[train_idx], e_all[train_idx]
    X_val, M_val, S_val, e_val = X_all[val_idx], M_all[val_idx], S_all[val_idx], e_all[val_idx]
    
    # Metadata
    metadata = {
        "graph_type": args.graph_type,
        "d": args.d,
        "edges": int(A_true.sum()),
        "mechanism": args.mechanism,
        "n_envs": args.n_envs,
        "T": args.T,
        "corruption_configs": corruption_configs,
        "seed": args.seed,
        "created": "2025-10-26"
    }
    
    # Save
    save_dataset(args.output, A_true, X_train, M_train, S_train, e_train,
                 X_val, M_val, S_val, e_val, metadata)


def run_factorial_sweep():
    """Run factorial sweep of hyperparameters."""
    graph_types = ["er", "sf"]
    d_values = [10, 20]
    mechanisms = ["linear", "mlp"]
    missing_types = ["mcar", "mar", "mnar"]
    missing_rates = [0.0, 0.2, 0.4]
    noise_scales = [0.1, 0.5, 1.0]
    
    total_configs = len(graph_types) * len(d_values) * len(mechanisms) * len(missing_types) * len(missing_rates) * len(noise_scales)
    print(f"📊 Generating {total_configs} synthetic datasets...")
    
    config_idx = 0
    for graph_type in graph_types:
        for d in d_values:
            for mechanism in mechanisms:
                for missing_type in missing_types:
                    for missing_rate in missing_rates:
                        for noise_scale in noise_scales:
                            config_idx += 1
                            
                            # Generate unique output directory
                            output_dir = f"data/interim/synth_{graph_type}_d{d}_{mechanism}_" \
                                        f"{missing_type}{int(missing_rate*100)}_noise{int(noise_scale*10)}"
                            
                            print(f"\n[{config_idx}/{total_configs}] {output_dir}")
                            
                            # Generate graph
                            seed = config_idx
                            if graph_type == "er":
                                edges = d * 2  # 2x sparsity
                                A_true = generate_er_dag(d, edges, seed=seed)
                            else:
                                A_true = generate_scale_free_dag(d, attachment=2, seed=seed)
                            
                            # Corruption configs
                            corruption_configs = [{
                                "missing_type": missing_type,
                                "missing_rate": missing_rate,
                                "noise_scale": noise_scale,
                                "drift_magnitude": 0.0
                            } for _ in range(3)]
                            
                            # Generate data
                            X_all, M_all, S_all, e_all = generate_multi_env_dataset(
                                A_true, n_envs=3, samples_per_env=300, T_per_sample=50,
                                mechanism=mechanism, corruption_configs=corruption_configs, seed=seed
                            )
                            
                            # Split
                            n_total = X_all.shape[0]
                            n_train = int(0.8 * n_total)
                            indices = np.random.RandomState(seed).permutation(n_total)
                            train_idx, val_idx = indices[:n_train], indices[n_train:]
                            
                            X_train, M_train, S_train, e_train = X_all[train_idx], M_all[train_idx], S_all[train_idx], e_all[train_idx]
                            X_val, M_val, S_val, e_val = X_all[val_idx], M_all[val_idx], S_all[val_idx], e_all[val_idx]
                            
                            # Metadata
                            metadata = {
                                "graph_type": graph_type,
                                "d": d,
                                "edges": int(A_true.sum()),
                                "mechanism": mechanism,
                                "missing_type": missing_type,
                                "missing_rate": missing_rate,
                                "noise_scale": noise_scale,
                                "n_envs": 3,
                                "seed": seed
                            }
                            
                            # Save
                            save_dataset(output_dir, A_true, X_train, M_train, S_train, e_train,
                                        X_val, M_val, S_val, e_val, metadata)
    
    print(f"\n✅ Factorial sweep complete: {total_configs} datasets generated!")


if __name__ == "__main__":
    main()
