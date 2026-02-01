#!/usr/bin/env python3
"""
Generate multi-environment corruption benchmarks for RC-GNN hypothesis testing.

Creates synthetic datasets with:
- Known ground truth causal structures
- Multiple environments with different corruption patterns
- Tunable MCAR/MAR/MNAR missingness (40-60%)
- Heteroscedastic noise and drift
- Policy-relevant edge tracking for H3

Outputs saved to: data/interim/synth_corrupted/

Benchmarks designed for:
- H1: Structural accuracy under compound corruptions
- H2: Stability improvement via invariance loss
- H3: Policy consistency validation

Usage:
    python scripts/generate_corruption_benchmarks.py --help
    python scripts/generate_corruption_benchmarks.py --benchmark h1_full
    python scripts/generate_corruption_benchmarks.py --benchmark h2_stability
    python scripts/generate_corruption_benchmarks.py --benchmark h3_policy
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

from scripts.synth_bench import (
    generate_er_dag,
    generate_scale_free_dag,
    generate_data_from_dag,
    apply_mcar,
    apply_mar,
    apply_mnar,
    add_heteroscedastic_noise,
    add_drift,
)


# ============================================================================
# Benchmark Generators
# ============================================================================

def generate_benchmark_h1_full(output_root="data/interim/synth_corrupted/h1_full", seed=42):
    """
    H1 Benchmark: Structural Accuracy Under Compound Corruptions
    
    Creates 12 datasets across:
    - 2 graph types: ER, Scale-Free
    - 3 corruption intensities: Light (20%), Medium (40%), Heavy (60%)
    - 2 mechanisms: Linear, MLP
    
    Each dataset: 4 environments, 500 samples/env, 50 timesteps
    Corruption pattern: Increases with environment index
    
    Success criterion:
    - RC-GNN SHD within 15% of oracle
    - Baselines degrade >40% under heavy corruption
    """
    os.makedirs(output_root, exist_ok=True)
    
    graph_types = ["er", "sf"]
    intensities = [
        {"name": "light", "missing": 0.2, "noise": 0.1, "drift": 0.0},
        {"name": "medium", "missing": 0.4, "noise": 0.3, "drift": 0.1},
        {"name": "heavy", "missing": 0.6, "noise": 0.5, "drift": 0.2},
    ]
    mechanisms = ["linear", "mlp"]
    
    configs = []
    
    for graph_type in graph_types:
        for intensity in intensities:
            for mechanism in mechanisms:
                config = {
                    "graph_type": graph_type,
                    "d": 15,
                    "edges": 30 if graph_type == "er" else None,
                    "attachment": 2 if graph_type == "sf" else None,
                    "mechanism": mechanism,
                    "n_envs": 4,
                    "samples_per_env": 500,
                    "T": 50,
                    "base_missing": intensity["missing"],
                    "base_noise": intensity["noise"],
                    "base_drift": intensity["drift"],
                    "intensity_name": intensity["name"],
                    "seed": seed,
                }
                configs.append(config)
    
    print(f"\n Generating H1 Benchmark: Structural Accuracy ({len(configs)} configs)")
    print("=" * 70)
    
    for idx, config in enumerate(configs, 1):
        config_name = f"{config['graph_type']}_d{config['d']}_" \
                     f"{config['mechanism']}_{config['intensity_name']}"
        output_dir = os.path.join(output_root, config_name)
        
        print(f"\n[{idx}/{len(configs)}] {config_name}")
        _generate_benchmark_config(output_dir, config)
    
    print(f"\n[DONE] H1 Benchmark complete: {len(configs)} datasets in {output_root}")
    return output_root


def generate_benchmark_h2_stability(output_root="data/interim/synth_corrupted/h2_stability", seed=42):
    """
    H2 Benchmark: Stability Improvement via Invariance Loss
    
    Creates 8 datasets with:
    - 2 graph types: ER, Scale-Free
    - 2 domain shifts: Moderate, Strong
    - 2 mechanisms: Linear, MLP
    
    Each dataset: 5 environments with INCREASING shift in (B^(e), noise, drift)
    
    Corruption pattern (per environment):
    - Env 0: Base level (20% missing, 0.1 noise, 0.0 drift)
    - Env 1-4: Increasing shift (40%, 50%, 60% missing; 0.3-0.6 noise; 0.1-0.3 drift)
    
    Success criterion:
    - Var_with_inv / Var_without_inv ≤ 0.4 (60% variance reduction)
    - All environments recover same causal structure
    """
    os.makedirs(output_root, exist_ok=True)
    
    graph_types = ["er", "sf"]
    shifts = [
        {"name": "moderate", "max_drift": 0.2},
        {"name": "strong", "max_drift": 0.4},
    ]
    mechanisms = ["linear", "mlp"]
    
    configs = []
    
    for graph_type in graph_types:
        for shift in shifts:
            for mechanism in mechanisms:
                config = {
                    "graph_type": graph_type,
                    "d": 12,
                    "edges": 24 if graph_type == "er" else None,
                    "attachment": 2 if graph_type == "sf" else None,
                    "mechanism": mechanism,
                    "n_envs": 5,
                    "samples_per_env": 400,
                    "T": 50,
                    "shift_type": "progressive", # Progressive shift across environments
                    "max_shift": shift["max_drift"],
                    "shift_name": shift["name"],
                    "seed": seed,
                }
                configs.append(config)
    
    print(f"\n Generating H2 Benchmark: Stability ({len(configs)} configs)")
    print("=" * 70)
    
    for idx, config in enumerate(configs, 1):
        config_name = f"{config['graph_type']}_d{config['d']}_" \
                     f"{config['mechanism']}_{config['shift_name']}"
        output_dir = os.path.join(output_root, config_name)
        
        print(f"\n[{idx}/{len(configs)}] {config_name}")
        _generate_benchmark_config(output_dir, config)
    
    print(f"\n[DONE] H2 Benchmark complete: {len(configs)} datasets in {output_root}")
    return output_root


def generate_benchmark_h3_policy(output_root="data/interim/synth_corrupted/h3_policy", seed=42):
    """
    H3 Benchmark: Expert Agreement on Policy Pathways
    
    Creates 4 datasets (ER & Scale-Free, Linear & MLP) with:
    - Known policy-relevant edges marked in metadata
    - 4 environments with increasing corruption
    - Focus on policy pathway stability
    
    Policy edges: Critical pathways for domain decisions
    (Marked in metadata for validation against expert knowledge)
    
    Success criterion:
    - >80% expert agreement on policy edges detected by RC-GNN
    - <60% agreement on baselines
    
    Datasets include:
    - policy_edges.json: List of (i, j) pairs marking important pathways
    - expert_validation.md: Template for expert annotation
    """
    os.makedirs(output_root, exist_ok=True)
    
    graph_types = ["er", "sf"]
    mechanisms = ["linear", "mlp"]
    
    configs = []
    
    for graph_type in graph_types:
        for mechanism in mechanisms:
            config = {
                "graph_type": graph_type,
                "d": 10,
                "edges": 20 if graph_type == "er" else None,
                "attachment": 2 if graph_type == "sf" else None,
                "mechanism": mechanism,
                "n_envs": 4,
                "samples_per_env": 600,
                "T": 50,
                "base_missing": 0.3,
                "base_noise": 0.2,
                "base_drift": 0.1,
                "track_policy_edges": True,
                "seed": seed,
            }
            configs.append(config)
    
    print(f"\n Generating H3 Benchmark: Policy Pathways ({len(configs)} configs)")
    print("=" * 70)
    
    for idx, config in enumerate(configs, 1):
        config_name = f"{config['graph_type']}_d{config['d']}_{config['mechanism']}"
        output_dir = os.path.join(output_root, config_name)
        
        print(f"\n[{idx}/{len(configs)}] {config_name}")
        _generate_benchmark_config(output_dir, config, track_policy=True)
    
    print(f"\n[DONE] H3 Benchmark complete: {len(configs)} datasets in {output_root}")
    return output_root


def _generate_benchmark_config(output_dir, config, track_policy=False):
    """Generate a single benchmark configuration."""
    os.makedirs(output_dir, exist_ok=True)
    
    seed = config.get("seed", 42)
    
    # Generate graph
    if config["graph_type"] == "er":
        A_true = generate_er_dag(config["d"], config["edges"], seed=seed)
    else:
        A_true = generate_scale_free_dag(config["d"], config["attachment"], seed=seed)
    
    print(f" Graph: {config['graph_type']} (d={config['d']}, edges={int(A_true.sum())})")
    
    # Generate per-environment corruption configs
    n_envs = config["n_envs"]
    corruption_configs = []
    
    if config.get("shift_type") == "progressive":
        # Progressive shift: gradually increase corruption across environments
        max_shift = config.get("max_shift", 0.2)
        
        for env_idx in range(n_envs):
            # Interpolate from base to max shift
            shift_factor = env_idx / (n_envs - 1) if n_envs > 1 else 0
            
            missing_rate = 0.2 + shift_factor * (0.6 - 0.2) # 20% -> 60%
            noise_scale = 0.1 + shift_factor * (0.6 - 0.1) # 0.1 -> 0.6
            drift = shift_factor * max_shift # 0 -> max_shift
            
            # Vary missing type per environment
            missing_types = ["mcar", "mar", "mnar"]
            missing_type = missing_types[env_idx % 3]
            
            corruption_configs.append({
                "missing_type": missing_type,
                "missing_rate": missing_rate,
                "noise_scale": noise_scale,
                "drift_magnitude": drift,
            })
    else:
        # Uniform corruption per environment (used for H1)
        base_missing = config.get("base_missing", 0.2)
        base_noise = config.get("base_noise", 0.1)
        base_drift = config.get("base_drift", 0.0)
        
        missing_types = ["mcar", "mar", "mnar"]
        
        for env_idx in range(n_envs):
            corruption_configs.append({
                "missing_type": missing_types[env_idx % 3],
                "missing_rate": base_missing,
                "noise_scale": base_noise,
                "drift_magnitude": base_drift,
            })
    
    # Generate data for all environments
    X_all = []
    M_all = []
    S_all = []
    e_all = []
    
    for env_idx in range(n_envs):
        corruption = corruption_configs[env_idx]
        
        for sample_idx in range(config["samples_per_env"]):
            # Generate clean signal
            S = generate_data_from_dag(
                A_true,
                config["T"],
                mechanism=config["mechanism"],
                noise_scale=0.1,
                seed=seed + env_idx * 10000 + sample_idx * 100
            )
            
            # Apply corruptions
            if corruption["missing_type"] == "mcar":
                M = apply_mcar(S, corruption["missing_rate"], seed=seed + env_idx * 10000 + sample_idx * 100 + 1000)
            elif corruption["missing_type"] == "mar":
                M = apply_mar(S, corruption["missing_rate"], seed=seed + env_idx * 10000 + sample_idx * 100 + 1000)
            else: # mnar
                M = apply_mnar(S, corruption["missing_rate"], seed=seed + env_idx * 10000 + sample_idx * 100 + 1000)
            
            # Add noise
            X = add_heteroscedastic_noise(S, M, corruption["noise_scale"], seed=seed + env_idx * 10000 + sample_idx * 100 + 2000)
            
            # Add drift
            X = add_drift(X, M, corruption["drift_magnitude"], drift_type="ar1", seed=seed + env_idx * 10000 + sample_idx * 100 + 3000)
            
            X_all.append(X)
            M_all.append(M)
            S_all.append(S)
            e_all.append(env_idx)
    
    X_all = np.stack(X_all, axis=0) # (N, T, d)
    M_all = np.stack(M_all, axis=0)
    S_all = np.stack(S_all, axis=0)
    e_all = np.array(e_all, dtype=np.int32)
    
    # Train/val split
    n_total = X_all.shape[0]
    n_train = int(0.8 * n_total)
    indices = np.random.RandomState(seed).permutation(n_total)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    X_train = X_all[train_idx]
    M_train = M_all[train_idx]
    S_train = S_all[train_idx]
    e_train = e_all[train_idx]
    
    X_val = X_all[val_idx]
    M_val = M_all[val_idx]
    S_val = S_all[val_idx]
    e_val = e_all[val_idx]
    
    # Select policy edges if tracking
    policy_edges = None
    if track_policy:
        # Policy edges: high-out-degree nodes and central edges
        out_degrees = A_true.sum(axis=1)
        high_out_nodes = np.argsort(-out_degrees)[:max(2, config["d"] // 3)]
        
        policy_edges = []
        for i in high_out_nodes:
            targets = np.where(A_true[i] > 0)[0]
            for j in targets[:2]: # Top 2 targets per high-degree node
                policy_edges.append((int(i), int(j)))
        
        policy_edges = list(set(policy_edges)) # Remove duplicates
    
    # Metadata
    metadata = {
        "graph_type": config["graph_type"],
        "d": config["d"],
        "edges": int(A_true.sum()),
        "mechanism": config["mechanism"],
        "n_envs": config["n_envs"],
        "T": config["T"],
        "corruption_configs": corruption_configs,
        "shift_type": config.get("shift_type", "uniform"),
        "seed": seed,
        "policy_edges": policy_edges,
        "train_samples": X_train.shape[0],
        "val_samples": X_val.shape[0],
    }
    
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
    
    print(f" [DONE] Saved: {output_dir}")
    print(f" Train: {X_train.shape[0]} samples, Val: {X_val.shape[0]} samples")
    if policy_edges:
        print(f" Policy edges: {len(policy_edges)} pathways")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate multi-environment corruption benchmarks for RC-GNN hypothesis testing"
    )
    
    parser.add_argument(
        "--benchmark",
        type=str,
        default="h1_full",
        choices=["h1_full", "h2_stability", "h3_policy", "all"],
        help="Benchmark to generate"
    )
    
    parser.add_argument(
        "--output_root",
        type=str,
        default="data/interim/synth_corrupted",
        help="Root output directory"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("RC-GNN CORRUPTION BENCHMARKS")
    print("=" * 70)
    
    if args.benchmark == "all":
        generate_benchmark_h1_full(os.path.join(args.output_root, "h1_full"), args.seed)
        generate_benchmark_h2_stability(os.path.join(args.output_root, "h2_stability"), args.seed)
        generate_benchmark_h3_policy(os.path.join(args.output_root, "h3_policy"), args.seed)
    elif args.benchmark == "h1_full":
        generate_benchmark_h1_full(os.path.join(args.output_root, "h1_full"), args.seed)
    elif args.benchmark == "h2_stability":
        generate_benchmark_h2_stability(os.path.join(args.output_root, "h2_stability"), args.seed)
    elif args.benchmark == "h3_policy":
        generate_benchmark_h3_policy(os.path.join(args.output_root, "h3_policy"), args.seed)
    
    print("\n[DONE] All benchmarks generated successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
