#!/usr/bin/env python3
"""
Visualize environment-specific structure variations.

Extracts and visualizes per-environment adjacency deltas from trained RC-GNN model.
Shows how the base causal structure adapts to different environments/regimes.

Usage:
  python scripts/visualize_environment_structure.py --checkpoint artifacts/checkpoints/rcgnn_best.pt
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pathlib import Path

import path_helper # noqa: F401
from src.models.rcgnn import RCGNN
from src.dataio.loaders import load_synth
import yaml


def extract_structures(model, val_loader, device='cpu'):
    """Extract base structure and per-environment deltas."""
    model.eval()
    
    structures = []
    with torch.no_grad():
        for batch in val_loader:
            for k in batch:
                batch[k] = batch[k].to(device)
            
            # Call forward method with batch data
            out = model(
                X=batch["X"],
                M=batch.get("M"),
                e=batch.get("e")
            )
            
            # Get adjacency matrix for each sample
            A = out["A"].cpu().numpy() # [B, d, d]
            B = A.shape[0]
            
            # Extract environment info
            e_batch = batch.get("e")
            if e_batch is not None:
                e_vals = e_batch.cpu().numpy().astype(int)
            else:
                e_vals = np.zeros(B, dtype=int)
            
            # Store structures
            for i in range(B):
                structures.append({'A': A[i], 'env': int(e_vals[i])})
    
    return structures


def compute_average_structure_by_env(structures, n_envs):
    """Compute average structure per environment."""
    env_structures = {e: [] for e in range(n_envs)}
    
    for struct in structures:
        env_structures[struct['env']].append(struct['A'])
    
    # Average per environment
    A_per_env = {}
    for e in range(n_envs):
        if env_structures[e]:
            A_per_env[e] = np.mean(env_structures[e], axis=0)
        else:
            A_per_env[e] = np.zeros((1, 1))
    
    return A_per_env


def plot_environment_comparison(A_per_env, output_path='artifacts/environment_structures.png'):
    """Plot side-by-side comparison of environment-specific structures."""
    n_envs = len(A_per_env)
    d = A_per_env[0].shape[0]
    
    fig, axes = plt.subplots(1, n_envs, figsize=(5 * n_envs, 5))
    if n_envs == 1:
        axes = [axes]
    
    vmin = min(A.min() for A in A_per_env.values())
    vmax = max(A.max() for A in A_per_env.values())
    
    for env_idx, ax in enumerate(axes):
        A = A_per_env[env_idx]
        im = ax.imshow(A, cmap='YlOrRd', vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_title(f'Environment {env_idx}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Target Features')
        ax.set_ylabel('Source Features')
        plt.colorbar(im, ax=ax, label='Edge Strength')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[DONE] Saved environment comparison: {output_path}")
    plt.close()


def plot_delta_analysis(A_per_env, output_path='artifacts/environment_deltas.png'):
    """Analyze and visualize environment deltas (differences from mean)."""
    n_envs = len(A_per_env)
    A_mean = np.mean([A for A in A_per_env.values()], axis=0)
    
    fig, axes = plt.subplots(1, n_envs, figsize=(5 * n_envs, 5))
    if n_envs == 1:
        axes = [axes]
    
    deltas = {}
    for env_idx in range(n_envs):
        deltas[env_idx] = A_per_env[env_idx] - A_mean
    
    vmax = max(np.abs(d).max() for d in deltas.values())
    
    for env_idx, ax in enumerate(axes):
        delta = deltas[env_idx]
        im = ax.imshow(delta, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
        ax.set_title(f'Environment {env_idx} Delta', fontsize=12, fontweight='bold')
        ax.set_xlabel('Target Features')
        ax.set_ylabel('Source Features')
        plt.colorbar(im, ax=ax, label='Difference from Mean')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[DONE] Saved delta analysis: {output_path}")
    plt.close()


def plot_structure_variation(A_per_env, output_path='artifacts/structure_variation.png'):
    """Quantify how much structure varies across environments."""
    n_envs = len(A_per_env)
    A_mean = np.mean([A for A in A_per_env.values()], axis=0)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel 1: Frobenius norm of deltas
    ax = axes[0]
    deltas_norm = []
    for env_idx in range(n_envs):
        delta = A_per_env[env_idx] - A_mean
        norm = np.linalg.norm(delta, 'fro')
        deltas_norm.append(norm)
    
    bars = ax.bar(range(n_envs), deltas_norm, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Environment')
    ax.set_ylabel('Frobenius Norm of Delta')
    ax.set_title('Structure Variation from Global Mean')
    ax.set_xticks(range(n_envs))
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, deltas_norm):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Panel 2: Pairwise differences
    ax = axes[1]
    pairwise_diffs = []
    labels = []
    for i in range(n_envs):
        for j in range(i+1, n_envs):
            diff = np.linalg.norm(A_per_env[i] - A_per_env[j], 'fro')
            pairwise_diffs.append(diff)
            labels.append(f'Env{i}-Env{j}')
    
    if pairwise_diffs:
        bars = ax.bar(range(len(pairwise_diffs)), pairwise_diffs, 
                     color='coral', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Environment Pair')
        ax.set_ylabel('Frobenius Norm Difference')
        ax.set_title('Pairwise Structure Differences')
        ax.set_xticks(range(len(pairwise_diffs)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars, pairwise_diffs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[DONE] Saved structure variation analysis: {output_path}")
    plt.close()


def save_structure_report(A_per_env, output_path='artifacts/environment_structures_report.txt'):
    """Save detailed report on environment-specific structures."""
    n_envs = len(A_per_env)
    A_mean = np.mean([A for A in A_per_env.values()], axis=0)
    
    report = []
    report.append("=" * 80)
    report.append("ENVIRONMENT-SPECIFIC STRUCTURE ANALYSIS")
    report.append("=" * 80)
    report.append("")
    
    report.append(f"Number of Environments: {n_envs}")
    report.append(f"Feature Dimension: {A_per_env[0].shape[0]}")
    report.append("")
    
    report.append("## GLOBAL STRUCTURE STATISTICS")
    report.append(f"Min: {A_mean.min():.6f}")
    report.append(f"Max: {A_mean.max():.6f}")
    report.append(f"Mean: {A_mean.mean():.6f}")
    report.append(f"Std: {A_mean.std():.6f}")
    report.append(f"Non-zero edges: {(A_mean > 0).sum()}/{A_mean.size} ({(A_mean > 0).sum()/A_mean.size*100:.1f}%)")
    report.append("")
    
    report.append("## PER-ENVIRONMENT STATISTICS")
    for env_idx in range(n_envs):
        A_env = A_per_env[env_idx]
        delta = A_env - A_mean
        delta_norm = np.linalg.norm(delta, 'fro')
        
        report.append(f"\nEnvironment {env_idx}:")
        report.append(f" Min: {A_env.min():.6f}")
        report.append(f" Max: {A_env.max():.6f}")
        report.append(f" Mean: {A_env.mean():.6f}")
        report.append(f" Non-zero edges: {(A_env > 0).sum()}/{A_env.size} ({(A_env > 0).sum()/A_env.size*100:.1f}%)")
        report.append(f" Delta norm (from global): {delta_norm:.6f}")
        report.append(f" Max change in any edge: {np.abs(delta).max():.6f}")
    
    report.append("")
    report.append("## STRUCTURE VARIATION")
    report.append(f"Variation Magnitude: {np.std([np.linalg.norm(A_per_env[e] - A_mean) for e in range(n_envs)]):.6f}")
    
    pairwise_diffs = []
    for i in range(n_envs):
        for j in range(i+1, n_envs):
            diff = np.linalg.norm(A_per_env[i] - A_per_env[j], 'fro')
            pairwise_diffs.append(diff)
    
    if pairwise_diffs:
        report.append(f"Avg Pairwise Difference: {np.mean(pairwise_diffs):.6f}")
        report.append(f"Max Pairwise Difference: {np.max(pairwise_diffs):.6f}")
    
    report.append("")
    report.append("## INTERPRETATION")
    
    total_variation = np.std([np.linalg.norm(A_per_env[e] - A_mean) for e in range(n_envs)])
    if total_variation < 0.01:
        report.append("[DONE] Minimal variation across environments")
        report.append(" -> Base causal structure is robust to environmental shifts")
    elif total_variation < 0.05:
        report.append("[DONE] Moderate variation across environments")
        report.append(" -> Model adapts structure moderately per environment")
    else:
        report.append("[WARN] High variation across environments")
        report.append(" -> Model learns environment-specific causal structures")
    
    report.append("")
    report.append("=" * 80)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"[DONE] Saved structure report: {output_path}")
    print("\n" + '\n'.join(report))


def main():
    parser = argparse.ArgumentParser(
        description="Visualize environment-specific causal structures"
    )
    parser.add_argument('--checkpoint', default='artifacts/checkpoints/rcgnn_best.pt',
                       help='Path to trained model checkpoint')
    parser.add_argument('--config-data', default='configs/data_uci.yaml',
                       help='Data configuration file')
    parser.add_argument('--config-model', default='configs/model.yaml',
                       help='Model configuration file')
    parser.add_argument('--export', default='artifacts',
                       help='Export directory for visualizations')
    parser.add_argument('--device', default='cpu',
                       help='Device to use (cpu or cuda)')
    
    args = parser.parse_args()
    
    # Load model
    if not os.path.exists(args.checkpoint):
        print(f"[FAIL] Error: Checkpoint not found at {args.checkpoint}")
        sys.exit(1)
    
    with open(args.config_data) as f:
        dc = yaml.safe_load(f)
    with open(args.config_model) as f:
        mc = yaml.safe_load(f)
    
    # Setup data
    dataset_dir = dc.get("dataset_dir", dc.get("dataset", "synth_small"))
    root = os.path.join(dc["paths"]["root"], "interim", dataset_dir)
    
    val_ds = load_synth(root, "val", seed=42)
    val_ld = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False)
    
    d = val_ds.X.shape[-1]
    n_envs_dataset = val_ds.n_envs if hasattr(val_ds, 'n_envs') else 2
    
    # Extract model config parameters
    latent_dim = mc.get("latent_dim", 16)
    hidden_dim = mc.get("hidden_dim", 32)
    sparsify_method = mc.get("sparsify", {}).get("method", "topk")
    topk_ratio = mc.get("sparsify", {}).get("topk_ratio", 0.1)
    
    # Load checkpoint to determine n_envs (checkpoints typically trained with n_envs=1)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    n_envs_ckpt = 1 # Default: most checkpoints use n_envs=1
    if "structure_learner.A_deltas.0" in checkpoint:
        # Count how many deltas exist
        delta_keys = [k for k in checkpoint.keys() if "A_deltas." in k]
        n_envs_ckpt = max([int(k.split(".")[1]) for k in delta_keys]) + 1
    
    # Use checkpoint n_envs (not dataset n_envs) to match saved weights
    model = RCGNN(
        d=d,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        n_envs=n_envs_ckpt,
        sparsify_method=sparsify_method,
        topk_ratio=topk_ratio,
        device=args.device
    )
    model.load_state_dict(checkpoint)
    model.to(args.device)
    print(f"[DONE] Loaded model from {args.checkpoint} (n_envs={n_envs_ckpt})")
    
    # Extract structures
    print("\n" + "=" * 80)
    print("EXTRACTING ENVIRONMENT-SPECIFIC STRUCTURES")
    print("=" * 80)
    
    structures = extract_structures(model, val_ld, args.device)
    print(f"[DONE] Extracted {len(structures)} structures")
    
    A_per_env = compute_average_structure_by_env(structures, n_envs_ckpt)
    print(f"[DONE] Computed average structures for {len(A_per_env)} environments")
    
    # Generate visualizations
    print("\n Generating visualizations...")
    plot_environment_comparison(A_per_env, 
                               os.path.join(args.export, 'environment_structures.png'))
    plot_delta_analysis(A_per_env, 
                       os.path.join(args.export, 'environment_deltas.png'))
    plot_structure_variation(A_per_env, 
                            os.path.join(args.export, 'structure_variation.png'))
    
    # Save report
    save_structure_report(A_per_env, 
                         os.path.join(args.export, 'environment_structures_report.txt'))
    
    print("\n[DONE] Environment-specific analysis complete!")


if __name__ == "__main__":
    main()
