#!/usr/bin/env python3
"""
Full training pipeline with validation and visualization.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support

# Add project root to path - find it first
for r, d, f in os.walk('/home/adetayo/Documents/CSCI Forms/Adetayo Research'):
    if 'rcgnn.py' in f and 'models' in r:
        repo_root = Path(r).parent.parent.parent
        break
else:
    repo_root = Path(__file__).parent.parent

os.chdir(repo_root)
sys.path.insert(0, str(repo_root))

import yaml
import torch
from torch.utils.data import DataLoader
from src.dataio.loaders import load_synth
from src.models.rcgnn import RCGNN
from src.training.optim import make_optimizer
from src.training.loop import train_epoch, eval_epoch


def compute_metrics(A_true, A_pred, threshold=0.5):
    """Compute evaluation metrics."""
    if A_true is None:
        return {}
    
    true_flat = A_true.flatten()
    pred_flat = A_pred.flatten()
    pred_binary = (pred_flat > threshold).astype(int)
    true_binary = (true_flat > 0.5).astype(int)
    
    prec, rec, f1, _ = precision_recall_fscore_support(
        true_binary, pred_binary, average='binary', zero_division=0
    )
    
    shd = np.sum(A_true != (A_pred > threshold).astype(int))
    
    return {
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'shd': shd,
        'threshold': threshold
    }


def plot_adjacency_matrices(A_true, A_pred, output_path, threshold=0.5):
    """Plot adjacency comparison."""
    fig, axes = plt.subplots(
        1, 3 if A_true is not None else 2, 
        figsize=(16 if A_true is not None else 11, 5)
    )
    if A_true is None:
        axes = [axes[0], axes[1]] # Make iterable
    
    # Predicted (continuous)
    ax = axes[1] if A_true is not None else axes[0]
    im = ax.imshow(A_pred, cmap='RdYlBu_r', vmin=0, vmax=A_pred.max())
    ax.set_title(f'Learned Adjacency (Continuous)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Target Node')
    ax.set_ylabel('Source Node')
    plt.colorbar(im, ax=ax, label='Edge Strength')
    
    # Binarized prediction
    A_pred_binary = (A_pred > threshold).astype(int)
    ax = axes[2] if A_true is not None else axes[1]
    im = ax.imshow(A_pred_binary, cmap='binary', vmin=0, vmax=1)
    ax.set_title(f'Learned Adjacency (Binarized, τ={threshold})', fontsize=12, fontweight='bold')
    ax.set_xlabel('Target Node')
    ax.set_ylabel('Source Node')
    plt.colorbar(im, ax=ax, label='Edge Present')
    
    # Ground truth
    if A_true is not None:
        ax = axes[0]
        im = ax.imshow(A_true, cmap='binary', vmin=0, vmax=1)
        ax.set_title('Ground Truth Adjacency', fontsize=12, fontweight='bold')
        ax.set_xlabel('Target Node')
        ax.set_ylabel('Source Node')
        plt.colorbar(im, ax=ax, label='Edge Present')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[DONE] Saved adjacency comparison to {output_path}")
    plt.close()


def plot_edge_distribution(A_pred, output_path):
    """Plot edge strength distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histogram
    nonzero = A_pred[A_pred > 0]
    ax = axes[0]
    ax.hist(nonzero, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(nonzero.mean(), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {nonzero.mean():.4f}')
    ax.axvline(np.median(nonzero), color='green', linestyle='--', linewidth=2,
               label=f'Median: {np.median(nonzero):.4f}')
    ax.set_xlabel('Edge Strength')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Non-Zero Edge Strengths')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Sparsity
    ax = axes[1]
    total_edges = A_pred.size
    zero_count = (A_pred == 0).sum()
    nonzero_count = total_edges - zero_count
    
    categories = ['Zero\nEdges', 'Non-Zero\nEdges']
    counts = [zero_count, nonzero_count]
    colors = ['lightgray', 'steelblue']
    
    bars = ax.bar(categories, counts, color=colors, edgecolor='black', linewidth=2)
    ax.set_ylabel('Count')
    ax.set_title('Edge Sparsity Statistics')
    ax.grid(axis='y', alpha=0.3)
    
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        percentage = (count / total_edges) * 100
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{percentage:.1f}%\n({int(count)})',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[DONE] Saved edge distribution to {output_path}")
    plt.close()


def plot_network_graph(A_pred, output_path, top_k=25):
    """Plot network graph visualization."""
    try:
        import networkx as nx
    except ImportError:
        print("[WARN] NetworkX not installed. Skipping network visualization.")
        return
    
    G = nx.DiGraph()
    n_nodes = A_pred.shape[0]
    
    for i in range(n_nodes):
        G.add_node(i)
    
    # Get top K edges
    edges_with_strength = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j and A_pred[i, j] > 0:
                edges_with_strength.append((i, j, A_pred[i, j]))
    
    edges_with_strength.sort(key=lambda x: x[2], reverse=True)
    top_edges = edges_with_strength[:top_k]
    
    for src, tgt, strength in top_edges:
        G.add_edge(src, tgt, weight=strength)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    node_sizes = [300 + 200 * (G.in_degree(n) + G.out_degree(n)) for n in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue',
                          edgecolors='black', linewidths=2, ax=ax)
    
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    max_weight = max(weights) if weights else 1
    edge_widths = [2 + 4 * (w / max_weight) for w in weights]
    
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='gray',
                          arrows=True, arrowsize=20, arrowstyle='->',
                          connectionstyle='arc3,rad=0.1', ax=ax, alpha=0.6)
    
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)
    
    edge_labels = {(u, v): f'{G[u][v]["weight"]:.3f}' for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, ax=ax)
    
    ax.set_title(f'Causal Graph Network (Top {top_k} Edges)', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[DONE] Saved network graph to {output_path}")
    plt.close()


def main():
    print("=" * 80)
    print("FULL TRAINING WITH VALIDATION & VISUALIZATION")
    print("=" * 80 + "\n")
    
    # Load configs
    with open('configs/data_uci.yaml') as f:
        dc = yaml.safe_load(f)
    with open('configs/model.yaml') as f:
        mc = yaml.safe_load(f)
    with open('configs/train.yaml') as f:
        tc = yaml.safe_load(f)
    
    # Clean old artifacts
    try:
        os.remove('artifacts/checkpoints/rcgnn_best.pt')
        os.remove('artifacts/adjacency/A_mean.npy')
    except:
        pass
    
    # Setup dataset
    dataset_dir = dc.get("dataset_dir", dc.get("dataset", "synth_small"))
    root = os.path.join(dc["paths"]["root"], "interim", dataset_dir)
    
    train_ds = load_synth(root, "train", seed=tc["seed"])
    val_ds = load_synth(root, "val", seed=tc["seed"]+1)
    
    train_ld = DataLoader(train_ds, batch_size=tc["batch_size"], shuffle=True)
    val_ld = DataLoader(val_ds, batch_size=1, shuffle=False)
    
    d = train_ds.X.shape[-1]
    model = RCGNN(d, mc)
    device = tc["device"]
    model.to(device)
    
    opt = make_optimizer(model.parameters(), tc)
    
    # Try loading ground truth
    A_true = None
    try:
        A_true = np.load(os.path.join(root, "A_true.npy"))
    except:
        pass
    
    print(f"Dataset: {dataset_dir} | Train: {len(train_ds)} | Val: {len(val_ds)}")
    print(f"Features: {d} | Batch size: {tc['batch_size']} | Epochs: {tc['epochs']}")
    print(f"Device: {device}\n")
    
    # Training loop
    print("-" * 80)
    print("TRAINING")
    print("-" * 80)
    
    best_shd = 1e9
    A_best = None
    
    for ep in range(tc["epochs"]):
        out = train_epoch(model, train_ld, opt, 
                         inv_weight=mc["loss"]["invariance"]["lambda_inv"],
                         device=device)
        ev = eval_epoch(model, val_ld, A_true=A_true, thr=0.5, device=device)
        
        shd_val = ev.get('shd', '-')
        print(f"Epoch {ep:02d} | Loss: {out['loss']:10.4f} | "
              f"L_rec: {out['L_rec']:10.4f} | L_acy: {out['L_acy']:10.4f} | SHD: {shd_val}")
        
        if A_true is None or ev.get("shd", 1e9) < best_shd:
            if A_true is not None:
                best_shd = ev["shd"]
            A_best = ev["A_mean"]
            os.makedirs("artifacts/checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "artifacts/checkpoints/rcgnn_best.pt")
            os.makedirs("artifacts/adjacency", exist_ok=True)
            np.save("artifacts/adjacency/A_mean.npy", A_best)
    
    print("\n" + "-" * 80)
    print("VALIDATION")
    print("-" * 80 + "\n")
    
    # Load best adjacency
    A_pred = np.load("artifacts/adjacency/A_mean.npy")
    
    print(f"[DONE] Best adjacency learned:")
    print(f" Shape: {A_pred.shape}")
    print(f" Min: {A_pred.min():.6f}, Max: {A_pred.max():.6f}, Mean: {A_pred.mean():.6f}")
    print(f" Sparsity: {(A_pred == 0).sum() / A_pred.size * 100:.1f}% zeros")
    print(f" Non-zero edges: {(A_pred != 0).sum()}/{A_pred.size}\n")
    
    if A_true is not None:
        print(f"[DONE] Ground truth adjacency:")
        print(f" Shape: {A_true.shape}")
        print(f" Non-zero edges: {(A_true > 0).sum()}/{A_true.size}\n")
        
        metrics = compute_metrics(A_true, A_pred, threshold=0.5)
        print(f"Evaluation Metrics (threshold=0.5):")
        print(f" Precision: {metrics['precision']:.4f}")
        print(f" Recall: {metrics['recall']:.4f}")
        print(f" F1-Score: {metrics['f1']:.4f}")
        print(f" SHD: {metrics['shd']}")
    
    print("\n" + "-" * 80)
    print("VISUALIZATION")
    print("-" * 80 + "\n")
    
    # Generate visualizations
    plot_adjacency_matrices(A_true, A_pred, "artifacts/adjacency_comparison.png")
    plot_edge_distribution(A_pred, "artifacts/edge_strength_dist.png")
    plot_network_graph(A_pred, "artifacts/causal_graph_network.png", top_k=25)
    
    print("\n" + "=" * 80)
    print("[DONE] TRAINING, VALIDATION & VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"\nOutput files:")
    print(f" artifacts/adjacency_comparison.png")
    print(f" artifacts/edge_strength_dist.png")
    print(f" artifacts/causal_graph_network.png")
    print(f" artifacts/checkpoints/rcgnn_best.pt")
    print(f" artifacts/adjacency/A_mean.npy")


if __name__ == "__main__":
    main()
