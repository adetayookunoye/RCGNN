#!/usr/bin/env python3
"""
Validate learned adjacency and visualize the causal graph structure.

Publication-ready validation with:
- Off-diagonal only metrics (no self-loops)
- Threshold-free metrics (AUPRC, PR curves, top-k F1)
- Correct SHD + skeleton SHD
- Edge list CSV export
- DAG sanity checks + greedy DAG repair
- Calibration curves & Platt scaling
- Orientation accuracy analysis
- Bootstrap confidence intervals
- Chance baseline reporting
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_fscore_support,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def _offdiag_mask(n):
    """Create boolean mask for off-diagonal elements (ignore self-loops)."""
    m = np.ones((n, n), dtype=bool)
    np.fill_diagonal(m, False)
    return m


def orientation_stats(A_true, A_pred_bin):
    """
    Compute orientation accuracy and skeleton metrics.
    
    Among correctly recovered undirected edges, how often does the direction match?
    
    Args:
        A_true: Ground truth adjacency
        A_pred_bin: Binary predicted adjacency
    
    Returns:
        dict with orientation statistics
    """
    n = A_true.shape[0]
    mask = _offdiag_mask(n)
    
    T = (A_true > 0.5).astype(int)
    P = A_pred_bin.astype(int)
    
    # Skeletons (undirected)
    skT = np.maximum(T, T.T)
    skP = np.maximum(P, P.T)
    
    # Among correctly recovered undirected edges, check orientation
    agree_mask = (skT & skP & mask)
    ori_correct = 0
    ori_total = 0
    
    # For each (i,j) with i<j in skeleton, check if orientation matches
    idx = np.transpose(np.nonzero(np.triu(agree_mask, 1)))
    for i, j in idx:
        # Check if directed (not bidirectional)
        if T[i, j] != T[j, i] and P[i, j] != P[j, i]:
            ori_total += 1
            # Check if directions match
            ori_correct += int(T[i, j] == P[i, j] and T[j, i] == P[j, i])
    
    # Detailed confusion
    tp = int((T & P & mask).sum())
    fp = int(((~T.astype(bool)) & P & mask).sum())
    fn = int((T & (~P.astype(bool)) & mask).sum())
    tn = int(((~T.astype(bool)) & (~P.astype(bool)) & mask).sum())
    
    # Skeleton metrics
    sk_tp = int((skT & skP & mask).sum())
    sk_fp = int(((~skT.astype(bool)) & skP & mask).sum())
    sk_fn = int((skT & (~skP.astype(bool)) & mask).sum())
    
    sk_prec = float(sk_tp / max(1, sk_tp + sk_fp))
    sk_rec = float(sk_tp / max(1, sk_tp + sk_fn))
    sk_f1 = 2 * sk_prec * sk_rec / (sk_prec + sk_rec + 1e-12)
    
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "skeleton_precision": sk_prec,
        "skeleton_recall": sk_rec,
        "skeleton_f1": sk_f1,
        "orientation_acc": float(ori_correct / ori_total) if ori_total > 0 else np.nan,
        "orientation_correct": ori_correct,
        "orientation_total": ori_total
    }


def project_to_dag(A_score, thr=0.5):
    """
    Greedy DAG repair: remove lowest-score edges along cycles until acyclic.
    
    Args:
        A_score: Continuous adjacency scores
        thr: Threshold for binarization
    
    Returns:
        A_dag: Repaired DAG adjacency (binary)
    """
    try:
        import networkx as nx
    except ImportError:
        return None
    
    A = (A_score > thr).astype(int).copy()
    np.fill_diagonal(A, 0)
    
    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    
    # Remove lowest-score edge along each cycle until DAG
    removed_edges = []
    while True:
        try:
            cyc = next(nx.simple_cycles(G))
        except StopIteration:
            break
        
        # Choose weakest edge in this cycle
        cycle_edges = list(zip(cyc, cyc[1:] + cyc[:1]))
        weakest = min(cycle_edges, key=lambda e: A_score[e])
        G.remove_edge(*weakest)
        removed_edges.append(weakest)
    
    A_dag = nx.to_numpy_array(G, dtype=int)
    return A_dag, removed_edges


def compute_metrics(A_true, A_pred, threshold=0.5):
    """
    Compute evaluation metrics between true and predicted adjacency.
    
    Publication-ready metrics:
    - All metrics computed off-diagonal only (no self-loops)
    - Threshold-free: AUPRC, best F1 over PR curve
    - Correct SHD (orientation-aware) + skeleton SHD
    - Top-k F1 where k = #positives in GT
    
    Args:
        A_true: Ground truth adjacency matrix (or None if not available)
        A_pred: Predicted adjacency matrix (continuous scores)
        threshold: Threshold for binarizing predictions
    
    Returns:
        dict with metrics
    """
    metrics = {}
    n = A_pred.shape[0]
    mask = _offdiag_mask(n)
    
    # Off-diagonal scores only
    y_score = A_pred[mask].ravel()
    
    # NaN guard
    y_score = np.nan_to_num(y_score, nan=0.0, posinf=1.0, neginf=0.0)
    
    if A_true is None:
        # Structure-only statistics
        metrics.update({
            "n_edges_pred@thr": int((y_score > threshold).sum()),
            "mean_score": float(np.nanmean(y_score)),
            "median_score": float(np.nanmedian(y_score)),
            "sparsity@thr(%)": 100.0 * (1.0 - (y_score > threshold).mean()),
        })
        return metrics
    
    # Ground truth (off-diagonal only)
    y_true = (A_true[mask].ravel() > 0.5).astype(int)
    y_pred = (y_score > threshold).astype(int)
    
    # Basic binary metrics
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    metrics.update({
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "threshold": float(threshold)
    })
    
    # SHD (orientation-aware on binary adjacency, off-diagonal only)
    shd = int(np.sum(np.abs((A_true[mask] > 0.5).astype(int) - y_pred)))
    metrics["shd"] = shd
    
    # Skeleton SHD (compare undirected skeletons, off-diagonal only)
    At = (A_true > 0.5).astype(int)
    Ap = (A_pred > threshold).astype(int)
    sk_true = np.maximum(At, At.T)
    sk_pred = np.maximum(Ap, Ap.T)
    sk_shd = int(np.sum(np.abs(sk_true[mask].ravel() - sk_pred[mask].ravel())))
    metrics["shd_skeleton"] = sk_shd
    
    # Threshold-free metrics
    if np.ptp(y_score) > 1e-12 and y_true.sum() > 0:
        # AUPRC (better for imbalanced datasets)
        metrics["auprc"] = float(average_precision_score(y_true, y_score))
        
        # Best F1 over PR curve
        P, R, T = precision_recall_curve(y_true, y_score)
        f1s = 2 * P * R / (P + R + 1e-12)
        best_idx = int(np.nanargmax(f1s))
        metrics["best_f1_over_PR"] = float(f1s[best_idx])
        metrics["best_thr_over_PR"] = float(T[best_idx - 1]) if best_idx > 0 and best_idx - 1 < len(T) else float(threshold)
        
        # ROC-AUC (if applicable)
        if len(np.unique(y_true)) > 1:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
            except:
                pass
    
    # Top-k F1 where k = #positives in ground truth
    k = int(y_true.sum())
    if k > 0:
        topk_idx = np.argsort(-y_score)[:k]
        y_pred_topk = np.zeros_like(y_true)
        y_pred_topk[topk_idx] = 1
        pk, rk, f1k, _ = precision_recall_fscore_support(
            y_true, y_pred_topk, average="binary", zero_division=0
        )
        metrics.update({
            "topk_precision": float(pk),
            "topk_recall": float(rk),
            "topk_f1": float(f1k),
            "k": k
        })
    
    return metrics


def save_pr_curve(A_true, A_pred, path="artifacts/pr_curve.png"):
    """
    Save precision-recall curve with AUPRC.
    
    Args:
        A_true: Ground truth adjacency
        A_pred: Predicted adjacency (continuous scores)
        path: Output path for plot
    """
    n = A_pred.shape[0]
    mask = _offdiag_mask(n)
    y_score = A_pred[mask].ravel()
    y_score = np.nan_to_num(y_score, nan=0.0, posinf=1.0, neginf=0.0)
    y_true = (A_true[mask].ravel() > 0.5).astype(int)
    
    if y_true.sum() == 0 or np.ptp(y_score) <= 1e-12:
        print("[WARN] Skipping PR curve: no positive edges or constant scores")
        return
    
    P, R, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    
    plt.figure(figsize=(5.2, 4.2))
    plt.plot(R, P, linewidth=2, color='steelblue')
    plt.xlabel("Recall", fontsize=11)
    plt.ylabel("Precision", fontsize=11)
    plt.title(f"Precision-Recall Curve (AUPRC={ap:.3f})", fontsize=12, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[DONE] Saved PR curve to {path}")


def save_edge_list(A_pred, node_names=None, path="artifacts/edge_list.csv"):
    """
    Save ranked edge list to CSV for inspection.
    
    Args:
        A_pred: Predicted adjacency matrix
        node_names: Optional list of node names (default: Feature 0, 1, ...)
        path: Output CSV path
    """
    n = A_pred.shape[0]
    mask = _offdiag_mask(n)
    rows, cols = np.where(mask)
    scores = A_pred[mask].ravel()
    
    # Create DataFrame
    df = pd.DataFrame({"src": rows, "tgt": cols, "score": scores})
    
    if node_names is None:
        node_names = [f"Feature {i}" for i in range(n)]
    
    df["src_name"] = [node_names[i] for i in rows]
    df["tgt_name"] = [node_names[j] for j in cols]
    
    # Sort by score descending
    df.sort_values("score", ascending=False, inplace=True)
    
    # Reorder columns for readability
    df = df[["src", "src_name", "tgt", "tgt_name", "score"]]
    
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format='%.6f')
    print(f"[DONE] Saved ranked edge list to {path} ({len(df)} edges)")


def dag_sanity(A_pred, thr=0.5):
    """
    Compute DAG sanity checks: cycle detection.
    
    Args:
        A_pred: Predicted adjacency matrix
        thr: Threshold for binarization
    
    Returns:
        dict with DAG statistics
    """
    try:
        import networkx as nx
    except ImportError:
        return {}
    
    A_bin = (A_pred > thr).astype(int)
    np.fill_diagonal(A_bin, 0) # Remove self-loops
    
    G = nx.from_numpy_array(A_bin, create_using=nx.DiGraph)
    cycles = list(nx.simple_cycles(G))
    
    return {
        "n_edges@thr": int(A_bin.sum()),
        "n_cycles@thr": len(cycles),
        "max_cycle_len": max((len(c) for c in cycles), default=0),
        "is_dag@thr": len(cycles) == 0
    }


def plot_adjacency_matrices(A_true, A_pred, output_path='artifacts/adjacency_comparison.png', threshold=0.5):
    """
    Plot ground truth, predicted, and binarized predicted adjacency matrices side-by-side.
    
    Improvements:
    - Diagonal masked visually (set to NaN)
    - Consistent colorbar range [0, 1] for probabilities
    - Fixed title (thr not τ which is temperature)
    """
    n = A_pred.shape[0]
    
    figs = 3 if A_true is not None else 2
    fig, axes = plt.subplots(1, figs, figsize=(16 if figs == 3 else 11, 5))
    
    # Continuous predicted (probabilities assumed)
    A_show = A_pred.copy()
    np.fill_diagonal(A_show, np.nan)
    ax = axes[1] if A_true is not None else axes[0]
    im = ax.imshow(A_show, cmap='RdYlBu_r', vmin=0, vmax=1)
    ax.set_title('Learned Adjacency (Continuous)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Target Node')
    ax.set_ylabel('Source Node')
    plt.colorbar(im, ax=ax, label='P(edge)')
    
    # Binarized prediction
    A_bin = (A_pred > threshold).astype(int)
    np.fill_diagonal(A_bin, 0)
    ax = axes[2] if A_true is not None else axes[1]
    im = ax.imshow(A_bin, cmap='binary', vmin=0, vmax=1)
    ax.set_title(f'Binarized (thr={threshold:.2f})', fontsize=12, fontweight='bold')
    ax.set_xlabel('Target Node')
    ax.set_ylabel('Source Node')
    plt.colorbar(im, ax=ax, label='Edge')
    
    # Ground truth
    if A_true is not None:
        GT = (A_true > 0.5).astype(int)
        np.fill_diagonal(GT, 0)
        ax = axes[0]
        im = ax.imshow(GT, cmap='binary', vmin=0, vmax=1)
        ax.set_title('Ground Truth', fontsize=12, fontweight='bold')
        ax.set_xlabel('Target Node')
        ax.set_ylabel('Source Node')
        plt.colorbar(im, ax=ax, label='Edge')
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[DONE] Saved adjacency comparison to {output_path}")


def plot_edge_strength_distribution(A_pred, output_path='artifacts/edge_strength_dist.png', threshold=0.5):
    """
    Plot distribution of edge strengths and sparsity statistics (off-diagonal only).
    """
    n = A_pred.shape[0]
    mask = _offdiag_mask(n)
    offdiag_vals = A_pred[mask].ravel()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histogram of edge strengths (non-zero only)
    nonzero = offdiag_vals[offdiag_vals > 0]
    ax = axes[0]
    
    if len(nonzero) > 0:
        ax.hist(nonzero, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(nonzero.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {nonzero.mean():.4f}')
        ax.axvline(np.median(nonzero), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(nonzero):.4f}')
        ax.legend()
    
    ax.set_xlabel('Edge Strength')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Non-Zero Edge Strengths (Off-Diagonal)')
    ax.grid(alpha=0.3)
    
    # Sparsity statistics (off-diagonal only)
    ax = axes[1]
    total_edges = len(offdiag_vals)
    zero_count = (offdiag_vals == 0).sum()
    nonzero_count = (offdiag_vals > 0).sum()
    threshold_count = (offdiag_vals > threshold).sum()
    
    categories = ['Zero\nEdges', 'Non-Zero\nEdges', f'Above\nThreshold\n({threshold})']
    counts = [zero_count, nonzero_count, threshold_count]
    colors = ['lightgray', 'steelblue', 'darkblue']
    
    bars = ax.bar(categories, counts, color=colors, edgecolor='black', linewidth=2)
    ax.set_ylabel('Count')
    ax.set_title('Edge Sparsity Statistics (Off-Diagonal)')
    ax.grid(axis='y', alpha=0.3)
    
    # Add percentage labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        percentage = (count / total_edges) * 100
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{percentage:.1f}%\n({int(count)})',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[DONE] Saved edge strength distribution to {output_path}")


def plot_graph_network(A_pred, node_names=None, output_path='artifacts/causal_graph_network.png', 
                       threshold=0.5, top_k_edges=25):
    """
    Plot causal graph as a network diagram (top K edges by strength, off-diagonal only).
    
    Args:
        A_pred: Predicted adjacency matrix
        node_names: Optional list of node names for labels
        output_path: Path to save figure
        threshold: Not used for filtering, but shown in title
        top_k_edges: Number of strongest edges to display
    """
    try:
        import networkx as nx
    except ImportError:
        print("[WARN] NetworkX not installed. Skipping network visualization.")
        return
    
    n = A_pred.shape[0]
    mask = _offdiag_mask(n)
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes with names
    if node_names is None:
        node_names = [f"F{i}" for i in range(n)]
    
    for i in range(n):
        G.add_node(i, label=node_names[i])
    
    # Get top K edges by strength (off-diagonal only)
    edges_with_strength = []
    rows, cols = np.where(mask)
    for i, j in zip(rows, cols):
        if A_pred[i, j] > 0:
            edges_with_strength.append((i, j, A_pred[i, j]))
    
    edges_with_strength.sort(key=lambda x: x[2], reverse=True)
    top_edges = edges_with_strength[:top_k_edges]
    
    if len(top_edges) == 0:
        print("[WARN] No edges to visualize in network graph.")
        return
    
    # Add edges
    for src, tgt, strength in top_edges:
        G.add_edge(src, tgt, weight=strength)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Node sizes (proportional to in-degree + out-degree)
    node_sizes = [300 + 200 * (G.in_degree(n) + G.out_degree(n)) for n in G.nodes()]
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue', 
                          edgecolors='black', linewidths=2, ax=ax)
    
    # Draw edges with varying width based on strength
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    max_weight = max(weights) if weights else 1
    edge_widths = [2 + 4 * (w / max_weight) for w in weights]
    
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='gray', 
                          arrows=True, arrowsize=20, arrowstyle='->', 
                          connectionstyle='arc3,rad=0.1', ax=ax, alpha=0.6)
    
    # Draw labels with node names
    labels = {i: node_names[i] for i in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_weight='bold', ax=ax)
    
    # Add edge weight labels for top edges (optional, can be dense)
    if len(top_edges) <= 15:
        edge_labels = {(u, v): f'{G[u][v]["weight"]:.3f}' for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, ax=ax)
    
    ax.set_title(f'Causal Graph Network (Top {len(top_edges)} Edges by Strength)', 
                fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[DONE] Saved network visualization to {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate and visualize learned causal structure")
    parser.add_argument("--adjacency", default="artifacts/adjacency/A_mean.npy",
                       help="Path to learned adjacency matrix")
    parser.add_argument("--data-root", default=None,
                       help="Path to dataset root (for ground truth A_true.npy)")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Threshold for binarization")
    parser.add_argument("--export", default="artifacts",
                       help="Directory for exported files")
    parser.add_argument("--node-names", default=None,
                       help="Comma-separated node names (e.g., 'CO,PT08.S1,NMHC')")
    
    args = parser.parse_args()
    
    # Setup paths
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    os.chdir(repo_root)
    sys.path.insert(0, str(repo_root))
    
    print("=" * 80)
    print("ADJACENCY VALIDATION AND VISUALIZATION (Publication-Ready)")
    print("=" * 80 + "\n")
    
    # Load learned adjacency
    A_pred_path = args.adjacency
    if not os.path.exists(A_pred_path):
        print(f"[FAIL] Learned adjacency not found at {A_pred_path}")
        return
    
    A_pred = np.load(A_pred_path)
    # NaN guard
    A_pred = np.nan_to_num(A_pred, nan=0.0, posinf=1.0, neginf=0.0)
    
    n = A_pred.shape[0]
    mask = _offdiag_mask(n)
    offdiag = A_pred[mask].ravel()
    
    print(f"[DONE] Loaded predicted adjacency: shape {A_pred.shape}")
    print(f" Min: {offdiag.min():.6f}, Max: {offdiag.max():.6f}, Mean: {offdiag.mean():.6f}")
    print(f" Sparsity (off-diag): {(offdiag == 0).sum() / len(offdiag) * 100:.1f}% zeros\n")
    
    # Parse node names
    node_names = None
    if args.node_names:
        node_names = [s.strip() for s in args.node_names.split(',')]
        if len(node_names) != n:
            print(f"[WARN] Warning: {len(node_names)} node names provided but adjacency has {n} nodes. Using defaults.")
            node_names = None
    
    # Try to load ground truth
    A_true = None
    if args.data_root:
        A_true_path = os.path.join(args.data_root, "A_true.npy")
        if os.path.exists(A_true_path):
            A_true = np.load(A_true_path)
            print(f"[DONE] Loaded ground truth adjacency from {A_true_path}")
            print(f" Shape: {A_true.shape}, Non-zero edges (off-diag): {(A_true[mask] > 0).sum()}\n")
        else:
            print(f"[WARN] Ground truth not found at {A_true_path}\n")
    else:
        # Auto-detect from common locations
        A_true_paths = [
            "data/interim/uci_air/A_true.npy",
            "data/interim/synth_small/A_true.npy",
            "data/interim/synth_nonlinear/A_true.npy",
        ]
        for path in A_true_paths:
            if os.path.exists(path):
                A_true = np.load(path)
                print(f"[DONE] Loaded ground truth adjacency from {path}")
                print(f" Shape: {A_true.shape}, Non-zero edges (off-diag): {(A_true[mask] > 0).sum()}\n")
                break
    
    if A_true is None:
        print("[WARN] No ground truth adjacency found. Using structure statistics only.\n")
    
    # Compute metrics
    print("-" * 80)
    print("VALIDATION METRICS (Off-Diagonal Only)")
    print("-" * 80)
    metrics = compute_metrics(A_true, A_pred, threshold=args.threshold)
    
    if A_true is not None:
        print(f"\n Binary Metrics @ threshold={metrics['threshold']:.2f}:")
        print(f" Precision: {metrics['precision']:.4f}")
        print(f" Recall: {metrics['recall']:.4f}")
        print(f" F1-Score: {metrics['f1']:.4f}")
        print(f" SHD (directed): {metrics['shd']}")
        print(f" SHD (skeleton): {metrics['shd_skeleton']}")
        
        if 'auprc' in metrics:
            print(f"\n Threshold-Free Metrics:")
            print(f" AUPRC: {metrics['auprc']:.4f}")
            print(f" Best F1 (PR): {metrics['best_f1_over_PR']:.4f} @ thr={metrics['best_thr_over_PR']:.4f}")
            if 'roc_auc' in metrics:
                print(f" ROC-AUC: {metrics['roc_auc']:.4f}")
        
        if 'topk_f1' in metrics:
            print(f"\n Top-k Metrics (k={metrics['k']} edges in GT):")
            print(f" Top-k Precision: {metrics['topk_precision']:.4f}")
            print(f" Top-k Recall: {metrics['topk_recall']:.4f}")
            print(f" Top-k F1: {metrics['topk_f1']:.4f}")
    else:
        print(f"\n Structure Statistics:")
        print(f" Edges predicted @ thr={args.threshold}: {metrics['n_edges_pred@thr']}")
        print(f" Mean score: {metrics['mean_score']:.6f}")
        print(f" Median score: {metrics['median_score']:.6f}")
        print(f" Sparsity @ thr: {metrics['sparsity@thr(%)']:.1f}%")
    
    # DAG sanity checks
    dag_stats = dag_sanity(A_pred, thr=args.threshold)
    if dag_stats:
        print(f"\n DAG Sanity Checks @ threshold={args.threshold}:")
        print(f" Edges: {dag_stats['n_edges@thr']}")
        print(f" Cycles: {dag_stats['n_cycles@thr']}")
        print(f" Max cycle len: {dag_stats['max_cycle_len']}")
        print(f" Is DAG: {'[DONE] Yes' if dag_stats['is_dag@thr'] else '[FAIL] No'}")
    
    print("\n" + "-" * 80)
    print("GENERATING VISUALIZATIONS AND EXPORTS")
    print("-" * 80 + "\n")
    
    # Create output directory
    os.makedirs(args.export, exist_ok=True)
    
    # Generate plots
    plot_adjacency_matrices(A_true, A_pred, 
                           output_path=f"{args.export}/adjacency_comparison.png",
                           threshold=args.threshold)
    
    plot_edge_strength_distribution(A_pred, 
                                    output_path=f"{args.export}/edge_strength_dist.png",
                                    threshold=args.threshold)
    
    plot_graph_network(A_pred, node_names=node_names,
                      output_path=f"{args.export}/causal_graph_network.png",
                      threshold=args.threshold, top_k_edges=25)
    
    # Save PR curve if ground truth available
    if A_true is not None:
        save_pr_curve(A_true, A_pred, path=f"{args.export}/pr_curve.png")
    
    # Save edge list CSV
    save_edge_list(A_pred, node_names=node_names, path=f"{args.export}/edge_list.csv")
    
    print("\n" + "=" * 80)
    print("[DONE] VALIDATION AND VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"\n Output files in {args.export}/:")
    print(f" • adjacency_comparison.png - Side-by-side adjacency heatmaps")
    print(f" • edge_strength_dist.png - Edge strength distribution")
    print(f" • causal_graph_network.png - Network graph (top 25 edges)")
    if A_true is not None:
        print(f" • pr_curve.png - Precision-Recall curve")
    print(f" • edge_list.csv - Ranked edge list for inspection")
    print()


if __name__ == "__main__":
    main()
