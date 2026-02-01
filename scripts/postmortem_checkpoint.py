#!/usr/bin/env python3
"""
Post-mortem analysis of RC-GNN checkpoint.

Extracts key diagnostics:
1. Edge distribution at multiple thresholds
2. Top-K edges with weights
3. Cycle measure h(A) (NOTEARS acyclicity)
4. Stability analysis (if training history available)

Usage:
    python scripts/postmortem_checkpoint.py \
        --checkpoint artifacts/unified_compound_mnar_noise_bias/best_model.pt \
        --data_dir data/interim/uci_air_c/compound_mnar_noise_bias \
        --output_dir artifacts/postmortem
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import numpy as np
import torch
from typing import Dict, List, Tuple

from src.models.rcgnn import RCGNN, notears_acyclicity


def compute_edge_stats(A: np.ndarray, thresholds: List[float] = [0.1, 0.2, 0.3, 0.5, 0.7]) -> Dict:
    """Compute edge statistics at multiple thresholds."""
    d = A.shape[0]
    max_edges = d * (d - 1) # No self-loops
    
    # Zero diagonal
    A_clean = A.copy()
    np.fill_diagonal(A_clean, 0)
    
    stats = {
        "d": d,
        "max_edges": max_edges,
        "A_max": float(A_clean.max()),
        "A_min": float(A_clean[A_clean > 0].min()) if (A_clean > 0).any() else 0.0,
        "A_mean": float(A_clean.mean()),
        "A_std": float(A_clean.std()),
        "edge_sum": float(A_clean.sum()),
        "thresholds": {}
    }
    
    for t in thresholds:
        binary = (A_clean > t).astype(int)
        n_edges = binary.sum()
        density = n_edges / max_edges
        stats["thresholds"][f"{t:.1f}"] = {
            "edges": int(n_edges),
            "density": float(density),
            "ratio_vs_d": float(n_edges / d) if d > 0 else 0.0
        }
    
    return stats


def get_top_k_edges(A: np.ndarray, k: int) -> List[Dict]:
    """Get top-K edges with weights."""
    d = A.shape[0]
    A_clean = A.copy()
    np.fill_diagonal(A_clean, 0)
    
    # Flatten and get top-k indices
    flat = A_clean.flatten()
    top_indices = np.argsort(flat)[::-1][:k]
    
    edges = []
    for idx in top_indices:
        i, j = idx // d, idx % d
        weight = flat[idx]
        edges.append({
            "edge": f"{i} -> {j}",
            "from": int(i),
            "to": int(j),
            "weight": float(weight)
        })
    
    return edges


def compute_notears_acyclicity(A: np.ndarray) -> float:
    """Compute h(A) = tr(e^A) - d (NOTEARS acyclicity constraint)."""
    A_tensor = torch.tensor(A, dtype=torch.float32)
    h = notears_acyclicity(A_tensor)
    return float(h.item())


def compare_with_ground_truth(A_pred: np.ndarray, A_true: np.ndarray, k: int = None) -> Dict:
    """Compare predicted adjacency with ground truth."""
    d = A_pred.shape[0]
    if k is None:
        k = int((A_true > 0).sum())
    
    # Clean matrices
    A_pred_clean = A_pred.copy()
    A_true_clean = A_true.copy()
    np.fill_diagonal(A_pred_clean, 0)
    np.fill_diagonal(A_true_clean, 0)
    
    # Get true edges
    true_edges = set(zip(*np.where(A_true_clean > 0)))
    n_true = len(true_edges)
    
    # Get top-k predicted edges
    flat = A_pred_clean.flatten()
    top_k_indices = np.argsort(flat)[::-1][:k]
    pred_edges = set()
    for idx in top_k_indices:
        i, j = idx // d, idx % d
        pred_edges.add((i, j))
    
    # Metrics
    tp = len(true_edges & pred_edges)
    fp = len(pred_edges - true_edges)
    fn = len(true_edges - pred_edges)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # SHD (structural hamming distance)
    A_pred_binary = (A_pred_clean > 0.5).astype(int)
    A_true_binary = (A_true_clean > 0.5).astype(int)
    shd = np.sum(A_pred_binary != A_true_binary)
    
    return {
        "k": k,
        "true_edges": n_true,
        "pred_edges_at_k": len(pred_edges),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "shd_at_0.5": int(shd),
        "true_edge_list": [f"{i}->{j}" for i, j in sorted(true_edges)],
        "pred_edge_list": [f"{i}->{j}" for i, j in sorted(pred_edges)],
        "correctly_found": [f"{i}->{j}" for i, j in sorted(true_edges & pred_edges)],
        "missed": [f"{i}->{j}" for i, j in sorted(true_edges - pred_edges)],
        "spurious": [f"{i}->{j}" for i, j in sorted(pred_edges - true_edges)]
    }


def analyze_training_history(history_path: Path) -> Dict:
    """Analyze training history for stability."""
    with open(history_path) as f:
        history = json.load(f)
    
    # Extract TopK-F1 trajectory
    topk_f1s = []
    epochs_with_best = []
    
    for entry in history:
        if "val" in entry and "topk_f1" in entry["val"]:
            topk_f1s.append(entry["val"]["topk_f1"])
        if "best_metric" in entry:
            epochs_with_best.append(entry["epoch"])
    
    return {
        "n_epochs": len(history),
        "topk_f1_trajectory": topk_f1s,
        "topk_f1_max": max(topk_f1s) if topk_f1s else 0,
        "topk_f1_std": float(np.std(topk_f1s)) if topk_f1s else 0,
        "epochs_with_improvement": epochs_with_best,
        "final_topk_f1": topk_f1s[-1] if topk_f1s else 0
    }


def main():
    parser = argparse.ArgumentParser(description="Post-mortem checkpoint analysis")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--adjacency", type=str, default=None, help="Path to A_best.npy (if separate)")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data directory")
    parser.add_argument("--output_dir", type=str, default="artifacts/postmortem", help="Output directory")
    parser.add_argument("--k", type=int, default=None, help="Top-K edges (default: true edge count)")
    args = parser.parse_args()
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("RC-GNN POST-MORTEM ANALYSIS")
    print("=" * 70)
    
    # Load data
    data_dir = Path(args.data_dir)
    A_true = np.load(data_dir / "A_true.npy")
    n_true_edges = int((A_true > 0).sum())
    d = A_true.shape[0]
    
    print(f"\nData: {data_dir}")
    print(f" d = {d}")
    print(f" True edges = {n_true_edges}")
    
    # Load predicted adjacency
    if args.adjacency:
        A_pred = np.load(args.adjacency)
        print(f"\nLoaded adjacency from: {args.adjacency}")
    else:
        # Try to find A_best.npy in same directory as checkpoint
        ckpt_dir = Path(args.checkpoint).parent
        if (ckpt_dir / "A_best.npy").exists():
            A_pred = np.load(ckpt_dir / "A_best.npy")
            print(f"\nLoaded adjacency from: {ckpt_dir / 'A_best.npy'}")
        else:
            # Load from checkpoint model
            print(f"\nLoading model from checkpoint to extract adjacency...")
            ckpt = torch.load(args.checkpoint, map_location="cpu")
            
            # Reconstruct model
            config = ckpt.get("config", {})
            model = RCGNN(
                d=d,
                latent_dim=config.get("latent_dim", 32),
                hidden_dim=config.get("hidden_dim", 64),
                n_regimes=config.get("n_regimes", 1),
            )
            model.load_state_dict(ckpt["model_state"])
            model.eval()
            
            A_pred = model.graph_learner.get_mean_adjacency().detach().numpy()
    
    k = args.k if args.k else n_true_edges
    
    # 1. Edge statistics
    print("\n" + "=" * 70)
    print("1. EDGE DISTRIBUTION")
    print("=" * 70)
    edge_stats = compute_edge_stats(A_pred)
    print(f" A_max: {edge_stats['A_max']:.4f}")
    print(f" A_mean: {edge_stats['A_mean']:.4f}")
    print(f" edge_sum: {edge_stats['edge_sum']:.2f}")
    print(f"\n Edges at thresholds:")
    for t, info in edge_stats["thresholds"].items():
        status = "[WARN] DENSE" if info["edges"] > 3 * n_true_edges else "[OK]"
        print(f" @{t}: {info['edges']:3d} edges ({info['density']*100:.1f}% density) {status}")
    
    # 2. Top-K edges
    print("\n" + "=" * 70)
    print(f"2. TOP-{k} EDGES (target_edges={n_true_edges})")
    print("=" * 70)
    top_edges = get_top_k_edges(A_pred, k)
    for i, e in enumerate(top_edges):
        print(f" {i+1:2d}. {e['edge']:8s} | weight={e['weight']:.4f}")
    
    # 3. Acyclicity
    print("\n" + "=" * 70)
    print("3. ACYCLICITY MEASURE")
    print("=" * 70)
    # Compute on soft A and binary A
    h_soft = compute_notears_acyclicity(A_pred)
    A_binary = (A_pred > 0.5).astype(float)
    h_binary = compute_notears_acyclicity(A_binary)
    
    print(f" h(A_soft): {h_soft:.4f}")
    print(f" h(A_binary): {h_binary:.4f}")
    if h_soft < 0.1:
        print(" [OK] Near-acyclic (good)")
    elif h_soft < 1.0:
        print(" [WARN] Some cycles present")
    else:
        print(" [FAIL] Significant cycles (h > 1)")
    
    # 4. Ground truth comparison
    print("\n" + "=" * 70)
    print("4. COMPARISON WITH GROUND TRUTH")
    print("=" * 70)
    comparison = compare_with_ground_truth(A_pred, A_true, k)
    print(f" TopK-F1 (K={k}): {comparison['f1']:.4f}")
    print(f" Precision: {comparison['precision']:.4f}")
    print(f" Recall: {comparison['recall']:.4f}")
    print(f" SHD (at 0.5): {comparison['shd_at_0.5']}")
    print(f"\n True edges found ({comparison['tp']}/{comparison['true_edges']}):")
    for e in comparison["correctly_found"]:
        print(f" [OK] {e}")
    print(f"\n Missed edges ({comparison['fn']}):")
    for e in comparison["missed"]:
        print(f" [X] {e}")
    print(f"\n Spurious edges ({comparison['fp']}):")
    for e in comparison["spurious"][:10]: # Limit output
        print(f" ? {e}")
    if len(comparison["spurious"]) > 10:
        print(f" ... and {len(comparison['spurious']) - 10} more")
    
    # 5. Training history (if available)
    ckpt_dir = Path(args.checkpoint).parent
    history_path = ckpt_dir / "training_history.json"
    if history_path.exists():
        print("\n" + "=" * 70)
        print("5. TRAINING HISTORY ANALYSIS")
        print("=" * 70)
        hist_analysis = analyze_training_history(history_path)
        print(f" Total epochs: {hist_analysis['n_epochs']}")
        print(f" Best TopK-F1: {hist_analysis['topk_f1_max']:.4f}")
        print(f" Final TopK-F1: {hist_analysis['final_topk_f1']:.4f}")
        print(f" TopK-F1 std: {hist_analysis['topk_f1_std']:.4f}")
        print(f" Improvement epochs: {hist_analysis['epochs_with_improvement']}")
    
    # Save results
    results = {
        "data_dir": str(data_dir),
        "checkpoint": str(args.checkpoint),
        "d": d,
        "true_edges": n_true_edges,
        "edge_stats": edge_stats,
        "top_k_edges": top_edges,
        "acyclicity": {
            "h_soft": h_soft,
            "h_binary": h_binary
        },
        "comparison": comparison
    }
    
    if history_path.exists():
        results["training_history"] = hist_analysis
    
    output_file = output_path / "postmortem_report.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 70)
    print(f"Report saved to: {output_file}")
    print("=" * 70)
    
    # Save adjacency matrix as well
    np.save(output_path / "A_analyzed.npy", A_pred)
    

if __name__ == "__main__":
    main()
