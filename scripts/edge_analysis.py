#!/usr/bin/env python3
"""
EDGE PREDICTION ANALYSIS
========================
Deep dive into why the model predicts certain edges:
1. Compare predicted edges vs true edges
2. Compute actual data correlations
3. Check if model is learning correlation instead of causation

Usage:
    python scripts/edge_analysis.py --data data/interim/uci_air_c/compound_mnar_bias
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
from pathlib import Path
from scipy import stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--adjacency", type=str, default=None)
    args = parser.parse_args()
    
    data_path = Path(args.data)
    
    # Load data
    X = np.load(data_path / "X.npy")
    A_true = np.load(data_path / "A_true.npy")
    
    if X.ndim == 3:
        N, T, d = X.shape
        X_flat = X.reshape(N * T, d)
    else:
        X_flat = X
        d = X.shape[1]
    
    print("=" * 70)
    print("EDGE PREDICTION ANALYSIS")
    print("=" * 70)
    print(f"Dataset: {data_path}")
    print(f"X shape: {X.shape}")
    print(f"d (variables): {d}")
    print()
    
    # Compute pairwise correlations
    print("Computing pairwise Pearson correlations...")
    corr_matrix = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            valid = ~(np.isnan(X_flat[:, i]) | np.isnan(X_flat[:, j]))
            if valid.sum() < 10:
                continue
            r, _ = stats.pearsonr(X_flat[valid, i], X_flat[valid, j])
            corr_matrix[i, j] = abs(r) if not np.isnan(r) else 0
    
    print()
    
    # Load predicted adjacency
    if args.adjacency:
        A_pred = np.load(args.adjacency)
    else:
        # Try to find
        paths = [
            data_path.parent / "artifacts" / f"unified_v8_{data_path.name}" / "A_best.npy",
            Path(f"artifacts/unified_v8_{data_path.name}/A_best.npy"),
        ]
        A_pred = None
        for p in paths:
            if p.exists():
                A_pred = np.load(p)
                break
        if A_pred is None:
            print("No adjacency found, using correlation as proxy")
            A_pred = corr_matrix
    
    np.fill_diagonal(A_pred, 0)
    np.fill_diagonal(corr_matrix, 0)
    
    # True edges
    true_edges = set(zip(*np.where(A_true > 0)))
    n_true = len(true_edges)
    
    print("=" * 70)
    print("GROUND TRUTH EDGES")
    print("=" * 70)
    print(f"{'Edge':<12} {'Corr |r|':<12} {'A_pred':<12} {'A_pred_rev':<12}")
    print("-" * 50)
    
    for (i, j) in sorted(true_edges):
        corr = corr_matrix[i, j]
        pred = A_pred[i, j]
        pred_rev = A_pred[j, i]
        direction_ok = "✓" if pred > pred_rev else "⇄"
        print(f"{i:>2} -> {j:<2}     {corr:>8.4f}     {pred:>8.4f}     {pred_rev:>8.4f}  {direction_ok}")
    
    print()
    
    # Top predicted edges
    print("=" * 70)
    print("MODEL'S TOP PREDICTED EDGES (vs Ground Truth)")
    print("=" * 70)
    
    flat = A_pred.flatten()
    top_idx = np.argsort(flat)[::-1][:n_true * 2]
    
    print(f"{'Rank':<5} {'Edge':<8} {'A_pred':<10} {'Corr':<10} {'Status':<15} {'Analysis'}")
    print("-" * 80)
    
    for rank, idx in enumerate(top_idx[:25], 1):
        i, j = idx // d, idx % d
        pred_w = flat[idx]
        corr_w = corr_matrix[i, j]
        
        if (i, j) in true_edges:
            status = "✓ CORRECT"
            analysis = ""
        elif (j, i) in true_edges:
            status = "⇄ REVERSED"
            analysis = f"Should be {j}->{i}"
        else:
            status = "✗ WRONG"
            # Check if this edge is due to confounding
            # i.e., is there a common ancestor?
            common_parents = []
            for k in range(d):
                if (k, i) in true_edges and (k, j) in true_edges:
                    common_parents.append(k)
            if common_parents:
                analysis = f"Confounded by node(s) {common_parents}"
            elif corr_w > 0.15:
                analysis = f"High corr but not causal"
            else:
                analysis = "Spurious?"
        
        print(f"{rank:<5} {i}->{j:<4} {pred_w:<10.4f} {corr_w:<10.4f} {status:<15} {analysis}")
    
    print()
    
    # Analyze specific wrong edges
    print("=" * 70)
    print("DETAILED ANALYSIS OF WRONG EDGES")
    print("=" * 70)
    
    wrong_edges = []
    for rank, idx in enumerate(top_idx[:n_true], 1):
        i, j = idx // d, idx % d
        if (i, j) not in true_edges and (j, i) not in true_edges:
            wrong_edges.append((i, j, flat[idx]))
    
    for (i, j, w) in wrong_edges[:5]:
        print(f"\nWRONG EDGE: {i} -> {j} (weight={w:.4f})")
        print(f"  Correlation: {corr_matrix[i, j]:.4f}")
        
        # Find paths through true edges
        print(f"  True edge paths involving {i} and {j}:")
        
        # Direct connections of i
        i_parents = [k for k in range(d) if (k, i) in true_edges]
        i_children = [k for k in range(d) if (i, k) in true_edges]
        
        # Direct connections of j
        j_parents = [k for k in range(d) if (k, j) in true_edges]
        j_children = [k for k in range(d) if (j, k) in true_edges]
        
        print(f"    Node {i}: parents={i_parents}, children={i_children}")
        print(f"    Node {j}: parents={j_parents}, children={j_children}")
        
        # Check for confounding (common parent)
        common_parents = set(i_parents) & set(j_parents)
        if common_parents:
            print(f"    ⚠️ CONFOUNDED: Common parents = {common_parents}")
            print(f"       This edge is due to confounding, not causation!")
        
        # Check for mediation (i->k->j or j->k->i)
        for k in i_children:
            if k in j_parents:
                print(f"    📍 MEDIATION: {i} -> {k} -> {j}")
        for k in j_children:
            if k in i_parents:
                print(f"    📍 MEDIATION: {j} -> {k} -> {i}")
    
    print()
    
    # Correlation vs Causation analysis
    print("=" * 70)
    print("CORRELATION vs CAUSATION ANALYSIS")
    print("=" * 70)
    
    # Top correlated pairs
    corr_flat = corr_matrix.flatten()
    top_corr_idx = np.argsort(corr_flat)[::-1][:n_true]
    
    corr_edges = set()
    for idx in top_corr_idx:
        i, j = idx // d, idx % d
        corr_edges.add((min(i,j), max(i,j)))  # Undirected
    
    true_skel = set((min(i,j), max(i,j)) for i,j in true_edges)
    
    print(f"Top {n_true} correlated pairs (undirected):")
    overlap = corr_edges & true_skel
    print(f"  Overlap with true skeleton: {len(overlap)}/{len(true_skel)}")
    print(f"  True edges in top correlations: {overlap}")
    print()
    
    only_corr = corr_edges - true_skel
    print(f"  High-correlation NON-CAUSAL edges:")
    for (i, j) in sorted(only_corr):
        c = max(corr_matrix[i,j], corr_matrix[j,i])
        # Check if confounded
        common_parents = []
        for k in range(d):
            if (k, i) in true_edges and (k, j) in true_edges:
                common_parents.append(k)
        reason = f"confounded by {common_parents}" if common_parents else "indirect path?"
        print(f"    {i}-{j}: corr={c:.4f} ({reason})")
    
    print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 70)
    
    # Check if model is just learning correlation
    pred_flat = A_pred.flatten()
    corr_with_pred = np.corrcoef(pred_flat, corr_flat)[0, 1]
    print(f"Correlation between A_pred and |r| matrix: {corr_with_pred:.4f}")
    
    if corr_with_pred > 0.7:
        print("⚠️ Model outputs are HIGHLY correlated with data correlation!")
        print("   The model is learning CORRELATION, not CAUSATION.")
        print()
        print("   Recommendations:")
        print("   1. Increase sparsity pressure to focus on fewer edges")
        print("   2. Use intervention/regime information more strongly")
        print("   3. Add explicit anti-correlation loss on confounded pairs")
    elif corr_with_pred > 0.4:
        print("~ Model outputs are MODERATELY correlated with data correlation")
        print("  Some causal learning, but still influenced by correlation")
    else:
        print("✓ Model outputs are NOT strongly correlated with data correlation")
        print("  Good sign - model may be learning causal structure")
    
    print()
    
    # Direction analysis
    print("DIRECTION ANALYSIS:")
    correct_dir = 0
    wrong_dir = 0
    for (i, j) in true_edges:
        if A_pred[i, j] > A_pred[j, i]:
            correct_dir += 1
        else:
            wrong_dir += 1
    
    print(f"  True edges with correct direction: {correct_dir}/{n_true}")
    print(f"  True edges with wrong direction: {wrong_dir}/{n_true}")
    
    if wrong_dir > correct_dir:
        print("  ⚠️ More wrong than correct directions!")
        print("     Check if GT convention matches model convention")
    
    print()


if __name__ == "__main__":
    main()
