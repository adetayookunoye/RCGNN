#!/usr/bin/env python3
"""
Standalone script to diagnose UNDIR=1.0 + DIR=0.0 contradiction.

This script:
1. Loads a fake adjacency matrix similar to training output
2. Extracts edges using the SAME method as train_rcgnn_unified.py
3. Prints both directed and undirected edge sets in detail
4. Tests the hypothesis that node indexing is misaligned

Usage:
    python debug_overlap_mismatch.py
"""

import numpy as np


def extract_edges_directed_method(A_pred_np, A_true_np, n_true_edges):
    """Extract top-K edges using DIRECTED method from training code."""
    d = A_pred_np.shape[0]
    
    # Flatten adjacency, mask diagonal, get top K
    A_flat = A_pred_np.flatten()
    flat_mask = np.ones(len(A_flat), dtype=bool)
    for i in range(d):
        flat_mask[i * d + i] = False  # mask diagonal
    
    # Get indices of top K in the FLAT directed space
    flat_valid = np.where(flat_mask)[0]
    flat_scores = A_flat[flat_valid]
    top_k_local = np.argsort(flat_scores)[-n_true_edges:]
    top_k_flat = flat_valid[top_k_local]
    
    # Unravel to (i,j) in the original directed space
    pred_edges_directed = set(zip(*np.unravel_index(top_k_flat, (d, d))))
    pred_edges_undirected = {tuple(sorted((i, j))) for i, j in pred_edges_directed}
    
    # True edges from ground truth
    true_indices = np.where(A_true_np > 0)
    true_edges_directed = set(zip(true_indices[0], true_indices[1]))
    true_edges_undirected = {tuple(sorted((i, j))) for i, j in true_edges_directed}
    
    return {
        'pred_edges_directed': pred_edges_directed,
        'pred_edges_undirected': pred_edges_undirected,
        'true_edges_directed': true_edges_directed,
        'true_edges_undirected': true_edges_undirected,
    }


def compute_overlaps(edges_dict):
    """Compute DIR and UNDIR overlaps."""
    pred_dir = edges_dict['pred_edges_directed']
    true_dir = edges_dict['true_edges_directed']
    pred_un = edges_dict['pred_edges_undirected']
    true_un = edges_dict['true_edges_undirected']
    
    inter_dir = pred_dir & true_dir
    inter_un = pred_un & true_un
    
    dir_overlap = len(inter_dir) / max(len(true_dir), 1)
    undir_overlap = len(inter_un) / max(len(true_un), 1)
    
    return {
        'dir_overlap': dir_overlap,
        'undir_overlap': undir_overlap,
        'inter_dir': inter_dir,
        'inter_un': inter_un,
    }


def test_case_1():
    """
    Test case from logs:
    - Pred (directed): [(11, 10), (6, 5), (7, 5)]
    - True (directed): [(0, 1), (2, 4), (10, 5)]
    - Observed: DIR=0.0, UNDIR=1.0
    
    This is IMPOSSIBLE if both are in the same coordinate system.
    """
    print("\n" + "="*70)
    print("TEST CASE 1: Reproduction from logs")
    print("="*70)
    
    # Fake adjacency matrices (13x13)
    d = 13
    n_true_edges = 3
    
    # Create pred adjacency with top weights at indices for (11,10), (6,5), (7,5)
    A_pred_np = np.zeros((d, d))
    A_pred_np[11, 10] = 0.900
    A_pred_np[6, 5] = 0.850
    A_pred_np[7, 5] = 0.800
    # Fill rest with lower values
    for i in range(d):
        for j in range(d):
            if i != j and A_pred_np[i, j] == 0:
                A_pred_np[i, j] = np.random.uniform(0.0, 0.3)
    
    # Create true adjacency with edges at (0,1), (2,4), (10,5)
    A_true_np = np.zeros((d, d))
    A_true_np[0, 1] = 1.0
    A_true_np[2, 4] = 1.0
    A_true_np[10, 5] = 1.0
    
    np.fill_diagonal(A_pred_np, 0)
    np.fill_diagonal(A_true_np, 0)
    
    edges = extract_edges_directed_method(A_pred_np, A_true_np, n_true_edges)
    overlaps = compute_overlaps(edges)
    
    print(f"\n[PRED]")
    print(f"  Directed:   {sorted(edges['pred_edges_directed'])}")
    print(f"  Undirected: {sorted(edges['pred_edges_undirected'])}")
    
    print(f"\n[TRUE]")
    print(f"  Directed:   {sorted(edges['true_edges_directed'])}")
    print(f"  Undirected: {sorted(edges['true_edges_undirected'])}")
    
    print(f"\n[INTERSECTION]")
    print(f"  Directed:   {sorted(overlaps['inter_dir'])}")
    print(f"  Undirected: {sorted(overlaps['inter_un'])}")
    
    print(f"\n[OVERLAP RESULTS]")
    print(f"  DIR:   {overlaps['dir_overlap']:.4f}")
    print(f"  UNDIR: {overlaps['undir_overlap']:.4f}")
    
    print(f"\n[ANALYSIS]")
    if overlaps['dir_overlap'] == 0.0 and overlaps['undir_overlap'] == 1.0:
        print("  ❌ REPRODUCED THE BUG: DIR=0.0 + UNDIR=1.0")
        print("  This should be IMPOSSIBLE if both use the same node indices!")
        print("\n  Possible causes:")
        print("  1. Pred edges are in one node ordering, true in another")
        print("  2. There's a permutation being applied inconsistently")
        print("  3. Undirected overlap calculation has a bug (not actually matching)")
    else:
        print(f"  ✓ No bug reproduced. Results are sensible.")
        print(f"    DIR overlap makes sense (no directed matches)")
        print(f"    UNDIR overlap = {overlaps['undir_overlap']:.4f} (reasonable)")


def test_case_2_with_permutation():
    """
    Test if PERMUTATION could cause this:
    - What if pred edges are in permuted node order?
    """
    print("\n" + "="*70)
    print("TEST CASE 2: With node permutation (hypothesis)")
    print("="*70)
    
    d = 13
    n_true_edges = 3
    
    # Original true edges
    true_edges_orig = [(0, 1), (2, 4), (10, 5)]
    
    # Apply a random permutation
    perm = np.random.permutation(d)
    inv_perm = np.argsort(perm)
    
    print(f"\nPermutation applied: {perm}")
    print(f"Inverse permutation: {inv_perm}")
    
    # Map true edges to permuted space
    true_edges_permuted = [(perm[i], perm[j]) for (i, j) in true_edges_orig]
    print(f"\nTrue edges (original):  {sorted(true_edges_orig)}")
    print(f"True edges (permuted):  {sorted(true_edges_permuted)}")
    
    # Now check: if we compare pred in original space vs true in permuted space
    pred_edges = [(11, 10), (6, 5), (7, 5)]
    
    pred_undirected = {tuple(sorted((i, j))) for i, j in pred_edges}
    true_undirected_perm = {tuple(sorted((i, j))) for i, j in true_edges_permuted}
    
    inter_un = pred_undirected & true_undirected_perm
    
    print(f"\n[MISMATCH SCENARIO]")
    print(f"  Pred (original space):     {sorted(pred_undirected)}")
    print(f"  True (permuted space):     {sorted(true_undirected_perm)}")
    print(f"  Intersection:              {sorted(inter_un)}")
    print(f"  Undirected overlap:        {len(inter_un) / max(len(true_undirected_perm), 1):.4f}")
    print(f"  -> Could explain 0 overlap if permutation scrambles indices!")


if __name__ == "__main__":
    test_case_1()
    test_case_2_with_permutation()
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("""
If TEST CASE 1 reproduces the bug (UNDIR=1.0 + DIR=0.0), then the issue is
that BOTH pred and true are being computed in the SAME coordinate system
but one is somehow inverted or relabeled at the undirected level.

More likely: The undirected extraction has a bug that causes it to
accidentally report 1.0 when it should report 0.0.

For debugging:
1. Print the FULL undirected edge sets (not just samples)
2. Verify they actually intersect
3. Check if there's a typo in the set construction (e.g., sorted() bug)
""")
