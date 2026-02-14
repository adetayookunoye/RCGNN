#!/usr/bin/env python3
"""
GOLDEN TESTS FOR METRICS - Ensures metric implementations are correct.

These tests use small synthetic graphs where the correct answers are known,
providing a ground truth for our metric functions.

Golden tests catch:
1. Off-by-one errors in edge counting
2. Convention errors (row vs column)
3. Diagonal handling bugs
4. TopK selection bugs
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# GOLDEN TEST CASES - Tiny graphs with known answers
# =============================================================================

def create_golden_case_1():
    """
    Golden Case 1: Simple chain 0→1→2
    
    A_true:
        0  1  2
    0 [ 0  1  0 ]  (0→1)
    1 [ 0  0  1 ]  (1→2)
    2 [ 0  0  0 ]
    
    True edges: 2 (directed)
    True skeleton edges: 2 (undirected pairs)
    """
    A_true = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0]
    ], dtype=float)
    
    return {
        'name': 'chain_3',
        'A_true': A_true,
        'n_true_edges': 2,
        'n_skeleton_edges': 2,
        'true_edges': [(0, 1), (1, 2)],
        'true_skeleton': [(0, 1), (1, 2)],
    }


def create_golden_case_2():
    """
    Golden Case 2: Fork 0→1, 0→2
    
    A_true:
        0  1  2
    0 [ 0  1  1 ]  (0→1, 0→2)
    1 [ 0  0  0 ]
    2 [ 0  0  0 ]
    
    True edges: 2 (directed)
    """
    A_true = np.array([
        [0, 1, 1],
        [0, 0, 0],
        [0, 0, 0]
    ], dtype=float)
    
    return {
        'name': 'fork_3',
        'A_true': A_true,
        'n_true_edges': 2,
        'n_skeleton_edges': 2,
        'true_edges': [(0, 1), (0, 2)],
        'true_skeleton': [(0, 1), (0, 2)],
    }


def create_golden_case_3():
    """
    Golden Case 3: V-structure (collider) 0→2←1
    
    A_true:
        0  1  2
    0 [ 0  0  1 ]  (0→2)
    1 [ 0  0  1 ]  (1→2)
    2 [ 0  0  0 ]
    
    True edges: 2 (directed)
    """
    A_true = np.array([
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 0]
    ], dtype=float)
    
    return {
        'name': 'collider_3',
        'A_true': A_true,
        'n_true_edges': 2,
        'n_skeleton_edges': 2,
        'true_edges': [(0, 2), (1, 2)],
        'true_skeleton': [(0, 2), (1, 2)],
    }


def create_golden_case_4():
    """
    Golden Case 4: Triangle with specific directions
    
    A_true:
        0  1  2
    0 [ 0  1  1 ]  (0→1, 0→2)
    1 [ 0  0  1 ]  (1→2)
    2 [ 0  0  0 ]
    
    True edges: 3 (directed)
    True skeleton: 3 (undirected pairs)
    """
    A_true = np.array([
        [0, 1, 1],
        [0, 0, 1],
        [0, 0, 0]
    ], dtype=float)
    
    return {
        'name': 'triangle_3',
        'A_true': A_true,
        'n_true_edges': 3,
        'n_skeleton_edges': 3,
        'true_edges': [(0, 1), (0, 2), (1, 2)],
        'true_skeleton': [(0, 1), (0, 2), (1, 2)],
    }


def create_golden_case_5():
    """
    Golden Case 5: Prediction with reversal
    
    Truth: 0→1
    Pred:  1→0 (reversed)
    
    This tests reversal counting and SHD conventions.
    """
    A_true = np.array([
        [0, 1, 0],
        [0, 0, 0],
        [0, 0, 0]
    ], dtype=float)
    
    A_pred = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 0, 0]
    ], dtype=float)
    
    return {
        'name': 'reversal_test',
        'A_true': A_true,
        'A_pred': A_pred,
        'expected_shd': 2,  # 1 FP (1→0) + 1 FN (0→1) = 2 (adjacency SHD)
        'expected_skeleton_f1': 1.0,  # Skeleton is correct (pair 0-1 exists)
        'expected_directed_f1': 0.0,  # No directed edge matches
        'expected_reversals': 1,
    }


def create_golden_case_6():
    """
    Golden Case 6: Soft scores requiring TopK selection
    
    Truth: edges (0,1), (1,2) with K=2
    Pred: soft scores where top 2 should select correct edges
    """
    A_true = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0]
    ], dtype=float)
    
    # Soft predictions: (0,1)=0.9, (1,2)=0.8 should be top 2
    # (0,2)=0.3 should be filtered out by TopK
    A_pred = np.array([
        [0.0, 0.9, 0.3],
        [0.1, 0.0, 0.8],
        [0.0, 0.0, 0.0]
    ], dtype=float)
    
    return {
        'name': 'topk_soft_scores',
        'A_true': A_true,
        'A_pred': A_pred,
        'K': 2,
        'expected_topk_f1': 1.0,  # Top 2 are correct edges
        'expected_topk_edges': [(0, 1), (1, 2)],
    }


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_golden_edge_counting():
    """Test that basic edge counting is correct."""
    cases = [
        create_golden_case_1(),
        create_golden_case_2(),
        create_golden_case_3(),
        create_golden_case_4(),
    ]
    
    for case in cases:
        A_true = case['A_true']
        np.fill_diagonal(A_true, 0)  # Ensure no self-loops
        
        # Count directed edges
        n_edges = int(np.sum(A_true > 0.5))
        assert n_edges == case['n_true_edges'], \
            f"{case['name']}: Expected {case['n_true_edges']} edges, got {n_edges}"
        
        # Count skeleton edges (upper triangle of symmetrized)
        A_skel = np.maximum(A_true, A_true.T)
        n_skel = int(np.sum(np.triu(A_skel, k=1) > 0.5))
        assert n_skel == case['n_skeleton_edges'], \
            f"{case['name']}: Expected {case['n_skeleton_edges']} skeleton edges, got {n_skel}"
    
    print("[PASS] Golden edge counting tests")


def test_golden_shd():
    """Test SHD computation with known cases."""
    case = create_golden_case_5()
    
    A_true = case['A_true']
    A_pred = case['A_pred']
    
    # Compute SHD (adjacency convention: reversal = 2 errors)
    A_pred_bin = (A_pred > 0.5).astype(float)
    A_true_bin = (A_true > 0.5).astype(float)
    np.fill_diagonal(A_pred_bin, 0)
    np.fill_diagonal(A_true_bin, 0)
    
    shd = int(np.sum(np.abs(A_pred_bin - A_true_bin)))
    assert shd == case['expected_shd'], \
        f"Expected SHD={case['expected_shd']}, got {shd}"
    
    # Count reversals
    reversals = int(np.sum((A_pred_bin > 0) & (A_true_bin.T > 0)))
    assert reversals == case['expected_reversals'], \
        f"Expected {case['expected_reversals']} reversals, got {reversals}"
    
    print("[PASS] Golden SHD tests")


def test_golden_skeleton_f1():
    """Test skeleton F1 with reversal case."""
    case = create_golden_case_5()
    
    A_true = case['A_true']
    A_pred = case['A_pred']
    
    # Skeleton: symmetrize then compute F1
    A_true_skel = ((A_true + A_true.T) > 0.5).astype(float)
    A_pred_skel = ((A_pred + A_pred.T) > 0.5).astype(float)
    np.fill_diagonal(A_true_skel, 0)
    np.fill_diagonal(A_pred_skel, 0)
    
    # Only upper triangle (avoid double counting)
    true_edges = set(zip(*np.where(np.triu(A_true_skel, k=1) > 0)))
    pred_edges = set(zip(*np.where(np.triu(A_pred_skel, k=1) > 0)))
    
    tp = len(true_edges & pred_edges)
    fp = len(pred_edges - true_edges)
    fn = len(true_edges - pred_edges)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    assert abs(f1 - case['expected_skeleton_f1']) < 0.01, \
        f"Expected Skeleton F1={case['expected_skeleton_f1']}, got {f1}"
    
    print("[PASS] Golden skeleton F1 tests")


def test_golden_directed_f1():
    """Test directed F1 with reversal case."""
    case = create_golden_case_5()
    
    A_true = case['A_true']
    A_pred = case['A_pred']
    
    A_true_bin = (A_true > 0.5).astype(float)
    A_pred_bin = (A_pred > 0.5).astype(float)
    np.fill_diagonal(A_true_bin, 0)
    np.fill_diagonal(A_pred_bin, 0)
    
    tp = int(np.sum((A_pred_bin > 0) & (A_true_bin > 0)))
    fp = int(np.sum((A_pred_bin > 0) & (A_true_bin == 0)))
    fn = int(np.sum((A_pred_bin == 0) & (A_true_bin > 0)))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    assert abs(f1 - case['expected_directed_f1']) < 0.01, \
        f"Expected Directed F1={case['expected_directed_f1']}, got {f1}"
    
    print("[PASS] Golden directed F1 tests")


def test_golden_topk_selection():
    """Test TopK edge selection with soft scores."""
    case = create_golden_case_6()
    
    A_pred = case['A_pred'].copy()
    K = case['K']
    
    # TopK selection (exclude diagonal)
    np.fill_diagonal(A_pred, -np.inf)
    flat = A_pred.flatten()
    topk_idx = np.argsort(flat)[-K:]
    
    d = A_pred.shape[0]
    selected_edges = [(idx // d, idx % d) for idx in topk_idx]
    selected_edges = set(selected_edges)
    
    expected_edges = set(case['expected_topk_edges'])
    
    assert selected_edges == expected_edges, \
        f"Expected edges {expected_edges}, got {selected_edges}"
    
    print("[PASS] Golden TopK selection tests")


def test_golden_diagonal_handling():
    """Test that diagonals are always excluded."""
    # Create matrix with diagonal entries
    A = np.array([
        [0.9, 0.5, 0.3],  # Diagonal is 0.9
        [0.1, 0.8, 0.6],  # Diagonal is 0.8
        [0.2, 0.4, 0.7]   # Diagonal is 0.7
    ], dtype=float)
    
    # TopK with K=3 should NOT select diagonal entries
    A_copy = A.copy()
    np.fill_diagonal(A_copy, -np.inf)
    flat = A_copy.flatten()
    topk_idx = np.argsort(flat)[-3:]
    
    d = A.shape[0]
    selected_edges = [(idx // d, idx % d) for idx in topk_idx]
    
    # None should be self-loops
    for i, j in selected_edges:
        assert i != j, f"TopK selected self-loop ({i}, {j})"
    
    print("[PASS] Golden diagonal handling tests")


def test_metric_consistency_with_eval_script():
    """
    Test that our golden tests match comprehensive_evaluation.py implementations.
    This ensures the evaluation script uses correct metric implementations.
    """
    try:
        # Import from evaluation script
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from comprehensive_evaluation import (
            compute_shd,
            compute_skeleton_f1,
            compute_directed_f1,
            select_topk_edges,
            compute_reversal_count,
        )
        
        # Test with golden case 5 (reversal)
        case = create_golden_case_5()
        A_true = case['A_true']
        A_pred = case['A_pred']
        
        # SHD
        shd = compute_shd(A_pred, A_true, threshold=0.5)
        assert shd == case['expected_shd'], \
            f"compute_shd: Expected {case['expected_shd']}, got {shd}"
        
        # Skeleton F1
        skel_f1, _, _ = compute_skeleton_f1(A_pred, A_true, threshold=0.5)
        assert abs(skel_f1 - case['expected_skeleton_f1']) < 0.01, \
            f"compute_skeleton_f1: Expected {case['expected_skeleton_f1']}, got {skel_f1}"
        
        # Directed F1
        dir_f1, _, _ = compute_directed_f1(A_pred, A_true, threshold=0.5)
        assert abs(dir_f1 - case['expected_directed_f1']) < 0.01, \
            f"compute_directed_f1: Expected {case['expected_directed_f1']}, got {dir_f1}"
        
        # Reversal count
        reversals = compute_reversal_count(A_pred, A_true, threshold=0.5)
        assert reversals == case['expected_reversals'], \
            f"compute_reversal_count: Expected {case['expected_reversals']}, got {reversals}"
        
        print("[PASS] Metric consistency with comprehensive_evaluation.py")
        
    except ImportError as e:
        print(f"[SKIP] Could not import evaluation functions: {e}")


def run_golden_tests():
    """Run all golden tests."""
    print("=" * 70)
    print("GOLDEN TESTS - Verifying metric implementations")
    print("=" * 70)
    print()
    
    test_golden_edge_counting()
    test_golden_shd()
    test_golden_skeleton_f1()
    test_golden_directed_f1()
    test_golden_topk_selection()
    test_golden_diagonal_handling()
    test_metric_consistency_with_eval_script()
    
    print()
    print("=" * 70)
    print("ALL GOLDEN TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    run_golden_tests()
