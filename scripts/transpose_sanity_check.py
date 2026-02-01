#!/usr/bin/env python3
"""
TRANSPOSE SANITY CHECK
======================
Quick diagnostic to detect direction bugs:
1. Load best checkpoint
2. Compute TopK-F1 as-is
3. Compute TopK-F1 on transpose
4. If transpose is better -> likely indexing or GT direction bug

Usage:
    python scripts/transpose_sanity_check.py --data data/interim/uci_air_c/compound_mnar_bias
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
from pathlib import Path


def compute_topk_f1(A_pred: np.ndarray, A_true: np.ndarray, k: int = None):
    """Compute TopK F1 score."""
    A_pred = A_pred.copy()
    A_true = (A_true > 0.5).astype(int)
    
    np.fill_diagonal(A_pred, 0)
    np.fill_diagonal(A_true, 0)
    
    n_true_edges = int(A_true.sum())
    if k is None:
        k = n_true_edges
    
    flat = A_pred.flatten()
    top_k_idx = np.argsort(flat)[-k:] if k > 0 else np.array([])
    
    pred_binary = np.zeros_like(flat, dtype=int)
    if len(top_k_idx) > 0:
        pred_binary[top_k_idx] = 1
    
    true_flat = A_true.flatten()
    
    TP = int(((pred_binary == 1) & (true_flat == 1)).sum())
    FP = int(((pred_binary == 1) & (true_flat == 0)).sum())
    FN = int(((pred_binary == 0) & (true_flat == 1)).sum())
    
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return {"f1": f1, "tp": TP, "fp": FP, "fn": FN, "k": k}


def compute_skeleton_f1(A_pred: np.ndarray, A_true: np.ndarray, k: int = None):
    """Compute skeleton (undirected) F1."""
    A_pred = A_pred.copy()
    A_true = (A_true > 0.5).astype(int)
    
    # Symmetrize
    A_pred_sym = np.maximum(A_pred, A_pred.T)
    A_true_sym = np.maximum(A_true, A_true.T)
    
    np.fill_diagonal(A_pred_sym, 0)
    np.fill_diagonal(A_true_sym, 0)
    
    # Upper triangle only
    A_pred_upper = np.triu(A_pred_sym)
    A_true_upper = np.triu(A_true_sym)
    
    n_true_edges = int(A_true_upper.sum())
    if k is None:
        k = n_true_edges
    
    flat = A_pred_upper.flatten()
    top_k_idx = np.argsort(flat)[-k:] if k > 0 else np.array([])
    
    pred_binary = np.zeros_like(flat, dtype=int)
    if len(top_k_idx) > 0:
        pred_binary[top_k_idx] = 1
    
    true_flat = A_true_upper.flatten()
    
    TP = int(((pred_binary == 1) & (true_flat == 1)).sum())
    
    precision = TP / (pred_binary.sum() + 1e-8)
    recall = TP / (true_flat.sum() + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return {"f1": f1, "tp": TP, "k": k}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to dataset")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (optional)")
    parser.add_argument("--adjacency", type=str, default=None, help="Path to A_mean.npy (optional)")
    args = parser.parse_args()
    
    data_path = Path(args.data)
    
    # Load ground truth
    A_true = np.load(data_path / "A_true.npy")
    n_true_edges = int((A_true > 0).sum())
    
    print("=" * 60)
    print("TRANSPOSE SANITY CHECK")
    print("=" * 60)
    print(f"Dataset: {data_path}")
    print(f"A_true shape: {A_true.shape}")
    print(f"True edges: {n_true_edges}")
    print()
    
    # Show A_true structure
    print("A_true edge list (i -> j):")
    edges = list(zip(*np.where(A_true > 0)))
    for i, j in edges[:20]:
        print(f" {i} -> {j}")
    if len(edges) > 20:
        print(f" ... ({len(edges) - 20} more)")
    print()
    
    # Load predicted adjacency
    if args.adjacency:
        A_pred = np.load(args.adjacency)
    elif args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        A_pred = ckpt.get("A_mean", None)
        if A_pred is None:
            print("ERROR: Checkpoint does not contain A_mean")
            return
        A_pred = A_pred.numpy() if isinstance(A_pred, torch.Tensor) else A_pred
    else:
        # Try to find latest adjacency
        artifact_paths = [
            Path("artifacts/adjacency/A_mean.npy"),
            Path("artifacts/unified_v8_compound_mnar_bias/A_mean.npy"),
            data_path.parent / "A_mean.npy",
        ]
        A_pred = None
        for p in artifact_paths:
            if p.exists():
                A_pred = np.load(p)
                print(f"Loaded: {p}")
                break
        
        if A_pred is None:
            print("ERROR: No adjacency found. Specify --adjacency or --checkpoint")
            return
    
    print(f"A_pred shape: {A_pred.shape}")
    print(f"A_pred range: [{A_pred.min():.4f}, {A_pred.max():.4f}]")
    print()
    
    # Compute metrics
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    # As-is
    topk_orig = compute_topk_f1(A_pred, A_true, k=n_true_edges)
    skel_orig = compute_skeleton_f1(A_pred, A_true, k=n_true_edges)
    
    print(f"ORIGINAL A_pred:")
    print(f" TopK-F1: {topk_orig['f1']:.4f} (TP={topk_orig['tp']}/{n_true_edges})")
    print(f" Skeleton-F1: {skel_orig['f1']:.4f} (TP={skel_orig['tp']})")
    print()
    
    # Transpose
    A_pred_T = A_pred.T
    topk_trans = compute_topk_f1(A_pred_T, A_true, k=n_true_edges)
    skel_trans = compute_skeleton_f1(A_pred_T, A_true, k=n_true_edges)
    
    print(f"TRANSPOSED A_pred.T:")
    print(f" TopK-F1: {topk_trans['f1']:.4f} (TP={topk_trans['tp']}/{n_true_edges})")
    print(f" Skeleton-F1: {skel_trans['f1']:.4f} (TP={skel_trans['tp']})")
    print()
    
    # Diagnosis
    print("=" * 60)
    print("DIAGNOSIS")
    print("=" * 60)
    
    if topk_trans['f1'] > topk_orig['f1'] + 0.05:
        print("[WARN] TRANSPOSE IS BETTER!")
        print(" This suggests a DIRECTION BUG:")
        print(" - A[i,j] indexing might be swapped")
        print(" - Or GT uses opposite convention (j->i instead of i->j)")
        print()
        print(" FIX: Check if model outputs A[j,i] for edge i->j")
    elif topk_orig['f1'] > topk_trans['f1'] + 0.05:
        print("[OK] ORIGINAL IS BETTER")
        print(" Direction encoding appears correct.")
    else:
        print("~ SIMILAR PERFORMANCE")
        print(" Model may not be learning direction well yet.")
        print(" Skeleton-F1 vs TopK-F1 gap indicates direction confusion.")
    
    print()
    
    # Direction ratio analysis
    dir_ratio_orig = topk_orig['f1'] / (skel_orig['f1'] + 1e-8)
    dir_ratio_trans = topk_trans['f1'] / (skel_trans['f1'] + 1e-8)
    
    print(f"Direction Ratios:")
    print(f" Original: {dir_ratio_orig:.2f} (TopK/Skel)")
    print(f" Transpose: {dir_ratio_trans:.2f} (TopK/Skel)")
    print()
    
    if skel_orig['f1'] > topk_orig['f1'] + 0.15:
        print("[WARN] SKELETON >> TOPK")
        print(" Model finds correct edges but wrong direction!")
        print(" Need stronger direction supervision.")
    
    # Show top predicted edges vs true
    print()
    print("=" * 60)
    print("TOP PREDICTED EDGES")
    print("=" * 60)
    
    A_pred_clean = A_pred.copy()
    np.fill_diagonal(A_pred_clean, 0)
    
    flat = A_pred_clean.flatten()
    top_idx = np.argsort(flat)[::-1][:n_true_edges * 2]
    
    print(f"{'Rank':<5} {'Edge':<10} {'Weight':<10} {'In GT?':<10} {'Reversed?'}")
    print("-" * 50)
    
    d = A_pred.shape[0]
    for rank, idx in enumerate(top_idx[:20], 1):
        i, j = idx // d, idx % d
        weight = flat[idx]
        in_gt = A_true[i, j] > 0
        reversed_gt = A_true[j, i] > 0
        
        status = "[OK] CORRECT" if in_gt else ("<-> REVERSED" if reversed_gt else "[X] WRONG")
        print(f"{rank:<5} {i}->{j:<6} {weight:<10.4f} {status}")
    
    print()


if __name__ == "__main__":
    main()
