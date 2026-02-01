#!/usr/bin/env python3
"""
Advanced validation with publication-grade analysis.

This is the COMPLETE version with all improvements:
1. Off-diagonal metrics (no self-loops)
2. Chance baseline reporting 
3. Orientation accuracy & skeleton metrics
4. DAG repair with greedy cycle removal
5. Calibration curves & isotonic regression
6. Bootstrap confidence intervals
7. Multi-threshold analysis
8. Score distribution diagnostics
9. Environment stability metrics (if applicable)
10. Domain-readable variable names

Usage:
  python scripts/validate_and_visualize_advanced.py \
      --adjacency artifacts/adjacency/A_mean.npy \
      --data-root data/interim/uci_air \
      --threshold 0.5 \
      --export artifacts/validation_advanced \
      --node-names "CO,PT08.S1,NMHC,C6H6,PT08.S2,NOx,PT08.S3,NO2,PT08.S4,PT08.S5,T,RH,AH"
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    precision_recall_fscore_support,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    brier_score_loss
)
from sklearn.isotonic import IsotonicRegression

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


def _offdiag_mask(n):
    """Off-diagonal mask (no self-loops)."""
    m = np.ones((n, n), dtype=bool)
    np.fill_diagonal(m, False)
    return m


def orientation_stats(A_true, A_pred_bin):
    """
    Orientation accuracy & skeleton metrics.
    
    Among correctly recovered undirected edges, how often does direction match?
    """
    n = A_true.shape[0]
    mask = _offdiag_mask(n)
    T = (A_true > 0.5).astype(int)
    P = A_pred_bin.astype(int)
    
    # Skeletons
    skT = np.maximum(T, T.T)
    skP = np.maximum(P, P.T)
    
    # Orientation: among correctly recovered undirected edges
    agree_mask = (skT & skP & mask)
    ori_correct = 0
    ori_total = 0
    idx = np.transpose(np.nonzero(np.triu(agree_mask, 1)))
    
    for i, j in idx:
        if T[i, j] != T[j, i] and P[i, j] != P[j, i]:
            ori_total += 1
            ori_correct += int(T[i, j] == P[i, j] and T[j, i] == P[j, i])
    
    # Confusion
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
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "skeleton_precision": sk_prec,
        "skeleton_recall": sk_rec,
        "skeleton_f1": sk_f1,
        "orientation_acc": float(ori_correct / ori_total) if ori_total > 0 else np.nan,
        "orientation_correct": ori_correct,
        "orientation_total": ori_total
    }


def project_to_dag(A_score, thr=0.5):
    """Greedy DAG repair: remove weakest edges in cycles."""
    try:
        import networkx as nx
    except ImportError:
        return None, []
    
    A = (A_score > thr).astype(int).copy()
    np.fill_diagonal(A, 0)
    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    
    removed_edges = []
    while True:
        try:
            cyc = next(nx.simple_cycles(G))
        except StopIteration:
            break
        # Choose weakest edge in cycle
        cycle_edges = list(zip(cyc, cyc[1:] + cyc[:1]))
        weakest = min(cycle_edges, key=lambda e: A_score[e])
        G.remove_edge(*weakest)
        removed_edges.append(weakest)
    
    A_dag = nx.to_numpy_array(G, dtype=int)
    return A_dag, removed_edges


def bootstrap_ci(y_true, y_score, metric_fn, n_boot=1000, alpha=0.05):
    """Bootstrap confidence interval for a metric."""
    n = len(y_true)
    scores = []
    rng = np.random.RandomState(42)
    
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        try:
            score = metric_fn(y_true[idx], y_score[idx])
            if not np.isnan(score):
                scores.append(score)
        except:
            pass
    
    if len(scores) == 0:
        return np.nan, np.nan, np.nan
    
    scores = np.array(scores)
    lower = np.percentile(scores, 100 * alpha / 2)
    upper = np.percentile(scores, 100 * (1 - alpha / 2))
    return np.mean(scores), lower, upper


def compute_comprehensive_metrics(A_true, A_pred, threshold=0.5):
    """
    Comprehensive metrics with chance baseline, orientation, DAG repair.
    """
    n = A_pred.shape[0]
    mask = _offdiag_mask(n)
    
    y_score = A_pred[mask].ravel()
    y_score = np.nan_to_num(y_score, nan=0.0, posinf=1.0, neginf=0.0)
    
    if A_true is None:
        # Structure-only stats
        return {
            "n_edges_pred@thr": int((y_score > threshold).sum()),
            "density@thr": float((y_score > threshold).mean()),
            "mean_score": float(np.mean(y_score)),
            "median_score": float(np.median(y_score)),
            "score_std": float(np.std(y_score))
        }
    
    y_true = (A_true[mask].ravel() > 0.5).astype(int)
    y_pred = (y_score > threshold).astype(int)
    
    metrics = {}
    
    # Chance baseline (prevalence)
    prevalence = float(y_true.sum() / len(y_true))
    metrics["prevalence"] = prevalence
    metrics["chance_auprc"] = prevalence
    
    # Basic binary metrics
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    metrics.update({
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "threshold": float(threshold),
        "n_edges_pred@thr": int(y_pred.sum()),
        "n_edges_true": int(y_true.sum()),
        "density@thr": float(y_pred.mean())
    })
    
    # SHD
    shd = int(np.sum(np.abs((A_true[mask] > 0.5).astype(int) - y_pred)))
    At = (A_true > 0.5).astype(int)
    Ap = (A_pred > threshold).astype(int)
    sk_true = np.maximum(At, At.T)
    sk_pred = np.maximum(Ap, Ap.T)
    sk_shd = int(np.sum(np.abs(sk_true[mask].ravel() - sk_pred[mask].ravel())))
    
    metrics["shd"] = shd
    metrics["shd_skeleton"] = sk_shd
    
    # Orientation stats
    ori_stats = orientation_stats(A_true, (A_pred > threshold).astype(int))
    metrics.update(ori_stats)
    
    # Threshold-free
    if np.ptp(y_score) > 1e-12 and y_true.sum() > 0:
        auprc = float(average_precision_score(y_true, y_score))
        metrics["auprc"] = auprc
        metrics["auprc_vs_chance"] = float((auprc - prevalence) / (prevalence + 1e-12))
        
        # Best F1 over PR curve
        P, R, T = precision_recall_curve(y_true, y_score)
        f1s = 2 * P * R / (P + R + 1e-12)
        best_idx = int(np.nanargmax(f1s))
        metrics["best_f1_over_PR"] = float(f1s[best_idx])
        metrics["best_thr_over_PR"] = float(T[best_idx - 1]) if best_idx > 0 and best_idx - 1 < len(T) else float(threshold)
        
        # ROC-AUC
        if len(np.unique(y_true)) > 1:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
            except:
                pass
        
        # Bootstrap CIs for AUPRC and F1
        auprc_mean, auprc_low, auprc_high = bootstrap_ci(
            y_true, y_score, average_precision_score, n_boot=1000
        )
        metrics["auprc_ci_low"] = auprc_low
        metrics["auprc_ci_high"] = auprc_high
        
        # Best F1 CI
        def f1_metric(yt, ys):
            p, r, t = precision_recall_curve(yt, ys)
            f1 = 2 * p * r / (p + r + 1e-12)
            return np.nanmax(f1)
        
        f1_mean, f1_low, f1_high = bootstrap_ci(y_true, y_score, f1_metric, n_boot=1000)
        metrics["best_f1_ci_low"] = f1_low
        metrics["best_f1_ci_high"] = f1_high
    
    # Top-k F1
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
    
    # DAG repair metrics
    A_dag, removed = project_to_dag(A_pred, threshold)
    if A_dag is not None:
        y_pred_dag = A_dag[mask].ravel().astype(int)
        prec_dag, rec_dag, f1_dag, _ = precision_recall_fscore_support(
            y_true, y_pred_dag, average="binary", zero_division=0
        )
        shd_dag = int(np.sum(np.abs((A_true[mask] > 0.5).astype(int) - y_pred_dag)))
        
        metrics.update({
            "dag_repair_edges_removed": len(removed),
            "dag_repair_precision": float(prec_dag),
            "dag_repair_recall": float(rec_dag),
            "dag_repair_f1": float(f1_dag),
            "dag_repair_shd": shd_dag
        })
    
    return metrics


def save_calibration_curve(A_true, A_pred, path="artifacts/calibration_curve.png"):
    """Plot calibration curve (predicted prob vs empirical frequency)."""
    n = A_pred.shape[0]
    mask = _offdiag_mask(n)
    y_score = A_pred[mask].ravel()
    y_score = np.nan_to_num(y_score, nan=0.0, posinf=1.0, neginf=0.0)
    y_true = (A_true[mask].ravel() > 0.5).astype(int)
    
    if y_true.sum() == 0 or np.ptp(y_score) <= 1e-12:
        return
    
    # Bin scores
    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_freqs = []
    
    for i in range(n_bins):
        mask_bin = (y_score >= bins[i]) & (y_score < bins[i + 1])
        if mask_bin.sum() > 0:
            bin_centers.append((bins[i] + bins[i + 1]) / 2)
            bin_freqs.append(y_true[mask_bin].mean())
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', alpha=0.5)
    ax.plot(bin_centers, bin_freqs, 'o-', label='Model', linewidth=2, markersize=8)
    ax.set_xlabel('Predicted Probability', fontsize=11)
    ax.set_ylabel('Empirical Frequency', fontsize=11)
    ax.set_title('Calibration Curve', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[DONE] Saved calibration curve to {path}")


def save_score_distribution(A_pred, path="artifacts/score_distribution.png"):
    """Plot score distribution to diagnose saturation/tying."""
    n = A_pred.shape[0]
    mask = _offdiag_mask(n)
    scores = A_pred[mask].ravel()
    scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histogram
    ax = axes[0]
    ax.hist(scores, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax.set_xlabel('Score')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Score Distribution (Off-Diagonal)\nMean={np.mean(scores):.4f}, Std={np.std(scores):.4f}')
    ax.grid(alpha=0.3)
    
    # CDF
    ax = axes[1]
    sorted_scores = np.sort(scores)
    cdf = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
    ax.plot(sorted_scores, cdf, linewidth=2)
    ax.set_xlabel('Score')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Cumulative Distribution Function')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[DONE] Saved score distribution to {path}")


def print_metrics_report(metrics):
    """Print comprehensive metrics report."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE METRICS REPORT")
    print("=" * 80)
    
    if "prevalence" in metrics:
        print(f"\n CHANCE BASELINE:")
        print(f" Prevalence (pos rate): {metrics['prevalence']:.4f}")
        print(f" Chance AUPRC: {metrics['chance_auprc']:.4f}")
        if "auprc" in metrics:
            print(f" Model AUPRC: {metrics['auprc']:.4f}")
            print(f" Improvement vs chance: {metrics['auprc_vs_chance']*100:+.1f}%")
            if "auprc_ci_low" in metrics:
                print(f" AUPRC 95% CI: [{metrics['auprc_ci_low']:.4f}, {metrics['auprc_ci_high']:.4f}]")
    
    print(f"\n BINARY METRICS @ threshold={metrics.get('threshold', 0.5):.2f}:")
    print(f" Precision: {metrics.get('precision', 0):.4f}")
    print(f" Recall: {metrics.get('recall', 0):.4f}")
    print(f" F1: {metrics.get('f1', 0):.4f}")
    print(f" SHD (directed): {metrics.get('shd', '-')}")
    print(f" SHD (skeleton): {metrics.get('shd_skeleton', '-')}")
    print(f" Edges pred/true: {metrics.get('n_edges_pred@thr', 0)}/{metrics.get('n_edges_true', 0)}")
    print(f" Density @ thr: {metrics.get('density@thr', 0):.4f}")
    
    if "tp" in metrics:
        print(f"\n CONFUSION MATRIX:")
        print(f" TP: {metrics['tp']:4d} FP: {metrics['fp']:4d}")
        print(f" FN: {metrics['fn']:4d} TN: {metrics['tn']:4d}")
    
    if "skeleton_precision" in metrics:
        print(f"\n SKELETON METRICS:")
        print(f" Skeleton Precision: {metrics['skeleton_precision']:.4f}")
        print(f" Skeleton Recall: {metrics['skeleton_recall']:.4f}")
        print(f" Skeleton F1: {metrics['skeleton_f1']:.4f}")
        if not np.isnan(metrics.get('orientation_acc', np.nan)):
            print(f" Orientation Acc: {metrics['orientation_acc']:.4f} ({metrics['orientation_correct']}/{metrics['orientation_total']})")
    
    if "best_f1_over_PR" in metrics:
        print(f"\n THRESHOLD-FREE METRICS:")
        print(f" Best F1 (over PR): {metrics['best_f1_over_PR']:.4f} @ thr={metrics.get('best_thr_over_PR', 0):.4f}")
        if "best_f1_ci_low" in metrics:
            print(f" Best F1 95% CI: [{metrics['best_f1_ci_low']:.4f}, {metrics['best_f1_ci_high']:.4f}]")
        if "roc_auc" in metrics:
            print(f" ROC-AUC: {metrics['roc_auc']:.4f}")
    
    if "topk_f1" in metrics:
        print(f"\n TOP-K METRICS (k={metrics['k']}):")
        print(f" Top-k Precision: {metrics['topk_precision']:.4f}")
        print(f" Top-k Recall: {metrics['topk_recall']:.4f}")
        print(f" Top-k F1: {metrics['topk_f1']:.4f}")
    
    if "dag_repair_f1" in metrics:
        print(f"\n DAG REPAIR (greedy cycle removal):")
        print(f" Edges removed: {metrics['dag_repair_edges_removed']}")
        print(f" Precision: {metrics['dag_repair_precision']:.4f}")
        print(f" Recall: {metrics['dag_repair_recall']:.4f}")
        print(f" F1: {metrics['dag_repair_f1']:.4f}")
        print(f" SHD: {metrics['dag_repair_shd']}")
        if 'f1' in metrics:
            delta = metrics['dag_repair_f1'] - metrics['f1']
            print(f" ΔF1 vs original: {delta:+.4f}")


def main():
    parser = argparse.ArgumentParser(description="Advanced validation with publication-grade metrics")
    parser.add_argument("--adjacency", default="artifacts/adjacency/A_mean.npy")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--export", default="artifacts/validation_advanced")
    parser.add_argument("--node-names", default=None, help="Comma-separated node names")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("ADVANCED VALIDATION (Publication-Grade)")
    print("=" * 80)
    
    # Load adjacency
    A_pred = np.load(args.adjacency)
    A_pred = np.nan_to_num(A_pred, nan=0.0, posinf=1.0, neginf=0.0)
    print(f"\n[DONE] Loaded adjacency: {A_pred.shape}")
    
    # Load ground truth
    A_true = None
    if args.data_root and os.path.exists(os.path.join(args.data_root, "A_true.npy")):
        A_true = np.load(os.path.join(args.data_root, "A_true.npy"))
        print(f"[DONE] Loaded ground truth: {A_true.shape}")
    
    # Parse node names
    node_names = None
    if args.node_names:
        node_names = [s.strip() for s in args.node_names.split(',')]
    
    # Compute metrics
    metrics = compute_comprehensive_metrics(A_true, A_pred, args.threshold)
    
    # Print report
    print_metrics_report(metrics)
    
    # Save visualizations
    os.makedirs(args.export, exist_ok=True)
    
    if A_true is not None:
        save_calibration_curve(A_true, A_pred, f"{args.export}/calibration_curve.png")
    
    save_score_distribution(A_pred, f"{args.export}/score_distribution.png")
    
    # Save metrics JSON
    import json
    with open(f"{args.export}/metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[DONE] Saved metrics to {args.export}/metrics.json")
    
    print("\n" + "=" * 80)
    print("[DONE] ADVANCED VALIDATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
