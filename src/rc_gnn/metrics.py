"""
Evaluation Metrics for RC-GNN v2

Implements standard causal discovery metrics:
- Structural Hamming Distance (SHD)
- Structural Intervention Distance (SID)
- F1, Precision, Recall
- AUROC, AUPRC
- False Discovery Rate (FDR)

Fixes critical bug: Excludes diagonal from SHD computation.
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional, Union
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


def threshold_adjacency(
    A: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Threshold soft adjacency to binary.
    
    Args:
        A: Soft adjacency [d, d] with values in [0, 1]
        threshold: Threshold for binarization
        
    Returns:
        A_binary: Binary adjacency [d, d]
    """
    if isinstance(A, torch.Tensor):
        A = A.detach().cpu().numpy()
    
    A_binary = (A > threshold).astype(np.float32)
    
    # Zero diagonal
    np.fill_diagonal(A_binary, 0)
    
    return A_binary


def compute_shd(
    A_pred: Union[torch.Tensor, np.ndarray],
    A_true: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.5,
    exclude_diagonal: bool = True,
) -> int:
    """
    Compute Structural Hamming Distance.
    
    SHD = # edge insertions + # edge deletions + # edge reversals
    
    For binary edges, this simplifies to counting differences
    in the symmetric difference of edge sets.
    
    CRITICAL: Excludes diagonal (self-loops) from computation.
    
    Args:
        A_pred: Predicted adjacency [d, d]
        A_true: Ground truth adjacency [d, d]
        threshold: Threshold for soft adjacency
        exclude_diagonal: Whether to exclude diagonal (should be True)
        
    Returns:
        shd: Structural Hamming Distance
    """
    if isinstance(A_pred, torch.Tensor):
        A_pred = A_pred.detach().cpu().numpy()
    if isinstance(A_true, torch.Tensor):
        A_true = A_true.detach().cpu().numpy()
    
    # Threshold
    A_pred_bin = (A_pred > threshold).astype(np.float32)
    A_true_bin = (A_true > threshold).astype(np.float32)
    
    # Zero diagonals
    if exclude_diagonal:
        np.fill_diagonal(A_pred_bin, 0)
        np.fill_diagonal(A_true_bin, 0)
    
    d = A_pred_bin.shape[0]
    
    # Count differences
    # We need to consider directed edges, not just undirected
    
    # Extra edges (false positives)
    extra = int(np.sum((A_pred_bin == 1) & (A_true_bin == 0)))
    
    # Missing edges (false negatives)
    missing = int(np.sum((A_pred_bin == 0) & (A_true_bin == 1)))
    
    # Reversed edges: edge exists in both but direction differs
    # For each pair (i,j), check if edge is reversed
    reverse = 0
    for i in range(d):
        for j in range(i + 1, d):
            # Ground truth has i->j, pred has j->i (or vice versa)
            if A_true_bin[i, j] == 1 and A_pred_bin[j, i] == 1 and A_true_bin[j, i] == 0 and A_pred_bin[i, j] == 0:
                reverse += 1
            elif A_true_bin[j, i] == 1 and A_pred_bin[i, j] == 1 and A_true_bin[i, j] == 0 and A_pred_bin[j, i] == 0:
                reverse += 1
    
    # Standard SHD: count simple differences (common definition)
    # Alternative: extra + missing - reverse (since reversals are counted twice)
    shd = int(np.sum(A_pred_bin != A_true_bin))
    
    return shd


def compute_edge_metrics(
    A_pred: Union[torch.Tensor, np.ndarray],
    A_true: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute edge-level classification metrics.
    
    Args:
        A_pred: Predicted adjacency [d, d]
        A_true: Ground truth adjacency [d, d]
        threshold: Threshold for binarization
        
    Returns:
        dict with precision, recall, f1, accuracy
    """
    if isinstance(A_pred, torch.Tensor):
        A_pred = A_pred.detach().cpu().numpy()
    if isinstance(A_true, torch.Tensor):
        A_true = A_true.detach().cpu().numpy()
    
    # Binarize
    A_pred_bin = (A_pred > threshold).astype(np.float32)
    A_true_bin = (A_true > threshold).astype(np.float32)
    
    # Zero diagonals
    np.fill_diagonal(A_pred_bin, 0)
    np.fill_diagonal(A_true_bin, 0)
    
    # Flatten (exclude diagonal)
    d = A_pred.shape[0]
    mask = ~np.eye(d, dtype=bool)
    
    y_pred = A_pred_bin[mask].flatten()
    y_true = A_true_bin[mask].flatten()
    
    # Handle edge case of no true edges
    if y_true.sum() == 0:
        if y_pred.sum() == 0:
            return {
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "accuracy": 1.0,
            }
        else:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "accuracy": float((y_pred == y_true).mean()),
            }
    
    # Compute metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    
    accuracy = float((y_pred == y_true).mean())
    
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": accuracy,
    }


def compute_auroc_auprc(
    A_pred: Union[torch.Tensor, np.ndarray],
    A_true: Union[torch.Tensor, np.ndarray],
) -> Dict[str, float]:
    """
    Compute AUROC and AUPRC from soft predictions.
    
    Args:
        A_pred: Predicted adjacency (soft) [d, d]
        A_true: Ground truth adjacency [d, d]
        
    Returns:
        dict with auroc, auprc
    """
    if isinstance(A_pred, torch.Tensor):
        A_pred = A_pred.detach().cpu().numpy()
    if isinstance(A_true, torch.Tensor):
        A_true = A_true.detach().cpu().numpy()
    
    d = A_pred.shape[0]
    
    # Exclude diagonal
    mask = ~np.eye(d, dtype=bool)
    y_score = A_pred[mask].flatten()
    y_true = (A_true[mask] > 0.5).astype(np.float32).flatten()
    
    # Handle edge cases
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return {"auroc": 0.5, "auprc": float(y_true.mean())}
    
    try:
        auroc = roc_auc_score(y_true, y_score)
    except ValueError:
        auroc = 0.5
    
    try:
        auprc = average_precision_score(y_true, y_score)
    except ValueError:
        auprc = float(y_true.mean())
    
    return {"auroc": float(auroc), "auprc": float(auprc)}


def compute_fdr_tpr(
    A_pred: Union[torch.Tensor, np.ndarray],
    A_true: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute False Discovery Rate and True Positive Rate.
    
    FDR = FP / (FP + TP)
    TPR = TP / (TP + FN) = Recall
    
    Args:
        A_pred: Predicted adjacency [d, d]
        A_true: Ground truth adjacency [d, d]
        threshold: Threshold for binarization
        
    Returns:
        dict with fdr, tpr
    """
    if isinstance(A_pred, torch.Tensor):
        A_pred = A_pred.detach().cpu().numpy()
    if isinstance(A_true, torch.Tensor):
        A_true = A_true.detach().cpu().numpy()
    
    A_pred_bin = (A_pred > threshold).astype(np.float32)
    A_true_bin = (A_true > threshold).astype(np.float32)
    
    np.fill_diagonal(A_pred_bin, 0)
    np.fill_diagonal(A_true_bin, 0)
    
    # Confusion matrix elements
    tp = np.sum((A_pred_bin == 1) & (A_true_bin == 1))
    fp = np.sum((A_pred_bin == 1) & (A_true_bin == 0))
    fn = np.sum((A_pred_bin == 0) & (A_true_bin == 1))
    tn = np.sum((A_pred_bin == 0) & (A_true_bin == 0))
    
    # FDR
    if tp + fp > 0:
        fdr = fp / (tp + fp)
    else:
        fdr = 0.0
    
    # TPR (Recall)
    if tp + fn > 0:
        tpr = tp / (tp + fn)
    else:
        tpr = 0.0
    
    return {"fdr": float(fdr), "tpr": float(tpr), "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


def compute_sid(
    A_pred: Union[torch.Tensor, np.ndarray],
    A_true: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.5,
) -> int:
    """
    Compute Structural Intervention Distance (SID).
    
    SID measures the number of interventional distributions
    that would be incorrectly estimated.
    
    Note: This is a simplified implementation. Full SID requires
    computing intervention effects, which is complex.
    Here we use a proxy based on causal ancestor differences.
    
    Args:
        A_pred: Predicted adjacency [d, d]
        A_true: Ground truth adjacency [d, d]
        threshold: Threshold for binarization
        
    Returns:
        sid: Structural Intervention Distance
    """
    if isinstance(A_pred, torch.Tensor):
        A_pred = A_pred.detach().cpu().numpy()
    if isinstance(A_true, torch.Tensor):
        A_true = A_true.detach().cpu().numpy()
    
    A_pred_bin = (A_pred > threshold).astype(np.float32)
    A_true_bin = (A_true > threshold).astype(np.float32)
    
    np.fill_diagonal(A_pred_bin, 0)
    np.fill_diagonal(A_true_bin, 0)
    
    d = A_pred_bin.shape[0]
    
    def get_ancestors(A, node):
        """Get all ancestors of a node via transitive closure."""
        visited = set()
        stack = [node]
        
        while stack:
            n = stack.pop()
            parents = np.where(A[:, n] > 0)[0]
            for p in parents:
                if p not in visited:
                    visited.add(p)
                    stack.append(p)
        
        return visited
    
    # Count mismatched ancestor sets
    sid = 0
    for i in range(d):
        ancestors_pred = get_ancestors(A_pred_bin, i)
        ancestors_true = get_ancestors(A_true_bin, i)
        
        # Symmetric difference
        sid += len(ancestors_pred.symmetric_difference(ancestors_true))
    
    return sid


def compute_all_metrics(
    A_pred: Union[torch.Tensor, np.ndarray],
    A_true: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute all causal discovery metrics.
    
    Args:
        A_pred: Predicted adjacency [d, d]
        A_true: Ground truth adjacency [d, d]
        threshold: Threshold for binarization
        
    Returns:
        dict with all metrics
    """
    metrics = {}
    
    # SHD
    metrics["shd"] = compute_shd(A_pred, A_true, threshold)
    
    # Edge metrics
    edge_metrics = compute_edge_metrics(A_pred, A_true, threshold)
    metrics.update(edge_metrics)
    
    # AUROC/AUPRC
    auc_metrics = compute_auroc_auprc(A_pred, A_true)
    metrics.update(auc_metrics)
    
    # FDR/TPR
    fdr_metrics = compute_fdr_tpr(A_pred, A_true, threshold)
    metrics.update(fdr_metrics)
    
    # SID
    metrics["sid"] = compute_sid(A_pred, A_true, threshold)
    
    # Graph statistics
    if isinstance(A_pred, torch.Tensor):
        A_pred_np = A_pred.detach().cpu().numpy()
    else:
        A_pred_np = A_pred
    
    if isinstance(A_true, torch.Tensor):
        A_true_np = A_true.detach().cpu().numpy()
    else:
        A_true_np = A_true
    
    A_pred_bin = (A_pred_np > threshold).astype(np.float32)
    A_true_bin = (A_true_np > threshold).astype(np.float32)
    np.fill_diagonal(A_pred_bin, 0)
    np.fill_diagonal(A_true_bin, 0)
    
    metrics["n_edges_pred"] = int(A_pred_bin.sum())
    metrics["n_edges_true"] = int(A_true_bin.sum())
    metrics["density_pred"] = float(A_pred_bin.mean())
    metrics["density_true"] = float(A_true_bin.mean())
    
    return metrics


def find_optimal_threshold(
    A_pred: Union[torch.Tensor, np.ndarray],
    A_true: Union[torch.Tensor, np.ndarray],
    metric: str = "f1",
    n_thresholds: int = 100,
) -> Tuple[float, float]:
    """
    Find optimal threshold that maximizes a metric.
    
    Args:
        A_pred: Predicted adjacency (soft) [d, d]
        A_true: Ground truth adjacency [d, d]
        metric: Metric to optimize ("f1", "shd", "auroc")
        n_thresholds: Number of thresholds to try
        
    Returns:
        best_threshold: Optimal threshold
        best_value: Best metric value
    """
    if isinstance(A_pred, torch.Tensor):
        A_pred = A_pred.detach().cpu().numpy()
    if isinstance(A_true, torch.Tensor):
        A_true = A_true.detach().cpu().numpy()
    
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    
    best_threshold = 0.5
    if metric == "shd":
        best_value = float("inf")
    else:
        best_value = -float("inf")
    
    for t in thresholds:
        if metric == "f1":
            value = compute_edge_metrics(A_pred, A_true, t)["f1"]
            if value > best_value:
                best_value = value
                best_threshold = t
        elif metric == "shd":
            value = compute_shd(A_pred, A_true, t)
            if value < best_value:
                best_value = value
                best_threshold = t
        elif metric == "precision":
            value = compute_edge_metrics(A_pred, A_true, t)["precision"]
            if value > best_value:
                best_value = value
                best_threshold = t
        elif metric == "recall":
            value = compute_edge_metrics(A_pred, A_true, t)["recall"]
            if value > best_value:
                best_value = value
                best_threshold = t
    
    return float(best_threshold), float(best_value)


class MetricsTracker:
    """
    Track metrics over training for reporting.
    """
    
    def __init__(self):
        self.history = []
        self.best_metrics = None
        self.best_epoch = None
    
    def update(
        self,
        epoch: int,
        A_pred: Union[torch.Tensor, np.ndarray],
        A_true: Union[torch.Tensor, np.ndarray],
        threshold: float = 0.5,
        extra_metrics: Optional[Dict] = None,
    ):
        """Update with new epoch's metrics."""
        metrics = compute_all_metrics(A_pred, A_true, threshold)
        metrics["epoch"] = epoch
        
        if extra_metrics:
            metrics.update(extra_metrics)
        
        self.history.append(metrics)
        
        # Track best (by F1)
        if self.best_metrics is None or metrics["f1"] > self.best_metrics["f1"]:
            self.best_metrics = metrics.copy()
            self.best_epoch = epoch
    
    def get_summary(self) -> Dict:
        """Get summary of training metrics."""
        if not self.history:
            return {}
        
        return {
            "best_epoch": self.best_epoch,
            "best_f1": self.best_metrics["f1"],
            "best_shd": self.best_metrics["shd"],
            "final_f1": self.history[-1]["f1"],
            "final_shd": self.history[-1]["shd"],
            "n_epochs": len(self.history),
        }
