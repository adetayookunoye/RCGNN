import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def evaluate_missingness(M_pred, M_true, threshold=0.5):
    """Evaluate missingness prediction performance.
    
    Args:
        M_pred: Predicted missingness probabilities [B,T,d] or [T,d]
        M_true: Ground truth missingness mask [B,T,d] or [T,d]
        threshold: Classification threshold for binary predictions
        
    Returns:
        metrics: Dictionary containing:
            - accuracy: Binary classification accuracy
            - auroc: Area under ROC curve
            - auprc: Area under PR curve
            - f1: F1 score
            - self_mask_rate: Proportion of self-masked values
            - temporal_consistency: Temporal smoothness score
    """
    if M_pred.dim() == 2:
        M_pred = M_pred.unsqueeze(0)
        M_true = M_true.unsqueeze(0)
        
    # Move to CPU for sklearn metrics
    M_pred_np = M_pred.detach().cpu().numpy()
    M_true_np = M_true.detach().cpu().numpy()
    
    # Flatten for binary metrics
    M_pred_flat = M_pred_np.reshape(-1)
    M_true_flat = M_true_np.reshape(-1)
    
    # Binary predictions
    M_pred_binary = (M_pred_flat > threshold).astype(np.float32)
    
    # Calculate metrics
    metrics = {}
    
    # Classification metrics
    metrics['accuracy'] = np.mean(M_pred_binary == M_true_flat)
    metrics['auroc'] = roc_auc_score(M_true_flat, M_pred_flat)
    metrics['auprc'] = average_precision_score(M_true_flat, M_pred_flat)
    
    # F1 score
    tp = np.sum((M_pred_binary == 1) & (M_true_flat == 1))
    fp = np.sum((M_pred_binary == 1) & (M_true_flat == 0))
    fn = np.sum((M_pred_binary == 0) & (M_true_flat == 1))
    metrics['f1'] = 2*tp / (2*tp + fp + fn)
    
    # Self-masking rate (per feature)
    metrics['self_mask_rate'] = np.mean(M_pred_np, axis=(0,1))
    
    # Temporal consistency 
    temp_diff = np.diff(M_pred_np, axis=1)
    metrics['temporal_consistency'] = 1.0 - np.mean(np.abs(temp_diff))
    
    return metrics

def missingness_pattern_analysis(M_pred, X, feature_names=None):
    """Analyze learned missingness patterns.
    
    Args:
        M_pred: Predicted missingness [B,T,d]
        X: Input data [B,T,d]
        feature_names: Optional list of feature names
        
    Returns:
        patterns: Dictionary of discovered patterns
    """
    if M_pred.dim() == 2:
        M_pred = M_pred.unsqueeze(0)
        X = X.unsqueeze(0)
        
    patterns = {}
    
    # Value-dependent missingness
    value_corr = torch.zeros(M_pred.shape[-1])
    for j in range(M_pred.shape[-1]):
        corr = torch.corrcoef(torch.stack([
            X[...,j].flatten(),
            M_pred[...,j].flatten()
        ]))[0,1]
        value_corr[j] = corr
        
    patterns['value_dependencies'] = value_corr
    
    # Cross-feature missingness correlation
    cross_corr = torch.corrcoef(M_pred.transpose(1,2).reshape(-1, M_pred.shape[-1]))
    patterns['cross_feature_corr'] = cross_corr
    
    # Temporal patterns
    temp_autocorr = torch.zeros(M_pred.shape[-1])
    for j in range(M_pred.shape[-1]):
        series = M_pred[...,j].mean(0) # Average over batch
        temp_autocorr[j] = torch.corrcoef(torch.stack([
            series[:-1], series[1:]
        ]))[0,1]
    patterns['temporal_autocorr'] = temp_autocorr
    
    if feature_names is not None:
        patterns['feature_names'] = feature_names
        
    return patterns