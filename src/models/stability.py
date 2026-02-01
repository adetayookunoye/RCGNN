"""Stability metrics for evaluating model robustness.

This module provides metrics to assess model stability across different types of 
corruptions and perturbations:
1. Missingness stability - how structure learning varies with missing rate
2. SNR stability - robustness to different noise levels
3. Distribution drift - stability under covariate shift
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from sklearn.metrics import jaccard_score

def edge_stability_curve(
    model: torch.nn.Module,
    data_generator: callable,
    corruption_range: List[float],
    corruption_type: str = 'missingness',
    n_samples: int = 100,
    n_repeats: int = 5
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Compute stability curves for edge recovery under increasing corruption.
    
    Args:
        model: Trained RC-GNN model
        data_generator: Function that generates data with specified corruption
        corruption_range: List of corruption levels to test (e.g. missing rates)
        corruption_type: Type of corruption ('missingness', 'snr', or 'drift')
        n_samples: Number of samples per corruption level
        n_repeats: Number of repeats for confidence intervals
        
    Returns:
        curves: Dictionary with keys:
            - 'mean': Mean stability score at each corruption level
            - 'std': Standard deviation of stability scores
            - 'levels': Corruption levels tested
        metrics: Aggregate stability metrics including:
            - 'auc': Area under stability curve
            - 'critical_level': Corruption level where stability drops below 0.5
            - 'relative_degradation': Rate of stability loss vs corruption
    """
    curves = {
        'mean': [],
        'std': [],
        'levels': corruption_range
    }
    
    # Track edges recovered at each corruption level
    edges_by_level = []
    
    with torch.no_grad():
        # Get baseline edges with no corruption
        clean_data = data_generator(0.0, n_samples)
        clean_adj = get_model_adjacency(model, clean_data)
        baseline_edges = adjacency_to_edges(clean_adj)
        
        # Test each corruption level
        for level in corruption_range:
            level_scores = []
            
            for _ in range(n_repeats):
                # Generate corrupted data
                corrupt_data = data_generator(level, n_samples)
                corrupt_adj = get_model_adjacency(model, corrupt_data)
                corrupt_edges = adjacency_to_edges(corrupt_adj)
                
                # Compute stability metrics
                stability = compute_edge_stability(baseline_edges, corrupt_edges)
                level_scores.append(stability)
                
            # Record statistics
            curves['mean'].append(np.mean(level_scores))
            curves['std'].append(np.std(level_scores))
            edges_by_level.append(corrupt_edges)
            
    # Convert to arrays
    curves['mean'] = np.array(curves['mean'])
    curves['std'] = np.array(curves['std'])
    
    # Compute aggregate metrics
    metrics = compute_aggregate_metrics(
        curves['mean'], 
        curves['levels'],
        edges_by_level,
        baseline_edges
    )
    
    return curves, metrics

def temporal_stability_curve(
    model: torch.nn.Module,
    data_generator: callable,
    drift_range: List[float],
    n_timesteps: int = 20,
    n_repeats: int = 5
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Compute stability curves for temporal structure under distribution drift.
    
    Specifically examines how well temporal dependencies are preserved under
    increasing distribution shift.
    
    Args:
        model: Trained RC-GNN model
        data_generator: Function generating temporally dependent data
        drift_range: List of drift magnitudes to test
        n_timesteps: Number of timesteps to generate
        n_repeats: Number of repeats for confidence intervals
        
    Returns:
        curves: Dictionary with stability curves
        metrics: Aggregate temporal stability metrics
    """
    curves = {
        'mean': [],
        'std': [],
        'levels': drift_range
    }
    
    with torch.no_grad():
        # Get baseline temporal structure
        clean_data = data_generator(0.0, n_timesteps)
        clean_adj_list = get_model_temporal_adjacency(model, clean_data)
        baseline_temporal = temporal_adjacency_to_edges(clean_adj_list)
        
        # Test each drift level
        for drift in drift_range:
            drift_scores = []
            
            for _ in range(n_repeats):
                # Generate data with drift
                drift_data = data_generator(drift, n_timesteps)
                drift_adj_list = get_model_temporal_adjacency(model, drift_data)
                drift_temporal = temporal_adjacency_to_edges(drift_adj_list)
                
                # Compute temporal stability
                stability = compute_temporal_stability(
                    baseline_temporal,
                    drift_temporal
                )
                drift_scores.append(stability)
                
            curves['mean'].append(np.mean(drift_scores))
            curves['std'].append(np.std(drift_scores))
            
    # Convert to arrays
    curves['mean'] = np.array(curves['mean'])
    curves['std'] = np.array(curves['std'])
    
    # Compute temporal stability metrics
    metrics = compute_temporal_metrics(
        curves['mean'],
        curves['levels']
    )
    
    return curves, metrics

def compute_edge_stability(
    edges1: List[Tuple[int, int]],
    edges2: List[Tuple[int, int]]
) -> float:
    """Compute stability between two edge sets using Jaccard similarity.
    
    Args:
        edges1: List of (source, target) edge tuples
        edges2: List of (source, target) edge tuples
        
    Returns:
        Jaccard similarity between edge sets
    """
    set1 = set(edges1)
    set2 = set(edges2)
    return len(set1.intersection(set2)) / len(set1.union(set2))

def compute_temporal_stability(
    temporal1: List[List[Tuple[int, int]]],
    temporal2: List[List[Tuple[int, int]]]
) -> float:
    """Compute stability between two temporal edge structures.
    
    Considers both edge preservation and lag consistency.
    
    Args:
        temporal1: List of edge sets per lag
        temporal2: List of edge sets per lag
        
    Returns:
        Combined temporal stability score
    """
    n_lags = len(temporal1)
    lag_scores = []
    
    for l in range(n_lags):
        # Compare edges at each lag
        lag_score = compute_edge_stability(temporal1[l], temporal2[l])
        # Weight earlier lags more heavily
        weight = np.exp(-l) # Exponential decay
        lag_scores.append(weight * lag_score)
        
    return np.sum(lag_scores) / np.sum([np.exp(-l) for l in range(n_lags)])

def compute_aggregate_metrics(
    stability_curve: np.ndarray,
    corruption_levels: np.ndarray,
    edges_by_level: List[List[Tuple[int, int]]],
    baseline_edges: List[Tuple[int, int]]
) -> Dict[str, float]:
    """Compute aggregate stability metrics from curves.
    
    Args:
        stability_curve: Array of stability scores
        corruption_levels: Array of corruption levels
        edges_by_level: List of edge sets for each level
        baseline_edges: Edge set with no corruption
        
    Returns:
        Dictionary of stability metrics
    """
    # Area under stability curve (normalized)
    auc = np.trapz(stability_curve, corruption_levels)
    auc_norm = auc / (corruption_levels[-1] - corruption_levels[0])
    
    # Find critical level where stability drops below 0.5
    above_thresh = stability_curve >= 0.5
    if not any(above_thresh):
        critical_level = corruption_levels[0]
    elif all(above_thresh):
        critical_level = corruption_levels[-1]
    else:
        idx = np.where(above_thresh)[0][-1]
        critical_level = corruption_levels[idx]
    
    # Compute rate of stability degradation
    deg_rate = np.polyfit(corruption_levels, stability_curve, deg=1)[0]
    
    # Edge set consistency
    edge_consistent = np.mean([
        compute_edge_stability(edges, baseline_edges)
        for edges in edges_by_level
    ])
    
    return {
        'auc': auc_norm,
        'critical_level': critical_level,
        'degradation_rate': deg_rate,
        'edge_consistency': edge_consistent
    }

def compute_temporal_metrics(
    stability_curve: np.ndarray,
    drift_levels: np.ndarray
) -> Dict[str, float]:
    """Compute aggregate metrics for temporal stability.
    
    Args:
        stability_curve: Array of temporal stability scores
        drift_levels: Array of drift magnitudes
        
    Returns:
        Dictionary of temporal stability metrics
    """
    # Area under temporal stability curve
    auc = np.trapz(stability_curve, drift_levels)
    auc_norm = auc / (drift_levels[-1] - drift_levels[0])
    
    # Critical drift level
    above_thresh = stability_curve >= 0.5
    if not any(above_thresh):
        critical_drift = drift_levels[0]
    elif all(above_thresh):
        critical_drift = drift_levels[-1]
    else:
        idx = np.where(above_thresh)[0][-1]
        critical_drift = drift_levels[idx]
    
    # Rate of temporal stability loss
    deg_rate = np.polyfit(drift_levels, stability_curve, deg=1)[0]
    
    return {
        'temporal_auc': auc_norm,
        'critical_drift': critical_drift,
        'temporal_degradation': deg_rate
    }

def adjacency_to_edges(adj: torch.Tensor) -> List[Tuple[int, int]]:
    """Convert adjacency matrix to list of edges.
    
    Args:
        adj: Adjacency matrix [d,d] or [B,d,d]
        
    Returns:
        List of (source, target) edge tuples
    """
    if adj.dim() == 3:
        adj = adj.mean(0) # Average over batch
        
    edges = []
    d = adj.shape[0]
    
    # Threshold for edge presence
    thresh = 0.5
    
    for i in range(d):
        for j in range(d):
            if adj[i,j] > thresh:
                edges.append((i,j))
                
    return edges

def temporal_adjacency_to_edges(
    adj_list: List[torch.Tensor]
) -> List[List[Tuple[int, int]]]:
    """Convert list of temporal adjacency matrices to edge sets.
    
    Args:
        adj_list: List of adjacency matrices [A^(1), ..., A^(L)]
        
    Returns:
        List of edge sets per lag
    """
    return [adjacency_to_edges(A) for A in adj_list]

# Helper functions for model inference
def get_model_adjacency(model, data):
    """Get adjacency matrix from model predictions."""
    with torch.no_grad():
        outputs = model(data)
        return outputs['adjacency']

def get_model_temporal_adjacency(model, data):
    """Get temporal adjacency matrices from model."""
    with torch.no_grad():
        outputs = model(data)
        return outputs['temporal_adjacency']