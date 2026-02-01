import pytest
import torch
import numpy as np
from src.models.stability import (
    compute_edge_stability,
    compute_temporal_stability,
    compute_aggregate_metrics,
    compute_temporal_metrics,
    edge_stability_curve,
    temporal_stability_curve
)

class MockModel(torch.nn.Module):
    def __init__(self, d=5, n_lags=3):
        super().__init__()
        self.d = d
        self.n_lags = n_lags
        
    def forward(self, x):
        # Return random adjacency for testing
        if hasattr(self, 'temporal'):
            return {
                'temporal_adjacency': [
                    torch.rand(self.d, self.d) for _ in range(self.n_lags)
                ]
            }
        return {'adjacency': torch.rand(self.d, self.d)}

def mock_data_generator(corruption_level, n_samples):
    """Generate mock data with corruption."""
    return torch.randn(n_samples, 10, 5) * (1 + corruption_level)

@pytest.fixture
def sample_edges():
    """Generate sample edge sets."""
    edges1 = [(0,1), (1,2), (2,3)]
    edges2 = [(0,1), (1,2), (1,3)]
    return edges1, edges2

@pytest.fixture
def sample_temporal():
    """Generate sample temporal edge sets."""
    temporal1 = [
        [(0,1), (1,2)],
        [(1,2), (2,3)],
        [(0,2)]
    ]
    temporal2 = [
        [(0,1), (1,2)],
        [(1,2), (1,3)],
        [(0,2)]
    ]
    return temporal1, temporal2

def test_edge_stability():
    """Test edge stability computation."""
    edges1 = [(0,1), (1,2), (2,3)]
    edges2 = [(0,1), (1,2), (1,3)]
    
    stability = compute_edge_stability(edges1, edges2)
    assert 0 <= stability <= 1
    
    # Perfect match
    assert compute_edge_stability(edges1, edges1) == 1.0
    
    # No overlap
    edges3 = [(3,4), (4,5)]
    assert compute_edge_stability(edges1, edges3) == 0.0

def test_temporal_stability(sample_temporal):
    """Test temporal stability computation."""
    temporal1, temporal2 = sample_temporal
    
    stability = compute_temporal_stability(temporal1, temporal2)
    assert 0 <= stability <= 1
    
    # Perfect match
    assert compute_temporal_stability(temporal1, temporal1) == 1.0
    
    # Earlier lags should matter more
    temporal3 = [
        temporal1[0], # Same first lag
        [(3,4), (4,5)], # Different later lags
        [(4,5)]
    ]
    stability_early_match = compute_temporal_stability(temporal1, temporal3)
    
    temporal4 = [
        [(3,4), (4,5)], # Different first lag
        temporal1[1], # Same later lags
        temporal1[2]
    ]
    stability_late_match = compute_temporal_stability(temporal1, temporal4)
    
    assert stability_early_match > stability_late_match

def test_aggregate_metrics(sample_edges):
    """Test computation of aggregate stability metrics."""
    stability_curve = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
    corruption_levels = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
    
    edges1, edges2 = sample_edges
    edges_by_level = [edges1, edges2] * 2 + [edges1]
    
    metrics = compute_aggregate_metrics(
        stability_curve,
        corruption_levels,
        edges_by_level,
        edges1
    )
    
    assert 'auc' in metrics
    assert 'critical_level' in metrics
    assert 'degradation_rate' in metrics
    assert metrics['auc'] > 0
    assert metrics['degradation_rate'] < 0 # Should decrease

def test_temporal_metrics():
    """Test computation of temporal stability metrics."""
    stability_curve = np.array([1.0, 0.9, 0.7, 0.5, 0.3])
    drift_levels = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    
    metrics = compute_temporal_metrics(stability_curve, drift_levels)
    
    assert 'temporal_auc' in metrics
    assert 'critical_drift' in metrics
    assert 'temporal_degradation' in metrics
    assert metrics['temporal_auc'] > 0
    assert metrics['temporal_degradation'] < 0

def test_stability_curves():
    """Test generation of stability curves."""
    model = MockModel()
    corruption_range = [0.0, 0.2, 0.4]
    
    # Edge stability
    curves, metrics = edge_stability_curve(
        model,
        mock_data_generator,
        corruption_range,
        n_samples=10,
        n_repeats=2
    )
    
    assert 'mean' in curves
    assert 'std' in curves
    assert 'levels' in curves
    assert curves['mean'].shape == (len(corruption_range),)
    assert all(0 <= s <= 1 for s in curves['mean'])
    
    # Temporal stability
    model.temporal = True
    curves, metrics = temporal_stability_curve(
        model,
        mock_data_generator,
        corruption_range,
        n_timesteps=10,
        n_repeats=2
    )
    
    assert 'mean' in curves
    assert 'std' in curves
    assert curves['mean'].shape == (len(corruption_range),)
    assert all(0 <= s <= 1 for s in curves['mean'])
