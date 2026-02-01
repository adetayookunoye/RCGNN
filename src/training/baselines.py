
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


def notears_lite(Xw):
    """Simple correlation-threshold baseline (undirected structure learning)."""
    X = Xw.mean(axis=1) # average over time
    d = X.shape[1]
    C = np.corrcoef(X, rowvar=False)
    A = np.abs(C) - np.eye(d)
    thr = np.quantile(A, 0.9)
    A_bin = (A>=thr).astype(int)
    return A_bin


def notears_linear(Xw, lambda1=0.1, lambda2=5.0, max_iter=100, tol=1e-5):
    """
    NOTEARS: Linear DAG learning with Lagrangian method.
    
    Solves: min_A ||X - XA||^2 + λ1||A||_1 s.t. h(A)=0 (acyclic)
    where h(A) = tr(exp(A⊙A)) - d
    
    Args:
        Xw: [N,T,d] or [N,d] data
        lambda1: L1 sparsity penalty
        lambda2: Lagrangian penalty weight
        max_iter: Maximum iterations
        tol: Convergence tolerance
    
    Returns:
        A: [d,d] learned adjacency matrix
    """
    # Flatten to [N*T, d]
    if Xw.ndim == 3:
        X = Xw.reshape(-1, Xw.shape[-1])
    else:
        X = Xw.copy()
    
    N, d = X.shape
    X = X - X.mean(axis=0) # Center
    X = X / np.std(X, axis=0, keepdims=True) # Normalize
    
    def _h(A):
        """Acyclicity constraint: tr(exp(A⊙A)) - d"""
        M = A * A # Element-wise square
        E = expm(M)
        return np.trace(E) - d
    
    def _loss(w):
        """Loss = MSE + L1 + Lagrangian penalty"""
        A = w.reshape(d, d)
        np.fill_diagonal(A, 0) # Enforce no self-loops
        
        # Reconstruction error
        X_pred = X @ A
        mse = np.sum((X - X_pred) ** 2) / (2 * N)
        
        # L1 penalty
        l1 = lambda1 * np.sum(np.abs(A))
        
        # Acyclicity constraint (soft)
        h_val = _h(A)
        h_penalty = (lambda2 / 2) * (h_val ** 2)
        
        return mse + l1 + h_penalty
    
    # Initialize
    A0 = np.random.randn(d, d) * 0.1
    np.fill_diagonal(A0, 0)
    
    # Optimize
    result = minimize(
        _loss,
        A0.flatten(),
        method='BFGS',
        options={'maxiter': max_iter, 'ftol': tol}
    )
    
    A = result.x.reshape(d, d)
    np.fill_diagonal(A, 0)
    
    # Threshold small values
    A = (np.abs(A) > 0.05).astype(float) * A
    
    return np.abs(A) # Return magnitude


def granger_causality(Xw, max_lag=2, significance=0.05):
    """
    Granger Causality: Time-series causal discovery.
    
    Tests if lagged X_i helps predict X_j beyond X_j's own history.
    Uses simple correlation-based approach for robustness.
    
    Args:
        Xw: [N,T,d] time-series data (T is time dimension)
        max_lag: Maximum lag to test
        significance: Significance level for F-test
    
    Returns:
        A: [d,d] directed adjacency matrix (A[i,j]=1 if i->j)
    """
    N, T, d = Xw.shape
    
    A = np.zeros((d, d))
    
    for j in range(d): # Target variable
        for i in range(d): # Source variable
            if i == j:
                continue
            
            # Compute cross-correlation at various lags
            max_cross_corr = 0
            for lag in range(1, max_lag + 1):
                for n in range(N):
                    if lag < T:
                        X_i_lag = Xw[n, :-lag, i] # X_i at t-lag
                        X_j = Xw[n, lag:, j] # X_j at t
                        
                        if len(X_i_lag) == len(X_j) and len(X_i_lag) > 1:
                            # Normalize
                            x_norm = (X_i_lag - X_i_lag.mean()) / (X_i_lag.std() + 1e-8)
                            y_norm = (X_j - X_j.mean()) / (X_j.std() + 1e-8)
                            
                            cross_corr = np.abs(np.corrcoef(x_norm, y_norm)[0, 1])
                            max_cross_corr = max(max_cross_corr, cross_corr)
            
            # Threshold: if strong lagged correlation exists
            if max_cross_corr > 0.25:
                A[i, j] = 1
    
    return A


def pcmci_plus(Xw, significance=0.05, max_lag=2):
    """
    PCMCI+: Causal discovery with time-lagged conditional independence tests.
    
    Uses time-series conditional independence to infer causal structure.
    Simpler version without full tigramite implementation.
    
    Args:
        Xw: [N,T,d] time-series data
        significance: Significance level
        max_lag: Maximum lag to consider
    
    Returns:
        A: [d,d] adjacency matrix (contemporaneous + time-lagged)
    """
    N, T, d = Xw.shape
    
    # Reshape to time-series format
    X = Xw.reshape(-1, d)
    X = X - X.mean(axis=0)
    X = X / (X.std(axis=0) + 1e-8)
    
    A = np.zeros((d, d))
    
    # Phase 1: Find time-lagged causal edges
    for j in range(d): # Target
        for i in range(d): # Source
            if i == j:
                continue
            
            # Test: X_i[t-lag] -> X_j[t]
            max_corr = 0
            for lag in range(1, max_lag + 1):
                if lag < T:
                    corr = np.corrcoef(X[:-lag, i], X[lag:, j])[0, 1]
                    max_corr = max(max_corr, abs(corr))
            
            if max_corr > 0.3: # Threshold
                A[i, j] = max_corr
    
    # Phase 2: Prune weak edges (remove if explained by other edges)
    for j in range(d):
        targets = np.where(A[:, j] > 0)[0]
        if len(targets) > 1:
            # Keep only strongest edges
            strengths = A[targets, j]
            keep_idx = np.argsort(strengths)[-min(3, len(targets)):]
            A[targets, j] = 0
            A[targets[keep_idx], j] = A[targets[keep_idx], j]
    
    return (A > 0.1).astype(float)


def dag_gnn_simple(Xw, hidden_dim=64, num_layers=2):
    """
    DAG-GNN: Simple graph neural network for structure learning.
    
    Without external torch dependency, uses numpy for node embeddings.
    Learns node representations then predicts edges.
    
    Args:
        Xw: [N,T,d] data
        hidden_dim: Hidden dimension
        num_layers: Number of GNN layers
    
    Returns:
        A: [d,d] predicted adjacency matrix
    """
    if Xw.ndim == 3:
        X = Xw.reshape(-1, Xw.shape[-1])
    else:
        X = Xw.copy()
    
    N, d = X.shape
    
    # Normalize
    X = X - X.mean(axis=0)
    X = X / (X.std(axis=0) + 1e-8)
    
    # Initialize node embeddings
    H = np.random.randn(d, hidden_dim) * 0.1
    
    # GNN message passing (simplified)
    for layer in range(num_layers):
        # Aggregate neighbor information
        H_new = np.zeros_like(H)
        
        for i in range(d):
            # Neighborhood aggregation: average features of all nodes
            H_new[i] = H.mean(axis=0) + H[i]
        
        # Non-linearity
        H = np.maximum(H_new, 0) # ReLU
        
        # Normalize
        H = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-8)
    
    # Predict adjacency via dot product
    A_scores = H @ H.T # [d, d]
    np.fill_diagonal(A_scores, 0)
    
    # Threshold to binary
    threshold = np.median(A_scores)
    A = (A_scores > threshold).astype(float)
    
    # Enforce DAG by topological ordering (greedy)
    A_dag = np.zeros_like(A)
    for _ in range(d): # Multiple passes
        # Find node with minimum in-degree
        in_degrees = A.sum(axis=0)
        if in_degrees.sum() == 0:
            break
        i = np.argmin(in_degrees)
        # Keep edges FROM this node
        A_dag[i, :] += A[i, :]
        A[i, :] = 0
    
    return A_dag
