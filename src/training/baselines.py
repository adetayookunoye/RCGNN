"""
Baseline Methods for Causal Structure Learning.

APPROPRIATE METHODS (for instantaneous/IID SEM data):
  - Classical constraint-based: PC
  - Score-based: GES-like greedy BIC, NOTEARS-penalty (single-shot Lagrangian)
  - Neural (optional): NOTEARS-MLP, GOLEM, GraN-DAG

NOT APPROPRIATE (temporal methods - DISABLED with RuntimeError):
  - granger_causality: assumes time-series with lagged dependencies
  - pcmci_plus: assumes time-series with lagged dependencies

IMPORTANT: All methods accept (Xw, Mw) where Mw is a missingness mask.
Missing entries (M=0) are mean-imputed before running any baseline.
Without this, zeros from X*M corrupt statistics.

OUTPUT TYPES:
  - PC: CPDAG adjacency (use skeleton + oriented_edges helpers)
  - GES-like: Binary DAG (our greedy BIC search, NOT full CPDAG GES)
  - NOTEARS, GOLEM, GraN-DAG: Edge scores (AUROC/AUPRC supported)
  - correlation_scores: Continuous |corr| matrix (AUROC/AUPRC supported)
  - correlation_threshold: Binary thresholded graph (F1/SHD meaningful)
  - DAG-GNN-inspired: Neural baseline (our impl, not canonical DAG-GNN)

For our benchmark, use ONLY the appropriate methods since data is IID.
"""

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
from scipy import stats
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# Check for optional dependencies
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    warnings.warn("PyTorch not found; neural baselines will use numpy fallback")


# ============================================================================
# UNIFIED PREPROCESSING (IDENTICAL FOR ALL BASELINES - REVIEWER-PROOF)
# ============================================================================

def preprocess_for_baseline(Xw, Mw=None, standardize=True):
    """
    Unified preprocessing for ALL baselines (ensures fairness).
    
    Steps (applied identically to all methods):
      1. Flatten to 2D if needed
      2. Mean-impute missing values (using observed entries only)
      3. Optionally standardize (zero mean, unit variance)
    
    CRITICAL: This ensures RC-GNN doesn't get "special treatment" - 
    all baselines see identically preprocessed data.
    
    Args:
        Xw: [N,T,d] or [N,d] or [T,d] data
        Mw: Same shape as Xw, 1=observed, 0=missing. If None, no imputation.
        standardize: If True, standardize columns after imputation
    
    Returns:
        X2: [N*T, d] or [N, d] preprocessed data (2D)
    """
    X2 = impute_with_mask(Xw, Mw)
    if standardize:
        X2 = _standardize(X2)
    return X2


def impute_with_mask(Xw, Mw=None):
    """
    Mean-impute missing entries using mask M.
    
    CRITICAL: Without this, baselines see zeros as real values,
    causing spurious correlations and wrong CI tests.
    
    Args:
        Xw: [N,T,d] or [N,d] or [T,d] data
        Mw: Same shape as Xw, 1=observed, 0=missing. If None, no imputation.
             MUST be binary {0,1} if provided.
    
    Returns:
        X2: [N*T, d] or [N, d] flattened and imputed data (2D)
        
    Raises:
        ValueError: If Mw is not binary {0,1}
    """
    X = Xw.copy()
    
    if Mw is None:
        # No mask, just flatten
        if X.ndim == 3:
            return X.reshape(-1, X.shape[-1])
        return X
    
    # FIX #6: Validate mask is binary {0,1}
    if not np.isin(Mw, [0, 1]).all():
        raise ValueError(
            "Mw must be binary {0,1} mask. Got values outside [0,1]. "
            "If using probability masks, threshold first."
        )
    
    # Flatten both to 2D
    if X.ndim == 3:
        X2 = X.reshape(-1, X.shape[-1])
        M2 = Mw.reshape(-1, Mw.shape[-1])
    else:
        X2 = X.copy()
        M2 = Mw.copy()
    
    # Column means over observed entries only
    col_sum = (X2 * M2).sum(axis=0)
    col_cnt = np.maximum(M2.sum(axis=0), 1.0)
    col_mean = col_sum / col_cnt
    
    # Impute where missing
    missing = (M2 <= 0)
    if missing.any():
        # For each missing entry, impute with column mean
        for j in range(X2.shape[1]):
            X2[missing[:, j], j] = col_mean[j]
    
    return X2


def _standardize(X):
    """Standardize columns to zero mean, unit variance."""
    return (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)


# ============================================================================
# BASELINE RUN LOGGING (REVIEWER-PROOF EVIDENCE)
# ============================================================================

def log_baseline_banner(method_name, mode="per_env", has_mask=True, temporal_disabled=True):
    """
    Print a one-line banner for baseline run (becomes rebuttal evidence).
    
    Example output:
      [BASELINE] PC | DATA=IID_SEM | mode=per_env | imputation=mask_mean | temporal=DISABLED
    """
    impute_str = "mask_mean" if has_mask else "none"
    temporal_str = "DISABLED" if temporal_disabled else "ENABLED"
    print(f"[BASELINE] {method_name} | DATA=IID_SEM | mode={mode} | "
          f"imputation={impute_str} | temporal={temporal_str}")


def assert_no_temporal_structure(config=None, strict=True):
    """
    Hard guard: refuse to run baselines if temporal_structure is enabled.
    
    This prevents accidental misuse of IID baselines on temporal data.
    
    Args:
        config: Dict with optional "temporal_structure" key
        strict: If True, raise RuntimeError; if False, just warn
        
    Raises:
        RuntimeError if temporal_structure is True and strict=True
    """
    if config is None:
        return  # No config, assume IID
    
    temporal = config.get("temporal_structure", False)
    if temporal:
        msg = ("FATAL: temporal_structure=True in config. "
               "IID baselines (PC, GES, NOTEARS, etc.) are NOT appropriate for "
               "time-series data with lagged dependencies. Use Granger/PCMCI instead, "
               "or set temporal_structure=False.")
        if strict:
            raise RuntimeError(msg)
        else:
            warnings.warn(msg)


# ============================================================================
# FAST DAG CHECKING (replaces expensive expm-based check)
# ============================================================================

def _has_cycle_dfs(A):
    """
    Check if adjacency matrix A has a cycle using DFS.
    
    O(d + e) instead of O(d³) for expm.
    
    Args:
        A: [d,d] adjacency matrix (A[i,j]>0 means i->j)
    
    Returns:
        True if cycle exists, False if DAG
    """
    d = A.shape[0]
    A_bin = (np.abs(A) > 1e-10)
    
    # 0=unvisited, 1=in current path, 2=finished
    state = np.zeros(d, dtype=int)
    
    def dfs(node):
        if state[node] == 1:
            return True  # Back edge = cycle
        if state[node] == 2:
            return False  # Already processed
        
        state[node] = 1  # Mark as in current path
        for child in np.where(A_bin[node, :])[0]:
            if dfs(child):
                return True
        state[node] = 2  # Mark as finished
        return False
    
    for node in range(d):
        if state[node] == 0:
            if dfs(node):
                return True
    return False


def _is_dag(A):
    """Check if A is a DAG (no cycles). Fast DFS-based."""
    return not _has_cycle_dfs(A)


# ============================================================================
# CPDAG HELPERS (for PC algorithm output)
# ============================================================================

def cpdag_to_skeleton(A):
    """
    Convert CPDAG to undirected skeleton.
    
    Edge is present if either A[i,j]>0 or A[j,i]>0.
    Use for skeleton-based metrics (Skeleton F1).
    
    Args:
        A: [d,d] CPDAG adjacency (may have bidirectional edges)
    
    Returns:
        S: [d,d] symmetric skeleton (undirected)
    """
    S = ((A + A.T) > 0).astype(int)
    np.fill_diagonal(S, 0)
    return S


def cpdag_oriented_edges(A):
    """
    Extract only oriented (directed) edges from CPDAG.
    
    An edge i->j is oriented iff A[i,j]>0 AND A[j,i]==0.
    Undirected edges (A[i,j]>0 AND A[j,i]>0) are excluded.
    Use for direction-sensitive metrics (Directed F1).
    
    Args:
        A: [d,d] CPDAG adjacency
    
    Returns:
        D: [d,d] only oriented edges (asymmetric)
    """
    D = ((A > 0) & (A.T == 0)).astype(int)
    np.fill_diagonal(D, 0)
    return D


# ============================================================================
# CLASSICAL CONSTRAINT-BASED METHODS
# ============================================================================

def pc_algorithm(Xw, Mw=None, alpha=0.05, max_cond_size=3):
    """
    PC Algorithm: Constraint-based causal discovery.
    
    Learns DAG skeleton via conditional independence tests, then orients edges.
    Uses Fisher's z-test for conditional independence.
    
    OUTPUT TYPE: CPDAG adjacency matrix
      - Some edges may be undirected (A[i,j]=1 AND A[j,i]=1)
      - Use cpdag_to_skeleton(A) for undirected skeleton metrics
      - Use cpdag_oriented_edges(A) for directed metrics (only count oriented edges)
      - AUROC/AUPRC not meaningful (binary output)
    
    Reference: Spirtes, Glymour, Scheines (2000) "Causation, Prediction, and Search"
    
    Args:
        Xw: [N,T,d] or [N,d] data (T dimension is flattened as additional samples)
        Mw: Missingness mask, same shape as Xw. 1=observed, 0=missing.
        alpha: Significance level for independence tests
        max_cond_size: Maximum conditioning set size
    
    Returns:
        A: [d,d] CPDAG adjacency matrix (may have bidirectional edges)
    """
    # Use unified preprocessing (Fix #3)
    X = preprocess_for_baseline(Xw, Mw, standardize=True)
    N, d = X.shape
    
    # Compute correlation matrix
    C = np.corrcoef(X, rowvar=False)
    
    def partial_corr(i, j, cond_set):
        """Compute partial correlation of X_i and X_j given X_cond_set."""
        if len(cond_set) == 0:
            return C[i, j]
        
        idx = list(cond_set)
        X_cond = X[:, idx]
        
        # Add intercept
        X_cond_aug = np.column_stack([np.ones(N), X_cond])
        
        # Regress X_i on conditioning set
        beta_i = np.linalg.lstsq(X_cond_aug, X[:, i], rcond=None)[0]
        resid_i = X[:, i] - X_cond_aug @ beta_i
        
        # Regress X_j on conditioning set
        beta_j = np.linalg.lstsq(X_cond_aug, X[:, j], rcond=None)[0]
        resid_j = X[:, j] - X_cond_aug @ beta_j
        
        # Correlation of residuals
        if resid_i.std() < 1e-8 or resid_j.std() < 1e-8:
            return 0.0
        return np.corrcoef(resid_i, resid_j)[0, 1]
    
    def fisher_z_test(r, n, k):
        """Fisher's z-test for partial correlation."""
        # Guard: need n - k - 3 > 0 for valid test
        if n - k - 3 <= 0:
            return 1.0  # Can't reject independence
        if abs(r) >= 1:
            return 0.0  # Perfectly correlated, not independent
        z = 0.5 * np.log((1 + r) / (1 - r))
        se = 1.0 / np.sqrt(n - k - 3)
        p_value = 2 * (1 - stats.norm.cdf(abs(z / se)))
        return p_value
    
    # Phase 1: Skeleton learning (find undirected edges)
    skeleton = np.ones((d, d)) - np.eye(d)
    sep_sets = {}
    
    for cond_size in range(max_cond_size + 1):
        for i in range(d):
            for j in range(i + 1, d):
                if skeleton[i, j] == 0:
                    continue
                
                # Use neighbors of i only (standard PC, not union)
                neighbors_i = set(np.where(skeleton[i, :] > 0)[0]) - {j}
                
                if len(neighbors_i) < cond_size:
                    continue
                
                # Test all conditioning sets of size cond_size
                for cond_set in combinations(neighbors_i, cond_size):
                    cond_set = set(cond_set)
                    r = partial_corr(i, j, cond_set)
                    p_value = fisher_z_test(r, N, len(cond_set))
                    
                    if p_value > alpha:  # Independent given cond_set
                        skeleton[i, j] = 0
                        skeleton[j, i] = 0
                        sep_sets[(i, j)] = cond_set
                        sep_sets[(j, i)] = cond_set
                        break
    
    # Phase 2: Orient v-structures (colliders)
    A = skeleton.copy()
    
    for k in range(d):
        neighbors_k = np.where(skeleton[k, :] > 0)[0]
        for i, j in combinations(neighbors_k, 2):
            if skeleton[i, j] > 0:
                continue
            
            sep = sep_sets.get((i, j), sep_sets.get((j, i), set()))
            if k not in sep:
                A[i, k] = 1
                A[k, i] = 0
                A[j, k] = 1
                A[k, j] = 0
    
    # Phase 3: Apply Meek's rules
    changed = True
    while changed:
        changed = False
        for i in range(d):
            for j in range(d):
                if A[i, j] > 0 and A[j, i] == 0:
                    for k in range(d):
                        if k == i or k == j:
                            continue
                        if A[j, k] > 0 and A[k, j] > 0:
                            if A[i, k] == 0 and A[k, i] == 0:
                                A[j, k] = 1
                                A[k, j] = 0
                                changed = True
    
    return A


def ges_algorithm(Xw, Mw=None, max_iter=100):
    """
    GES-like Greedy BIC DAG Search (forward-backward).
    
    DISCLAIMER (for paper):
      "We implement a greedy BIC DAG search in the spirit of GES (forward-backward),
       using a fast DFS-based cycle check. This is NOT full equivalence-class GES
       over CPDAGs, but a practical approximation for DAG recovery."
    
    OUTPUT TYPE: Binary DAG adjacency
      - Returns a DAG (not CPDAG equivalence class)
      - AUROC/AUPRC not meaningful (binary output)
    
    Reference: Chickering (2002) "Optimal Structure Identification With Greedy Search"
    
    Args:
        Xw: [N,T,d] or [N,d] data
        Mw: Missingness mask
        max_iter: Maximum iterations per phase
    
    Returns:
        A: [d,d] binary DAG adjacency matrix
    """
    # Use unified preprocessing (Fix #3)
    X = preprocess_for_baseline(Xw, Mw, standardize=True)
    N, d = X.shape
    
    def local_score(j, parents):
        """BIC score for node j given parents."""
        if len(parents) == 0:
            residual = X[:, j]
        else:
            X_pa = X[:, list(parents)]
            beta = np.linalg.lstsq(X_pa, X[:, j], rcond=None)[0]
            residual = X[:, j] - X_pa @ beta
        
        rss = np.sum(residual ** 2)
        k = len(parents) + 1
        bic = N * np.log(rss / N + 1e-10) + k * np.log(N)
        return -bic
    
    # Initialize empty graph
    A = np.zeros((d, d))
    
    # Forward phase: add edges
    for _ in range(max_iter):
        best_gain = 0
        best_edge = None
        
        for i in range(d):
            for j in range(d):
                if i == j or A[i, j] > 0:
                    continue
                
                parents_j = set(np.where(A[:, j] > 0)[0])
                old_score = local_score(j, parents_j)
                new_score = local_score(j, parents_j | {i})
                gain = new_score - old_score
                
                # Fast cycle check
                A_test = A.copy()
                A_test[i, j] = 1
                if gain > best_gain and _is_dag(A_test):
                    best_gain = gain
                    best_edge = (i, j)
        
        if best_edge is None:
            break
        A[best_edge[0], best_edge[1]] = 1
    
    # Backward phase: remove edges
    for _ in range(max_iter):
        best_gain = 0
        best_edge = None
        
        for i in range(d):
            for j in range(d):
                if A[i, j] == 0:
                    continue
                
                parents_j = set(np.where(A[:, j] > 0)[0])
                old_score = local_score(j, parents_j)
                new_score = local_score(j, parents_j - {i})
                gain = new_score - old_score
                
                if gain > best_gain:
                    best_gain = gain
                    best_edge = (i, j)
        
        if best_edge is None:
            break
        A[best_edge[0], best_edge[1]] = 0
    
    return A


# ============================================================================
# SCORE-BASED METHODS
# ============================================================================

def correlation_scores(Xw, Mw=None):
    """
    Correlation scores baseline: continuous |correlation| matrix.
    
    OUTPUT TYPE: Continuous edge scores (AUROC/AUPRC meaningful)
    
    Returns absolute correlation as edge scores (no thresholding).
    Use with TopK selection or threshold at evaluation time.
    
    Args:
        Xw: [N,T,d] or [N,d] data
        Mw: Missingness mask
    
    Returns:
        A: [d,d] continuous edge scores (symmetric, 0 on diagonal)
    """
    X = preprocess_for_baseline(Xw, Mw, standardize=True)
    d = X.shape[1]
    C = np.corrcoef(X, rowvar=False)
    A = np.abs(C)
    np.fill_diagonal(A, 0)
    return A


def correlation_threshold(Xw, Mw=None, quantile=0.9):
    """
    Correlation-threshold baseline: binary thresholded graph.
    
    OUTPUT TYPE: Binary adjacency (AUROC/AUPRC not meaningful)
    
    Uses absolute correlation thresholded at given quantile.
    For continuous scores, use correlation_scores() instead.
    
    Args:
        Xw: [N,T,d] or [N,d] data
        Mw: Missingness mask
        quantile: Threshold quantile (default 0.9 = top 10% correlations)
    
    Returns:
        A: [d,d] binary undirected adjacency (symmetric)
    """
    A = correlation_scores(Xw, Mw)
    thr = np.quantile(A, quantile)
    return (A >= thr).astype(float)


def correlation_baseline(Xw, Mw=None, quantile=0.9):
    """
    DEPRECATED: Use correlation_scores() or correlation_threshold() instead.
    
    This function is kept for backward compatibility but returns SCORES
    (continuous) for proper AUROC/AUPRC evaluation. Use TopK at eval time.
    """
    # Return continuous scores (not thresholded) for proper AUROC/AUPRC
    return correlation_scores(Xw, Mw)


# Alias for backward compatibility
def notears_lite(Xw, Mw=None):
    """Alias for correlation_scores (backward compatibility)."""
    return correlation_scores(Xw, Mw)


def notears_linear(Xw, Mw=None, lambda1=0.1, lambda2=5.0, max_iter=100, tol=1e-5):
    """
    NOTEARS-penalty: Linear DAG learning with acyclicity penalty.
    
    NOTE: This is a single-shot optimization with soft acyclicity penalty,
    NOT the full augmented Lagrangian NOTEARS. For full NOTEARS, use 
    external libraries like `cdt` or `notears` package.
    
    OUTPUT TYPE: Continuous edge scores (AUROC/AUPRC supported)
    
    Solves: min_A ||X - XA||^2 + λ1||A||_1 + (λ2/2) h(A)²
    where h(A) = tr(exp(A⊙A)) - d
    
    Args:
        Xw: [N,T,d] or [N,d] data
        Mw: Missingness mask
        lambda1: L1 sparsity penalty
        lambda2: Acyclicity penalty weight
        max_iter: Maximum iterations
        tol: Convergence tolerance
    
    Returns:
        A: [d,d] learned adjacency matrix (continuous edge magnitudes)
    """
    # Use unified preprocessing (Fix #3)
    X = preprocess_for_baseline(Xw, Mw, standardize=True)
    N, d = X.shape
    
    def _h(A):
        """Acyclicity constraint: tr(exp(A⊙A)) - d"""
        M = A * A
        E = expm(M)
        return np.trace(E) - d
    
    def _loss(w):
        """Loss = MSE + L1 + soft acyclicity penalty"""
        A = w.reshape(d, d)
        np.fill_diagonal(A, 0)
        
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
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': tol}
    )
    
    A = result.x.reshape(d, d)
    np.fill_diagonal(A, 0)
    
    # Threshold small values
    A = (np.abs(A) > 0.05).astype(float) * A
    
    return np.abs(A)


# ============================================================================
# TEMPORAL METHODS - DISABLED (raise error if called)
# ============================================================================

def granger_causality(*args, **kwargs):
    """
    DISALLOWED: Granger causality assumes time-series with lagged dependencies.
    
    Our benchmark data is IID SEM (instantaneous, no lag terms).
    Using Granger would be methodologically incorrect.
    """
    raise RuntimeError(
        "DISALLOWED: Granger causality assumes time-series with lagged "
        "dependencies. Our benchmark data is IID SEM (instantaneous). "
        "Use PC, GES, or NOTEARS-based methods instead."
    )


def pcmci_plus(*args, **kwargs):
    """
    DISALLOWED: PCMCI+ assumes time-series with lagged dependencies.
    
    Our benchmark data is IID SEM (instantaneous, no lag terms).
    Using PCMCI+ would be methodologically incorrect.
    """
    raise RuntimeError(
        "DISALLOWED: PCMCI+ assumes time-series with lagged dependencies. "
        "Our benchmark data is IID SEM (instantaneous). "
        "Use PC, GES, or NOTEARS-based methods instead."
    )


# ============================================================================
# NEURAL METHODS
# ============================================================================

def notears_mlp(Xw, Mw=None, hidden_dim=32, lambda1=0.01, lambda2=5.0, max_iter=200, lr=0.01):
    """
    NOTEARS-MLP: Nonlinear DAG learning with MLP mechanisms.
    
    Extends NOTEARS to nonlinear relationships using neural networks.
    Each variable X_j = MLP_j(X_{Pa(j)}) + noise.
    
    OUTPUT TYPE: Continuous edge scores (AUROC/AUPRC supported)
    
    REQUIRES: PyTorch (numpy fallback disabled for practical runtime)
    
    Reference: Zheng et al. (2020) "Learning Sparse Nonparametric DAGs"
    
    Args:
        Xw: [N,T,d] or [N,d] data
        Mw: Missingness mask
        hidden_dim: MLP hidden dimension
        lambda1: L1 sparsity penalty
        lambda2: Lagrangian penalty for acyclicity
        max_iter: Maximum optimization iterations
        lr: Learning rate
    
    Returns:
        A: [d,d] learned adjacency matrix (continuous edge weights)
        
    Raises:
        RuntimeError: If PyTorch not available (numpy fallback disabled)
    """
    # Use unified preprocessing (Fix #3)
    X = preprocess_for_baseline(Xw, Mw, standardize=True)
    N, d = X.shape
    
    # Fix #4: Disable numpy fallback (too slow, produces unreliable results)
    if not HAS_TORCH:
        raise RuntimeError(
            "NOTEARS-MLP requires PyTorch for practical runtime. "
            "NumPy fallback disabled (numerical gradients too slow/unstable). "
            "Install PyTorch or use notears_linear() instead."
        )
    
    return _notears_mlp_torch(X, d, hidden_dim, lambda1, lambda2, max_iter, lr)


def _notears_mlp_numpy(X, d, hidden_dim, lambda1, lambda2, max_iter):
    """NumPy fallback for NOTEARS-MLP (simplified)."""
    N = X.shape[0]
    
    # Initialize adjacency weights
    A = np.random.randn(d, d) * 0.1
    np.fill_diagonal(A, 0)
    
    # MLP weights for each node (simplified: just use A as gating)
    W1 = np.random.randn(d, d, hidden_dim) * 0.1  # Input -> hidden
    W2 = np.random.randn(d, hidden_dim) * 0.1     # Hidden -> output (one per node)
    
    def forward(X, A, j):
        """Forward pass for node j."""
        # Mask inputs by adjacency
        X_masked = X * A[:, j]  # [N, d]
        # Simple nonlinear transform
        h = np.tanh(X_masked @ W1[j])  # [N, hidden]
        out = h @ W2[j]  # [N]
        return out
    
    def h_acyclic(A):
        """Acyclicity constraint."""
        M = A * A
        return np.trace(expm(M)) - d
    
    for iteration in range(max_iter):
        # Compute loss and gradients numerically
        loss = 0
        grad_A = np.zeros_like(A)
        
        for j in range(d):
            pred = forward(X, A, j)
            residual = X[:, j] - pred
            loss += np.sum(residual ** 2) / (2 * N)
            
            # Numerical gradient for A[:, j]
            eps = 1e-5
            for i in range(d):
                if i == j:
                    continue
                A[i, j] += eps
                pred_plus = forward(X, A, j)
                loss_plus = np.sum((X[:, j] - pred_plus) ** 2) / (2 * N)
                A[i, j] -= 2 * eps
                pred_minus = forward(X, A, j)
                loss_minus = np.sum((X[:, j] - pred_minus) ** 2) / (2 * N)
                A[i, j] += eps
                grad_A[i, j] = (loss_plus - loss_minus) / (2 * eps)
        
        # Add L1 penalty gradient
        grad_A += lambda1 * np.sign(A)
        
        # Add acyclicity penalty gradient
        h_val = h_acyclic(A)
        grad_h = 2 * A * expm(A * A)
        grad_A += lambda2 * h_val * grad_h
        
        # Update
        A -= 0.01 * grad_A
        np.fill_diagonal(A, 0)
        
        if iteration % 50 == 0:
            print(f"  NOTEARS-MLP iter {iteration}: loss={loss:.4f}, h={h_val:.4f}")
    
    return np.abs(A)


def _notears_mlp_torch(X, d, hidden_dim, lambda1, lambda2, max_iter, lr):
    """PyTorch implementation of NOTEARS-MLP."""
    X_tensor = torch.tensor(X, dtype=torch.float32)
    
    class MLPBlock(nn.Module):
        def __init__(self, d, hidden):
            super().__init__()
            self.fc1 = nn.Linear(d, hidden)
            self.fc2 = nn.Linear(hidden, 1)
            self.A = nn.Parameter(torch.randn(d) * 0.1)  # Edge weights to this node
        
        def forward(self, X):
            # Mask inputs by adjacency weights
            X_masked = X * self.A.unsqueeze(0)
            h = torch.tanh(self.fc1(X_masked))
            return self.fc2(h).squeeze(-1)
    
    # One MLP per node
    mlps = nn.ModuleList([MLPBlock(d, hidden_dim) for _ in range(d)])
    optimizer = optim.Adam(mlps.parameters(), lr=lr)
    
    def get_A():
        """Extract adjacency matrix from MLP weights."""
        A = torch.zeros(d, d)
        for j, mlp in enumerate(mlps):
            A[:, j] = mlp.A
        return A
    
    def h_acyclic(A):
        """Acyclicity constraint using matrix exponential."""
        M = A * A
        E = torch.matrix_exp(M)
        return torch.trace(E) - d
    
    for iteration in range(max_iter):
        optimizer.zero_grad()
        
        # Reconstruction loss
        loss = 0
        for j, mlp in enumerate(mlps):
            pred = mlp(X_tensor)
            loss += torch.mean((X_tensor[:, j] - pred) ** 2)
        
        # L1 penalty
        A = get_A()
        l1_loss = lambda1 * torch.sum(torch.abs(A))
        
        # Acyclicity penalty
        h_val = h_acyclic(A)
        h_loss = lambda2 * h_val * h_val
        
        total_loss = loss + l1_loss + h_loss
        total_loss.backward()
        optimizer.step()
        
        # Zero out diagonal
        with torch.no_grad():
            for j, mlp in enumerate(mlps):
                mlp.A[j] = 0
    
    A_final = get_A().detach().numpy()
    np.fill_diagonal(A_final, 0)
    return np.abs(A_final)


def golem(Xw, Mw=None, lambda1=0.01, lambda2=5.0, equal_variances=True, max_iter=200, lr=0.01):
    """
    GOLEM: Gradient-based Optimization for DAG Learning with M-matrices.
    
    Uses likelihood-based score with soft sparsity and acyclicity constraints.
    More stable than NOTEARS due to better parameterization.
    
    OUTPUT TYPE: Continuous edge scores (AUROC/AUPRC supported)
    
    REQUIRES: PyTorch (numpy fallback disabled for practical runtime)
    
    Reference: Ng et al. (2020) "On the Role of Sparsity and DAG Constraints for 
               Learning Linear DAGs"
    
    Args:
        Xw: [N,T,d] or [N,d] data
        Mw: Missingness mask
        lambda1: L1 sparsity penalty
        lambda2: Acyclicity penalty weight
        equal_variances: If True, assume equal noise variances
        max_iter: Maximum iterations
        lr: Learning rate
    
    Returns:
        A: [d,d] learned adjacency matrix (continuous edge weights)
        
    Raises:
        RuntimeError: If PyTorch not available (numpy fallback disabled)
    """
    # Use unified preprocessing (Fix #3)
    X = preprocess_for_baseline(Xw, Mw, standardize=True)
    N, d = X.shape
    
    # Compute sample covariance
    cov = X.T @ X / N
    
    # Fix #4: Disable numpy fallback (too slow, produces unreliable results)
    if not HAS_TORCH:
        raise RuntimeError(
            "GOLEM requires PyTorch for practical runtime. "
            "NumPy fallback disabled (numerical gradients too slow/unstable). "
            "Install PyTorch or use notears_linear() instead."
        )
    
    return _golem_torch(cov, d, lambda1, lambda2, equal_variances, max_iter, lr)


def _golem_numpy(cov, d, lambda1, lambda2, equal_variances, max_iter):
    """NumPy implementation of GOLEM."""
    # Initialize adjacency
    A = np.random.randn(d, d) * 0.01
    np.fill_diagonal(A, 0)
    
    def h_acyclic(A):
        M = A * A
        return np.trace(expm(M)) - d
    
    def likelihood_loss(A):
        """Negative log-likelihood (Gaussian)."""
        I_minus_A = np.eye(d) - A
        
        # Log determinant term
        sign, logdet = np.linalg.slogdet(I_minus_A)
        if sign <= 0:
            return 1e10  # Invalid (not a DAG direction)
        
        # Reconstruction term
        if equal_variances:
            # tr((I-A) cov (I-A)^T)
            residual_cov = I_minus_A @ cov @ I_minus_A.T
            loss = np.trace(residual_cov) - 2 * logdet
        else:
            # Sum of log variances
            residual_cov = I_minus_A @ cov @ I_minus_A.T
            variances = np.diag(residual_cov)
            loss = np.sum(np.log(variances + 1e-8)) - 2 * logdet
        
        return loss
    
    for iteration in range(max_iter):
        # Numerical gradient
        grad_A = np.zeros_like(A)
        eps = 1e-5
        
        for i in range(d):
            for j in range(d):
                if i == j:
                    continue
                A[i, j] += eps
                loss_plus = likelihood_loss(A) + lambda1 * np.sum(np.abs(A))
                A[i, j] -= 2 * eps
                loss_minus = likelihood_loss(A) + lambda1 * np.sum(np.abs(A))
                A[i, j] += eps
                grad_A[i, j] = (loss_plus - loss_minus) / (2 * eps)
        
        # Add acyclicity gradient
        h_val = h_acyclic(A)
        grad_h = 2 * A * expm(A * A)
        grad_A += lambda2 * h_val * grad_h
        
        # Update
        A -= 0.1 * grad_A
        np.fill_diagonal(A, 0)
    
    return np.abs(A)


def _golem_torch(cov, d, lambda1, lambda2, equal_variances, max_iter, lr):
    """PyTorch implementation of GOLEM."""
    cov_tensor = torch.tensor(cov, dtype=torch.float32)
    A = nn.Parameter(torch.randn(d, d) * 0.01)
    optimizer = optim.Adam([A], lr=lr)
    
    def h_acyclic(A):
        M = A * A
        E = torch.matrix_exp(M)
        return torch.trace(E) - d
    
    for iteration in range(max_iter):
        optimizer.zero_grad()
        
        # Mask diagonal
        A_masked = A * (1 - torch.eye(d))
        
        I_minus_A = torch.eye(d) - A_masked
        
        # Likelihood loss
        logdet = torch.logdet(I_minus_A)
        if torch.isnan(logdet) or torch.isinf(logdet):
            continue
        
        residual_cov = I_minus_A @ cov_tensor @ I_minus_A.T
        
        if equal_variances:
            loss = torch.trace(residual_cov) - 2 * logdet
        else:
            variances = torch.diag(residual_cov)
            loss = torch.sum(torch.log(variances + 1e-8)) - 2 * logdet
        
        # Penalties
        l1_loss = lambda1 * torch.sum(torch.abs(A_masked))
        h_val = h_acyclic(A_masked)
        h_loss = lambda2 * h_val * h_val
        
        total_loss = loss + l1_loss + h_loss
        total_loss.backward()
        optimizer.step()
    
    A_final = A.detach().numpy()
    np.fill_diagonal(A_final, 0)
    return np.abs(A_final)


# ============================================================================
# GRAPH NEURAL NETWORK METHODS (simplified - honest naming)
# ============================================================================

def gnn_score_heuristic(Xw, Mw=None, hidden_dim=64, num_layers=2):
    """
    GNN-score heuristic: Simple embedding + dot-product baseline.
    
    NOTE: This is NOT the full DAG-GNN algorithm. It's a simplified
    heuristic using random embeddings and greedy DAG enforcement.
    Included as a neural baseline reference point when PyTorch unavailable.
    
    OUTPUT TYPE: Binary adjacency (heuristic, not continuous scores)
    
    Args:
        Xw: [N,T,d] data
        Mw: Missingness mask
        hidden_dim: Embedding dimension
        num_layers: Number of message-passing layers
    
    Returns:
        A: [d,d] predicted adjacency matrix
    """
    # Use unified preprocessing (Fix #3)
    X = preprocess_for_baseline(Xw, Mw, standardize=True)
    N, d = X.shape
    
    # Initialize node embeddings based on feature statistics
    feat_mean = X.mean(axis=0)  # [d]
    feat_std = X.std(axis=0) + 1e-8
    H = np.column_stack([feat_mean, feat_std, np.random.randn(d, hidden_dim-2) * 0.1])
    
    # Message passing (simplified)
    for layer in range(num_layers):
        H_new = np.zeros_like(H)
        for i in range(d):
            H_new[i] = H.mean(axis=0) + H[i]
        H = np.maximum(H_new, 0)  # ReLU
        H = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-8)
    
    # Predict adjacency via dot product
    A_scores = H @ H.T
    np.fill_diagonal(A_scores, 0)
    
    # Threshold
    threshold = np.median(A_scores)
    A = (A_scores > threshold).astype(float)
    
    # Enforce DAG via greedy topological ordering
    A_dag = np.zeros_like(A)
    for _ in range(d):
        in_degrees = A.sum(axis=0)
        if in_degrees.sum() == 0:
            break
        i = np.argmin(in_degrees)
        A_dag[i, :] = A[i, :]
        A[:, i] = 0
        A[i, :] = 0
    
    return A_dag


def dag_gnn(Xw, Mw=None, hidden_dim=64, num_layers=3, max_iter=300, lr=0.001, lambda_h=1.0):
    """
    DAG-GNN-inspired: Neural adjacency learning baseline (our implementation).
    
    DISCLAIMER (for paper):
      "DAG-GNN-inspired neural baseline (our implementation). Uses GNN-style
       encoder with learnable adjacency and NOTEARS-style h(A) constraint.
       This is NOT the canonical DAG-GNN VAE formulation from Yu et al. (2019),
       but a simplified neural baseline for comparison."
    
    OUTPUT TYPE: Continuous edge scores (AUROC/AUPRC supported)
    
    REQUIRES: PyTorch (fallback to gnn_score_heuristic without PyTorch)
    
    Args:
        Xw: [N,T,d] or [N,d] data
        Mw: Missingness mask
        hidden_dim: Hidden dimension for GNN
        num_layers: Number of GNN layers
        max_iter: Maximum training iterations
        lr: Learning rate
        lambda_h: Acyclicity penalty weight
    
    Returns:
        A: [d,d] learned adjacency matrix (continuous edge weights)
    """
    # Use unified preprocessing (Fix #3)
    X = preprocess_for_baseline(Xw, Mw, standardize=True)
    N, d = X.shape
    
    if HAS_TORCH:
        return _dag_gnn_torch(X, d, hidden_dim, num_layers, max_iter, lr, lambda_h)
    else:
        # Fallback to simple heuristic (honest: not DAG-GNN)
        warnings.warn("PyTorch not available; using gnn_score_heuristic (simplified, not DAG-GNN)")
        return gnn_score_heuristic(Xw, Mw, hidden_dim, num_layers)


def _dag_gnn_torch(X, d, hidden_dim, num_layers, max_iter, lr, lambda_h):
    """PyTorch implementation of DAG-GNN."""
    X_tensor = torch.tensor(X, dtype=torch.float32)
    N = X.shape[0]
    
    class DAGGNN(nn.Module):
        def __init__(self, d, hidden_dim, num_layers):
            super().__init__()
            self.d = d
            
            # Adjacency matrix (learnable)
            self.A = nn.Parameter(torch.randn(d, d) * 0.1)
            
            # Encoder: node features -> embeddings
            self.encoder = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            
            # GNN layers
            self.gnn_layers = nn.ModuleList([
                nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
            ])
            
            # Decoder: embeddings -> reconstructed features
            self.decoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
        
        def forward(self, X):
            # X: [N, d]
            batch_size = X.shape[0]
            
            # Get adjacency (mask diagonal, apply sigmoid for soft edges)
            A = torch.sigmoid(self.A) * (1 - torch.eye(self.d))
            
            # Encode each node's feature
            H = self.encoder(X.unsqueeze(-1))  # [N, d, hidden]
            
            # GNN message passing
            for layer in self.gnn_layers:
                # Aggregate: H_new[j] = sum_i A[i,j] * H[i]
                H_agg = torch.einsum('ij,nid->njd', A, H)  # [N, d, hidden]
                H = torch.relu(layer(H_agg) + H)  # Residual connection
            
            # Decode
            X_recon = self.decoder(H).squeeze(-1)  # [N, d]
            
            return X_recon, A
        
        def h_acyclic(self):
            """Acyclicity constraint."""
            A = torch.sigmoid(self.A) * (1 - torch.eye(self.d))
            M = A * A
            E = torch.matrix_exp(M)
            return torch.trace(E) - self.d
    
    model = DAGGNN(d, hidden_dim, num_layers)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for iteration in range(max_iter):
        optimizer.zero_grad()
        
        X_recon, A = model(X_tensor)
        
        # Reconstruction loss
        recon_loss = torch.mean((X_tensor - X_recon) ** 2)
        
        # Sparsity loss
        sparse_loss = 0.01 * torch.sum(torch.abs(A))
        
        # Acyclicity loss
        h_val = model.h_acyclic()
        h_loss = lambda_h * h_val * h_val
        
        total_loss = recon_loss + sparse_loss + h_loss
        total_loss.backward()
        optimizer.step()
    
    # Extract final adjacency
    with torch.no_grad():
        A_final = torch.sigmoid(model.A) * (1 - torch.eye(d))
    
    return A_final.numpy()


def gran_dag(Xw, Mw=None, hidden_dim=64, num_layers=2, max_iter=300, lr=0.001):
    """
    GraN-DAG: Gradient-based Neural DAG Learning (our implementation).
    
    Uses neural networks to model nonlinear causal mechanisms with
    gradient-based structure learning and augmented Lagrangian.
    
    OUTPUT TYPE: Continuous edge scores (AUROC/AUPRC supported)
    
    REQUIRES: PyTorch (numpy fallback disabled for practical runtime)
    
    Reference: Lachapelle et al. (2020) "Gradient-Based Neural DAG Learning"
    
    Args:
        Xw: [N,T,d] or [N,d] data
        Mw: Missingness mask
        hidden_dim: Hidden dimension
        num_layers: Number of hidden layers per mechanism
        max_iter: Maximum iterations
        lr: Learning rate
    
    Returns:
        A: [d,d] learned adjacency matrix (continuous edge weights)
        
    Raises:
        RuntimeError: If PyTorch not available (numpy fallback disabled)
    """
    # Use unified preprocessing (Fix #3)
    X = preprocess_for_baseline(Xw, Mw, standardize=True)
    N, d = X.shape
    
    # Fix #4: Disable numpy fallback
    if not HAS_TORCH:
        raise RuntimeError(
            "GraN-DAG requires PyTorch for practical runtime. "
            "NumPy fallback disabled. Install PyTorch or use notears_linear()."
        )
    
    return _gran_dag_torch(X, d, hidden_dim, num_layers, max_iter, lr)


def _gran_dag_torch(X, d, hidden_dim, num_layers, max_iter, lr):
    """PyTorch implementation of GraN-DAG."""
    X_tensor = torch.tensor(X, dtype=torch.float32)
    N = X.shape[0]
    
    class GraNDAG(nn.Module):
        def __init__(self, d, hidden_dim, num_layers):
            super().__init__()
            self.d = d
            
            # Connectivity matrix (learnable, will be masked)
            self.W = nn.Parameter(torch.randn(d, d) * 0.1)
            
            # Neural network for each variable
            self.nets = nn.ModuleList()
            for j in range(d):
                layers = [nn.Linear(d, hidden_dim), nn.LeakyReLU()]
                for _ in range(num_layers - 1):
                    layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU()])
                layers.append(nn.Linear(hidden_dim, 1))
                self.nets.append(nn.Sequential(*layers))
        
        def forward(self, X):
            # X: [N, d]
            
            # Soft adjacency (sigmoid + mask diagonal)
            A = torch.sigmoid(self.W) * (1 - torch.eye(self.d))
            
            # For each variable, predict from parents
            X_pred = torch.zeros_like(X)
            for j in range(self.d):
                # Mask input by adjacency weights
                X_masked = X * A[:, j].unsqueeze(0)  # [N, d]
                X_pred[:, j] = self.nets[j](X_masked).squeeze(-1)
            
            return X_pred, A
        
        def h_acyclic(self):
            """Acyclicity constraint using trace exponential."""
            A = torch.sigmoid(self.W) * (1 - torch.eye(self.d))
            M = A * A
            E = torch.matrix_exp(M)
            return torch.trace(E) - self.d
    
    model = GraNDAG(d, hidden_dim, num_layers)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Augmented Lagrangian parameters
    rho = 1.0
    alpha = 0.0
    h_prev = float('inf')
    
    for iteration in range(max_iter):
        optimizer.zero_grad()
        
        X_pred, A = model(X_tensor)
        
        # Reconstruction loss (negative log-likelihood)
        recon_loss = torch.mean((X_tensor - X_pred) ** 2)
        
        # Sparsity penalty
        sparse_loss = 0.01 * torch.sum(torch.abs(A))
        
        # Augmented Lagrangian for acyclicity
        h_val = model.h_acyclic()
        h_loss = alpha * h_val + 0.5 * rho * h_val * h_val
        
        total_loss = recon_loss + sparse_loss + h_loss
        total_loss.backward()
        optimizer.step()
        
        # Update Lagrangian multipliers every 100 iterations
        if (iteration + 1) % 100 == 0:
            with torch.no_grad():
                h_new = model.h_acyclic().item()
                if h_new > 0.25 * h_prev:
                    rho *= 10
                h_prev = h_new
                alpha += rho * h_new
    
    # Extract final adjacency
    with torch.no_grad():
        A_final = torch.sigmoid(model.W) * (1 - torch.eye(d))
    
    return A_final.numpy()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_all_baselines():
    """
    Return dict of all appropriate baseline methods for IID SEM data.
    
    All methods accept (Xw, Mw) where:
      - Xw: [N,T,d] or [N,d] data
      - Mw: Missingness mask (1=observed, 0=missing), or None
    
    Missing values are mean-imputed before running any baseline.
    """
    return {
        # Classical constraint-based
        "PC": pc_algorithm,
        "GES": ges_algorithm,
        
        # Score-based (linear)
        "NOTEARS": notears_linear,  # Single-shot penalty version
        "GOLEM": golem,
        
        # Neural (nonlinear)
        "NOTEARS-MLP": notears_mlp,
        "GraN-DAG": gran_dag,
        "DAG-GNN": dag_gnn,
        
        # Simple baselines
        "Correlation": correlation_baseline,
    }


def get_quick_baselines():
    """Return fast baselines suitable for quick evaluation."""
    return {
        "PC": pc_algorithm,
        "GES": ges_algorithm,
        "NOTEARS": notears_linear,
        "Correlation": correlation_baseline,
    }


def get_temporal_baselines():
    """
    Return dict of temporal baseline methods.
    
    WARNING: These are DISABLED and will raise RuntimeError if called!
    They are NOT appropriate for IID SEM data.
    """
    return {
        "Granger": granger_causality,
        "PCMCI+": pcmci_plus,
    }