
import numpy as np
def shd(A_hat, A_true):
    return int(np.sum(A_hat != A_true))

def jaccard_edges(A1, A2):
    e1 = (A1>0).astype(int).flatten()
    e2 = (A2>0).astype(int).flatten()
    inter = (e1 & e2).sum()
    union = ((e1 + e2) > 0).sum()
    return inter / max(1, union)

def pairwise_l1(As):
    As = [a.astype(float) for a in As]
    n = len(As)
    s = 0.0; c = 0
    for i in range(n):
        for j in range(i+1, n):
            s += np.abs(As[i]-As[j]).mean()
            c += 1
    return s / max(1,c)


def adjacency_variance(A_dict_by_env):
    """
    Compute cross-environment adjacency variance (Frobenius norm).
    
    Measures structural stability across environments.
    Paper metric: Var_{e,e'}[||A^(e) - A^(e')||_F]
    
    Args:
        A_dict_by_env: Dict mapping env_id -> adjacency matrix A [d, d]
        
    Returns:
        float: Variance of pairwise Frobenius distances
    """
    if not A_dict_by_env or len(A_dict_by_env) < 2:
        return 0.0
    
    env_ids = sorted(A_dict_by_env.keys())
    A_list = [A_dict_by_env[eid].astype(float) for eid in env_ids]
    
    # Compute pairwise Frobenius distances
    distances = []
    for i in range(len(A_list)):
        for j in range(i+1, len(A_list)):
            diff = A_list[i] - A_list[j]
            frob_dist = np.sqrt(np.sum(diff ** 2)) # Frobenius norm
            distances.append(frob_dist)
    
    if not distances:
        return 0.0
    
    # Return variance of distances
    return float(np.var(distances))


def edge_set_jaccard(A_dict_by_env, threshold=0.5):
    """
    Compute Jaccard similarity of edge sets across environments.
    
    Measures consistency of discovered edges across regimes.
    Paper metric: E_{e,e'}[J(E^(e), E^(e'))]
    
    Args:
        A_dict_by_env: Dict mapping env_id -> adjacency matrix A [d, d]
        threshold: Threshold for binarizing adjacency (default 0.5)
        
    Returns:
        float: Mean Jaccard similarity across all environment pairs
    """
    if not A_dict_by_env or len(A_dict_by_env) < 2:
        return 1.0 # Perfect similarity if only one environment
    
    env_ids = sorted(A_dict_by_env.keys())
    A_list = [A_dict_by_env[eid].astype(float) for eid in env_ids]
    
    # Binarize: edges = (A > threshold)
    E_list = [(A > threshold).astype(int).flatten() for A in A_list]
    
    # Compute pairwise Jaccard similarities
    similarities = []
    for i in range(len(E_list)):
        for j in range(i+1, len(E_list)):
            e_i = E_list[i]
            e_j = E_list[j]
            intersection = (e_i & e_j).sum()
            union = ((e_i + e_j) > 0).sum()
            jaccard = intersection / max(1, union)
            similarities.append(jaccard)
    
    if not similarities:
        return 1.0
    
    # Return mean Jaccard similarity
    return float(np.mean(similarities))


def policy_consistency(A_dict_by_env, policy_edges, threshold=0.5):
    """
    Measure stability of specified policy-relevant causal pathways.
    
    Tracks whether important domain pathways are consistently discovered.
    Paper metric: Stability of "domain-relevant causal pathways"
    
    Args:
        A_dict_by_env: Dict mapping env_id -> adjacency matrix A [d, d]
        policy_edges: List of tuples [(i, j), ...] representing policy-relevant edges
                     (e.g., [(0, 1), (2, 3)] = edges 0->1 and 2->3)
        threshold: Threshold for detecting edge presence (default 0.5)
        
    Returns:
        dict: {
            'consistency': float in [0, 1], # How consistently edges appear
            'presence': float in [0, 1], # How many policy edges detected
            'variance': float, # Cross-env variance in presence
        }
    """
    if not A_dict_by_env or not policy_edges:
        return {
            'consistency': 1.0,
            'presence': 0.0,
            'variance': 0.0,
        }
    
    env_ids = sorted(A_dict_by_env.keys())
    results_per_env = []
    
    for env_id in env_ids:
        A = A_dict_by_env[env_id].astype(float)
        
        # Check which policy edges are present
        edge_detected = []
        for i, j in policy_edges:
            if i < A.shape[0] and j < A.shape[1]:
                is_present = A[i, j] > threshold
                edge_detected.append(1.0 if is_present else 0.0)
        
        if edge_detected:
            # Fraction of policy edges detected in this environment
            presence_frac = np.mean(edge_detected)
        else:
            presence_frac = 0.0
        
        results_per_env.append(presence_frac)
    
    if not results_per_env:
        return {
            'consistency': 1.0,
            'presence': 0.0,
            'variance': 0.0,
        }
    
    # Consistency: how many environments have similar presence
    # Higher = more consistent across environments
    mean_presence = np.mean(results_per_env)
    
    # For consistency: measure how tight the distribution is
    # Perfect consistency = all envs have same presence (low variance)
    variance_presence = float(np.var(results_per_env))
    
    # Consistency score: 1 - normalized_variance
    max_variance = 0.25 # Max possible variance for binary variable
    consistency_score = max(0.0, 1.0 - (variance_presence / max_variance))
    
    return {
        'consistency': float(consistency_score),
        'presence': float(mean_presence),
        'variance': variance_presence,
    }
