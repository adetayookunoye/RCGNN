# Evaluation Metrics: Graph Recovery & Forecasting

## 1. Graph Recovery Metrics (Directed Edges)

### A. Core Metrics: Precision, Recall, F1

**Setup:**
- `A_true` ∈ {0, 1}^{d×d} with true directed edges (causal convention: A[i,j]="i causes j")
- `A_pred` ∈ [0, 1]^{d×d} with predicted edge scores
- Threshold at τ (default τ=0.5): `A_binary = (A_pred > τ).astype(int)`

**Calculation:**

```python
def compute_directed_metrics(A_pred, A_true, threshold=0.5):
 """
 Args:
 A_pred: [d, d] predicted adjacency scores
 A_true: [d, d] ground truth binary adjacency
 threshold: edge probability threshold

 Returns:
 dict with precision, recall, f1, tp, fp, fn
 """
 d = A_pred.shape[0]

 # Threshold predictions
 A_binary = (A_pred > threshold).astype(int)

 # Remove self-loops (diagonal)
 np.fill_diagonal(A_binary, 0)
 np.fill_diagonal(A_true, 0)

 # Flatten for counting
 pred_edges = A_binary.flatten() > 0
 true_edges = A_true.flatten() > 0

 # True positives, false positives, false negatives
 tp = np.sum(pred_edges & true_edges)
 fp = np.sum(pred_edges & ~true_edges)
 fn = np.sum(~pred_edges & true_edges)

 # Metrics
 precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
 recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
 f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

 return {
 'precision': precision,
 'recall': recall,
 'f1': f1,
 'tp': tp,
 'fp': fp,
 'fn': fn,
 }
```

**Report for directed edges:**
- Precision (TP / (TP + FP))
- Recall (TP / (TP + FN))
- F1 (harmonic mean)
- Count: TP, FP, FN

### B. Structural Hamming Distance (SHD) - Directed

**Definition:** Number of edge differences to transform predicted to true graph.

```python
def compute_shd_directed(A_pred, A_true, threshold=0.5):
 """
 SHD counts:
 - Missing edges (in true, not in pred)
 - Extra edges (in pred, not in true)
 - Reversed edges (A[i,j] in pred, A[j,i] in true)
 """
 A_binary = (A_pred > threshold).astype(int)
 np.fill_diagonal(A_binary, 0)
 np.fill_diagonal(A_true, 0)

 d = A_binary.shape[0]

 # Missing and extra in same direction
 missing = np.sum((A_true & ~A_binary) > 0)
 extra = np.sum((A_binary & ~A_true) > 0)

 # Reversed: count pairs (i,j) where pred has i->j but true has j->i
 reversed_edges = 0
 for i in range(d):
 for j in range(d):
 if i != j and A_binary[i, j] > 0 and A_true[j, i] > 0:
 reversed_edges += 1

 shd = missing + extra + reversed_edges
 return shd
```

### C. Orientation Accuracy

**Definition:** Among correctly recovered skeleton edges, what fraction have correct direction?

```python
def compute_orientation_accuracy(A_pred, A_true, threshold=0.5):
 """
 For each undirected edge (i,j) or (j,i) in true:
 - If correctly predicted in right direction: [OK]
 - If predicted but reversed: [X]
 - If missing: (not counted)
 """
 A_binary = (A_pred > threshold).astype(int)
 np.fill_diagonal(A_binary, 0)
 np.fill_diagonal(A_true, 0)

 d = A_binary.shape[0]

 correct_direction = 0
 wrong_direction = 0

 for i in range(d):
 for j in range(i+1, d): # Upper triangle only (undirected edge)
 true_ij = A_true[i, j]
 true_ji = A_true[j, i]
 pred_ij = A_binary[i, j]
 pred_ji = A_binary[j, i]

 # If true edge exists
 if true_ij + true_ji > 0:
 # Check if predicted in correct direction
 if true_ij > 0 and pred_ij > 0:
 correct_direction += 1
 elif true_ji > 0 and pred_ji > 0:
 correct_direction += 1
 elif pred_ij > 0 or pred_ji > 0:
 wrong_direction += 1

 total = correct_direction + wrong_direction
 orientation_accuracy = correct_direction / total if total > 0 else 0.0

 return {
 'orientation_accuracy': orientation_accuracy,
 'correct_direction': correct_direction,
 'wrong_direction': wrong_direction,
 }
```

---

## 2. Graph Recovery Metrics (Skeleton / Undirected)

### A. Skeleton Precision, Recall, F1

```python
def compute_skeleton_metrics(A_pred, A_true, threshold=0.5):
 """
 Ignore edge direction; only count undirected edge presence.
 """
 A_binary = (A_pred > threshold).astype(int)
 np.fill_diagonal(A_binary, 0)
 np.fill_diagonal(A_true, 0)

 d = A_binary.shape[0]

 # Undirected edges
 skel_pred = np.zeros_like(A_binary)
 skel_true = np.zeros_like(A_true)

 for i in range(d):
 for j in range(d):
 if A_binary[i, j] > 0 or A_binary[j, i] > 0:
 skel_pred[i, j] = 1
 skel_pred[j, i] = 1
 if A_true[i, j] > 0 or A_true[j, i] > 0:
 skel_true[i, j] = 1
 skel_true[j, i] = 1

 # Symmetrize and remove diagonal
 skel_pred = np.triu(skel_pred)
 skel_true = np.triu(skel_true)

 # Count edges (upper triangle only to avoid double-counting)
 pred_edges = np.sum(skel_pred)
 true_edges = np.sum(skel_true)
 common_edges = np.sum(skel_pred & skel_true)

 tp = common_edges
 fp = pred_edges - common_edges
 fn = true_edges - common_edges

 precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
 recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
 f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

 return {
 'precision': precision,
 'recall': recall,
 'f1': f1,
 'tp': tp,
 'fp': fp,
 'fn': fn,
 }
```

### B. SHD Skeleton

```python
def compute_shd_skeleton(A_pred, A_true, threshold=0.5):
 """
 SHD on skeleton (undirected): missing + extra edges.
 """
 A_binary = (A_pred > threshold).astype(int)
 np.fill_diagonal(A_binary, 0)
 np.fill_diagonal(A_true, 0)

 d = A_binary.shape[0]

 # Convert to skeleton (undirected)
 skel_pred = np.zeros_like(A_binary)
 skel_true = np.zeros_like(A_true)

 for i in range(d):
 for j in range(i+1, d):
 if A_binary[i, j] > 0 or A_binary[j, i] > 0:
 skel_pred[i, j] = 1
 skel_pred[j, i] = 1
 if A_true[i, j] > 0 or A_true[j, i] > 0:
 skel_true[i, j] = 1
 skel_true[j, i] = 1

 # SHD = missing + extra
 shd = np.sum((skel_true & ~skel_pred)) + np.sum((skel_pred & ~skel_true))
 return shd
```

---

## 3. Threshold-Free Metrics

### A. AUPRC (Area Under Precision-Recall Curve)

```python
def compute_auprc(A_pred, A_true):
 """
 Compute precision-recall curve over all threshold values,
 then compute area under curve.
 """
 from sklearn.metrics import precision_recall_curve, auc

 # Flatten and remove diagonal
 d = A_pred.shape[0]
 mask = ~np.eye(d, dtype=bool)

 pred_scores = A_pred[mask].flatten()
 true_binary = (A_true[mask] > 0).astype(int).flatten()

 # Compute PR curve
 precision, recall, _ = precision_recall_curve(true_binary, pred_scores)
 auprc = auc(recall, precision)

 return auprc
```

### B. ROC-AUC for Edge Detection

```python
def compute_roc_auc(A_pred, A_true):
 """
 ROC-AUC: edge presence (positive) vs non-presence (negative).
 """
 from sklearn.metrics import roc_auc_score

 d = A_pred.shape[0]
 mask = ~np.eye(d, dtype=bool)

 pred_scores = A_pred[mask].flatten()
 true_binary = (A_true[mask] > 0).astype(int).flatten()

 roc_auc = roc_auc_score(true_binary, pred_scores)
 return roc_auc
```

### C. F1 vs K Curve

```python
def compute_f1_vs_k(A_pred, A_true):
 """
 For K in {0.5E, 0.75E, E, 1.5E, 2.0E} where E = number of true edges:
 - Select top-K edges by score
 - Compute directed F1
 """
 n_true_edges = np.sum(A_true > 0)
 k_values = [0.5, 0.75, 1.0, 1.5, 2.0]
 results = {}

 for k_frac in k_values:
 k = max(1, int(k_frac * n_true_edges))

 # Select top-K edges
 d = A_pred.shape[0]
 np.fill_diagonal(A_pred, -np.inf) # Don't select diagonal

 flat_indices = np.argsort(A_pred.flatten())[-k:]
 A_topk = np.zeros_like(A_pred)
 A_topk.flat[flat_indices] = 1

 # Compute F1
 f1 = compute_directed_metrics(A_topk, A_true, threshold=0.5)['f1']
 results[f'F1@{k_frac:.2f}E'] = f1

 return results
```

---

## 4. Forecasting Metrics (If Applicable)

### A. MAE & RMSE

```python
def compute_mae_rmse(y_true, y_pred):
 """
 Args:
 y_true: [test_size, horizon, d] ground truth
 y_pred: [test_size, horizon, d] predictions
 """
 mae = np.mean(np.abs(y_true - y_pred))
 rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

 # Also per-node for granularity
 mae_per_node = np.mean(np.abs(y_true - y_pred), axis=(0, 1)) # [d]
 rmse_per_node = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=(0, 1))) # [d]

 return {
 'mae': mae,
 'rmse': rmse,
 'mae_per_node': mae_per_node,
 'rmse_per_node': rmse_per_node,
 }
```

### B. NLL (Negative Log-Likelihood)

```python
def compute_nll(y_true, y_pred_mean, y_pred_std):
 """
 For probabilistic predictions: -log p(y_true | μ, σ)
 Assumes Gaussian: p(y|μ,σ) = N(y; μ, σ²)
 """
 eps = 1e-8
 y_pred_std = np.clip(y_pred_std, eps, None)

 nll = -np.mean(
 -0.5 * np.log(2 * np.pi * y_pred_std**2)
 - 0.5 * ((y_true - y_pred_mean) / y_pred_std) ** 2
 )

 return nll
```

### C. Calibration: ECE (Expected Calibration Error)

```python
def compute_ece(y_true, y_pred_mean, y_pred_std, n_bins=10):
 """
 ECE measures if predicted uncertainties match actual errors.
 Lower ECE = better calibration.

 Bin predictions by uncertainty; compute mean error in each bin.
 """
 eps = 1e-8
 abs_errors = np.abs(y_true - y_pred_mean)
 y_pred_std = np.clip(y_pred_std, eps, None)

 # Bin by predicted std
 bin_edges = np.percentile(y_pred_std, np.linspace(0, 100, n_bins + 1))
 bin_indices = np.digitize(y_pred_std, bin_edges) - 1

 ece = 0.0
 for i in range(n_bins):
 mask = bin_indices == i
 if mask.sum() > 0:
 empirical_std = np.std(abs_errors[mask])
 predicted_std = np.mean(y_pred_std[mask])
 ece += mask.sum() / len(y_pred_std) * np.abs(empirical_std - predicted_std)

 return ece
```

### D. Coverage of Prediction Intervals

```python
def compute_coverage(y_true, y_pred_mean, y_pred_std, alpha=0.95):
 """
 What fraction of true values fall within [μ - z*σ, μ + z*σ]
 where z is the (1-alpha)/2 quantile of N(0,1)?
 """
 from scipy.stats import norm

 z = norm.ppf((1 + alpha) / 2)
 lower = y_pred_mean - z * y_pred_std
 upper = y_pred_mean + z * y_pred_std

 in_interval = (y_true >= lower) & (y_true <= upper)
 coverage = np.mean(in_interval)

 return {
 'coverage': coverage,
 'nominal_level': alpha,
 'difference': coverage - alpha, # Ideal = 0
 }
```

---

## 5. Reporting Summary

For each dataset + seed, report:

```python
metrics = {
 # Directed metrics
 'directed_precision': 0.92,
 'directed_recall': 0.95,
 'directed_f1': 0.93,
 'directed_shd': 1,
 'orientation_accuracy': 0.87,

 # Skeleton metrics
 'skeleton_precision': 1.00,
 'skeleton_recall': 1.00,
 'skeleton_f1': 1.00,
 'skeleton_shd': 0,

 # Threshold-free
 'auprc': 0.98,
 'roc_auc': 0.97,
 'f1_at_0.5e': 0.85,
 'f1_at_1.0e': 0.93,
 'f1_at_1.5e': 0.88,

 # Forecasting (if applicable)
 'mae': 0.45,
 'rmse': 0.62,
 'nll': -0.15,
 'ece': 0.08,
 'coverage_95': 0.94,
}

# Save
with open(f'artifacts/metrics_{dataset}_{seed}.json', 'w') as f:
 json.dump(metrics, f, indent=2)
```

---

## 6. Statistical Reporting (§8)

### A. Aggregate Across Seeds

For N seeds, report:

```python
def aggregate_metrics(metrics_list):
 """
 metrics_list: list of dicts (one per seed)
 """
 import pandas as pd

 df = pd.DataFrame(metrics_list)
 summary = {
 'mean': df.mean().to_dict(),
 'std': df.std().to_dict(),
 'min': df.min().to_dict(),
 'max': df.max().to_dict(),
 }

 # 95% CI
 t_val = 1.96 # for large n; use t-distribution for small n
 ci_lower = summary['mean'] - t_val * summary['std'] / np.sqrt(len(metrics_list))
 ci_upper = summary['mean'] + t_val * summary['std'] / np.sqrt(len(metrics_list))

 summary['ci_lower'] = ci_lower
 summary['ci_upper'] = ci_upper

 return summary
```

### B. Paired Statistical Tests

```python
from scipy.stats import wilcoxon

def paired_test(method_a_scores, method_b_scores):
 """
 Wilcoxon signed-rank test for paired samples.
 Null hypothesis: distributions are equal.
 """
 stat, pval = wilcoxon(method_a_scores, method_b_scores)

 # Effect size (rank-biserial correlation)
 n = len(method_a_scores)
 r = 1 - (2 * stat) / (n * (n + 1))

 return {
 'test_statistic': stat,
 'p_value': pval,
 'effect_size_r': r,
 'significant': pval < 0.05,
 }
```

Example output:

```
RC-GNN vs NOTEARS (Directed F1):
 RC-GNN mean: 0.920 ± 0.030
 NOTEARS mean: 0.650 ± 0.050

 Wilcoxon p-value: 0.008 **
 Effect size (r): 0.75 (large)

 Conclusion: RC-GNN significantly outperforms NOTEARS (p < 0.01).
```

---

## Checklist: Metrics Implementation

- [ ] Directed precision/recall/F1 implemented
- [ ] SHD directed calculated correctly
- [ ] Orientation accuracy computed on recovered skeleton
- [ ] Skeleton precision/recall/F1 implemented
- [ ] SHD skeleton calculated
- [ ] AUPRC and ROC-AUC computed
- [ ] F1 vs K curve generated for K ∈ {0.5E, 0.75E, E, 1.5E, 2.0E}
- [ ] Forecasting metrics (MAE, RMSE, NLL, ECE, coverage) if applicable
- [ ] Metrics aggregated across 5+ seeds with mean ± std
- [ ] Statistical significance tests (Wilcoxon) implemented
- [ ] All metrics saved to JSON per dataset + seed

---

**Next:** See [baselines.md](baselines.md) for baseline implementation.
