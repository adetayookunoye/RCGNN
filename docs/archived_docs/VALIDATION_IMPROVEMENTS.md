# Validation Script Improvements — Publication-Ready

**Enhanced**: `scripts/validate_and_visualize.py`  
**Date**: October 26, 2025

## 🎯 Overview

The validation script has been completely overhauled to be **publication-ready** with robust metrics, better visualizations, and comprehensive exports. All improvements follow best practices for causal discovery evaluation.

---

## ✨ Key Improvements

### 1. **Off-Diagonal Only Evaluation** ✅
- **Problem**: Self-loops artificially inflate/deflate metrics
- **Solution**: All metrics computed on off-diagonal elements only
- **Implementation**: `_offdiag_mask(n)` function masks diagonal in all computations
- **Impact**: More accurate metrics that ignore self-connections

### 2. **Threshold-Free Metrics** ✅
- **Added**:
  - **AUPRC** (Area Under Precision-Recall Curve) — better for imbalanced datasets
  - **Best F1 over PR curve** — optimal threshold-free performance
  - **ROC-AUC** — standard ranking metric (when applicable)
- **Why**: Publications require threshold-agnostic evaluation; single threshold can be misleading
- **Example Output**:
  ```
  AUPRC:           0.1397
  Best F1 (PR):    0.2857 @ thr=0.0000
  ROC-AUC:         0.6154
  ```

### 3. **Correct SHD Computation** ✅
- **SHD (directed)**: Orientation-aware structural Hamming distance on off-diagonal only
- **SHD (skeleton)**: Undirected skeleton comparison (optional, useful for structure recovery)
- **Implementation**:
  ```python
  # Directed SHD (off-diagonal)
  shd = int(np.sum(np.abs((A_true[mask] > 0.5).astype(int) - y_pred)))
  
  # Skeleton SHD
  sk_true = np.maximum(At, At.T)
  sk_pred = np.maximum(Ap, Ap.T)
  sk_shd = int(np.sum(np.abs(sk_true[mask] - sk_pred[mask])))
  ```

### 4. **Top-k F1 Metrics** ✅
- **What**: Evaluate precision/recall/F1 on top-k predicted edges where k = #edges in ground truth
- **Why**: Tests if model ranks true edges highly (threshold-free ranking quality)
- **Example Output**:
  ```
  Top-k Metrics (k=13 edges in GT):
     Top-k Precision: 0.3077
     Top-k Recall:    0.3077
     Top-k F1:        0.3077
  ```

### 5. **DAG Sanity Checks** ✅
- **Metrics**:
  - Number of edges at threshold
  - Number of cycles detected (via NetworkX)
  - Maximum cycle length
  - Is DAG? (boolean)
- **Why**: Catch acyclicity constraint failures, debug structure learning
- **Example Output**:
  ```
  DAG Sanity Checks @ threshold=0.5:
     Edges:          15
     Cycles:         2
     Max cycle len:  3
     Is DAG:         ❌ No
  ```

### 6. **Ranked Edge List Export (CSV)** ✅
- **What**: Exports all edges sorted by predicted strength with src/tgt indices and names
- **File**: `artifacts/validation_pub_ready/edge_list.csv`
- **Columns**: `src | src_name | tgt | tgt_name | score`
- **Why**: Easy inspection in Excel/Pandas, manual domain validation, supplementary material
- **Example**:
  ```csv
  src,src_name,tgt,tgt_name,score
  0,Feature 0,1,Feature 1,0.519518
  7,Feature 7,5,Feature 5,0.519518
  ```

### 7. **Precision-Recall Curve Plot** ✅
- **What**: PR curve visualization with AUPRC in title
- **File**: `artifacts/validation_pub_ready/pr_curve.png`
- **Why**: Essential for publications; shows performance across all thresholds
- **Implementation**: Handles edge cases (no positives, constant scores)

### 8. **Improved Visualizations** ✅

#### Adjacency Heatmaps:
- Diagonal masked visually (set to NaN → white in plot)
- Consistent colorbar range `[0, 1]` for probability interpretation
- Fixed title: "thr" (threshold) not "τ" (temperature)
- 3-panel: Ground Truth | Continuous Predictions | Binarized

#### Edge Distribution:
- Off-diagonal only histogram
- 3-category bar chart: Zero | Non-Zero | Above Threshold
- Shows threshold effect clearly

#### Network Graph:
- Optional node name labels (publication-ready feature names)
- Top-k edges only (default 25) to avoid clutter
- Edge width proportional to strength
- Conditional edge labels (only if ≤15 edges)

### 9. **Robustness & Safety** ✅
- **NaN guard**: `np.nan_to_num(A_pred, nan=0.0, posinf=1.0, neginf=0.0)` applied everywhere
- **Edge case handling**: No positives, constant scores, missing ground truth
- **Zero-division protection**: `zero_division=0` in sklearn metrics

### 10. **Command-Line Interface** ✅
- **Flexible arguments**:
  ```bash
  python scripts/validate_and_visualize.py \
      --adjacency artifacts/adjacency/A_mean.npy \
      --data-root data/interim/uci_air \
      --threshold 0.5 \
      --export artifacts/validation_pub_ready \
      --node-names "CO,PT08.S1,NMHC,C6H6,PT08.S2,NOx,PT08.S3,NO2,PT08.S4,PT08.S5,T,RH,AH"
  ```
- **Auto-detection**: Ground truth auto-detected from common paths if `--data-root` not specified

---

## 📊 Example Output

### Console Output:
```
================================================================================
ADJACENCY VALIDATION AND VISUALIZATION (Publication-Ready)
================================================================================

✅ Loaded predicted adjacency: shape (13, 13)
   Min: 0.000000, Max: 0.519518, Mean: 0.049954
   Sparsity (off-diag): 90.4% zeros

✅ Loaded ground truth adjacency from data/interim/uci_air/A_true.npy
   Shape: (13, 13), Non-zero edges (off-diag): 13

--------------------------------------------------------------------------------
VALIDATION METRICS (Off-Diagonal Only)
--------------------------------------------------------------------------------

🎯 Binary Metrics @ threshold=0.50:
   Precision:  0.2667
   Recall:     0.3077
   F1-Score:   0.2857
   SHD (directed):   20
   SHD (skeleton):   38

🎯 Threshold-Free Metrics:
   AUPRC:           0.1397
   Best F1 (PR):    0.2857 @ thr=0.0000
   ROC-AUC:         0.6154

🎯 Top-k Metrics (k=13 edges in GT):
   Top-k Precision: 0.3077
   Top-k Recall:    0.3077
   Top-k F1:        0.3077

🔍 DAG Sanity Checks @ threshold=0.5:
   Edges:          15
   Cycles:         2
   Max cycle len:  3
   Is DAG:         ❌ No
```

### Generated Files:
```
artifacts/validation_pub_ready/
├── adjacency_comparison.png    # 3-panel heatmap comparison
├── edge_strength_dist.png      # Distribution + sparsity bar chart
├── causal_graph_network.png    # NetworkX graph (top 25 edges)
├── pr_curve.png                # Precision-Recall curve
└── edge_list.csv               # Ranked edge list (156 rows)
```

---

## 🔬 Use Cases

### 1. **Quick Validation After Training**
```bash
python scripts/validate_and_visualize.py \
    --adjacency artifacts/adjacency/A_mean.npy \
    --data-root data/interim/synth_small \
    --threshold 0.3
```

### 2. **Publication Figures with Node Names**
```bash
python scripts/validate_and_visualize.py \
    --adjacency artifacts/adjacency/A_mean.npy \
    --data-root data/interim/uci_air \
    --threshold 0.5 \
    --export paper_figures \
    --node-names "CO,PT08.S1,NMHC,C6H6,PT08.S2,NOx,PT08.S3,NO2,PT08.S4,PT08.S5,T,RH,AH"
```

### 3. **Structure-Only (No Ground Truth)**
```bash
python scripts/validate_and_visualize.py \
    --adjacency artifacts/adjacency/A_mean.npy \
    --threshold 0.4 \
    --export structure_analysis
```

### 4. **Edge Inspection for Domain Validation**
```bash
# Generate edge list, then inspect in Pandas:
python scripts/validate_and_visualize.py --adjacency artifacts/adjacency/A_mean.npy
python -c "import pandas as pd; df = pd.read_csv('artifacts/validation_pub_ready/edge_list.csv'); print(df.head(20))"
```

---

## 📝 What Changed (Code Level)

### Removed:
- ❌ `import seaborn` (unused)
- ❌ Diagonal elements in all metric computations
- ❌ Unsafe SHD (included diagonal)
- ❌ vmax=A_pred.max() (inconsistent colorbars)

### Added:
- ✅ `_offdiag_mask(n)` — universal off-diagonal mask
- ✅ `save_pr_curve()` — PR curve export
- ✅ `save_edge_list()` — CSV export with Pandas
- ✅ `dag_sanity()` — cycle detection via NetworkX
- ✅ AUPRC, top-k F1, skeleton SHD in `compute_metrics()`
- ✅ NaN guards everywhere
- ✅ Command-line argument parser
- ✅ Node name support in network graph

### Improved:
- 🔧 `compute_metrics()`: Now computes 10+ metrics including threshold-free
- 🔧 `plot_adjacency_matrices()`: Diagonal masked, consistent colorbars
- 🔧 `plot_edge_strength_distribution()`: Off-diagonal only, 3-category bars
- 🔧 `plot_graph_network()`: Node names, conditional edge labels

---

## 🚀 Next Steps

1. **Use edge_list.csv for domain expert review** — validate top edges against air quality sensor physics
2. **Include PR curve in paper** — shows robustness across thresholds
3. **Report AUPRC + top-k F1** — more informative than single-threshold F1
4. **Fix cycles** — 2 cycles detected at thr=0.5; tune acyclicity penalty or post-process
5. **Try multiple thresholds** — generate validation reports at [0.1, 0.3, 0.5, 0.7] and compare

---

## 📚 References

Best practices implemented from:
- DAG-GNN (Yu et al., 2019): threshold-free evaluation, skeleton SHD
- NOTEARS (Zheng et al., 2018): acyclicity checking
- Graphical Model Literature: off-diagonal metrics, PR curves for ranking

---

## 🎯 Summary

The script is now **publication-ready** with:
- ✅ Correct off-diagonal evaluation
- ✅ Threshold-free metrics (AUPRC, top-k F1)
- ✅ Comprehensive exports (CSV, PR curves)
- ✅ DAG sanity checks
- ✅ Robust handling of edge cases
- ✅ Beautiful, interpretable visualizations

**No more seaborn. No more diagonal bias. Ready for ICML/NeurIPS/UAI submissions!** 🎉
