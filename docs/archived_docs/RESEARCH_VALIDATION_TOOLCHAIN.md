# Research Validation Toolchain Documentation

This document covers the 4 new research scripts created for comprehensive RC-GNN analysis and validation.

## Quick Start

All scripts assume:
- ✅ Model trained with `python scripts/train_rcgnn.py`
- ✅ Artifacts exist: `artifacts/checkpoints/rcgnn_best.pt` and `artifacts/adjacency/A_mean.npy`
- ✅ Data available: `data/interim/uci_air/` with X.npy, A_true.npy, etc.

### One-Command Full Pipeline

```bash
bash scripts/reproducibility_pipeline.sh --data-root data/interim/uci_air
```

This executes all 6 steps automatically and creates timestamped experiment directory with all results.

---

## 1. Threshold Optimization (`optimize_threshold.py`)

**Purpose**: Find the optimal binary threshold for the learned continuous adjacency matrix.

### Problem Solved
- RC-GNN outputs continuous adjacencies (0-0.1 range typically)
- Need binary 0/1 edges for comparison with ground truth
- Different thresholds trade off precision vs recall
- This script finds optimal threshold via F1 maximization

### Usage

```bash
python scripts/optimize_threshold.py \
    --adjacency artifacts/adjacency/A_mean.npy \
    --data-root data/interim/uci_air \
    --export artifacts \
    --threshold 0.5  # (optional, default=0.5 for metric computation)
```

### Output Files

**Visualizations:**
- `artifacts/threshold_analysis.png` - 4-panel analysis
  - Top-left: Precision-Recall curve
  - Top-right: F1 score vs threshold
  - Bottom-left: SHD vs threshold
  - Bottom-right: Edge sparsity vs threshold
  
- `artifacts/threshold_comparison.png` - Table of top 10 thresholds ranked by F1

**Reports:**
- `artifacts/threshold_report.txt` - Text report with:
  - Best threshold found
  - Optimal metrics (precision, recall, F1)
  - Recommendations for binarization
  - Interpretation of trade-offs

### Key Functions

```python
compute_metrics_at_threshold(A_pred, A_true, threshold)
  → precision, recall, F1, SHD, TP, FP, FN

find_optimal_threshold(A_pred, A_true, thresholds=None)
  → Grid search over 50 log-spaced thresholds
  → Returns best_threshold + all results

plot_threshold_analysis(results, output_path)
  → Creates 4-panel visualization

save_threshold_report(best_threshold, results, output_path)
  → Generates text report with recommendations
```

### Example Output

```
================================================================================
THRESHOLD OPTIMIZATION REPORT
================================================================================

Grid Search Results:
  Total thresholds tested: 50
  Threshold range: [0.0001, 0.0845]
  
OPTIMAL THRESHOLD: 0.0123

Optimal Metrics:
  Precision: 0.8941
  Recall: 0.7854
  F1 Score: 0.8353
  SHD: 18
  
Top 5 Thresholds by F1:
  1. 0.0123 (F1=0.8353)
  2. 0.0189 (F1=0.8241)
  3. 0.0078 (F1=0.8109)
  ...

RECOMMENDATION: Use threshold 0.0123 for binary adjacency matrix
```

---

## 2. Environment-Specific Structure Analysis (`visualize_environment_structure.py`)

**Purpose**: Extract and visualize how the learned causal structure adapts across different environments.

### Problem Solved
- RC-GNN learns per-environment structure via A_delta
- Global structure A learned, then adjusted per-environment
- Need to visualize these adjustments to understand robustness
- This script extracts and visualizes per-environment structures

### Usage

```bash
python scripts/visualize_environment_structure.py \
    --checkpoint artifacts/checkpoints/rcgnn_best.pt \
    --config-data configs/data_uci.yaml \
    --config-model configs/model.yaml \
    --export artifacts
```

### Output Files

**Visualizations:**
- `artifacts/environment_comparison.png` - Side-by-side heatmaps
  - Shows adjacency matrix for each environment
  - Allows visual comparison of structural differences
  
- `artifacts/environment_deltas.png` - Delta analysis
  - RdBu colormap: red=strong in this env, blue=weak
  - Shows differences from global mean structure
  - Reveals which edges are environment-specific
  
- `artifacts/structure_variation.png` - Variation quantification
  - Bar charts of Frobenius norms (variation magnitude)
  - Pairwise differences between environments
  - Identifies which environments differ most

**Reports:**
- `artifacts/environment_report.txt` - Analysis with:
  - Per-environment variation statistics
  - Most variable edges across environments
  - Interpretation of robustness
  - Recommendations for environment-specific analysis

### Key Functions

```python
extract_structures(model, val_loader, device)
  → Forward pass on validation set
  → Returns per-environment adjacencies

compute_average_structure_by_env(structures, n_envs)
  → Averages adjacencies per environment
  → Returns dict: {env_id: A_env}

plot_environment_comparison(A_per_env, output_path)
  → Side-by-side heatmaps

plot_delta_analysis(A_per_env, output_path)
  → RdBu colormap visualization of deltas

plot_structure_variation(A_per_env, output_path)
  → Frobenius norms + pairwise differences
```

### Example Output

```
================================================================================
ENVIRONMENT-SPECIFIC STRUCTURE ANALYSIS REPORT
================================================================================

Number of environments: 10
Global structure edges: 132

Per-Environment Variation:
  Env 1: 5.234 (Frobenius norm from mean)
  Env 2: 3.821
  Env 3: 6.145 (most different)
  ...
  
Most Variable Edges (changing across environments):
  - F1 → F8: range [0.032, 0.089] (std=0.018)
  - F3 → F9: range [0.028, 0.076] (std=0.015)
  - F5 → F8: range [0.025, 0.063] (std=0.012)

Interpretation:
  High variation (edges present in some envs, weak in others)
  suggests these relationships are environment-dependent.
  
  Could indicate:
  1. Causal mechanism changes with environment
  2. Different measurement noise per environment
  3. Unobserved confounders that vary
```

---

## 3. Baseline Comparison (`compare_baselines.py`)

**Purpose**: Compare RC-GNN learned structure against baseline methods.

### Baselines Included

1. **Correlation-based**: Edge weight = |correlation coefficient|
   - Simplest baseline
   - Assumes linear relationships
   - No causal interpretation
   
2. **NOTears-lite**: Greedy thresholding of correlation matrix
   - Simple causality heuristic
   - Mimics NOTears without full optimization
   - Lightweight baseline

3. **RC-GNN**: Learned causal structure (learned model)
   - Full RC-GNN with all components
   - Ground truth comparison target

### Usage

```bash
python scripts/compare_baselines.py \
    --data-root data/interim/uci_air \
    --adjacency artifacts/adjacency/A_mean.npy \
    --export artifacts \
    --threshold 0.5  # (optional, default=0.5)
```

### Output Files

**Visualizations:**
- `artifacts/baseline_comparison.png` - Grouped bar chart
  - Precision, Recall, F1 for each method
  - SHD (Structural Hamming Distance)
  - Direct performance comparison
  
- `artifacts/adjacency_methods_comparison.png` - Side-by-side adjacencies
  - Ground truth
  - Correlation-based result
  - NOTears-lite result
  - RC-GNN result
  - Visual inspection of structure differences

**Reports:**
- `artifacts/baseline_comparison_report.txt` - Summary with:
  - Metrics table (precision, recall, F1, SHD)
  - Best method for each metric
  - Interpretation and conclusions
  - Recommendations

### Key Functions

```python
compute_correlation_adjacency(X, threshold=None)
  → Correlation-based method

compute_notears_lite_adjacency(X, threshold=0.1)
  → NOTears-lite method

compute_metrics(A_pred, A_true, threshold=0.5)
  → precision, recall, F1, SHD

compare_methods(X, A_true, A_rcgnn, threshold=0.5)
  → Compare all methods

plot_method_comparison(results, output_path)
  → Grouped bar chart

plot_adjacency_comparison(methods, A_true, output_path)
  → Side-by-side heatmaps
```

### Example Output

```
================================================================================
BASELINE COMPARISON
================================================================================

RC-GNN:
  Precision: 0.8941
  Recall: 0.7854
  F1 Score: 0.8353
  SHD: 18
  TP/FP/FN: 104/13/31

Correlation:
  Precision: 0.4521
  Recall: 0.6234
  F1 Score: 0.5234
  SHD: 87
  TP/FP/FN: 82/100/53

NOTears-lite:
  Precision: 0.6234
  Recall: 0.5834
  F1 Score: 0.6027
  SHD: 62
  TP/FP/FN: 77/47/58

================================================================================
BEST METHODS
================================================================================

Best F1 Score: RC-GNN (0.8353)
Best Precision: RC-GNN (0.8941)
Best Recall: Correlation (0.6234)
Best SHD: RC-GNN (18)

✅ RC-GNN substantially outperforms baselines on all metrics
```

---

## 4. Reproducibility Pipeline (`reproducibility_pipeline.sh`)

**Purpose**: Execute complete end-to-end RC-GNN workflow with proper logging and artifact organization.

### What It Does

1. **Step 1**: Environment verification (PyTorch, NumPy, YAML)
2. **Step 2**: Train RC-GNN model
3. **Step 3**: Generate visualizations
4. **Step 4**: Optimize threshold
5. **Step 5**: Analyze per-environment structures
6. **Step 6**: Compare against baselines

### Usage

```bash
# Basic usage (uses default UCI Air Quality data)
bash scripts/reproducibility_pipeline.sh

# With custom data
bash scripts/reproducibility_pipeline.sh --data-root path/to/data

# With custom configs
bash scripts/reproducibility_pipeline.sh \
    --data-root data/interim/uci_air \
    --config-data configs/data_uci.yaml \
    --config-model configs/model.yaml
```

### Output Structure

```
artifacts/experiments/YYYYMMDD_HHMMSS/
├── EXPERIMENT_SUMMARY.txt          # This summary
├── logs/                            # Step logs
│   ├── 01_train.log
│   ├── 02_visualize.log
│   ├── 03_threshold_optimize.log
│   ├── 04_environment_structure.log
│   └── 05_baseline_comparison.log
├── visualizations/                  # All output visualizations
│   ├── learned_adjacency_mean.png
│   ├── network_graph.png
│   ├── threshold_analysis.png
│   ├── threshold_pr_curve.png
│   ├── environment_comparison.png
│   ├── environment_deltas.png
│   ├── structure_variation.png
│   ├── baseline_comparison.png
│   ├── adjacency_methods_comparison.png
│   ├── (all reports .txt files)
│   └── ...
├── checkpoints/
│   └── rcgnn_best.pt                # Trained model
└── adjacency/
    └── A_mean.npy                   # Learned adjacency
```

### Example Execution

```bash
$ bash scripts/reproducibility_pipeline.sh --data-root data/interim/uci_air

╔════════════════════════════════════════════════════════════════════════════════╗
║ RC-GNN Reproducibility Pipeline
║ Timestamp: 20250116_153042
║ Experiment Dir: artifacts/experiments/20250116_153042
╚════════════════════════════════════════════════════════════════════════════════╝

[STEP 1/6] Verifying environment...
✅ PyTorch: 2.0.1
✅ NumPy: 1.24.3
✅ PyYAML available
✅ Environment verified

[STEP 2/6] Training RC-GNN model...
         Data: data/interim/uci_air
         Configs: configs/data_uci.yaml, configs/model.yaml, configs/train.yaml
Epoch 0: loss=0.1523
Epoch 1: loss=0.0892
...

[STEP 3/6] Generating visualizations...
✅ Visualization complete

[STEP 4/6] Optimizing binary threshold...
✅ Threshold optimization complete

[STEP 5/6] Analyzing per-environment structures...
✅ Environment structure analysis complete

[STEP 6/6] Comparing against baseline methods...
✅ Baseline comparison complete

[FINAL] Collecting artifacts...
✅ Copied visualizations
✅ Copied checkpoints
✅ Copied adjacency matrices

✅ Experiment summary saved

╔════════════════════════════════════════════════════════════════════════════════╗
║ RC-GNN Reproducibility Pipeline Completed Successfully!
║ Results: artifacts/experiments/20250116_153042
╚════════════════════════════════════════════════════════════════════════════════╝
```

---

## Integration with Existing Workflow

### Training Pipeline
```bash
# Original training script (still works)
python scripts/train_rcgnn.py configs/data.yaml configs/model.yaml configs/train.yaml

# New: Training with automatic visualization
python scripts/train_and_visualize.py configs/data.yaml configs/model.yaml configs/train.yaml

# Standalone validation/visualization
python scripts/validate_and_visualize.py configs/data.yaml configs/model.yaml configs/eval.yaml
```

### New Analysis Tools (Use After Training)
```bash
# Individual scripts (run separately)
python scripts/optimize_threshold.py --adjacency artifacts/adjacency/A_mean.npy --data-root data/interim/uci_air

python scripts/visualize_environment_structure.py --checkpoint artifacts/checkpoints/rcgnn_best.pt --config-data configs/data_uci.yaml --config-model configs/model.yaml

python scripts/compare_baselines.py --data-root data/interim/uci_air --adjacency artifacts/adjacency/A_mean.npy

# Or run all together with reproducibility pipeline
bash scripts/reproducibility_pipeline.sh --data-root data/interim/uci_air
```

---

## Troubleshooting

### Script Issues

**ModuleNotFoundError: No module named 'path_helper'**
- Run scripts from project root: `cd rcgnn/`
- Or ensure scripts/ is in PYTHONPATH

**FileNotFoundError: artifacts/adjacency/A_mean.npy**
- Run training first: `python scripts/train_rcgnn.py ...`
- Check artifacts directory exists

**FileNotFoundError: data not found**
- Verify data location: `ls data/interim/uci_air/`
- Pass correct --data-root path

**CUDA/Device errors**
- Ensure `configs/train.yaml` has `device: "cpu"`
- Or change to available GPU if needed

### Visualization Issues

**Empty or blank plots**
- Check adjacency matrix is not all zeros: `python -c "import numpy as np; A=np.load('artifacts/adjacency/A_mean.npy'); print(A.min(), A.max())"`
- Verify data has correct ground truth: `python -c "import numpy as np; A_true=np.load('data/interim/uci_air/A_true.npy'); print((A_true>0).sum())"`

**Colors look wrong in heatmaps**
- Heatmaps use 'YlOrRd' colormap - check matplotlib version
- Deltas use 'RdBu' colormap - centered at 0

### Performance Issues

**Scripts running very slow**
- Reduce config batch size: `configs/train.yaml` → `batch_size: 4`
- Reduce epochs for testing: `configs/train.yaml` → `epochs: 2`
- Use smaller dataset subset if available

**Out of memory**
- Reduce batch size further
- Reduce number of environments if applicable
- Run on CPU: check `configs/train.yaml` has `device: "cpu"`

---

## Example Complete Workflow

```bash
# 1. Setup environment
conda activate rc-gnn
cd rcgnn

# 2. Quick training run (8 epochs, CPU)
python scripts/train_rcgnn.py \
    configs/data_uci.yaml \
    configs/model.yaml \
    configs/train.yaml

# 3. Analyze with all tools
python scripts/optimize_threshold.py \
    --adjacency artifacts/adjacency/A_mean.npy \
    --data-root data/interim/uci_air

python scripts/visualize_environment_structure.py \
    --checkpoint artifacts/checkpoints/rcgnn_best.pt \
    --config-data configs/data_uci.yaml \
    --config-model configs/model.yaml

python scripts/compare_baselines.py \
    --data-root data/interim/uci_air \
    --adjacency artifacts/adjacency/A_mean.npy

# 4. View results
# - Check visualizations in artifacts/
# - Read reports (*.txt) for interpretation
# - Compare metrics across methods
```

---

## Citation & Reproducibility

These tools ensure reproducibility by:
✅ Logging all configurations used
✅ Timestamping experiments
✅ Saving all outputs organized by run
✅ Providing detailed reports and metrics
✅ Supporting shell script automation

For publication-ready experiments:
```bash
bash scripts/reproducibility_pipeline.sh --data-root data/interim/uci_air \
    2>&1 | tee experiment_run.log
```

All artifacts are self-contained in timestamped directory - ready for archival or supplementary material.
