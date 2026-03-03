# RC-GNN Experiments Guide

## Quick Reproduction (One Command)

```bash
# Full pipeline: synthetic data → training → evaluation
python run_full_pipeline.py --config configs/train.yaml --seed 1337

# Or step-by-step:
python scripts/synth_bench.py --out data/interim/synth_small  # Generate data
python scripts/train_rcgnn.py configs/data.yaml configs/model.yaml configs/train.yaml  # Train
python scripts/eval_rcgnn.py artifacts/adjacency/A_mean.npy data/interim/synth_small  # Evaluate
```

## Experiment Protocol

### 1. Data Preparation

Synthetic datasets are generated with known ground truth graphs:

```bash
python scripts/synth_bench.py --d 12 --N 1000 --T 50 --n_regimes 3 --seed 42 --out data/interim/synth_small
```

This creates:
- `X.npy`: Observed data [N, T, d]
- `M.npy`: Missingness mask [N, T, d] (1=observed, 0=missing)
- `e.npy`: Regime labels [N]
- `A_true.npy`: Ground truth adjacency [d, d]

### 2. Training

```bash
python scripts/train_rcgnn.py configs/data.yaml configs/model.yaml configs/train.yaml
```

Outputs:
- `artifacts/checkpoints/rcgnn_best.pt`: Best model checkpoint
- `artifacts/adjacency/A_mean.npy`: Learned adjacency (mean across regimes)
- `run_manifest.json`: Full provenance record

### 3. Evaluation

#### Standard Metrics

```bash
python scripts/eval_rcgnn.py artifacts/adjacency/A_mean.npy data/interim/synth_small --topk oracle
```

Metrics computed:
- **SHD** (Structural Hamming Distance): Lower is better
- **Directed F1**: Precision/recall for directed edges
- **Skeleton F1**: Precision/recall ignoring direction
- **AUROC/AUPRC**: Threshold-free ranking metrics

#### TopK Calibration Protocol

The TopK calibration determines how many edges to select from continuous scores:

1. **Oracle** (`--topk oracle`): Use ground truth edge count. Valid for method comparison but not deployment.
2. **Fixed** (`--topk fixed --k 13`): Use fixed K. Realistic for deployment.
3. **Sensitivity** (`--sensitivity`): Report metrics across K range.

**IMPORTANT**: For fair comparison, all methods (RC-GNN, NOTEARS, PC, etc.) use the SAME K.

### 4. Baseline Comparison

```bash
python scripts/run_baselines.py --method notears_lite --config configs/data.yaml
python scripts/run_baselines.py --method pc --config configs/data.yaml
python scripts/run_baselines.py --method ges --config configs/data.yaml
```

Then comprehensive evaluation:

```bash
python scripts/comprehensive_evaluation.py --artifacts-dir artifacts_using --output artifacts/evaluation_report.json
```

## Metric Conventions

### SHD (Structural Hamming Distance)

```
SHD = |Edges in Pred but not True| + |Edges in True but not Pred|
    = FP + FN (for directed graphs)
```

- Convention: Diagonal entries are ALWAYS excluded
- Lower is better, 0 is perfect

### Skeleton vs Directed Metrics

| Metric | Description |
|--------|-------------|
| Directed F1 | Edge (i→j) counted only if exact match |
| Skeleton F1 | Edge (i,j) counted if either i→j or j→i exists |

For symmetric methods (Correlation), skeleton F1 is the primary metric.

### CPDAG Handling

Constraint-based methods (PC, GES) output CPDAGs with undirected edges.
We handle this by:
1. Matching undirected ↔ undirected as correct
2. Directed pred vs undirected true counts as half-correct

## Ablation Studies

### 1. Component Ablation

Remove components to measure contribution:

```yaml
# configs/ablation_no_mnar.yaml
missingness:
  strategy: "mcar"  # Disable MNAR modeling
```

### 2. Corruption Robustness

Test across corruption levels:

```bash
for level in clean moderate compound_full; do
    python scripts/train_rcgnn.py configs/data_${level}.yaml configs/model.yaml configs/train.yaml
done
```

### 3. Sensitivity to K

```bash
python scripts/sensitivity_analysis.py --k-range 5,50 --step 5
```

## Artifact Structure

```
artifacts/
├── checkpoints/
│   └── rcgnn_best.pt          # Model weights
├── adjacency/
│   └── A_mean.npy             # Learned graph
├── evaluation_report.json     # Full metrics
├── run_manifest.json          # Provenance
└── visualizations/
    ├── adjacency_heatmap.png
    └── training_curves.png
```

## Reproducing Paper Results

### Table 1: Main Results

```bash
# Generate all corruption scenarios
./scripts/run_paper_experiments.sh

# Aggregate results
python scripts/aggregate_results.py artifacts/experiments/ --output paper/tables/main_results.tex
```

### Figure 2: Sensitivity Curves

```bash
python scripts/plot_sensitivity.py artifacts/experiments/ --output paper/figures/sensitivity.pdf
```

## Troubleshooting

### "RuntimeError: CUDA out of memory"

Reduce batch size or use CPU:

```yaml
# configs/train.yaml
training:
  device: "cpu"
  batch_size: 16
```

### "AssertionError: Sanity check failed"

Check that:
1. Input matrices are square
2. Diagonal is zero
3. No NaN/Inf values

### "WARN: TopK sanity check failed"

This means the selected TopK has fewer edges than requested (sparse input).
Check that the model has converged.

## Test Suite

Run all tests before submission:

```bash
pytest tests/ -v

# Key tests:
pytest tests/test_oracle_audit.py -v      # Verify no oracle leakage
pytest tests/test_golden_metrics.py -v    # Verify metric implementations
pytest tests/test_training_step.py -v     # Verify training loop
```
