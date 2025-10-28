# Synthetic Corruption Benchmarks for Hypothesis Testing

Generated: October 28, 2025

## Benchmarks Summary

All benchmarks have been successfully generated and are ready for hypothesis testing. See `data/interim/synth_corrupted_*/` directories.

### H1: Structural Accuracy Under Missingness

Tests: Can RC-GNN recover true causal structure despite missing data?

#### H1 Easy (`synth_corrupted_h1_easy`)
- **Graph**: Erdős-Rényi, 15 nodes, 30 edges
- **Mechanism**: Linear
- **Environments**: 3
- **Samples**: 1200 train / 300 val (50 timesteps)
- **Corruption**: MCAR 10-20%, noise 0.1-0.2, drift 0.0-0.1
- **Use case**: Baseline/sanity check, quick iteration
- **Expected RC-GNN SHD**: < 5 (oracle: 0)

#### H1 Medium (`synth_corrupted_h1_medium`)
- **Graph**: Erdős-Rényi, 15 nodes, 30 edges
- **Mechanism**: MLP (nonlinear)
- **Environments**: 4
- **Samples**: 1920 train / 480 val (50 timesteps)
- **Corruption**: Mixed MCAR/MAR/MNAR, 20-30% missing, noise 0.2-0.4, drift 0.1-0.2
- **Use case**: Main H1 test, realistic complexity
- **Expected RC-GNN SHD**: < 8-10

#### H1 Hard (`synth_corrupted_h1_hard`)
- **Graph**: Scale-Free, 20 nodes, ~40 edges
- **Mechanism**: MLP (nonlinear)
- **Environments**: 5
- **Samples**: 2800 train / 700 val (50 timesteps)
- **Corruption**: Mixed MCAR/MAR/MNAR, 35-55% missing, noise 0.25-0.5, drift 0.1-0.3
- **Use case**: Challenge test, high corruption stress-test
- **Expected RC-GNN SHD**: < 15-20, baselines degrade >40%

### H2: Stability Improvement via Invariance Loss

Tests: Does invariance loss reduce cross-environment adjacency variance?

#### H2 Multi-Env (`synth_corrupted_h2_multi_env`)
- **Graph**: Erdős-Rényi, 20 nodes, 40 edges
- **Mechanism**: Linear (simpler)
- **Environments**: 5 (key for stability)
- **Samples**: 1600 train / 400 val (50 timesteps)
- **Corruption**: MCAR only, but 15-35% missing, graduated noise/drift per env
- **Use case**: Clean test of stability metrics, focus on environment diversity
- **Expected variance ratio**: Var_with_inv / Var_without_inv ≤ 0.4 (60% reduction)

#### H2 Stability (`synth_corrupted_h2_stability`)
- **Graph**: Erdős-Rényi, 15 nodes, 25 edges
- **Mechanism**: MLP (nonlinear)
- **Environments**: 4
- **Samples**: 1600 train / 400 val (50 timesteps)
- **Corruption**: Mixed MAR/MNAR (adversarial), 20-45% missing, noise 0.2-0.4, drift 0.1-0.25
- **Use case**: Stress-test invariance loss effectiveness
- **Expected variance ratio**: Var_with_inv / Var_without_inv ≤ 0.5

### H3: Expert Agreement on Policy-Relevant Pathways

Tests: Do learned adjacencies align with domain expert knowledge?

#### H3 Policy (`synth_corrupted_h3_policy`)
- **Graph**: Erdős-Rényi, 25 nodes, 50 edges
- **Mechanism**: MLP
- **Environments**: 4
- **Samples**: 1920 train / 480 val (50 timesteps)
- **Corruption**: Mixed MCAR/MAR/MNAR, 20-30% missing, noise 0.2-0.4, drift 0.05-0.2
- **Policy edges (simulated domain knowledge)**: 
  - (2→5), (2→8), (5→12), (8→12), (12→20) — represents air quality causal chains
  - Example: Node 2 (vehicle traffic) → Nodes 5,8 (precursor pollutants) → Node 12 (PM2.5)
- **Use case**: Policy consistency evaluation, `policy_consistency()` metric
- **Expected metrics**:
  - With RC-GNN: policy_consistency ≥ 0.75, presence ≥ 0.9
  - With baselines: policy_consistency ≤ 0.5, presence ≤ 0.7

## Data Structure

Each benchmark directory contains:

```
data/interim/synth_corrupted_{name}/
├── A_true.npy          # True adjacency (d, d)
├── X_train.npy         # Observed data (N_train, T, d)
├── X_val.npy           # Observed validation (N_val, T, d)
├── M_train.npy         # Missingness masks (same shape)
├── M_val.npy
├── S_train.npy         # Clean signal before corruption (same shape)
├── S_val.npy
├── e_train.npy         # Environment labels (N_train,)
├── e_val.npy           # Environment labels (N_val,)
└── meta.json           # Metadata (graph params, corruption configs, etc.)
```

## Quick Start: Training Models

### H1 Easy (fastest, sanity check)
```bash
python scripts/train_rcgnn.py configs/train.yaml --data_root data/interim/synth_corrupted_h1_easy
```

### H1 Medium (main H1 test)
```bash
python scripts/train_rcgnn.py configs/train.yaml --data_root data/interim/synth_corrupted_h1_medium --epochs 200
```

### H1 Hard (stress test)
```bash
python scripts/train_rcgnn.py configs/train.yaml --data_root data/interim/synth_corrupted_h1_hard --epochs 300 --batch_size 16
```

### H2 Multi-Env (stability test)
```bash
python scripts/train_rcgnn.py configs/train.yaml --data_root data/interim/synth_corrupted_h2_multi_env --model.loss.lambda_inv 1.0
```

### H2 Stability (invariance loss effectiveness)
```bash
# Train WITH invariance (λ_inv = 1.0)
python scripts/train_rcgnn.py configs/train.yaml --data_root data/interim/synth_corrupted_h2_stability --model.loss.lambda_inv 1.0 --output_dir artifacts/h2_with_inv

# Train WITHOUT invariance (λ_inv = 0)
python scripts/train_rcgnn.py configs/train.yaml --data_root data/interim/synth_corrupted_h2_stability --model.loss.lambda_inv 0.0 --output_dir artifacts/h2_without_inv

# Compare stability metrics
python scripts/eval_rcgnn.py --checkpoint artifacts/h2_with_inv/rcgnn_best.pt --data_root data/interim/synth_corrupted_h2_stability
```

### H3 Policy (policy consistency)
```bash
python scripts/train_rcgnn.py configs/train.yaml --data_root data/interim/synth_corrupted_h3_policy --epochs 250
# Then evaluate with policy_consistency() metric
```

## Expected Hypothesis Results

| Hypothesis | Benchmark | Success Criterion | Expected RC-GNN | Baselines (average) |
|------------|-----------|-------------------|-----------------|---------------------|
| H1 | h1_easy | SHD < 5 | ✅ ~2-3 | ❌ 10-15 |
| H1 | h1_medium | SHD < 10 | ✅ ~6-8 | ❌ 15-25 |
| H1 | h1_hard | SHD < 20 & baseline degradation > 40% | ✅ ~12-18 | ❌ 25-40 |
| H2 | h2_multi_env | Var_ratio ≤ 0.4 (60% reduction) | ✅ ~0.35-0.45 | ❌ 0.8-1.0 |
| H2 | h2_stability | Var_ratio ≤ 0.5 (50% reduction) | ✅ ~0.45-0.55 | ❌ 0.85-1.0 |
| H3 | h3_policy | policy_consistency ≥ 0.75 | ✅ ~0.80-0.90 | ❌ 0.45-0.60 |

## Implementation Timeline

- **Created**: 2025-10-28
- **Scripts**:
  - `scripts/synth_corruption_benchmark.py` — Benchmark generator (6 benchmarks)
  - `src/training/metrics.py` — Stability metrics (adjacency_variance, edge_set_jaccard, policy_consistency)
  - `src/training/loop.py` — Multi-environment evaluation (eval_epoch_multi_env)
  
- **Next Steps**:
  - Run H1/H2/H3 hypothesis tests (Week 3)
  - Generate baseline comparisons (NOTEARS, DCDI, DECI, MissDAG)
  - Produce Results section figures and tables
  
## Notes

- All corruptions are reproducible (seed-based random generation)
- Ground truth adjacencies (`A_true.npy`) are saved for SHD/F1 computation
- Multi-environment labels (`e_*.npy`) enable per-environment evaluation
- Clean signals (`S_*.npy`) available for oracle performance benchmarking
- Metadata (`meta.json`) documents all corruption parameters for reproducibility
