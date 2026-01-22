# Reproducibility: Methodology & Artifacts

Reproducibility is critical for publication. This document ensures all results can be verified by independent reviewers.

---

## 1. Fixed Random Seeds

**Seed List (Ordered, Use Exactly These):**

```python
SEEDS = [42, 1337, 2024, 99, 777]
```

**Rationale:**
- Seed 42: Standard ML seed (widely recognized)
- Seed 1337: Established reproducibility seed
- Seed 2024: Year of submission
- Seed 99: Common in ML benchmarks
- Seed 777: Lucky number (also standard)

**CRITICAL:** Use these exact seeds for all runs. Report results as **mean ± std** across all 5 seeds.

### Implementation

```python
# scripts/train_rcgnn.py
def main(args):
    SEEDS = [42, 1337, 2024, 99, 777]
    results_per_seed = []
    
    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"Run: Seed {seed}")
        print(f"{'='*60}")
        
        # Set all random states
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Load data with same seed
        ds_train, ds_val, ds_test = load_synth(
            root=args.data_root,
            split=args.split,
            seed=seed,
        )
        
        # Train
        model, metrics = train_and_eval(ds_train, ds_val, ds_test)
        results_per_seed.append(metrics)
    
    # Aggregate
    print(f"\n{'='*60}")
    print("FINAL RESULTS (Mean ± Std across 5 seeds)")
    print(f"{'='*60}")
    for key in results_per_seed[0].keys():
        values = [r[key] for r in results_per_seed]
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{key}: {mean_val:.4f} ± {std_val:.4f}")
    
    return results_per_seed
```

---

## 2. Data Splits: Reproducible Indices

**Protocol:** Save split indices as JSON to reproduce exact train/val/test splits.

### A. Generating Splits

```python
def generate_and_save_splits(
    n_samples, 
    train_ratio=0.6, 
    val_ratio=0.2, 
    test_ratio=0.2,
    seed=42,
    output_dir='artifacts/splits',
):
    """
    Generate reproducible train/val/test indices.
    Uses time-aware contiguous split (no shuffling).
    """
    np.random.seed(seed)
    
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    n_test = n_samples - n_train - n_val
    
    # Contiguous split (time-series friendly)
    idx_train = np.arange(0, n_train)
    idx_val = np.arange(n_train, n_train + n_val)
    idx_test = np.arange(n_train + n_val, n_samples)
    
    # Save
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    splits = {
        'train_indices': idx_train.tolist(),
        'val_indices': idx_val.tolist(),
        'test_indices': idx_test.tolist(),
        'ratios': {
            'train': train_ratio,
            'val': val_ratio,
            'test': test_ratio,
        },
        'seed': seed,
        'n_samples': n_samples,
    }
    
    with open(f'{output_dir}/splits_seed{seed}.json', 'w') as f:
        json.dump(splits, f, indent=2)
    
    return splits

# In training script:
splits = generate_and_save_splits(
    n_samples=len(ds),
    seed=args.seed,
    output_dir='artifacts/splits',
)
ds_train = Subset(ds, splits['train_indices'])
ds_val = Subset(ds, splits['val_indices'])
ds_test = Subset(ds, splits['test_indices'])
```

### B. Verifying Splits

```python
def verify_splits(seed=42, split_file='artifacts/splits/splits_seed42.json'):
    """
    Load saved splits and verify no data leakage.
    """
    with open(split_file, 'r') as f:
        splits = json.load(f)
    
    train_indices = set(splits['train_indices'])
    val_indices = set(splits['val_indices'])
    test_indices = set(splits['test_indices'])
    
    # Check no overlap
    assert len(train_indices & val_indices) == 0, "Train-Val overlap!"
    assert len(train_indices & test_indices) == 0, "Train-Test overlap!"
    assert len(val_indices & test_indices) == 0, "Val-Test overlap!"
    
    # Check coverage
    all_indices = train_indices | val_indices | test_indices
    assert len(all_indices) == splits['n_samples'], "Missing indices!"
    
    print(f"✓ Split verification passed (seed {seed})")
    return splits
```

---

## 3. Hyperparameter Search Space

**CRITICAL:** Hyperparameters tuned on validation loss **only**, NOT on ground-truth A.

### A. Hyperparameter Ranges

```yaml
# configs/hyperparameter_space.yaml
hyperparameters:
  # Architectural
  hidden_dim: [32, 64, 128]  # Default: 64
  n_layers: [2, 3]           # Default: 2
  dropout: [0.0, 0.1, 0.2]   # Default: 0.1
  
  # Training
  learning_rate: [1e-4, 5e-4, 1e-3]  # Default: 1e-3
  batch_size: [32, 64, 128]          # Default: 64
  weight_decay: [0.0, 1e-5, 1e-4]    # Default: 1e-4
  
  # Sparsification
  lambda_sparsity: [0.01, 0.1, 0.5]  # Default: 0.1
  lambda_dag: [1.0, 5.0, 10.0]       # Default: 10.0
  
  # Corruption modeling
  lambda_mnar: [0.1, 1.0, 10.0]      # Default: 1.0
  lambda_bias: [0.1, 1.0, 10.0]      # Default: 1.0
```

### B. Hyperparameter Tuning Procedure

```python
def tune_hyperparameters(ds_train, ds_val, search_space, n_trials=20):
    """
    Bayesian optimization over hyperparameter space.
    Objective: minimize validation loss (NOT A_true-based).
    """
    import optuna
    
    def objective(trial):
        # Sample hyperparameters
        config = {
            'hidden_dim': trial.suggest_int('hidden_dim', 32, 128),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True),
            'lambda_sparsity': trial.suggest_float('lambda_sparsity', 0.01, 0.5, log=True),
            'lambda_dag': trial.suggest_float('lambda_dag', 1.0, 10.0),
        }
        
        # Train model (short run for speed)
        model = train(
            ds_train, ds_val,
            config=config,
            epochs=20,  # Short trial
            early_stopping_metric='val_loss',  # NOT A_true
            early_stopping_patience=3,
        )
        
        # Evaluate on validation set
        val_loss = evaluate_loss(model, ds_val)
        
        return val_loss
    
    # Run Bayesian optimization
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        direction='minimize',
        sampler=sampler,
    )
    study.optimize(objective, n_trials=n_trials)
    
    # Log best hyperparameters
    best_params = study.best_params
    print(f"Best hyperparameters (val_loss={study.best_value:.4f}):")
    print(json.dumps(best_params, indent=2))
    
    return best_params

# In training:
if args.tune_hps:
    best_params = tune_hyperparameters(
        ds_train, ds_val,
        search_space='configs/hyperparameter_space.yaml',
        n_trials=20,
    )
    # Save for reproducibility
    with open('artifacts/checkpoints/best_hps.json', 'w') as f:
        json.dump(best_params, f, indent=2)
else:
    # Use default hyperparameters
    with open('configs/model.yaml', 'r') as f:
        best_params = yaml.safe_load(f)
```

---

## 4. Compute Requirements & Environment

### A. System Specifications

Document for reproducibility:

```yaml
# artifacts/environment/compute_specs.yaml
compute:
  device: "cuda"  # or "cpu"
  gpu_model: "NVIDIA A100"  # if GPU
  gpu_count: 1
  gpu_memory_gb: 40
  cpu_cores: 16
  cpu_memory_gb: 64

software:
  python_version: "3.10.12"
  pytorch_version: "2.1.2"
  pytorch_cuda: "12.1"
  
dependencies:
  - name: "torch"
    version: "2.1.2"
  - name: "numpy"
    version: "1.24.3"
  - name: "pandas"
    version: "2.0.3"
  - name: "scikit-learn"
    version: "1.3.0"
  - name: "tigramite"
    version: "5.0.1"
  - name: "notears"
    version: "0.2.0"

walltime_per_run_seconds: 3600  # 1 hour per seed
total_walltime_hours: 20  # 5 seeds × 4 hours
```

### B. Generating Environment Report

```bash
#!/bin/bash
# scripts/generate_environment_report.sh

echo "=== Python Environment ===" > artifacts/environment/environment_report.txt
python --version >> artifacts/environment/environment_report.txt
echo "" >> artifacts/environment/environment_report.txt

echo "=== PyTorch ===" >> artifacts/environment/environment_report.txt
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" >> artifacts/environment/environment_report.txt
echo "" >> artifacts/environment/environment_report.txt

echo "=== Package Versions ===" >> artifacts/environment/environment_report.txt
pip freeze | grep -E "torch|numpy|pandas|sklearn|scipy" >> artifacts/environment/environment_report.txt

echo "Environment report saved to artifacts/environment/environment_report.txt"
```

---

## 5. Code Release Checklist

**Before publication, ensure:**

- [ ] Main training script: `scripts/train_rcgnn_unified.py` (include git hash/commit)
- [ ] All config files: `configs/*.yaml` (frozen versions)
- [ ] Data loaders: `src/dataio/loaders.py` (exact same loading logic)
- [ ] Model architecture: `src/models/rcgnn.py` (frozen version)
- [ ] Random seeds: Fixed list [42, 1337, 2024, 99, 777]
- [ ] Split indices: Saved JSON files in `artifacts/splits/`
- [ ] Hyperparameter files: Best parameters in `artifacts/checkpoints/best_hps.json`
- [ ] Environment: Full pip freeze in `artifacts/environment/requirements_exact.txt`
- [ ] README: Instructions for reproducing all results
- [ ] License: Clearly stated (MIT, Apache, GPL, etc.)

---

## 6. Artifact Organization

**Exact Directory Structure:**

```
artifacts/
├── checkpoints/
│   ├── rcgnn_best.pt              # Best model checkpoint
│   ├── best_hps.json              # Best hyperparameters
│   └── training_log.json          # Training metrics over epochs
├── adjacency/
│   ├── A_mean.npy                 # Mean adjacency over seeds
│   ├── A_std.npy                  # Std of adjacency
│   └── seed_*.npy                 # Per-seed adjacencies
├── metrics/
│   ├── results_seed42.json        # Results for seed 42
│   ├── results_seed1337.json      # Results for seed 1337
│   ├── ...
│   └── results_aggregated.json    # Mean ± std over all seeds
├── splits/
│   ├── splits_seed42.json         # Train/val/test indices
│   ├── splits_seed1337.json
│   └── ...
├── baselines/
│   ├── correlation_results.json
│   ├── notears_results.json
│   └── ...
├── ablations/
│   ├── results_no_3stage.json
│   ├── results_no_signal_enc.json
│   └── ...
├── robustness/
│   ├── missingness_sweep.json
│   ├── corruption_modes.json
│   ├── ood_regime_test.json
│   └── ...
├── environment/
│   ├── environment_report.txt
│   ├── requirements_exact.txt
│   └── compute_specs.yaml
└── README.md                      # Artifact guide
```

---

## 7. Reporting Template

### Main Paper Results Table

```markdown
## Table 1: Main Results (ID + LODO)

| Protocol | Directed F1 | Skeleton F1 | SHD | AUPRC | ROC-AUC |
|----------|-------------|-------------|-----|-------|---------|
| **RC-GNN (ID)** | 0.92 ± 0.03 | 1.00 ± 0.00 | 1.0 ± 0.2 | 0.95 ± 0.02 | 0.98 ± 0.01 |
| **RC-GNN (LODO)** | 0.86 ± 0.06 | 0.98 ± 0.02 | 1.8 ± 0.5 | 0.92 ± 0.04 | 0.96 ± 0.02 |
| NOTEARS | 0.68 ± 0.07 | 0.75 ± 0.09 | 7 ± 1 | 0.78 ± 0.06 | 0.82 ± 0.05 |
| Correlation | 0.31 ± 0.05 | 0.50 ± 0.08 | 19 ± 2 | 0.45 ± 0.08 | 0.52 ± 0.09 |

**Result:** RC-GNN achieves 92% directed F1 on in-distribution data and 86% on leave-one-regime-out evaluation, substantially outperforming baselines. Results are mean ± std over 5 random seeds (42, 1337, 2024, 99, 777).
```

### Supplementary: Seed-by-Seed Breakdown

```markdown
## Table S1: Per-Seed Results (Transparency)

| Seed | Directed F1 | Skeleton F1 | SHD | Training Time (s) |
|------|-------------|-------------|-----|-------------------|
| 42 | 0.94 | 1.00 | 0.8 | 2145 |
| 1337 | 0.91 | 1.00 | 1.2 | 2156 |
| 2024 | 0.93 | 1.00 | 0.9 | 2138 |
| 99 | 0.90 | 1.00 | 1.0 | 2152 |
| 777 | 0.92 | 1.00 | 1.5 | 2141 |
| **Mean** | **0.92** | **1.00** | **1.0** | **2146** |
| **Std** | **0.03** | **0.00** | **0.2** | **9** |
```

---

## 8. Statistical Significance Testing

### Wilcoxon Signed-Rank Test

```python
from scipy.stats import wilcoxon

def compare_methods(results_method_a, results_method_b, metric='directed_f1'):
    """
    Wilcoxon signed-rank test: compare two methods across seeds.
    """
    scores_a = np.array([r[metric] for r in results_method_a])
    scores_b = np.array([r[metric] for r in results_method_b])
    
    # Wilcoxon test
    statistic, p_value = wilcoxon(scores_a, scores_b)
    
    # Effect size: rank-biserial correlation
    n = len(scores_a)
    r = 1 - (2 * statistic) / (n * (n + 1))
    
    # Print results
    print(f"Wilcoxon Signed-Rank Test: {metric}")
    print(f"  Method A: {scores_a.mean():.4f} ± {scores_a.std():.4f}")
    print(f"  Method B: {scores_b.mean():.4f} ± {scores_b.std():.4f}")
    print(f"  p-value: {p_value:.6f}")
    print(f"  Effect size (r): {r:.4f}")
    
    if p_value < 0.05:
        print(f"  ✓ Significant difference (α=0.05)")
    else:
        print(f"  ✗ No significant difference")
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'effect_size': r,
        'mean_a': scores_a.mean(),
        'std_a': scores_a.std(),
        'mean_b': scores_b.mean(),
        'std_b': scores_b.std(),
    }

# Example:
results_rc_gnn = [...]  # 5 seed results
results_notears = [...]  # 5 seed results
comparison = compare_methods(results_rc_gnn, results_notears, metric='directed_f1')
```

---

## 9. Reproducibility Checklist

- [ ] Five fixed random seeds documented and used consistently
- [ ] Train/val/test splits saved as JSON (no accidental replay)
- [ ] Hyperparameters tuned on validation loss only (no A_true)
- [ ] Best hyperparameters saved and documented
- [ ] Full environment (pip freeze) committed or documented
- [ ] Compute requirements (GPU/CPU/memory) specified
- [ ] Training time per run recorded
- [ ] Results reported as mean ± std over 5 seeds
- [ ] Statistical significance tests performed (Wilcoxon)
- [ ] Per-seed results available in appendix
- [ ] Code repository includes exact version tags (git commit hash)
- [ ] README includes step-by-step reproduction instructions

---

## 10. Publication Checklist

### Code & Data

- [ ] Main script versioned and frozen (git tag)
- [ ] All configs YAML files included
- [ ] Data loading scripts deterministic
- [ ] Random seed fix documented
- [ ] No hardcoded paths (use relative or configurable)

### Reporting

- [ ] Table 1: Main results (mean ± std)
- [ ] Table 2: Ablations (component contributions)
- [ ] Table 3: Baseline comparisons
- [ ] Table 4: Robustness (missingness / corruption sweep)
- [ ] Appendix S1: Per-seed results (transparency)
- [ ] Appendix S2: Hyperparameter search results
- [ ] Appendix S3: Compute requirements

### Transparency

- [ ] p-values and effect sizes reported
- [ ] Confidence intervals provided
- [ ] Dataset specifications (n, t, d, n_regimes, edge density)
- [ ] Code availability statement
- [ ] Data availability statement (synthetic: code; real: links)
- [ ] Conflicts of interest (if any)

---

**Next:** See [templates.md](templates.md) for paper figure/table templates.
