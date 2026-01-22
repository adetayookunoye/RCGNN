# Robustness Evaluation: Stress Testing

Robustness studies stress RC-GNN under realistic perturbations: increasing missingness, corruption combinations, out-of-distribution regimes, and cross-regime generalization.

---

## 1. Stress Test 1: Missingness Sweep

**Hypothesis:** RC-GNN handles missing data via imputation; performance should degrade gracefully as missingness increases.

### A. Experimental Design

**Missingness Levels:** 0%, 10%, 20%, 40%, 60%, 80%

**For each level:**
- Generate new datasets by randomly masking X
- Keep ground-truth A, e (regimes), S (corruption pattern) fixed
- Retrain RC-GNN on masked data
- Evaluate on test set (also masked at same rate)

**Datasets:** All 3 (extreme, compound_full, compound_mnar_bias)

**Seeds:** 5 per configuration

### B. Expected Behavior

| Missingness | Directed F1 | Skeleton F1 | Trend |
|-------------|-------------|-------------|-------|
| 0% | 0.92 | 1.00 | Baseline |
| 10% | 0.90 | 0.99 | Slight decay |
| 20% | 0.87 | 0.98 | Gradual |
| 40% | 0.80 | 0.95 | Accelerating |
| 60% | 0.65 | 0.85 | Steep |
| 80% | 0.35 | 0.55 | Collapse |

**Key Metric:** F1 degradation should be smooth, not abrupt (no cliff).

### C. Implementation

```python
def stress_test_missingness(dataset='extreme', n_seeds=5):
    """
    Test RC-GNN robustness to increasing missingness.
    """
    missingness_rates = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8]
    results = {}
    
    for miss_rate in missingness_rates:
        print(f"\nTesting missingness rate: {miss_rate:.1%}")
        ablation_results = []
        
        for seed in range(n_seeds):
            # Load original data
            ds_train, ds_val, ds_test = load_synth(
                root='data/interim/synth_small',
                split='train',
                seed=seed,
                dataset=dataset,
            )
            
            # Apply missingness mask
            def apply_missing(batch):
                X = batch['X'].clone()
                M = batch['M'].clone()
                # Random masking
                n, t, d = X.shape
                mask = torch.rand(n, t, d) < miss_rate
                M[mask] = False
                batch['X_masked'] = X
                batch['M'] = M
                return batch
            
            ds_train_masked = torch.utils.data.TensorDataset(
                *[apply_missing(ds_train[i]) for i in range(len(ds_train))]
            )
            ds_test_masked = torch.utils.data.TensorDataset(
                *[apply_missing(ds_test[i]) for i in range(len(ds_test))]
            )
            
            # Train and evaluate
            metrics = train_and_eval(
                ds_train_masked,
                ds_val,
                ds_test_masked,
                seed=seed,
            )
            ablation_results.append(metrics)
        
        # Aggregate
        results[f'{miss_rate:.1%}'] = {
            'directed_f1': np.mean([r['directed_f1'] for r in ablation_results]),
            'directed_f1_std': np.std([r['directed_f1'] for r in ablation_results]),
            'skeleton_f1': np.mean([r['skeleton_f1'] for r in ablation_results]),
            'skeleton_f1_std': np.std([r['skeleton_f1'] for r in ablation_results]),
        }
    
    return results
```

---

## 2. Stress Test 2: Corruption Combinations

**Hypothesis:** RC-GNN handles compound corruptions (MNAR + bias + noise); performance depends on corruption severity and type.

### A. Experimental Design

**Corruption Modes:**

1. **None**: Clean data (baseline)
2. **Noise only**: Gaussian noise, σ = {0.1, 0.5, 1.0}
3. **MNAR only**: Structured missingness (no bias)
4. **Bias only**: Systematic sensor bias (no MNAR)
5. **MNAR + Bias**: Compound corruption (our target)
6. **Extreme**: All + high noise (stress test)

**For each corruption mode:**
- Train on corrupted data
- Evaluate on test (same corruption applied)
- 5 seeds per configuration

### B. Expected Performance Matrix

| Corruption Type | Directed F1 | Skeleton F1 | SHD | Remarks |
|-----------------|-------------|-------------|-----|---------|
| Clean (0%) | 0.92 | 1.00 | 1.0 | Perfect |
| Noise σ=0.1 | 0.90 | 0.99 | 1.2 | Minimal impact |
| Noise σ=0.5 | 0.88 | 0.98 | 1.5 | Slight |
| Noise σ=1.0 | 0.82 | 0.95 | 2.2 | Moderate |
| MNAR only | 0.85 | 0.97 | 1.8 | Imputation works |
| Bias only | 0.88 | 0.98 | 1.5 | Bias encoder handles |
| MNAR + Bias | 0.80 | 0.94 | 2.5 | Compound effect |
| Extreme (all) | 0.60 | 0.80 | 4.0 | Severe degradation |

### C. Implementation

```python
def stress_test_corruptions(dataset='extreme', n_seeds=5):
    """
    Test RC-GNN robustness to different corruption modes.
    """
    corruption_modes = [
        ('clean', {'noise_std': 0.0, 'mnar_rate': 0.0, 'bias_scale': 0.0}),
        ('noise_0.1', {'noise_std': 0.1, 'mnar_rate': 0.0, 'bias_scale': 0.0}),
        ('noise_0.5', {'noise_std': 0.5, 'mnar_rate': 0.0, 'bias_scale': 0.0}),
        ('noise_1.0', {'noise_std': 1.0, 'mnar_rate': 0.0, 'bias_scale': 0.0}),
        ('mnar_only', {'noise_std': 0.0, 'mnar_rate': 0.3, 'bias_scale': 0.0}),
        ('bias_only', {'noise_std': 0.0, 'mnar_rate': 0.0, 'bias_scale': 0.5}),
        ('mnar_bias', {'noise_std': 0.0, 'mnar_rate': 0.3, 'bias_scale': 0.5}),
        ('extreme', {'noise_std': 1.0, 'mnar_rate': 0.4, 'bias_scale': 1.0}),
    ]
    
    results = {}
    for mode_name, corruption_params in corruption_modes:
        print(f"\nTesting corruption mode: {mode_name}")
        ablation_results = []
        
        for seed in range(n_seeds):
            # Generate corrupted data
            ds_train, ds_val, ds_test = load_synth(
                root='data/interim/synth_small',
                split='train',
                seed=seed,
                dataset=dataset,
                corruption_params=corruption_params,
            )
            
            # Train and evaluate
            metrics = train_and_eval(
                ds_train, ds_val, ds_test,
                seed=seed,
            )
            ablation_results.append(metrics)
        
        # Aggregate
        results[mode_name] = {
            'directed_f1': np.mean([r['directed_f1'] for r in ablation_results]),
            'directed_f1_std': np.std([r['directed_f1'] for r in ablation_results]),
            'skeleton_f1': np.mean([r['skeleton_f1'] for r in ablation_results]),
            'skeleton_f1_std': np.std([r['skeleton_f1'] for r in ablation_results]),
        }
    
    return results
```

---

## 3. Stress Test 3: Out-of-Distribution (OOD) Regime

**Hypothesis:** RC-GNN should generalize to unseen regime types via learned structure + environment deltas.

### A. Experimental Design

**Protocol: Leave-One-Out (LOO) per Regime**

For datasets with multiple regimes (extreme: 5 regimes, compound_full: 3 regimes):

1. **Train on:** Regimes 0, 1, 2, 3 (hold out regime 4)
2. **Test on:** Held-out regime 4
3. **Repeat:** Hold out each regime once
4. **Metric:** Is A_pred generalizable to unseen regime?

**Expected:** 
- Skeleton F1 should remain high (structure is shared)
- Directed F1 may degrade (directionality varies per regime)
- OOD F1 / ID F1 ratio: 0.8–0.95 (minor degradation acceptable)

### B. Implementation

```python
def stress_test_ood_regime(dataset='extreme', n_seeds=5):
    """
    Test generalization to left-out regime.
    """
    from src.dataio.loaders import load_synth_by_regime
    
    results = {'id': {}, 'ood': {}}
    
    for seed in range(n_seeds):
        print(f"\nSeed {seed}: LOO regime test")
        
        # Load data by regime
        data_by_regime = load_synth_by_regime(
            root='data/interim/synth_small',
            dataset=dataset,
            seed=seed,
        )  # {regime_id: {'X': ..., 'A_true': ...}}
        
        n_regimes = len(data_by_regime)
        regime_ids = list(data_by_regime.keys())
        
        for loo_regime in regime_ids:
            # Train on all regimes except loo_regime
            train_regimes = [r for r in regime_ids if r != loo_regime]
            ds_train = combine_regimes(
                [data_by_regime[r] for r in train_regimes]
            )
            ds_test_ood = data_by_regime[loo_regime]
            
            # Train
            model = train_and_eval(
                ds_train,
                None,  # No validation set
                ds_test_ood,
                seed=seed,
            )
            
            # Evaluate on OOD regime
            metrics_ood = evaluate(model, ds_test_ood)
            
            # Evaluate on ID regimes (average over train regimes)
            metrics_id_list = [
                evaluate(model, data_by_regime[r])
                for r in train_regimes[:2]  # Subset for speed
            ]
            metrics_id = {
                'directed_f1': np.mean([m['directed_f1'] for m in metrics_id_list]),
                'skeleton_f1': np.mean([m['skeleton_f1'] for m in metrics_id_list]),
            }
            
            # Store
            results['ood'][f'seed{seed}_regime{loo_regime}'] = metrics_ood
            results['id'][f'seed{seed}_regime{loo_regime}'] = metrics_id
    
    return results
```

---

## 4. Stress Test 4: Leave-One-Regime-Out (LORO) Generalization

**Protocol:** Similar to LOO above; focus on regime generalization across dataset shift.

**Expected Results Table:**

| Regime | Train Data | Test Regime | ID Dir F1 | OOD Dir F1 | Drop % |
|--------|-----------|-------------|-----------|------------|--------|
| 0 | 1,2,3,4 | 0 | 0.92 | 0.88 | -4% |
| 1 | 0,2,3,4 | 1 | 0.92 | 0.86 | -6% |
| 2 | 0,1,3,4 | 2 | 0.92 | 0.87 | -5% |
| 3 | 0,1,2,4 | 3 | 0.92 | 0.85 | -7% |
| 4 | 0,1,2,3 | 4 | 0.92 | 0.84 | -8% |
| **Avg** | — | — | **0.92** | **0.86** | **-6%** |

---

## 5. Stress Test 5: Dataset Shift (Domain Adaptation)

**Hypothesis:** RC-GNN trained on synthetic should partially transfer to real data.

### A. Experimental Design

**Cross-dataset:**
1. Train RC-GNN on `synthetic` (perfect, high regimes)
2. Evaluate on `compound_full` or `compound_mnar_bias` without retraining
3. Measure drop in performance

**Expected:**
- Synthetic → Compound: -20% to -35% (expected; different data distribution)
- Extreme → Compound: -15% to -25% (more regimes help transfer)

### B. Implementation

```python
def stress_test_domain_shift(src_dataset='extreme', tgt_dataset='compound_full', n_seeds=3):
    """
    Train on source dataset, evaluate on target (no fine-tuning).
    """
    results = []
    
    for seed in range(n_seeds):
        # Train on source
        ds_src_train, ds_src_val, _ = load_synth(
            root='data/interim/synth_small',
            split='train',
            seed=seed,
            dataset=src_dataset,
        )
        
        model = train(ds_src_train, ds_src_val, seed=seed)
        
        # Evaluate on target (zero-shot)
        ds_tgt_train, ds_tgt_val, ds_tgt_test = load_synth(
            root='data/interim/synth_small',
            split='test',
            seed=seed,
            dataset=tgt_dataset,
        )
        
        metrics_tgt = evaluate(model, ds_tgt_test)
        results.append(metrics_tgt)
    
    return results
```

---

## 6. Robustness Results Template

### Summary Table

| Stress Test | Condition | Metric | Baseline | Result | Drop % |
|-------------|-----------|--------|----------|--------|--------|
| **Missingness** | 20% missing | Dir F1 | 0.92 | 0.87 | -5% ✓ |
| | 40% missing | Dir F1 | 0.92 | 0.80 | -13% ✓ |
| | 60% missing | Dir F1 | 0.92 | 0.65 | -29% ✓ |
| **Corruption** | MNAR only | Dir F1 | 0.92 | 0.85 | -7% ✓ |
| | Bias only | Dir F1 | 0.92 | 0.88 | -4% ✓ |
| | MNAR+Bias | Dir F1 | 0.92 | 0.80 | -13% ✓ |
| | Extreme combo | Dir F1 | 0.92 | 0.60 | -35% ✓ |
| **OOD Regime** | Leave-out 1 | Dir F1 (ID) | 0.92 | 0.92 | 0% ✓ |
| | Leave-out 1 | Dir F1 (OOD) | 0.92 | 0.86 | -6% ✓ |
| **Domain Shift** | Synth → Real | Dir F1 | 0.92 | 0.68 | -26% ⚠️ |

**Legend:**
- ✓ Expected graceful degradation
- ⚠️ Significant drop (may indicate distribution shift)

---

## 7. Robustness Checklist

- [ ] Missingness sweep (0%, 10%, 20%, 40%, 60%, 80%) implemented
- [ ] All missingness configs run for 5 seeds
- [ ] Corruption modes (clean, noise, MNAR, bias, compound, extreme) tested
- [ ] LOO regime tests (hold out each regime, train on others)
- [ ] Cross-dataset domain shift tests (synthetic → real)
- [ ] Results aggregated in `artifacts/robustness/`
- [ ] Summary table created with mean ± std
- [ ] Visualization: robustness curves (missingness %, corruption type vs F1)

---

## 8. Visualization Templates

### Figure: Missingness Impact

```python
import matplotlib.pyplot as plt

missingness_rates = [0, 0.1, 0.2, 0.4, 0.6, 0.8]
dir_f1_means = [0.92, 0.90, 0.87, 0.80, 0.65, 0.35]
dir_f1_stds = [0.03, 0.04, 0.05, 0.06, 0.08, 0.12]

fig, ax = plt.subplots()
ax.errorbar(missingness_rates, dir_f1_means, yerr=dir_f1_stds, marker='o', label='Directed F1')
ax.axhline(y=0.5, color='r', linestyle='--', label='Baseline (correlation)')
ax.set_xlabel('Missingness Rate')
ax.set_ylabel('Directed F1')
ax.set_title('RC-GNN Robustness to Missingness')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('artifacts/robustness/missingness_curve.png', dpi=300)
```

### Figure: Corruption Comparison

```python
corruption_types = ['Clean', 'Noise', 'MNAR', 'Bias', 'MNAR+Bias', 'Extreme']
dir_f1 = [0.92, 0.82, 0.85, 0.88, 0.80, 0.60]

fig, ax = plt.subplots()
bars = ax.bar(corruption_types, dir_f1, color=['green', 'yellow', 'yellow', 'yellow', 'orange', 'red'])
ax.axhline(y=0.5, color='gray', linestyle='--', label='Baseline')
ax.set_ylabel('Directed F1')
ax.set_title('RC-GNN Performance Under Corruption')
ax.set_ylim([0, 1.0])
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}', ha='center', va='bottom')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('artifacts/robustness/corruption_comparison.png', dpi=300)
```

---

**Next:** See [reproducibility.md](reproducibility.md) for reproducibility checklist.
