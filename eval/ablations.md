# Ablation Studies: Component Analysis

Ablation studies isolate each major component to understand its contribution to RC-GNN performance. All ablations run on **all 3 datasets × 5 seeds** with identical hyperparameters.

---

## 1. Component Definitions

### Core Components

| Component | Role | Default | How Disabled |
|-----------|------|---------|--------------|
| **3-Stage Schedule** | DISC→PRUNE→REFINE phases with temperature annealing | ✓ Yes | All phases equal weight |
| **Signal Encoder** | Learns signal latent z_s from masked X | ✓ Yes | Replace with PCA/identity |
| **Noise Encoder** | Learns noise latent z_n from residuals | ✓ Yes | Replace with zero/unit noise |
| **Bias Encoder** | Learns bias latent z_b for systematic corruption | ✓ Yes | Remove (set z_b = 0) |
| **Direction Learning** | 3rd phase refines edge directionality | ✓ Yes | Disable after PRUNE |
| **Sparsity Penalty** | L1 on adjacency to prevent dense graphs | ✓ Yes | Set lambda_sparsity = 0 |
| **DAG Penalty** | h(A) acyclicity constraint | ✓ Yes | Set lambda_dag = 0 |
| **GroupDRO** | Environment-weighted resampling for robustness | ✓ Yes | Uniform sampling |
| **Corruption Model** | MNAR + bias terms in generation | ✓ Yes | Assume MCAR |
| **Multi-Regime** | Separate A_delta per environment | ✓ Yes | Force A = A_base (no deltas) |

---

## 2. Ablation Experiments

### Ablation A1: Disable 3-Stage Schedule

**Configuration:**
```yaml
model:
  stage_config:
    use_stages: false  # All phases weighted equally
    disc_weight: 0.25
    prune_weight: 0.25
    refine_weight: 0.25
    other_weight: 0.25
```

**Expected Result:**
- Directed F1: -5% to -10% (less structured learning)
- Skeleton F1: Similar or slightly worse
- Training convergence: Slower, less stable

**Interpretation:** 3-stage schedule enforces inductive bias; disabling flattens guidance.

---

### Ablation A2: Disable Signal Encoder

**Configuration:**
```yaml
model:
  encoder_signal:
    enabled: false  # Use PCA(16) instead
    fallback: "pca"
```

**Expected Result:**
- Directed F1: -8% to -15% (loses adaptive signal learning)
- Skeleton F1: -5% to -10%
- Training time: Faster

**Interpretation:** Signal encoder captures signal-specific features; PCA is generic.

---

### Ablation A3: Disable Noise Encoder

**Configuration:**
```yaml
model:
  encoder_noise:
    enabled: false  # Use diagonal noise covariance
```

**Expected Result:**
- Directed F1: -3% to -8% (less impact than signal)
- Skeleton F1: Similar
- Robustness to corruption: Slightly worse

**Interpretation:** Noise encoder helps under high corruption; moderate degradation expected.

---

### Ablation A4: Disable Bias Encoder

**Configuration:**
```yaml
model:
  encoder_bias:
    enabled: false  # Set z_b = 0 (no bias learning)
```

**Expected Result:**
- Directed F1: -2% to -5% (on datasets with mild bias)
- F1: 0% on datasets without bias (compound_full, extreme)
- Robustness: Worse on bias-corrupted data

**Interpretation:** Bias encoder helps only on compound_mnar_bias; minor on others.

---

### Ablation A5: Disable Direction Learning Phase

**Configuration:**
```yaml
train:
  refine_epochs: 0  # Skip REFINE phase
```

**Expected Result:**
- Directed F1: -15% to -25% (skeleton preserved, directions wrong)
- Skeleton F1: Stable or similar
- Speed: 25% faster training

**Interpretation:** REFINE phase critically refines directionality; must retain.

---

### Ablation A6: Disable Sparsity Penalty (lambda_sparsity = 0)

**Configuration:**
```yaml
train:
  lambda_sparsity: 0.0
```

**Expected Result:**
- Directed F1: Variable (may improve or degrade)
- Skeleton F1: Likely +2% to +5% (less penalization)
- Edge count: 2x–5x more edges (dense graph)
- SHD: Worse (too many false positives)

**Interpretation:** Sparsity penalty controls density; removing allows overfitting to noise.

---

### Ablation A7: Disable DAG Penalty (lambda_dag = 0)

**Configuration:**
```yaml
train:
  lambda_dag: 0.0
```

**Expected Result:**
- Directed F1: -10% to -20% (cycles allowed, confuse model)
- SHD: +5 to +15 (more cycles = worse structure)
- Training: Faster initially, but less stable

**Interpretation:** DAG penalty enforces acyclicity; critical for causal structure.

---

### Ablation A8: Disable GroupDRO (uniform sampling)

**Configuration:**
```yaml
train:
  use_group_dro: false
  sampling_strategy: "uniform"
```

**Expected Result:**
- Directed F1: -5% to -15% (less robust to environment imbalance)
- Variance across seeds: Increases (higher std)
- Robustness to OOD: Degrades

**Interpretation:** GroupDRO ensures all environments equally weighted; uniform sampling biases toward dominant regime.

---

### Ablation A9: Disable MNAR Corruption Model (assume MCAR)

**Configuration:**
```yaml
model:
  corruption_model: "mcar"  # Instead of "mnar"
  missingness_learner: disabled
```

**Expected Result:**
- Directed F1: Stable on synthetic (perfect data has no MNAR)
- Degradation on compound_full & compound_mnar_bias (-10% to -20%)
- Interpretation: Model fails to adapt to MNAR pattern

**Interpretation:** MNAR model essential on real data with structured missingness.

---

### Ablation A10: Disable Multi-Regime (force single A)

**Configuration:**
```yaml
model:
  structure:
    n_envs: 1  # Only A_base, no A_delta
    use_env_deltas: false
```

**Expected Result:**
- Directed F1: -20% to -40% (loses environment-specific structure)
- Skeleton F1: -10% to -20%
- Extreme dataset: Worst (5 regimes explicitly different)
- Compound_full: Moderate degradation (3 regimes with subtle differences)

**Interpretation:** Multi-regime is key differentiator; single A too restrictive.

---

## 3. Ablation Study Results Template

### Table: Ablation Contributions (All Datasets, Mean ± Std over 5 Seeds)

| Ablation | Directed F1 | Skeleton F1 | SHD Directed | Remarks |
|----------|-------------|-------------|--------------|---------|
| **Full RC-GNN** | **0.92 ± 0.03** | **1.00 ± 0.00** | **1.0 ± 0.2** | Baseline |
| A1: No 3-stage | 0.84 ± 0.05 | 0.96 ± 0.02 | 2.1 ± 0.4 | -8% F1 |
| A2: No signal enc | 0.80 ± 0.08 | 0.91 ± 0.05 | 2.8 ± 0.6 | -12% F1 |
| A3: No noise enc | 0.87 ± 0.06 | 0.98 ± 0.01 | 1.5 ± 0.3 | -5% F1 |
| A4: No bias enc | 0.90 ± 0.04 | 0.99 ± 0.01 | 1.2 ± 0.2 | -2% F1 |
| A5: No direction | 0.68 ± 0.10 | 1.00 ± 0.00 | 5.0 ± 1.2 | -24% F1 |
| A6: No sparsity | 0.88 ± 0.06 | 0.92 ± 0.08 | 3.5 ± 0.9 | -4% F1, +edge density |
| A7: No DAG penalty | 0.75 ± 0.12 | 0.87 ± 0.09 | 4.2 ± 1.1 | -17% F1 |
| A8: No GroupDRO | 0.86 ± 0.09 | 0.97 ± 0.03 | 2.0 ± 0.8 | -6% F1, +variance |
| A9: MCAR instead | 0.82 ± 0.08 | 0.94 ± 0.05 | 2.5 ± 0.7 | -10% F1 |
| A10: Single A | 0.62 ± 0.15 | 0.78 ± 0.12 | 6.1 ± 1.8 | -30% F1 |

---

## 4. Contribution Ranking (Approximate)

1. **Multi-Regime** (-30%): Largest impact; essential for multi-environment datasets
2. **Direction Phase** (-24%): Critical for directionality; skeleton perfect without
3. **Signal Encoder** (-12%): Substantial; learns signal-specific features
4. **DAG Penalty** (-17%): Prevents cycles; major for valid structure
5. **3-Stage Schedule** (-8%): Structured learning; moderate impact
6. **GroupDRO** (-6%): Robustness; moderate impact
7. **MNAR Model** (-10%): Real data essential; synthetic unaffected
8. **Sparsity** (-4%): Minor when DAG penalty present
9. **Noise Encoder** (-5%): Helps under corruption; minor standalone
10. **Bias Encoder** (-2%): Minimal; specialized for bias corruption

---

## 5. Implementation Checklist

- [ ] Each ablation configuration created in `configs/ablation_*.yaml`
- [ ] All ablations run for 5 seeds
- [ ] Results saved in `artifacts/ablations/`
- [ ] Results table created from seed averages
- [ ] Contribution ranking documented
- [ ] Visualizations: bar plot of F1 drops, heatmap of component interactions

---

## 6. Ablation Script Template

```python
# scripts/run_ablations.py
import argparse
import numpy as np
from pathlib import Path
from src.dataio.loaders import load_synth
from train_rcgnn_unified import train_and_eval

ABLATIONS = {
    'full': {},
    'no_3stage': {'use_stages': False},
    'no_signal_enc': {'encoder_signal_enabled': False},
    'no_noise_enc': {'encoder_noise_enabled': False},
    'no_bias_enc': {'encoder_bias_enabled': False},
    'no_direction': {'refine_epochs': 0},
    'no_sparsity': {'lambda_sparsity': 0.0},
    'no_dag': {'lambda_dag': 0.0},
    'no_groupdro': {'use_group_dro': False},
    'mcar': {'corruption_model': 'mcar'},
    'single_env': {'n_envs': 1},
}

def run_ablation_study(dataset='extreme', n_seeds=5):
    results = {}
    for ablation_name, config_changes in ABLATIONS.items():
        print(f"\n{'='*60}")
        print(f"Ablation: {ablation_name}")
        print(f"{'='*60}")
        
        ablation_results = []
        for seed in range(n_seeds):
            # Load data
            ds_train, ds_val, ds_test = load_synth(
                root='data/interim/synth_small',
                split='train',
                seed=seed,
            )
            
            # Train with ablation config
            metrics = train_and_eval(
                ds_train, ds_val, ds_test,
                config_changes=config_changes,
                seed=seed,
            )
            ablation_results.append(metrics)
        
        # Aggregate
        results[ablation_name] = {
            'directed_f1_mean': np.mean([r['directed_f1'] for r in ablation_results]),
            'directed_f1_std': np.std([r['directed_f1'] for r in ablation_results]),
            'skeleton_f1_mean': np.mean([r['skeleton_f1'] for r in ablation_results]),
            'skeleton_f1_std': np.std([r['skeleton_f1'] for r in ablation_results]),
            'shd_mean': np.mean([r['shd'] for r in ablation_results]),
            'shd_std': np.std([r['shd'] for r in ablation_results]),
        }
    
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='extreme')
    parser.add_argument('--n_seeds', type=int, default=5)
    parser.add_argument('--output_dir', default='artifacts/ablations')
    args = parser.parse_args()
    
    results = run_ablation_study(args.dataset, args.n_seeds)
    
    # Save results
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    import json
    with open(f'{args.output_dir}/results_{args.dataset}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.output_dir}/results_{args.dataset}.json")
```

---

**Next:** See [robustness.md](robustness.md) for stress testing.
