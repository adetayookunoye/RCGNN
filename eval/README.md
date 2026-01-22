# RC-GNN Publication-Ready Evaluation Pipeline

## Overview

This directory contains the complete evaluation protocol for RC-GNN causal discovery under missingness and compound sensor corruptions. All documentation is designed for publication submission and reviewer scrutiny.

**Key principles:**
- ✅ **No ground-truth leakage** in training, hyperparameter selection, or early stopping
- ✅ **Multi-dataset generalization** (in-domain + leave-one-dataset-out)
- ✅ **Threshold-free metrics** (not just TopK-F1)
- ✅ **Statistical rigor** (mean ± std, paired tests, effect sizes)
- ✅ **Reproducibility** (fixed seeds, exact splits, hyperparameter search space documented)

---

## Document Structure

| Document | Purpose |
|----------|---------|
| [protocol.md](protocol.md) | Data splits, train/val/test protocol, ground-truth rules |
| [metrics.md](metrics.md) | Graph recovery & forecasting metrics (definitions + code) |
| [baselines.md](baselines.md) | Comparison methods (correlation, NOTEARS, DAG-GNN, PCMCI+, etc.) |
| [ablations.md](ablations.md) | Component removal studies |
| [robustness.md](robustness.md) | Stress tests: missingness, corruption, OOD generalization |
| [reproducibility.md](reproducibility.md) | Checklist + artifact release |
| [templates.md](templates.md) | Table/figure templates for paper |

---

## Quick-Start Evaluation Checklist

### Phase 1: Setup (Day 1)
- [ ] Read [protocol.md](protocol.md) — understand train/val/test split rules
- [ ] Choose datasets (extreme, compound_full, compound_mnar_bias, etc.)
- [ ] Fix random seeds: `[42, 1337, 2024, 99, 777]` (minimum 5)
- [ ] Set `A_true=None` in all training code
- [ ] Verify no leakage (see [protocol.md §2](protocol.md))

### Phase 2: Baselines (Day 2-3)
- [ ] Implement/run NOTEARS (linear + nonlinear)
- [ ] Implement/run correlation + Granger baseline
- [ ] Run DAG-GNN if feasible
- [ ] Save results in `artifacts/baseline_results/`

### Phase 3: RC-GNN Training (Day 4-5)
- [ ] Train 5× RC-GNN on each dataset
- [ ] Use config from [reproducibility.md](reproducibility.md)
- [ ] Save checkpoints + logs

### Phase 4: Evaluation (Day 6)
- [ ] Run [metrics.md](metrics.md) evaluation suite
- [ ] Compute directed F1, skeleton F1, SHD, orientation accuracy
- [ ] Compute threshold-free metrics (AUPRC, ROC-AUC, F1 vs K)
- [ ] Compute forecasting metrics (MAE, RMSE, calibration)
- [ ] Report mean ± std across 5 seeds

### Phase 5: Analysis (Day 7-8)
- [ ] Statistical significance tests ([metrics.md §8](metrics.md))
- [ ] Ablation studies ([ablations.md](ablations.md))
- [ ] Robustness evaluation ([robustness.md](robustness.md))
- [ ] Generate tables/figures per [templates.md](templates.md)

---

## Datasets & Ground Truth

| Dataset | Size | Nodes | Edges | Regimes | Corruption | File |
|---------|------|-------|-------|---------|------------|------|
| extreme | 9448 | 24 | 13 | 5 | High (all types) | `data/interim/uci_air_c/extreme` |
| compound_full | 9448 | 24 | 13 | 3 | Moderate (MNAR) | `data/interim/uci_air_c/compound_full` |
| compound_mnar_bias | 9448 | 24 | 13 | 3 | Moderate (MNAR + bias) | `data/interim/uci_air_c/compound_mnar_bias` |

**A_true format:** each dataset contains `A_true.npy` (13×13 adjacency, causal convention: A[i,j]="i causes j")

---

## Key Evaluation Claims (Reviewers Will Check)

### ✅ Claim 1: Causal Recovery Under Missingness
- Report graph metrics (F1, SHD, orientation) across 0%, 20%, 40%, 60%, 80% missing rates
- Show that RC-GNN outperforms baselines at all missing rates

### ✅ Claim 2: Robustness to Corruption Regimes
- Test on each regime combination (compound alone, mnar_bias alone, both, etc.)
- Report worst-case performance across corruption types

### ✅ Claim 3: Multi-Dataset Generalization
- In-domain: train on all, test on held-out per dataset
- LODO: train on all-but-one, test on held-out dataset
- Show generalization without per-dataset tuning

### ✅ Claim 4: No Leakage
- Explicit "leakage test": run training with `A_true=None`
- Performance should match main results
- Include in appendix as sanity check

### ✅ Claim 5: Calibration Under Corruption
- If using uncertainty estimates: report ECE, coverage vs nominal
- Show predictions degrade gracefully under corruption

---

## Required Metrics Summary

### Graph Recovery (Directed)
- **Precision** = TP / (TP + FP)
- **Recall** = TP / (TP + FN)
- **F1** = 2 × Precision × Recall / (Precision + Recall)
- **SHD** = # wrongly oriented + # missing + # extra
- **Orientation accuracy** = correct orientations / recovered edges

### Graph Recovery (Skeleton/Undirected)
- Same as above but ignore edge direction

### Threshold-Free
- **AUPRC** over all edge threshold values
- **ROC-AUC** for edge detection (positive vs negative)
- **F1 vs K** for K ∈ {0.5E, 0.75E, E, 1.5E, 2.0E}

### Forecasting (if applicable)
- **MAE**, **RMSE**
- **NLL** (negative log-likelihood if probabilistic)
- **ECE** (expected calibration error)
- **Coverage** of 95% prediction intervals

---

## Statistical Reporting

For each metric and dataset:

```
Metric (dataset): M ± σ (95% CI: [L, U])
p-value vs baseline: p < 0.05 *
Effect size (Cohen's d): d = 0.50
```

Example:
```
Directed F1 (extreme):     0.92 ± 0.03  (95% CI: [0.89, 0.95])  p < 0.001 ***
Directed F1 (compound_full): 0.65 ± 0.08  (95% CI: [0.57, 0.73])  p = 0.012 *
```

---

## Output Artifacts

After full evaluation, you should have:

```
artifacts/
├── evaluation_results/
│   ├── {dataset}_{seed}/
│   │   ├── metrics.json          # All graph metrics
│   │   ├── forecasting.json      # MAE, RMSE, NLL, ECE
│   │   ├── pred_edges.npy        # Predicted adjacency
│   │   └── pred_scores.npy       # Edge probabilities
│   ├── summary_table.csv         # Mean ± std across seeds
│   └── statistical_tests.txt     # p-values, effect sizes
├── baseline_results/
│   ├── notears_{dataset}_{seed}/ # NOTEARS results
│   ├── correlation_{dataset}_{seed}/
│   └── ...
└── ablations/
    ├── no_3stage_{dataset}_{seed}/
    ├── no_direction_phase_{dataset}_{seed}/
    └── ...
```

---

## Writing the Methods Section (Publication Ready)

Use this template:

---

**Evaluation Protocol.** We evaluate (i) causal graph recovery under missingness/corruption and (ii) forecasting performance. We employ time-aware train/validation/test splits (60/20/20 on contiguous segments) and repeat all experiments across 5 random seeds. **Ground-truth graphs are used only for evaluation and never during training, hyperparameter selection, or early stopping.** To assess robustness, we report directed/skeleton precision/recall/F1, SHD (structural Hamming distance), and orientation accuracy, plus threshold-free metrics (AUPRC, ROC-AUC, F1 vs K curve). We compare against correlation, Granger, NOTEARS (linear and nonlinear), and DAG-GNN baselines. Robustness is evaluated across missingness rates (0%–80%) and corruption type combinations (compound, MNAR, compound+MNAR). Generalization is assessed via leave-one-dataset-out (LODO) and in-domain protocols. For all metrics, we report mean ± standard deviation across seeds and perform paired statistical tests (Wilcoxon signed-rank, α=0.05). Ablation studies remove each major component (3-stage schedule, direction-only phase, sparsity/DAG penalties, GroupDRO). All hyperparameters are tuned on validation data without access to test adjacency.

---

## Next Steps

1. **Start with [protocol.md](protocol.md)** — understand data splits and leakage prevention
2. **Implement baselines** — use [baselines.md](baselines.md) for exact methods
3. **Run RC-GNN** — follow reproducibility checklist in [reproducibility.md](reproducibility.md)
4. **Evaluate** — use metric definitions in [metrics.md](metrics.md)
5. **Analyze results** — robustness tests in [robustness.md](robustness.md), ablations in [ablations.md](ablations.md)
6. **Generate paper artifacts** — use [templates.md](templates.md)

---

## Questions? Reviewer Preparedness

This pipeline directly addresses common causal discovery paper rejections:

| Reviewer Concern | How We Address It |
|------------------|-------------------|
| "How do you prevent leakage from A_true?" | [protocol.md §2](protocol.md) + explicit leakage test |
| "Why only TopK-F1?" | [metrics.md](metrics.md) reports AUPRC, ROC-AUC, F1 vs K |
| "What about strong baselines?" | [baselines.md](baselines.md) includes NOTEARS, DAG-GNN, PCMCI+ |
| "Does it generalize?" | LODO + in-domain in [robustness.md](robustness.md) |
| "Is it robust to corruption?" | Stress tests in [robustness.md](robustness.md) |
| "What's your hyperparameter search?" | [reproducibility.md](reproducibility.md) full space + selection criterion |
| "Statistical significance?" | Paired tests in [metrics.md §8](metrics.md) |

---

**Version:** 1.0  
**Last updated:** Jan 21, 2026  
**Status:** Publication-ready template
