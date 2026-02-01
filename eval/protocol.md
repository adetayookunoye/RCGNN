# Evaluation Protocol: Data Splits & Ground-Truth Rules

## 1. Data Splits (Time-Aware, No Leakage)

### A. Split Strategy

For each dataset (extreme, compound_full, compound_mnar_bias):

**60/20/20 split on contiguous time segments:**

```python
# Pseudocode
total_length = 9448 samples

train_end = int(0.6 * total_length) # 5668 samples
val_end = train_end + int(0.2 * total_length) # 5668 + 1889 = 7557
test_end = total_length # 9448

train_idx = np.arange(0, train_end)
val_idx = np.arange(train_end, val_end)
test_idx = np.arange(val_end, test_end)

assert len(train_idx) == 5668
assert len(val_idx) == 1889
assert len(test_idx) == 1891
```

**Why contiguous?**
- Time-series data -> no random shuffle (temporal order matters)
- Avoids data leakage (test window is strictly after training window)
- Realistic deployment scenario (train on past, test on future)

### B. Regime Inclusion

**CRITICAL:** Each split **must contain all regimes** (unless doing explicit OOD-regime tests).

Check at dataset load:

```python
# Verify regime balance
train_regimes = np.unique(e[train_idx]) # e = regime labels
val_regimes = np.unique(e[val_idx])
test_regimes = np.unique(e[test_idx])

assert len(train_regimes) == num_regimes, f"Train missing regimes: {train_regimes}"
assert len(val_regimes) == num_regimes, f"Val missing regimes: {val_regimes}"
assert len(test_regimes) == num_regimes, f"Test missing regimes: {test_regimes}"
```

For datasets with 3 regimes (compound_full, compound_mnar_bias):
- Train: regimes 0, 1, 2 present
- Val: regimes 0, 1, 2 present
- Test: regimes 0, 1, 2 present

For datasets with 5 regimes (extreme):
- Train: regimes 0, 1, 2, 3, 4 present
- Val: regimes 0, 1, 2, 3, 4 present
- Test: regimes 0, 1, 2, 3, 4 present

---

## 2. Unified Training Protocols

Define two evaluation settings:

### Protocol 1: In-Domain (ID)

**Setup:**
- Train one model on **all datasets** combined (or per-dataset variants)
- Evaluate on held-out test windows from **each dataset separately**

**Output:**
- Per-dataset metrics: (extreme_test_f1, compound_full_test_f1, compound_mnar_bias_test_f1)
- Summary: report per-dataset and average across datasets

**When to use:**
- If claiming "multi-dataset robustness without retraining per-dataset"

### Protocol 2: Leave-One-Dataset-Out (LODO)

**Setup:**
- For each dataset D_test:
 - Train on all-other-datasets (combine train + val from all except D_test)
 - Evaluate on held-out test window of D_test

**Example (3 datasets):**

| Test Dataset | Train On | Val On | Test On |
|--------------|----------|--------|---------|
| extreme | compound_full + compound_mnar_bias | (from those) | extreme |
| compound_full | extreme + compound_mnar_bias | (from those) | compound_full |
| compound_mnar_bias | extreme + compound_full | (from those) | compound_mnar_bias |

**Output:**
- Per-dataset generalization: how well does model trained on *other* datasets predict on *this* dataset?
- Summary: report each LODO result + average

**When to use:**
- Strong generalization story: "model trained on diverse corruptions generalizes to unseen dataset"

---

## 3. Ground-Truth Access Rules (CRITICAL FOR PUBLICATION)

### [DONE] A_true is ALLOWED for:

- **Evaluation only** (computing metrics on test set)
- Sanity checks (verifying splits contain all edges)
- Post-hoc analysis (edge frequency, stability plots)

### [FAIL] A_true is FORBIDDEN during:

- **Training loss** (never use A_true in L_recon, L_inv, L_dag, etc.)
- **Early stopping** (never use test adj to decide when to stop)
- **Hyperparameter tuning** (never use A_true to select λ, τ, etc.)
- **Sparsity/temperature schedule** (never make decisions based on A_true)
- **Validation metric calculation** (use only validation reconstruction loss, not adjacency metrics)
- **Threshold selection** (never threshold edges using A_true as oracle)

### Required Validation Metrics (No A_true)

Use these for model selection / early stopping:

```python
# ALLOWED (does not leak A_true)
val_loss = recon_loss(val_batch) + λ_sparsity * L_sparsity(A) + λ_dag * L_dag(A)

# FORBIDDEN (leaks A_true)
val_f1 = compute_f1(A_pred, A_true[val_idx]) # [FAIL] NO!
val_adjacency_metric = anything involving A_true # [FAIL] NO!
```

---

## 4. Multiple Random Seeds

### A. Seed Protocol

Fix seed list for reproducibility:

```python
SEEDS = [42, 1337, 2024, 99, 777] # Minimum 5 seeds
# If making "robustness" claims, use 10 seeds
SEEDS = [42, 1337, 2024, 99, 777, 123, 456, 789, 2048, 4096]
```

For each seed:
1. Set `np.random.seed(seed)`
2. Set `torch.manual_seed(seed)` and `torch.cuda.manual_seed(seed)`
3. Run full pipeline (train RC-GNN + all baselines)
4. Save results in `results/{dataset}_{seed}/`

### B. Reporting Results

For each metric M and dataset D:

```
Metric M (dataset D):
 seed_42: 0.820
 seed_1337: 0.795
 seed_2024: 0.835
 seed_99: 0.801
 seed_777: 0.812

 Mean ± Std: 0.812 ± 0.016
 95% CI: [0.791, 0.833] (via t-distribution with df=4)
```

Then compute paired statistical test vs baselines (see metrics.md §8).

---

## 5. Leakage Sanity Test (Include in Paper)

### Required Experiment

**Goal:** Prove that performance doesn't degrade when A_true is completely hidden during training.

**Setup:**

1. Train RC-GNN normally (A_true available only for final evaluation)
2. Train RC-GNN with explicit `A_true = None` everywhere in training code (set it to None at load time)
3. Compare final test metrics

**Expected result:**

```
Config | Directed F1 | Skeleton F1 | SHD |
--------|------------|------------|-----|
Normal (A_true hidden) | 0.920 | 1.000 | 1.0 |
Leakage test (A_true=None) | 0.920 | 1.000 | 1.0 |
Difference | 0.000 | 0.000 | 0.0 |
```

If they match -> **no leakage** [DONE]

### Code Implementation

```python
# In train_rcgnn_unified.py, add:
if args.leakage_test:
 A_true = None # Suppress A_true everywhere

# Verify it's never accessed:
assert A_true is None
# Later during training, if any code tries to use A_true -> error
```

### Reporting in Paper

**Appendix A: Leakage Test**

"To verify that ground-truth adjacencies do not leak into training, we ran RC-GNN with explicit `A_true = None` enforced throughout training and hyperparameter tuning. Results on test sets show no degradation (Table A.1), confirming that model performance is driven solely by the training loss and validation metrics."

---

## 6. Hyperparameter Tuning (Validation Only)

### A. Search Space

Do NOT tune on test data. Use validation loss + early stopping:

```python
# Grid/random search over hyperparameters
hyperparams = {
 'lr': [0.0001, 0.0005, 0.001],
 'latent_dim': [16, 32, 64],
 'λ_sparsity': [0.01, 0.001, 0.0001],
 'λ_dag': [0.01, 0.1, 1.0],
}

best_hp = None
best_val_loss = np.inf

for hp_combo in random_search(hyperparams, n_combos=30):
 # Train on train set
 model.train(train_loader, ...)

 # Validate on val set (using ONLY validation loss, not A_true)
 val_loss = model.evaluate(val_loader) # Reconstruction + sparsity, no adj metrics

 if val_loss < best_val_loss:
 best_val_loss = val_loss
 best_hp = hp_combo

# Now evaluate best_hp on held-out test set
test_metrics = model.evaluate(test_loader) # Compute all metrics including A_true
```

### B. Early Stopping

```python
# GOOD: Early stopping on validation LOSS (no A_true)
early_stop = EarlyStopping(
 metric='val_loss', # reconstruction loss only
 patience=10,
 mode='min'
)

# BAD: Early stopping on validation F1 (leaks A_true)
early_stop = EarlyStopping(
 metric='val_f1_directed', # [FAIL] NO!
 patience=10,
 mode='max'
)
```

---

## 7. Split Reproducibility

### A. Save Exact Indices

For each dataset, save the exact split indices:

```python
# After splitting
splits = {
 'train_idx': train_idx.tolist(),
 'val_idx': val_idx.tolist(),
 'test_idx': test_idx.tolist(),
 'seed': seed,
 'dataset': dataset_name,
}

with open(f'artifacts/splits/{dataset_name}_{seed}_splits.json', 'w') as f:
 json.dump(splits, f)
```

### B. Verification Script

```bash
# For reproducibility, verify splits are identical in reproduction
python scripts/verify_splits.py \
 --saved_splits artifacts/splits/extreme_42_splits.json \
 --dataset extreme
# Should print: "[DONE] Splits match"
```

---

## 8. Dataset-Specific Rules

### extreme (5 regimes, high corruption)

```python
dataset = 'extreme'
num_regimes = 5
num_samples = 9448
true_edges = 13

# Must verify all 5 regimes in train/val/test
assert max(e[train_idx]) == 4, "Train must include regimes 0-4"
```

### compound_full & compound_mnar_bias (3 regimes, moderate corruption)

```python
dataset = 'compound_full' # or 'compound_mnar_bias'
num_regimes = 3
num_samples = 9448
true_edges = 13

# Must verify all 3 regimes in train/val/test
assert max(e[train_idx]) == 2, "Train must include regimes 0-2"
```

---

## Checklist for Protocol Implementation

- [ ] Time-aware 60/20/20 split implemented (contiguous, not random)
- [ ] All regimes present in train/val/test (verified with assertions)
- [ ] A_true hidden during training (no early stopping on adj metrics)
- [ ] Validation uses only loss, not A_true-based metrics
- [ ] 5+ random seeds fixed and listed
- [ ] Split indices saved in `artifacts/splits/`
- [ ] Leakage test scenario runnable
- [ ] Hyperparameter search documented (space + selection criterion)
- [ ] Both ID and LODO protocols definable (or explicitly chosen)

---

**Next:** See [metrics.md](metrics.md) for exact evaluation metrics implementation.
