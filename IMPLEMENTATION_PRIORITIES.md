# 🎯 Implementation Priorities: Ranked by Impact & Urgency

**Purpose**: Clear, actionable priority list for next implementation sprint.

**Last Updated**: October 27, 2025

---

## Priority Matrix: Impact vs Effort

```
                    HIGH EFFORT
                        ↑
                        │
          HARD, BIG WIN  │  HARD, NECESSARY
                        │
                        │
                        ├─────────────→ HIGH IMPACT
                        │
                        │
       EASY, NICE        │  EASY, QUICK WIN
         TO HAVE         │
                        ↓
                    LOW EFFORT
```

---

## 🔴 PRIORITY 1: Infrastructure (Must Start Monday)

### 1.1 **Stability Metrics** ⚡ EASIEST FIRST
- **Impact**: 🔴 CRITICAL—Validates paper's core innovation
- **Effort**: 2-4 hours
- **Why First**: Quick win; enables H1/H2 experiments; tests evaluation framework

**Deliverable**: `src/training/metrics.py` additions

```python
def adjacency_variance(A_dict):
    """Frobenius norm variance across environments."""
    
def edge_set_jaccard(A_dict, threshold=0.5):
    """Jaccard similarity of thresholded edge sets."""
    
def policy_consistency(A_dict, policy_edges):
    """Stability of specified pathways."""
```

**Success Criteria**:
- ✅ Functions return scalar values (no NaN/Inf)
- ✅ Tested on dummy data: known variance should compute correctly
- ✅ Integrated into eval_epoch: metrics printed after each epoch

**Blocking**: Nothing—can do immediately
**Blocked By**: Nothing

---

### 1.2 **Synthetic Corruption Benchmark** 🏗️ PRIORITY
- **Impact**: 🔴 CRITICAL—Enables H1/H2 hypothesis tests
- **Effort**: 6-8 hours
- **Why Next**: Depends on stability metrics; needed for experiments

**Deliverable**: Extended `scripts/synth_bench.py` + `data/interim/synth_corrupted/`

```python
def generate_multi_env_corruption(
    G_star,           # Known causal graph
    T=5000,           # Time steps
    n_envs=4,         # Number of environments
    missingness_rates=[0.4, 0.5, 0.5, 0.6],  # Per environment
    missingness_types=["MCAR", "MAR", "MNAR", "MCAR"],
    noise_levels=[0.1, 0.15, 0.1, 0.2],
    drift_strengths=[0.02, 0.03, 0.02, 0.05],
    seed=42
):
    """Generate multi-environment benchmarks with known ground truth."""
```

**Output Format**:
```
data/interim/synth_corrupted/
├── meta.json                # {n_nodes, n_edges, n_envs, missingness_rates, ...}
├── A_true.npy              # Ground truth [d, d]
├── X.npy                   # Observations [T, d]
├── M.npy                   # Missingness [T, d]
├── e.npy                   # Environment indices [T]
└── drift_params.npy        # For reference
```

**Success Criteria**:
- ✅ A_true recoverable: SHD = 0 if perfect recovery
- ✅ Missingness realistic: 40-60% entries missing
- ✅ Multi-env corruption: Each environment has distinct pattern
- ✅ Datasets save/load correctly

**Blocking**: H1/H2 experiments
**Blocked By**: Nothing—can parallelize with other tasks

---

### 1.3 **GRU-D Masking-Aware Imputer** 🧠 COMPLEX
- **Impact**: 🔴 CRITICAL—Handles missingness as paper describes
- **Effort**: 8-12 hours
- **Why Last in Phase 1**: Complex; other tasks don't depend on it; can do in parallel

**Deliverable**: `src/models/encoders.py` additions

```python
class GRUDImputer(nn.Module):
    """Imputes missing values using GRU-D architecture."""
    
    def forward(self, X, M):
        """
        Args:
            X: [B, T, d] observations (with NaN for missing)
            M: [B, T, d] missingness mask (0=missing, 1=observed)
        
        Returns:
            X_imputed: [B, T, d] imputed values
            uncertainty: [B, T, d] uncertainty estimates
        """

class TriLatentEncoderV2(nn.Module):
    """Updated to use masking-aware imputer."""
    
    def __init__(self, input_dim, latent_dim=16, hidden_dim=32):
        super().__init__()
        self.imputer = GRUDImputer(input_dim, hidden_dim)
        self.signal_encoder = ...
        self.noise_encoder = ...
        self.bias_encoder = ...
    
    def forward(self, X, M, e):
        # CHANGED: Add imputation step
        X_imputed, uncertainty = self.imputer(X, M)
        z_s = self.signal_encoder(X_imputed)
        z_n = self.noise_encoder(uncertainty)
        z_b = self.bias_encoder(e)
        return z_s, z_n, z_b
```

**Success Criteria**:
- ✅ Handles MCAR/MAR/MNAR: No crashes on different patterns
- ✅ Uncertainty meaningful: High where data missing
- ✅ Imputation quality: RMSE on held-out missing values < baseline
- ✅ Gradients flow: Backprop through imputer works

**Blocking**: Higher fidelity experiments (optional)
**Blocked By**: Nothing

**Note**: Can use simplified time-aware imputation if full GRU-D complex:
```python
class SimpleTimeAwareImputer(nn.Module):
    def forward(self, X, M):
        # Forward fill missing values
        # Add learned uncertainty based on local variance
        # Return: X_imputed, uncertainty
```

---

## 🟠 PRIORITY 2: Theory Validation (Week 2)

### 2.1 **Identifiability Test** (Proposition 1)
- **Impact**: 🔴 HIGH—Validates paper's core theoretical claim
- **Effort**: 6-8 hours
- **Depends On**: Synthetic corruption benchmark (Priority 1.2)

**Deliverable**: `scripts/test_identifiability.py`

```python
def test_identifiability():
    """
    H0: SHD remains low (close to oracle) despite environment shifts
    H1: RC-GNN can recover G_star from multi-environment corruption
    """
    
    # 1. Generate ground truth SCM
    G_star = generate_scm(d=20, edge_density=0.3)
    
    # 2. Create 4 environments with different corruptions
    datasets = []
    for env_id in range(4):
        X, M, e = generate_multi_env_corruption(
            G_star,
            n_envs=1,
            missingness_rate=0.5,
            noise_level=0.1 * (1 + 0.2 * env_id),  # Vary noise
            drift_strength=0.02 * (1 + env_id)
        )
        datasets.append((X, M, e))
    
    # 3. Run RC-GNN on combined multi-environment data
    model = train_rc_gnn(datasets)
    
    # 4. Measure: SHD vs oracle
    A_pred = model.get_adjacency()
    shd = compute_shd(A_pred, G_star)
    
    # 5. Plot: SHD vs environment diversity
    # Expected: SHD increases slightly with environment shift
    #           but remains close to oracle (validates identifiability)
```

**Expected Results**:
- SHD within 20% of oracle despite 4 corruption regimes
- Demonstrates: Partial Invariance assumption holds

---

### 2.2 **Stability Bound Verification** (Proposition 2)
- **Impact**: 🔴 HIGH—Validates stability improvement guarantee
- **Effort**: 4-6 hours
- **Depends On**: Synthetic corruption benchmark (Priority 1.2)

**Deliverable**: `scripts/test_stability_bound.py`

```python
def test_stability_bound():
    """
    Verify: Var_{e,e'}[A^(e) - A^(e')] <= (2/lambda_inv) * E[L_recon + L_disent]
    """
    
    # 1. Train RC-GNN with lambda_inv = 0.1
    model = train_rc_gnn(..., lambda_inv=0.1)
    
    # 2. Extract adjacency matrices per environment
    A_by_env = {}
    for env_id in range(n_envs):
        A_by_env[env_id] = model.get_adjacency(env_idx=env_id)
    
    # 3. Measure: Actual variance
    actual_variance = adjacency_variance(A_by_env)  # Using metric from Priority 1.1
    
    # 4. Compute: Theoretical bound
    L_recon_mean = ...  # From training log
    L_disent_mean = ...  # From training log
    theoretical_bound = (2.0 / 0.1) * (L_recon_mean + L_disent_mean)
    
    # 5. Report: Ratio (should be < 1.0)
    ratio = actual_variance / theoretical_bound
    assert ratio <= 1.0, f"Bound violated: {ratio} > 1.0"
    
    return {
        "actual_variance": actual_variance,
        "theoretical_bound": theoretical_bound,
        "ratio": ratio,
        "validation": "PASS" if ratio <= 1.0 else "FAIL"
    }
```

**Expected Results**:
- Ratio < 1.0 (actual variance ≤ bound)
- Demonstrates: Stability improvement guarantee holds

---

## 🔴 PRIORITY 3: Hypothesis Testing (Week 3)

### 3.1 **H1: Structural Accuracy Under Missingness** 🎯 MAIN PAPER CLAIM
- **Impact**: 🔴 CRITICAL—Core hypothesis in abstract
- **Effort**: 10-12 hours
- **Depends On**: Priorities 1.1, 1.2 (metrics, synthetic data)
- **Blocks**: Paper's main narrative

**Deliverable**: `scripts/run_h1_experiment.py` + `experiments/h1_results.csv`

```python
def run_h1_experiment():
    """
    Hypothesis: RC-GNN maintains structural accuracy (SHD within 15% of oracle)
                under high missingness (40-60%) where baselines degrade (>40% SHD increase)
    """
    
    # 1. Generate synthetic benchmark: 4-5 environments, 40-60% missingness
    benchmark = generate_h1_benchmark()
    
    # 2. Run all methods
    results = {}
    for method in ["RCGNN", "NOTEARS", "DCDI", "DECI", "MissDAG"]:
        print(f"Running {method}...")
        model = train_model(method, benchmark)
        A_pred = model.get_adjacency()
        
        # 3. Measure: SHD, Precision, Recall, F1
        metrics = evaluate_structure(A_pred, benchmark["A_true"])
        results[method] = metrics
    
    # 4. Compare against oracle
    shd_oracle = 0  # Perfect recovery
    for method, metrics in results.items():
        shd_increase = (metrics["shd"] - shd_oracle) / shd_oracle * 100
        print(f"{method}: SHD={metrics['shd']:.1f} (+{shd_increase:.1f}%)")
    
    # 5. Validate H1
    rc_gnn_shd = results["RCGNN"]["shd"]
    baseline_shds = [results[m]["shd"] for m in ["NOTEARS", "DCDI", "DECI"]]
    
    h1_pass = (
        rc_gnn_shd <= 0.15 * len(benchmark["A_true"]) and  # SHD within 15% of oracle
        all(b_shd > rc_gnn_shd * 1.4 for b_shd in baseline_shds)  # Baselines >40% worse
    )
    
    return h1_pass, results
```

**Expected Output**:
```
H1 Results:
└─ Dataset: synth_4env_40_60_missingness
   ├─ RC-GNN:    SHD=2.3 (17% of oracle)  ✓ PASS
   ├─ NOTEARS:   SHD=5.1 (50% of oracle)
   ├─ DCDI:      SHD=4.7 (47% of oracle)
   ├─ DECI:      SHD=4.2 (42% of oracle)
   └─ MissDAG:   SHD=3.8 (38% of oracle)
   
Hypothesis: PASS ✓
  ✓ RC-GNN SHD 2.3 within 15% of oracle (threshold: 3.3)
  ✓ All baselines >40% worse than RC-GNN
```

**Success Criteria**:
- ✅ RC-GNN SHD ≤ 15% of oracle
- ✅ All baselines SHD increase > 40% relative to RC-GNN
- ✅ Statistical significance: p < 0.05 (t-test)

**Timeline**: 3-4 days

---

### 3.2 **H2: Stability Improvement via Invariance** 📉 CORE INNOVATION
- **Impact**: 🔴 CRITICAL—Validates structure-level invariance
- **Effort**: 6-8 hours
- **Depends On**: Priority 1.1 (stability metrics)
- **Blocks**: Ablation studies

**Deliverable**: `scripts/run_h2_experiment.py` + `experiments/h2_results.pdf`

```python
def run_h2_experiment():
    """
    Hypothesis: Invariance loss reduces cross-environment adjacency variance >60%
                compared to ablated version (lambda_inv=0)
    """
    
    # 1. Generate multi-environment benchmark: 4-5 environments, known G_star
    benchmark = generate_h2_benchmark(n_envs=4)
    
    # 2. Train WITH invariance
    print("Training RC-GNN with lambda_inv=0.1...")
    model_with_inv = train_rc_gnn(benchmark, lambda_inv=0.1)
    A_with_inv = model_with_inv.get_adjacency_per_env()  # Dict: env_id -> A
    var_with_inv = adjacency_variance(A_with_inv)
    
    # 3. Train WITHOUT invariance
    print("Training RC-GNN with lambda_inv=0.0...")
    model_no_inv = train_rc_gnn(benchmark, lambda_inv=0.0)
    A_no_inv = model_no_inv.get_adjacency_per_env()
    var_no_inv = adjacency_variance(A_no_inv)
    
    # 4. Compute: Variance reduction
    reduction_pct = (1 - var_with_inv / var_no_inv) * 100
    
    # 5. Validate H2
    h2_pass = reduction_pct > 60  # 60% reduction
    
    return h2_pass, {
        "variance_with_inv": var_with_inv,
        "variance_without_inv": var_no_inv,
        "reduction_percent": reduction_pct
    }
```

**Expected Output**:
```
H2 Results:
├─ Variance WITH invariance (λ=0.1):    0.032
├─ Variance WITHOUT invariance (λ=0.0): 0.081
└─ Reduction: 60.5% ✓ PASS

Hypothesis: PASS ✓
  ✓ Variance reduced from 0.081 to 0.032 (60.5% reduction > 60% threshold)
```

**Success Criteria**:
- ✅ Variance reduction > 60%
- ✅ Reduction statistically significant (p < 0.05)
- ✅ Edge-set Jaccard also improves (policy consistency)

**Timeline**: 2-3 days

---

### 3.3 **H3: Expert Agreement on Policy-Relevant Pathways** 👥 UTILITY VALIDATION
- **Impact**: 🟡 HIGH—Demonstrates real-world applicability
- **Effort**: 8-10 hours (expert scheduling) or 4-6 hours (literature-based)
- **Depends On**: None (can parallelize)

**Deliverable**: `experiments/h3_expert_validation_report.pdf` OR `h3_literature_validation.csv`

**Option A: Real Expert Evaluation** (Slower, more credible)
```python
def run_h3_expert_validation():
    """
    Present learned graphs to air quality expert, score policy relevance
    """
    
    # 1. Run RC-GNN and baselines on UCI Air
    benchmark = load_uci_air()
    results = {}
    for method in ["RCGNN", "NOTEARS", "DCDI"]:
        model = train_model(method, benchmark)
        results[method] = model.get_adjacency()
    
    # 2. Create expert survey
    questions = [
        "Which variable most influences PM2.5? (expected: Industry/Traffic)",
        "How strong is the connection between Temperature and Humidity? (expected: Strong)",
        "Does RH influence CO? (expected: Yes, indirect through T)",
        # ... more policy questions
    ]
    
    # 3. Present anonymized graphs (don't reveal which is RC-GNN)
    # Score: 1-5 confidence on 5 questions per method
    
    # 4. Analyze results
    expert_scores = {
        "RC-GNN": [4.5, 4.0, 5.0, 4.2, 3.8],  # Average: 4.3
        "NOTEARS": [3.2, 3.5, 2.8, 3.1, 2.9],  # Average: 3.1
        "DCDI": [3.0, 3.2, 3.1, 2.9, 3.0],     # Average: 3.04
    }
    
    rc_gnn_agree = sum(expert_scores["RC-GNN"]) / len(expert_scores["RC-GNN"])
    baseline_agree = np.mean([expert_scores["NOTEARS"], expert_scores["DCDI"]])
    
    h3_pass = rc_gnn_agree > 4.0 and rc_gnn_agree > 1.3 * baseline_agree
    
    return h3_pass, expert_scores
```

**Option B: Literature-Based Validation** (Faster, reproducible)
```python
def run_h3_literature_validation():
    """
    Score learned graphs against known air quality causal relationships
    from published domain literature
    """
    
    # 1. Define policy-relevant edges from literature
    policy_edges = {
        "PM2.5 <- Traffic": {"confidence": 0.95, "sources": ["WHO, 2021", ...]},
        "PM2.5 <- Industry": {"confidence": 0.90, "sources": [...]}
        # ... known causal relationships
    }
    
    # 2. Run RC-GNN and baselines
    results = {}
    for method in ["RCGNN", "NOTEARS", "DCDI"]:
        A_pred = train_and_extract(method, benchmark)
        results[method] = evaluate_policy_alignment(A_pred, policy_edges)
    
    # 3. Score: % of policy edges correctly recovered
    rc_gnn_recovery = results["RCGNN"]["edge_recovery_pct"]  # e.g., 85%
    baseline_recovery = np.mean([results[m]["edge_recovery_pct"] for m in baselines])
    
    h3_pass = rc_gnn_recovery > 80 and rc_gnn_recovery > baseline_recovery
    
    return h3_pass, results
```

**Success Criteria**:
- ✅ Expert agreement > 80% (or literature recovery > 80%)
- ✅ RC-GNN outperforms baselines by >20%

**Timeline**: 1-2 weeks (expert) or 2-3 days (literature)

---

## 🟡 PRIORITY 4: Baselines & Ablations (Week 4)

### 4.1 **Unified Baseline Evaluation** 
- **Effort**: 4-6 hours
- **Depends On**: H1/H2 experiments

```python
def benchmark_all_methods():
    """Run all methods on all datasets with unified evaluation."""
    
    datasets = [
        "synth_4env_40_50_60_missingness",
        "uci_air",
        "synthetic_identifiability",
    ]
    
    methods = ["RCGNN", "NOTEARS", "DCDI", "DECI", "MissDAG"]
    
    results = pd.DataFrame(
        columns=["Dataset", "Method", "SHD", "Precision", "Recall", "F1", "Variance"]
    )
    
    for dataset in datasets:
        for method in methods:
            print(f"Running {method} on {dataset}...")
            A_pred = train_model(method, dataset)
            metrics = evaluate_all(A_pred, dataset)
            results = results.append(metrics)
    
    return results
```

### 4.2 **Ablation Studies**
- **Effort**: 3-4 hours
- **Depends On**: Nothing

```python
def run_ablations():
    """Measure contribution of each component."""
    
    configs = {
        "RC-GNN (Full)": {"inv": 0.1, "disent": 0.01, "recon": 1.0},
        "RC-GNN \\ Inv": {"inv": 0.0, "disent": 0.01, "recon": 1.0},
        "RC-GNN \\ Disent": {"inv": 0.1, "disent": 0.0, "recon": 1.0},
        "RC-GNN \\ Recon": {"inv": 0.1, "disent": 0.01, "recon": 0.0},
    }
    
    results = {}
    for name, config in configs.items():
        model = train_rc_gnn(..., **config)
        results[name] = evaluate_structure(model)
    
    return results
```

---

## 📋 Execution Checklist

### Monday (Oct 28)
- [ ] Stability metrics (Priority 1.1): 4 hours
  - Code review: paper Eqs. 11-12
  - Implement: 3 metrics functions
  - Test: dummy data validation

### Tuesday (Oct 29)
- [ ] Synthetic benchmark (Priority 1.2): 8 hours
  - Design corruption parameters
  - Generate datasets
  - Validate: A_true recoverable

### Wednesday-Thursday (Oct 30-31)
- [ ] GRU-D imputer (Priority 1.3): 8 hours
  - Implement: GRUDImputer class
  - Integrate: Into TriLatentEncoder
  - Test: Imputation quality on simulated missingness

### Friday (Nov 1)
- [ ] Week 1 recap: All infrastructure complete
- [ ] Begin: Identifiability test (Priority 2.1)

### Week 2 (Nov 3-8)
- [ ] Identifiability & stability bound tests: 12 hours
- [ ] Prepare: Hypothesis test setups

### Week 3 (Nov 8-13)
- [ ] Run: H1, H2, H3 experiments: 20+ hours
- [ ] Parallelize with baselines

### Week 4 (Nov 13-15)
- [ ] Baseline comparisons: 6 hours
- [ ] Results writing: 8 hours

---

## 🎯 Success Definition

**By November 15, 2025, you will have:**

✅ **Code**:
- GRU-D imputer integrated and tested
- Stability metrics computing correctly
- Synthetic multi-environment benchmarks generated
- All loss components validated

✅ **Theory**:
- Proposition 1 (identifiability) empirically verified
- Proposition 2 (stability bound) empirically verified

✅ **Experiments**:
- H1: RC-GNN SHD within 15% of oracle; baselines degrade >40%
- H2: Invariance reduces variance >60%
- H3: >80% expert/literature agreement

✅ **Paper**:
- Results section complete with figures and tables
- All claims backed by experimental evidence
- Baselines properly compared
- Ready for submission

---

**You've got this. 🚀 Start with Priority 1.1 (stability metrics) Monday morning.**
