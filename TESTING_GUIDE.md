# 🧪 Week 1 Infrastructure Testing Guide

**Status**: ✅ **ALL TESTS PASSING**

This guide shows you how to test and validate all Week 1 infrastructure before running hypothesis tests.

---

## 🚀 Quick Start (5 minutes)

### Run All Quick Tests
```bash
python scripts/test_week1_infrastructure.py --test quick
```

**Output**: Tests stability metrics, benchmarks, and multi-environment evaluation

**Expected**: ✅ All 3 tests pass

---

## 🔍 Detailed Test Suite

### Test 1: Stability Metrics (Unit Tests)
```bash
python scripts/test_week1_infrastructure.py --test metrics
```

**What it tests**:
- ✅ `adjacency_variance()` — Computes Frobenius norm variance
- ✅ `edge_set_jaccard()` — Computes Jaccard similarity of edge sets
- ✅ `policy_consistency()` — Tracks policy-relevant pathways
- ✅ Stability sensitivity — Verifies 100,000× difference between high/low stability

**Sample output**:
```
[Test 1.1] adjacency_variance()
  ✅ PASS: variance = 0.178633

[Test 1.2] edge_set_jaccard()
  ✅ PASS: jaccard similarity = 0.576190

[Test 1.3] policy_consistency()
  ✅ PASS: policy metrics = {'consistency': 0.944, 'presence': 0.917, 'variance': 0.014}

[Test 1.4] Stability Sensitivity (High vs Low)
  High stability variance: 0.000000
  Low stability variance:  0.047460
  Ratio (low/high):        474598753.2×
  ✅ PASS: Sensitivity verified (ratio > 100×)
```

---

### Test 2: Synthetic Benchmarks (Data Validation)
```bash
python scripts/test_week1_infrastructure.py --test benchmarks
```

**What it tests**:
- ✅ All 6 benchmarks exist and have correct directory structure
- ✅ All required files present (A_true, X, M, S, e, metadata)
- ✅ Data shapes are correct
- ✅ Metadata is valid JSON with expected fields

**Benchmarks checked**:
- h1_easy: 1200 train samples, 300 val
- h1_medium: 1920 train samples, 480 val
- h1_hard: 2800 train samples, 700 val
- h2_multi_env: 1600 train samples, 400 val
- h2_stability: 1600 train samples, 400 val
- h3_policy: 1920 train samples, 480 val

**Sample output**:
```
[Test 3] Checking h1_easy...
  ✅ PASS:
     - Shape: (1200, 50, 15) (samples, timesteps, features)
     - Edges: 30
     - Environments: 3
     - Size: 12.9 MB
```

---

### Test 3: Multi-Environment Evaluation (Integration Test)
```bash
python scripts/test_week1_infrastructure.py --test eval
```

**What it tests**:
- ✅ Load real benchmark data
- ✅ Create RC-GNN model
- ✅ Run eval_epoch_multi_env() function
- ✅ Compute stability metrics on real data
- ✅ Verify all expected metric keys are present

**Sample output**:
```
[Test 2.3] Running eval_epoch_multi_env()
  ✓ Metrics computed successfully
  ✓ Keys: ['recon_loss', 'A_mean', 'shd', 'adjacency_variance', 'edge_set_jaccard']
  ✓ Multi-environment metrics available:
    - adjacency_variance: 0.000000
    - edge_set_jaccard: 1.000000
```

---

### Test 4: End-to-End Training (Full Pipeline)
```bash
python scripts/test_week1_infrastructure.py --test training
```

**What it tests**:
- ✅ Load benchmark data into PyTorch DataLoader
- ✅ Create RC-GNN model with correct architecture
- ✅ Run forward pass (no errors)
- ✅ Compute loss and backpropagate
- ✅ Verify loss decreases over epochs
- ✅ Run evaluation with eval_epoch_multi_env()

**Duration**: ~2-3 minutes (trains for 3 epochs on small batch)

---

## 🎯 Testing Before Hypothesis Tests

### Recommended Testing Sequence

**Step 1: Quick validation (5 min)**
```bash
python scripts/test_week1_infrastructure.py --test quick
```
→ Ensures all basic components work

**Step 2: Unit tests (2 min)**
```bash
python scripts/test_week1_infrastructure.py --test metrics
```
→ Validates metric calculations with synthetic data

**Step 3: Data validation (1 min)**
```bash
python scripts/test_week1_infrastructure.py --test benchmarks
```
→ Confirms all 6 benchmarks are ready

**Step 4: Integration test (3 min)**
```bash
python scripts/test_week1_infrastructure.py --test eval
```
→ Tests metrics on real benchmark data

**Step 5: Full pipeline (3 min)**
```bash
python scripts/test_week1_infrastructure.py --test training
```
→ Validates end-to-end training loop

**Total time**: ~15 minutes

---

## 🧪 Manual Testing (Advanced)

### Test Stability Metrics Directly in Python

```python
from src.training.metrics import adjacency_variance, edge_set_jaccard, policy_consistency
import numpy as np

# Create test adjacency matrices
A1 = np.random.rand(10, 10)
A2 = np.random.rand(10, 10)
A3 = np.random.rand(10, 10)

A_by_env = {0: A1, 1: A2, 2: A3}

# Test 1: Adjacency variance
var = adjacency_variance(A_by_env)
print(f"Variance: {var}")  # Should be > 0

# Test 2: Edge-set Jaccard
jac = edge_set_jaccard(A_by_env, threshold=0.5)
print(f"Jaccard: {jac}")  # Should be in [0, 1]

# Test 3: Policy consistency
policy_edges = [(0, 1), (1, 2), (2, 3)]
pol = policy_consistency(A_by_env, policy_edges)
print(f"Policy: {pol}")  # Should have 'consistency', 'presence', 'variance' keys
```

---

### Test Multi-Environment Evaluation Directly

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from src.models.rcgnn import RCGNN
from src.training.loop import eval_epoch_multi_env

# Load a benchmark
data_root = "data/interim/synth_corrupted_h1_easy"
X_val = np.load(f"{data_root}/X_val.npy")
A_true = np.load(f"{data_root}/A_true.npy")

# Create simple model
d = X_val.shape[-1]
model = RCGNN(d=d, latent_dim=8, hidden_dim=16, n_envs=3, device="cpu")

# Create dataloader
X_t = torch.from_numpy(X_val).float()
e_t = torch.zeros(X_val.shape[0], dtype=torch.long)  # Dummy environment labels
loader = DataLoader(TensorDataset(X_t, e_t), batch_size=32)

# Run evaluation
with torch.no_grad():
    metrics = eval_epoch_multi_env(
        model=model,
        eval_loader=loader,
        A_true=A_true,
        device="cpu",
        threshold=0.5,
        policy_edges=[(0, 1), (1, 2), (2, 3)]  # Optional
    )

print(metrics)
# Output: {
#   'shd': 45.0,
#   'precision': 0.12,
#   'recall': 0.15,
#   'f1': 0.13,
#   'auc': 0.58,
#   'adjacency_variance': 0.0001,
#   'edge_set_jaccard': 0.85,
#   'policy_consistency': 0.75,
#   'policy_presence': 0.9,
#   'policy_variance': 0.05
# }
```

---

### Test Benchmark Data Loading

```python
import numpy as np
import json
from pathlib import Path

# Load benchmark
bench = "h1_easy"
data_root = f"data/interim/synth_corrupted_{bench}"

# Load metadata
with open(f"{data_root}/meta.json") as f:
    meta = json.load(f)

print(f"Graph type: {meta['graph_type']}")
print(f"Nodes: {meta['d']}")
print(f"Edges: {meta['edges']}")
print(f"Environments: {meta['n_envs']}")
print(f"Mechanism: {meta['mechanism']}")
print(f"Corruption configs: {meta['corruption_configs']}")

# Load data
A_true = np.load(f"{data_root}/A_true.npy")
X_train = np.load(f"{data_root}/X_train.npy")
M_train = np.load(f"{data_root}/M_train.npy")
S_train = np.load(f"{data_root}/S_train.npy")
e_train = np.load(f"{data_root}/e_train.npy")

print(f"\nData shapes:")
print(f"A_true: {A_true.shape}")
print(f"X_train: {X_train.shape} (samples, timesteps, features)")
print(f"M_train: {M_train.shape} (missingness: 1=observed, 0=missing)")
print(f"S_train: {S_train.shape} (clean signals before corruption)")
print(f"e_train: {e_train.shape} (environment labels)")

print(f"\nEnvironment distribution:")
for env in range(meta['n_envs']):
    count = (e_train == env).sum()
    print(f"  Env {env}: {count} samples")

print(f"\nMissing data rate:")
for env in range(meta['n_envs']):
    mask_env = M_train[e_train == env]
    missing_rate = 1 - mask_env.mean()
    print(f"  Env {env}: {missing_rate*100:.1f}% missing")
```

---

## ✅ Test Coverage Summary

| Component | Test Type | Status | Command |
|-----------|-----------|--------|---------|
| Stability Metrics | Unit | ✅ PASS | `--test metrics` |
| Benchmarks | Data Validation | ✅ PASS | `--test benchmarks` |
| Multi-Env Evaluation | Integration | ✅ PASS | `--test eval` |
| End-to-End Training | Full Pipeline | ✅ PASS | `--test training` |

---

## 🔧 Troubleshooting

### Issue: "Benchmark not found"
**Solution**: Generate benchmarks first
```bash
python scripts/synth_corruption_benchmark.py --benchmark h1_easy --seed 42
```

### Issue: "Module not found" errors
**Solution**: Ensure you're running from project root
```bash
cd /path/to/rcgnn
python scripts/test_week1_infrastructure.py --test quick
```

### Issue: Test runs but gives warnings
**Solution**: This is normal for untrained models. Metrics should still compute without errors.
Verify by checking for "✅ PASS" in output.

### Issue: Out of memory
**Solution**: Test suite uses small batches (8-32 samples). If still failing, reduce batch sizes in test file.

---

## 📊 Expected Test Output

### Stability Metrics Test
```
✅ TEST 1 PASSED: All stability metrics working correctly
```

### Benchmarks Test
```
✅ TEST 3 PASSED: All benchmarks valid and ready
```

### Multi-Environment Evaluation Test
```
✅ TEST 2 PASSED: eval_epoch_multi_env() working correctly
```

### Training Test
```
✅ TEST 4 PASSED: End-to-end training works correctly
```

### Overall Summary
```
🎉 ALL TESTS PASSED! Infrastructure ready for hypothesis testing
```

---

## 📈 Next Steps After Testing

Once all tests pass:

1. **Read benchmarks guide**
   ```bash
   cat BENCHMARK_SUMMARY.md
   ```

2. **Start H1 hypothesis test**
   ```bash
   python scripts/train_rcgnn.py configs/data_h1_easy.yaml configs/model.yaml configs/train.yaml --epochs 100
   ```

3. **Test on H2 benchmarks** (stability comparison)
   ```bash
   # Compare with/without invariance loss
   python scripts/train_rcgnn.py configs/data_h2_stability.yaml configs/model.yaml configs/train.yaml --model.loss.lambda_inv 1.0
   ```

4. **Test on H3 benchmarks** (policy consistency)
   ```bash
   python scripts/train_rcgnn.py configs/data_h3_policy.yaml configs/model.yaml configs/train.yaml --epochs 250
   ```

---

## 💡 Tips

- **Quick smoke test before long runs**: Always run `--test quick` before starting hypothesis tests
- **Debug metrics**: If metrics seem wrong, use `--test metrics` to verify calculation correctness
- **Check data quality**: Run `--test benchmarks` to ensure all 6 benchmarks are ready
- **Validate pipeline**: Run `--test training` before committing to multi-hour hypothesis runs

---

## 📝 Test Results Log

After running tests, save results:

```bash
# Save full test output
python scripts/test_week1_infrastructure.py --test all > test_results.log 2>&1

# View results
cat test_results.log
```

---

**All set!** Your Week 1 infrastructure is tested and ready for hypothesis testing. 🚀

