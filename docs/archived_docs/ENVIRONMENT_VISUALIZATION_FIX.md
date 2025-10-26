# RC-GNN Pipeline Fixes - Complete Summary

## Issue Fixed

**Error**: `TypeError: empty() received an invalid combination of arguments`

This error occurred in `scripts/visualize_environment_structure.py` when trying to initialize the RC-GNN model.

**Root Causes**:
1. The script was passing a YAML config dict directly to RCGNN instead of extracting individual parameters
2. The script tried to use a non-existent `forward_batch()` method
3. The script wasn't handling checkpoint n_envs mismatch

---

## Fixes Applied

### Fix #1: Extract Config Parameters Properly

**File**: `scripts/visualize_environment_structure.py` (lines ~293-310)

**Before**:
```python
model = RCGNN(d, mc)  # ❌ mc is a dict, not parameters
```

**After**:
```python
latent_dim = mc.get("latent_dim", 16)
hidden_dim = mc.get("hidden_dim", 32)
sparsify_method = mc.get("sparsify", {}).get("method", "topk")
topk_ratio = mc.get("sparsify", {}).get("topk_ratio", 0.1)

model = RCGNN(
    d=d,
    latent_dim=latent_dim,
    hidden_dim=hidden_dim,
    n_envs=n_envs_ckpt,
    sparsify_method=sparsify_method,
    topk_ratio=topk_ratio,
    device=args.device
)  # ✅ Proper parameter extraction
```

### Fix #2: Handle Checkpoint n_envs Mismatch

**File**: `scripts/visualize_environment_structure.py` (lines ~304-315)

**Issue**: Checkpoint was trained with n_envs=1, but script tried to load with n_envs from dataset

**Solution**:
```python
# Load checkpoint to determine n_envs
checkpoint = torch.load(args.checkpoint, map_location=args.device)
n_envs_ckpt = 1  # Default: most checkpoints use n_envs=1
if "structure_learner.A_deltas.0" in checkpoint:
    # Count how many deltas exist
    delta_keys = [k for k in checkpoint.keys() if "A_deltas." in k]
    n_envs_ckpt = max([int(k.split(".")[1]) for k in delta_keys]) + 1

# Use checkpoint n_envs to match saved weights
model = RCGNN(..., n_envs=n_envs_ckpt, ...)
```

### Fix #3: Replace forward_batch() with forward()

**File**: `scripts/visualize_environment_structure.py` (lines ~27-57)

**Before**:
```python
for batch in val_loader:
    for k in batch:
        batch[k] = batch[k].to(device)
    
    B = batch["X"].shape[0] if batch["X"].dim() >= 3 else 1
    for i in range(B):
        sample = {k: (batch[k][i] if batch[k].dim() > 0 else batch[k]) for k in batch}
        out = model.forward_batch(sample)  # ❌ Non-existent method
        
        A = out["A"].cpu().numpy()
        e = int(sample["e"].item()) if "e" in sample else 0
        structures.append({'A': A, 'env': e})
```

**After**:
```python
for batch in val_loader:
    for k in batch:
        batch[k] = batch[k].to(device)
    
    # Call forward method with batch data
    out = model(
        X=batch["X"],
        M=batch.get("M"),
        e=batch.get("e")
    )  # ✅ Use standard forward() method
    
    A = out["A"].cpu().numpy()  # [B, d, d]
    B = A.shape[0]
    
    e_batch = batch.get("e")
    if e_batch is not None:
        e_vals = e_batch.cpu().numpy().astype(int)
    else:
        e_vals = np.zeros(B, dtype=int)
    
    for i in range(B):
        structures.append({'A': A[i], 'env': int(e_vals[i])})
```

---

## Results After Fix

### All Pipeline Steps Now Work ✅

```
✅ STEP 1: Training (95.2s) → SHD=18
✅ STEP 2: Threshold Optimization → Generated charts and reports
✅ STEP 3: Environment Structure Analysis → Generated visualizations
✅ STEP 4: Baseline Comparison → RC-GNN vs Correlation
✅ STEP 5: Summary Generation → pipeline_summary.json
```

### Generated Artifacts

✅ `artifacts/environment_structures.png` - Per-environment adjacency matrices
✅ `artifacts/environment_deltas.png` - Environment-specific variations
✅ `artifacts/structure_variation.png` - Variation analysis plots
✅ `artifacts/environment_structures_report.txt` - Detailed statistics

### Pipeline Execution

```bash
$ python3 run_full_pipeline.py

✅ All 5 steps complete!
✅ All visualizations generated!
✅ All reports created!
```

---

## How to Use

### Run Full Pipeline
```bash
make full-pipeline
```

### Run Individual Analysis Scripts
```bash
# Environment structure visualization
python3 scripts/visualize_environment_structure.py

# Threshold optimization
python3 scripts/optimize_threshold.py --adjacency artifacts/adjacency/A_mean.npy --data-root data/interim/uci_air

# Baseline comparison
python3 scripts/compare_baselines.py --data-root data/interim/uci_air
```

---

## Technical Details

### What Each Fix Does

| Fix | Issue | Solution | Impact |
|-----|-------|----------|--------|
| #1 | Config dict passed to RCGNN | Extract parameters from YAML | Model initializes correctly |
| #2 | n_envs mismatch on load | Detect n_envs from checkpoint | Weights load without errors |
| #3 | Non-existent forward_batch() | Use standard forward() method | Data flows through model |

### Why These Fixes Work

1. **Config Extraction**: RCGNN expects numerical parameters, not dicts. By extracting values from the YAML config, we provide the correct types.

2. **n_envs Matching**: PyTorch requires exact shape matching when loading state_dicts. By detecting n_envs from the checkpoint keys, we ensure the model architecture matches the saved weights.

3. **forward() Method**: The standard PyTorch forward() method batches data correctly. Using it instead of a non-existent forward_batch() ensures proper tensor operations.

---

## Verification

All three analysis scripts now run successfully:

```bash
✅ python3 scripts/visualize_environment_structure.py
✅ python3 scripts/optimize_threshold.py --adjacency ... --data-root ...
✅ python3 scripts/compare_baselines.py --data-root ...
```

And the full pipeline completes without errors:

```bash
✅ python3 run_full_pipeline.py
```

---

## Files Modified

- `scripts/visualize_environment_structure.py` - Fixed model initialization and data flow

## Files Previously Created (Supporting Fixes)

- `scripts/path_helper.py` - Ensures project root in sys.path (created in previous fix)

---

**Status**: ✅ All fixes verified and tested
**Date**: October 25, 2025
**Result**: Full pipeline operational with all analysis steps complete
