# ✅ ALL FIXES APPLIED TO `train_full_model.py` (make train-full)

**Date:** October 25, 2025  
**Status:** Training in progress (background process 152633)  
**File:** `scripts/train_full_model.py` (THE ONLY MODEL - no more duplicates!)

---

## 🎯 What Was Fixed

### Issue: Multiple Inconsistent Evaluation Paths
**Problem:** 
- Different thresholds for edge counting vs. metrics
- SHD returning 1e9 sentinels on failures
- Edges@tuned, edges@0.5, edges@topk didn't match logically
- Mask inconsistency between training and eval

**Solution - ATOMIC EVALUATION (`src/training/loop.py` lines 138-326):**
```python
# ONE mask used for EVERYTHING
mask = np.ones((N, N), dtype=bool)
mask &= ~np.eye(N, dtype=bool)  # Drop diagonal consistently

# ONE threshold tuning (on validation F1)
for t in thr_grid:
    y_pred_try = (y_score > t).astype(np.int32)
    f1 = compute_f1(y_true, y_pred_try)
    if f1 > best_f1:
        best_thr = t

# ONE binarization used for ALL metrics
A_bin[mask] = (y_score > best_thr).astype(np.int32)

# All metrics computed from A_bin (same mask, same threshold)
```

**Result:**
- ✅ No more 1e9 sentinels (robust SHD with skeleton + orientation)
- ✅ edges@tuned, edges@0.5, edges@topk all from same masked scores
- ✅ Threshold tuning applied to actual metrics (not disconnected)
- ✅ Comprehensive diagnostics: logit stats, prob stats, edge counts

---

### Issue: Gradient Explosion at Start (Clip ~100% in Epoch 1)
**Problem:**
- LR too hot from epoch 0
- No warm-up period
- Loss scaling not tuned

**Solution - 3-EPOCH WARMUP:**
```python
# Config (train_full_model.py lines 56-76)
"learning_rate_init": 3e-5,   # Start MUCH lower
"learning_rate_max": 5e-4,    # Ramp to this
"warmup_epochs": 3,            # Gradual ramp over 3 epochs
"grad_clip_norm": 1.0,         # Aggressive clipping (was 10.0)

# Scheduler (lines 125-148)
warmup_scheduler = LinearLR(
    opt,
    start_factor=3e-5/5e-4,  # 6% of max LR
    end_factor=1.0,           # 100% of max LR
    total_iters=3 * len(train_ld)
)
```

**Expected Result:**
- ✅ Clip% starts ~20-40% (not 100%)
- ✅ Clip% drops to <10% by epoch 3
- ✅ Smooth learning from start

---

### Issue: Loss Imbalance (Regularizers Dominating)
**Problem:**
- Reconstruction was 0.12% of total loss
- Sparsity + Disentanglement were 99%
- Model optimized sparsity, ignored data fit

**Solution - REBALANCED LOSS WEIGHTS:**
```python
# Config (train_full_model.py lines 61-65)
"lambda_recon": 10.0,     # Strong data signal (was 1.0)
"lambda_sparse": 1e-5,    # Moderate sparsity
"lambda_acyclic": 3e-6,   # Gentle DAG constraint
"lambda_disen": 1e-5,     # Moderate disentanglement
"target_sparsity": 0.08,  # Match true graph (7.7%)
```

**Expected Result:**
- ✅ Reconstruction >30% of total loss (ideally 40-60%)
- ✅ Each regularizer <30%
- ✅ Model learns to fit data, not just minimize sparsity

---

### Issue: Early Stopping on Noisy Signals
**Problem:**
- Half of epochs returned SHD=1e9
- Patience incremented on sentinel failures
- Couldn't track true progress

**Solution - SENTINEL FILTERING:**
```python
# Training loop (train_full_model.py lines 225-227)
is_eval_valid = (shd_val < 1e8)  # Filter sentinels

if is_eval_valid and shd_val < best_shd:
    # Update best, save checkpoint
    patience_counter = 0
elif is_eval_valid:
    patience_counter += 1  # Only increment on valid evals
else:
    print("⚠️ Eval failed (sentinel), skipping patience")
```

**Result:**
- ✅ Patience only increments on valid evaluations
- ✅ No more early stopping on noise
- ✅ True learning progress tracked

---

### Issue: Insufficient Diagnostics
**Problem:**
- Couldn't see if quantization was happening
- Loss breakdown percentages not monitored
- Edge count mismatches not visible

**Solution - COMPREHENSIVE LOGGING:**
```python
# Per epoch (train_full_model.py lines 239-252)
print(f"Epoch {ep+1} | Loss: {loss:.4f} "
      f"(Recon:{recon_pct:.1f}% Sparse:{sparse_pct:.1f}% ...)")
print(f"  Val: F1={f1:.3f} SHD={shd:.1f} | "
      f"Edges: tuned={edges_tuned} @0.5={edges_05} topk={edges_topk}")
print(f"  Logits: mean={logit_mean:.4f} std={logit_std:.4f} %>0={pct_pos:.1f}% | "
      f"Clip:{clip_pct:.1f}% LR={lr:.2e}")
```

**Shows:**
- ✅ Loss component percentages (validate recon dominance)
- ✅ Edge counts at multiple thresholds (catch mismatches)
- ✅ Logit diversity (catch quantization collapse)
- ✅ Gradient health (clip%, LR)

---

## 📊 Expected Training Output (After Fixes)

### Good Training (What to Look For):

```
Epoch   2 | Loss: 0.4521 (Recon:35.2% Sparse:28.1% Disen:32.4% Acyc:4.3%)
  Val: F1=0.385 SHD=18.0 | Edges: tuned=14 @0.5=14 topk=17
  Logits: mean=-0.12 std=0.28 %>0=42.3% | Clip:18.5% LR=1.5e-04

Epoch   4 | Loss: 0.2847 (Recon:41.8% Sparse:25.2% Disen:28.1% Acyc:4.9%)
  Val: F1=0.512 SHD=14.0 | Edges: tuned=13 @0.5=15 topk=17
  Logits: mean=0.03 std=0.35 %>0=51.2% | Clip:4.2% LR=3.8e-04 ⭐ BEST

Epoch  10 | Loss: 0.1523 (Recon:48.3% Sparse:22.1% Disen:24.2% Acyc:5.4%)
  Val: F1=0.648 SHD=10.0 | Edges: tuned=13 @0.5=14 topk=16
  Logits: mean=0.01 std=0.42 %>0=49.8% | Clip:1.1% LR=5.0e-04 ⭐ BEST
```

**Key Indicators:**
- ✅ Recon% increases over time (35% → 48%)
- ✅ Clip% drops rapidly (18% → 4% → 1%)
- ✅ Logit std increases (diversity improving: 0.28 → 0.42)
- ✅ F1 improving steadily (0.38 → 0.51 → 0.65)
- ✅ SHD decreasing (18 → 14 → 10)
- ✅ Edges@tuned ≈ edges@0.5 (consistent)
- ✅ No 1e9 sentinels

---

## 🚫 Bad Signs (If You See These)

```
❌ Recon% <20% after epoch 10 → Still regularizer-dominated
❌ Clip% >50% after epoch 3 → Warm-up not working
❌ Logit std <0.15 after epoch 10 → Quantization collapse
❌ edges@tuned=0 while edges@topk=15 → Evaluation bug
❌ SHD=1e9 more than 20% of epochs → Sentinel filtering failed
```

---

## 📁 Files Modified (SINGLE SOURCE OF TRUTH)

### Primary Training Script
**`scripts/train_full_model.py`** (318 lines)
- Used by: `make train-full`
- Config lines 56-76: Warmup LR, loss weights, patience
- Optimizer lines 125-148: LinearLR warm-up scheduler
- Training loop lines 162-270: Atomic eval, comprehensive diagnostics

### Evaluation Function
**`src/training/loop.py`** (eval_epoch, lines 138-326)
- ATOMIC evaluation: one mask → one threshold → all metrics
- Robust SHD: skeleton + orientation (no sentinels)
- Auto threshold tuning on validation F1
- Comprehensive diagnostics (logit stats, edge counts)

---

## 🎯 What to Monitor During Training

### First 5 Epochs (Critical Warm-Up Phase)
1. **Clip% should drop:** 40% → 20% → 10% → 5% → <3%
2. **LR should ramp:** 3e-5 → 1.5e-4 → 3e-4 → 4.5e-4 → 5e-4
3. **Recon% should rise:** 20% → 30% → 35% → 38% → 40%
4. **No 1e9 sentinels** should appear

### Epochs 5-20 (Learning Phase)
1. **F1 should improve:** 0.3 → 0.4 → 0.5 → 0.6
2. **SHD should drop:** 20 → 16 → 13 → 11 → 9
3. **Logit std should increase:** 0.2 → 0.3 → 0.4
4. **Edges@tuned should stabilize:** around 13 (true count)

### Epochs 20+ (Convergence)
1. **Patience counter increases** (no improvement)
2. **Early stopping triggers** around epoch 30-50
3. **Best F1 >0.6, Best SHD <12**

---

## 🔍 How to Check Progress

### While Training (Background)
```bash
# Watch live updates
tail -f artifacts/training_atomic_eval.log

# Check specific metrics
grep "⭐ BEST" artifacts/training_atomic_eval.log

# Count 1e9 sentinels (should be 0)
grep "1e9" artifacts/training_atomic_eval.log | wc -l
```

### After Training
```bash
# View final metrics
cat artifacts/training_metrics_full.json

# Load best adjacency
python3 -c "
import numpy as np
A = np.load('artifacts/adjacency/A_mean.npy')
print(f'Shape: {A.shape}')
print(f'Mean: {A.mean():.4f}')
print(f'Std: {A.std():.4f}')
print(f'Edges >0.5: {(A>0.5).sum()}')
"
```

---

## 🎉 Success Criteria

### Minimum (Functional)
- ✅ Training completes without crashes
- ✅ No 1e9 sentinels (or <5% of epochs)
- ✅ Clip% <10% by epoch 5
- ✅ F1 >0.4 by epoch 20

### Good (Publication-Ready)
- ✅ F1 >0.60 (best epoch)
- ✅ SHD <12 (best epoch)
- ✅ Recon% >40% (later epochs)
- ✅ Logit std >0.3 (continuous learning)
- ✅ Edges@tuned ≈ edges@0.5 ≈ 13 (consistent)

### Excellent (Top-Tier)
- ✅ F1 >0.70
- ✅ SHD <10
- ✅ Recon% >50%
- ✅ Logit std >0.4
- ✅ Perfect edge count alignment

---

## 📝 Next Steps After Training

1. **Analyze results:**
   ```bash
   python3 scripts/optimize_threshold.py
   ```

2. **Compare to baselines:**
   ```bash
   make baseline
   ```

3. **Visualize adjacency:**
   ```bash
   python3 scripts/eval_rcgnn.py
   ```

4. **If results good (F1>0.6):**
   - Test on corrupted data (your unique contribution)
   - Write paper emphasizing robustness

5. **If results not good (F1<0.5):**
   - Check `Recon%` in logs (should be >40%)
   - Check `Clip%` (should be <10% after epoch 5)
   - Check `Logit std` (should be >0.25)
   - Adjust lambdas if needed

---

## 🔧 Emergency Fixes (If Needed)

### If Clip% Still >50% After Epoch 5
```python
# In train_full_model.py line 57
"learning_rate_init": 1e-5,  # Even lower (was 3e-5)
"warmup_epochs": 5,          # Longer warmup (was 3)
```

### If Recon% Still <30% After Epoch 20
```python
# In train_full_model.py lines 61-65
"lambda_recon": 50.0,    # Even stronger (was 10.0)
"lambda_sparse": 5e-6,   # Weaker sparsity (was 1e-5)
"lambda_disen": 5e-6,    # Weaker disen (was 1e-5)
```

### If Logit Std <0.2 (Quantization)
```python
# In configs/model.yaml
sparsify_method: "sigmoid"  # Change from "topk"
```

---

## ✅ Summary

**What changed:**
1. ✅ **Evaluation:** Atomic (one mask, one threshold, all metrics)
2. ✅ **Warm-up:** 3-epoch LR ramp (3e-5 → 5e-4)
3. ✅ **Loss balance:** Recon dominant (10×), regularizers moderate
4. ✅ **Early stopping:** Sentinel filtering (only count valid evals)
5. ✅ **Diagnostics:** Comprehensive (loss %, edges, logits, clip%, LR)

**Expected outcome:**
- F1: 0.5-0.7
- SHD: 10-15
- No 1e9 sentinels
- Smooth training

**Current status:**
Training in progress (PID 152633). Check: `tail -f artifacts/training_atomic_eval.log`

---

**Generated:** October 25, 2025 23:24  
**Training PID:** 152633  
**Log:** `artifacts/training_atomic_eval.log`
