# 📊 RC-GNN Publication-Quality Analysis & Fix Plan

**Date:** October 25, 2025  
**Status:** CRITICAL ISSUES IDENTIFIED - Requires Major Fixes  
**Current Performance:** F1=0.276, SHD=21 (Target: F1>0.7, SHD<10)

---

## 🔍 EXECUTIVE SUMMARY

Your RC-GNN model has **successfully avoided gradient explosion** and **basic training issues**, but exhibits **critical performance problems** that prevent publication:

### Current State
- ✅ **Gradient stability achieved** (clipping dropped from 99% → 0%)
- ✅ **Training converges smoothly** (no NaN/Inf)
- ✅ **Structure learning works** (predicts 16 edges vs 13 true edges)
- ❌ **POOR DISCRIMINATION**: Only 4/13 edges correct (30.8% true positive rate)
- ❌ **HIGH FALSE POSITIVES**: 12 incorrect edges predicted
- ❌ **QUANTIZATION ARTIFACT**: All predicted logits are exactly 0.0 or 0.5195 (binary)
- ❌ **NO CONTINUOUS LEARNING**: Model stuck in discrete states

### Performance Gap to Publication Standard
| Metric | Current | Target | Gap |
|--------|---------|--------|-----|
| F1 Score | 0.276 | >0.70 | **-61%** |
| SHD | 21 | <10 | **+110%** |
| Precision | 0.25 | >0.70 | **-64%** |
| Recall | 0.308 | >0.70 | **-56%** |

---

## 🔴 ROOT CAUSE ANALYSIS: 5 CRITICAL ISSUES

### **Issue 1: QUANTIZATION COLLAPSE (MOST CRITICAL)**

**Symptoms:**
```
Predicted logits: ALL values are either 0.0 or 0.51951784
Mean logit: 0.049185 (very low)
Std logit:  0.152097 (very low)
% Positive: 9.5% (too sparse)
```

**Root Cause:**
The adjacency matrix is **quantized to only 2 discrete values** instead of learning continuous confidence scores. This is caused by:

1. **Gumbel-Softmax temperature decay too aggressive**
   - Temperature likely dropped to near-zero by epoch 17
   - Converts soft attention to hard binary selection
   - Located in: `src/models/structure.py` → `StructureLearner.temperature()`

2. **Sparsification method producing hard assignments**
   - Current: `topk` or `sparsemax` forcing discrete selections
   - Should: Allow continuous gradients with soft sparsity

**Evidence from logs:**
- Epoch 1: `edge_logit_stats.mean = 0.4116` (diverse)
- Epoch 6: `edge_logit_stats.mean = -0.000014` (collapsed to zero)
- Epoch 17: Only two unique values in entire 13×13 matrix

**Impact:** Model cannot learn fine-grained edge confidence → poor discrimination

---

### **Issue 2: REGULARIZATION STILL TOO STRONG**

**Symptoms:**
```
Epoch 17 loss breakdown:
  Reconstruction: 0.12% of total loss  ← PRIMARY LEARNING SIGNAL
  Sparsity:      45.29% of total loss  ← DOMINATES
  Disentangle:   54.36% of total loss  ← DOMINATES
  Acyclicity:     0.23% of total loss
```

**Root Cause:**
Despite reducing λ values, **sparsity + disentanglement still dominate** the training signal:
- `λ_sparse = 1e-5`: Still 360× stronger than reconstruction (45% vs 0.12%)
- `λ_disen = 1e-5`: Still 453× stronger than reconstruction (54% vs 0.12%)
- Reconstruction loss itself is very small (0.000124), indicating model stopped improving data fit

**What's happening:**
1. Model quickly learns to minimize sparsity (make graph empty)
2. Model learns to maximize disentanglement (separate latents)
3. Model ignores reconstruction quality (actual data fit)
4. Structure learning starved of signal

**Expected for publication:**
- Reconstruction should be >40% of total loss
- Regularizers should be <30% each

---

### **Issue 3: EVALUATION OSCILLATION & EARLY STOPPING TOO EARLY**

**Symptoms:**
```
Epoch  2: F1=0.286, SHD=20.0, edges=15 ✅ BEST
Epoch  4: F1=0.143, SHD=22.0, edges=15
Epoch  6: F1=0.000, SHD=1e9,  edges=0  ← COLLAPSE
Epoch  8: F1=0.138, SHD=23.0, edges=16
Epoch 10: F1=0.143, SHD=22.0, edges=15
Epoch 16: F1=0.143, SHD=22.0, edges=15
Epoch 17: Early stopping triggered (patience=15)
```

**Root Causes:**
1. **SHD=1e9 sentinel pollution**: Epochs 3,5,7,9,11,13,15,17 return error sentinel
   - `eval_epoch_robust()` returning 1e9 when evaluation fails
   - Causes: threshold tuning finds no valid threshold, shape mismatches, NaN predictions
   
2. **No improvement after epoch 2**: Best SHD stays at 20.0 for 15 epochs
   - Model converged prematurely
   - Early stopping patience=15 is appropriate, but learning stalled

3. **Threshold instability**: Best threshold oscillates wildly
   - Epoch 1: threshold=0.5
   - Epoch 2: threshold=0.0 (accepts all edges)
   - Epoch 6: threshold=0.5
   - This indicates model not learning stable confidence scores

**Impact:** Can't assess true learning progress; early stopping on noise

---

### **Issue 4: TEMPERATURE SCHEDULE TOO AGGRESSIVE**

**Current Implementation (from structure.py):**
```python
# Temperature annealing in StructureLearner
self.register_buffer('step', torch.tensor(0))
# Likely: temp = max(0.1, 1.0 * 0.99^step)
```

**Problem:**
- By epoch 17 (let's say 500 steps), temp ≈ 0.1 * 0.99^500 ≈ 0.00066
- At such low temperatures, Gumbel-Softmax becomes **deterministic hard selection**
- Gradients vanish, continuous learning stops

**Evidence:**
- Epoch 1: Logits have good diversity (mean=0.41, std=0.064)
- Epoch 5: Logits collapse (mean=0.00007, std=0.00010)
- Epoch 17: Only 2 unique values across entire matrix

**For publication:**
- Temperature should stay >0.3 for first 50% of training
- Final temperature should be >0.1 to maintain gradients

---

### **Issue 5: NO DIVERSITY IN PREDICTIONS**

**Symptom:**
ALL 16 predicted edges have **identical confidence** (logit = 0.5195)

**This reveals:**
1. Model learned a **shared bias term** instead of individual edge probabilities
2. All positive edges share same initialization or constraint
3. No per-edge discrimination learned

**Root cause likely in:**
- `StructureLearner` initialization: All edges initialized to same value
- Lack of edge-specific features or attention
- Shared environmental deltas not providing enough variation

**For publication:**
- Should have continuous distribution of confidence scores
- Different edges should have different learned strengths

---

## ✅ COMPREHENSIVE FIX PLAN (7 Steps)

### **FIX 1: STOP TEMPERATURE DECAY (HIGHEST PRIORITY)**

**What:** Keep temperature constant at 1.0 or use very gentle decay

**Why:** Allow continuous gradients throughout training

**Implementation:**
```python
# In src/models/structure.py - StructureLearner
def temperature(self):
    """Temperature for Gumbel-Softmax (fixed for stable training)"""
    # OLD: return max(self.temp_min, self.temp_init * (self.temp_decay ** self.step))
    # NEW: Fixed temperature
    return 1.0  # Or use very slow decay: max(0.5, 1.0 * 0.9999^step)
```

**Expected impact:** Prevent quantization collapse, allow continuous learning

---

### **FIX 2: REBALANCE LOSSES (CRITICAL)**

**What:** Make reconstruction PRIMARY loss component (>50% of total)

**Why:** Model needs strong signal to fit data, not just minimize regularizers

**Implementation:**
```python
# In scripts/train_rcgnn_fixed.py
tc = {
    # OLD VALUES:
    # "lambda_recon": 10.0,     # 0.12% of loss
    # "lambda_sparse": 1e-5,    # 45% of loss
    # "lambda_disen": 1e-5,     # 54% of loss
    # "lambda_acyclic": 3e-6,   # 0.23% of loss
    
    # NEW VALUES (100× stronger reconstruction):
    "lambda_recon": 1000.0,      # Should be >50% of loss
    "lambda_sparse": 1e-5,       # Keep same
    "lambda_disen": 1e-6,        # Reduce 10× (was dominating)
    "lambda_acyclic": 3e-6,      # Keep same
    "target_sparsity": 0.08,     # Match true graph density (7.7%)
}
```

**Validation:** After training, check loss breakdown:
```
Expected:
  Reconstruction: >50%
  Sparsity: <25%
  Disentangle: <20%
  Acyclic: <5%
```

---

### **FIX 3: EXTEND TRAINING & IMPROVE EARLY STOPPING**

**What:** 
1. Increase max epochs to 200
2. Increase patience to 30
3. Filter out 1e9 sentinels from early stopping logic

**Why:** 
- Model needs more time to learn with gentler temperature
- Current early stopping triggered on noise (1e9 sentinels)

**Implementation:**
```python
# In scripts/train_rcgnn_fixed.py
tc = {
    "epochs": 200,           # Was: 100
    "patience": 30,          # Was: 15
    "eval_frequency": 2,     # Keep: evaluate every 2 epochs
}

# In training loop, filter sentinels:
if val_shd < 1e8:  # Only consider valid evaluations
    if val_shd < best_shd:
        best_shd = val_shd
        patience_counter = 0
    else:
        patience_counter += 1
```

---

### **FIX 4: ADD EDGE-SPECIFIC INITIALIZATION**

**What:** Initialize adjacency logits with small random noise

**Why:** Break symmetry, allow model to learn different confidence per edge

**Implementation:**
```python
# In src/models/structure.py - StructureLearner.__init__()
# OLD:
# self.A_base = nn.Parameter(torch.zeros(d, d) + 0.5)

# NEW: Add small random noise
torch.manual_seed(seed)
init_logits = torch.zeros(d, d) + torch.randn(d, d) * 0.1  # Noise ~ N(0, 0.1)
self.A_base = nn.Parameter(init_logits)
```

**Expected:** Each edge starts with unique value → learns unique confidence

---

### **FIX 5: SWITCH SPARSIFICATION TO CONTINUOUS METHOD**

**What:** Use `entmax` instead of `topk` for differentiable sparsity

**Why:** 
- `topk` produces hard selections (0 or 1)
- `entmax` produces continuous sparse distributions
- Maintains gradients while encouraging sparsity

**Implementation:**
```python
# In configs/model.yaml
structure:
  sparsify:
    method: "entmax"  # Was: "topk" or "sparsemax"
    alpha: 1.5        # Entmax parameter (1.5 = good sparsity)
    k: null           # Not used for entmax
```

**If entmax not available:**
```python
# Alternative: Use sigmoid with L1 penalty (already have λ_sparse)
# In src/models/structure.py - StructureLearner.forward()
A_logits = self.A_base + env_delta
A_soft = torch.sigmoid(A_logits)  # Continuous in [0,1]
# Let λ_sparse handle sparsity via L1 penalty
return A_soft, A_logits
```

---

### **FIX 6: ADD LEARNING RATE RESTART**

**What:** Cosine annealing with warm restarts every 50 epochs

**Why:** 
- Helps escape local minima
- Current training plateaued at epoch 2
- Periodic LR boosts can find better solutions

**Implementation:**
```python
# In scripts/train_rcgnn_fixed.py
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Replace current scheduler:
# cosine_scheduler = CosineAnnealingLR(opt, T_max=...)

# NEW:
cosine_scheduler = CosineAnnealingWarmRestarts(
    opt, 
    T_0=50 * len(train_ld),      # Restart every 50 epochs
    T_mult=1,                     # Keep same cycle length
    eta_min=1e-5                  # Minimum LR
)
```

---

### **FIX 7: IMPROVE EVALUATION ROBUSTNESS**

**What:** Better handle edge cases in evaluation to eliminate 1e9 sentinels

**Why:** Half of evaluations return 1e9 error sentinel → can't track progress

**Implementation:**
```python
# In src/training/eval_robust.py - evaluate_adj()

# Add at start of function:
if A_pred.shape != A_true.shape:
    print(f"WARNING: Shape mismatch {A_pred.shape} vs {A_true.shape}")
    # Resize or crop instead of returning error
    min_size = min(A_pred.shape[0], A_true.shape[0])
    A_pred = A_pred[:min_size, :min_size]
    A_true = A_true[:min_size, :min_size]

# When no valid threshold found:
if best_f1 == 0 and best_threshold is None:
    # Instead of returning 1e9, use default threshold
    best_threshold = 0.3  # Reasonable default
    print(f"WARNING: No valid threshold found, using default {best_threshold}")
    # Compute metrics at this threshold
    A_bin = (A_pred >= best_threshold).astype(int)
    # ... compute TP, FP, FN, SHD ...
```

---

## 📈 EXPECTED RESULTS AFTER FIXES

### Performance Targets (Publication Quality)
| Metric | Current | After Fixes | Target |
|--------|---------|-------------|--------|
| F1 Score | 0.276 | 0.60-0.75 | >0.70 |
| SHD | 21 | 8-12 | <10 |
| Precision | 0.25 | 0.65-0.80 | >0.70 |
| Recall | 0.308 | 0.55-0.70 | >0.70 |
| True Positive | 4/13 | 9-11/13 | >10/13 |
| False Positive | 12 | 2-5 | <3 |

### Training Dynamics (Expected)
- **Epochs 1-50:** Warm-up phase, reconstruction loss drops from 0.001 → 0.0001
- **Epochs 50-100:** Structure refinement, F1 climbs from 0.3 → 0.6
- **Epochs 100-150:** Fine-tuning, F1 reaches 0.7+
- **Epochs 150-200:** Convergence or early stop

### Loss Breakdown (Expected at Epoch 100)
```
Reconstruction: 55-65% (PRIMARY - model fits data well)
Sparsity:       20-25% (forces graph sparsity)
Disentangle:    10-15% (latent separation)
Acyclic:        2-5%   (DAG constraint)
```

### Logit Distribution (Expected)
```
Mean logit:    ~0 (balanced)
Std logit:     0.3-0.6 (good discrimination)
Min/Max:       -2 to +3 (wide range)
Unique values: 169 (all different, continuous learning)
```

---

## 🚀 IMPLEMENTATION ORDER

### Phase 1: Critical Fixes (Do First) ⚡
**Time: 30 minutes**
1. ✅ Fix 1: Stop temperature decay
2. ✅ Fix 2: Rebalance losses (1000× reconstruction)
3. ✅ Fix 3: Extend training (200 epochs, patience=30)

**Run training:** `python3 scripts/train_rcgnn_fixed.py`  
**Expected:** Logits become continuous, reconstruction dominates loss

### Phase 2: Refinement (If Phase 1 Works) 🔧
**Time: 1 hour**
4. ✅ Fix 4: Edge-specific initialization
5. ✅ Fix 5: Continuous sparsification (entmax or sigmoid)
6. ✅ Fix 7: Robust evaluation

**Run training:** Check F1 improves to >0.5

### Phase 3: Advanced (If Stuck at Local Minimum) 🎯
**Time: 1 hour**
7. ✅ Fix 6: LR warm restarts
8. Optional: Hyperparameter sweep (λ_recon ∈ [100, 500, 1000, 2000])

---

## 📊 VALIDATION CHECKLIST

After implementing fixes, verify:

### Training Health
- [ ] Loss components: Reconstruction >50% of total
- [ ] Logit diversity: Std >0.2, unique values >100
- [ ] No 1e9 sentinels in validation metrics
- [ ] Gradient clipping <10% after epoch 10
- [ ] Learning rate visible in logs, restarts working

### Performance Metrics
- [ ] F1 score >0.60 by epoch 100
- [ ] SHD <15 by epoch 100
- [ ] True positives ≥8/13
- [ ] False positives <8

### Qualitative
- [ ] Predicted adjacency matrix shows variation (not all 0.0 or 0.52)
- [ ] High-confidence edges align with true graph structure
- [ ] Training log shows smooth improvement (not oscillation)

---

## 🎯 PUBLICATION READINESS CRITERIA

To publish RC-GNN results, you need:

### Minimum Requirements
- ✅ F1 >0.70 (current: 0.276) ❌
- ✅ SHD <10 (current: 21) ❌
- ✅ Stable training (current: YES) ✅
- ✅ Reproducible (current: YES, seed=1337) ✅

### Competitive Requirements (vs. Baselines)
Compare to:
- **NOTEARS**: F1 ≈ 0.65, SHD ≈ 12 (classic baseline)
- **GraN-DAG**: F1 ≈ 0.72, SHD ≈ 8 (SOTA structure learning)
- **DAG-GNN**: F1 ≈ 0.68, SHD ≈ 10 (graph neural baseline)

**Your RC-GNN should:** F1 >0.75, SHD <8 to claim improvement

### Robustness Claims (Original Paper Goal)
- Test on corrupted data (missing values, sensor noise)
- Show RC-GNN maintains F1>0.70 under 30% corruption
- Baselines drop to F1<0.50 under same corruption
- This is your **unique contribution**

---

## 📝 NEXT STEPS

### Immediate (Today)
1. **Implement Phase 1 fixes** (temperature + loss rebalancing + extend training)
2. **Run training for 200 epochs** (~10-15 minutes)
3. **Check logit diversity** and loss breakdown

### This Week
4. **Implement Phase 2 fixes** if results improve but <0.6 F1
5. **Run corrupted data experiments** (your unique contribution)
6. **Compare to baselines** (NOTEARS, GraN-DAG)

### Publication Prep
7. **Write results section** with ablation study (which fixes matter most?)
8. **Create visualizations** (adjacency matrices, training curves, ablation plots)
9. **Draft paper** emphasizing robustness to corruption

---

## 🔗 FILES TO MODIFY

### Critical Files
1. `src/models/structure.py` - Fix temperature decay (line ~80-100)
2. `scripts/train_rcgnn_fixed.py` - Rebalance losses (line ~68-93), extend epochs (line ~70)
3. `src/training/eval_robust.py` - Improve error handling (line ~30-60)

### Optional Files
4. `configs/model.yaml` - Change sparsify.method to "entmax"
5. `scripts/train_rcgnn_fixed.py` - Add LR warm restarts (line ~175-210)

---

## ⚠️ WARNINGS

### Do NOT do these:
- ❌ Reduce λ_sparse further (already at 1e-5)
- ❌ Increase batch size (will reduce gradient signal)
- ❌ Use very high learning rate (will destabilize)
- ❌ Remove regularizers entirely (will overfit)

### DO these:
- ✅ **Increase λ_recon** massively (100-1000×)
- ✅ **Fix temperature** (stop decay)
- ✅ **Train longer** (200+ epochs)
- ✅ **Monitor loss breakdown** every epoch

---

## 📚 REFERENCES FOR FIXES

### Temperature in Gumbel-Softmax
- Jang et al. 2017: "Categorical Reparameterization with Gumbel-Softmax"
- Recommendation: τ ∈ [0.5, 1.0] for learning, τ=0.1 only for final inference

### Structure Learning Baselines
- NOTEARS (Zheng et al. 2018): "DAGs with NO TEARS"
- GraN-DAG (Lachapelle et al. 2020): "Gradient-Based Neural DAG Learning"
- DAG-GNN (Yu et al. 2019): "DAG-GNN: DAG Structure Learning with Graph Neural Networks"

### Loss Balancing
- Multi-task learning: reconstruction should dominate early, regularizers kick in later
- Your case: reconstruction TOO WEAK (0.12% vs should be >50%)

---

**Generated:** October 25, 2025  
**Author:** AI Analysis System  
**Status:** READY FOR IMPLEMENTATION
