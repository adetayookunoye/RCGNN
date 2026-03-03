# 🔥 EXTREME Hyperparameters for UCI Air Quality Dataset

**Date:** October 25, 2025  
**Problem:** Real-world UCI dataset causes gradient explosion (100% clipping) and model collapse  
**Solution:** Ultra-conservative learning rates + extreme loss rebalancing

---

## 🚨 Why UCI Air Quality Is Difficult

### Dataset Characteristics:
- **13 features** (CO, NOx, temperature, humidity, etc.)
- **6,613 training samples** (time series)
- **Real-world noise** (sensor drift, missing values)
- **Complex dependencies** (weather, pollution, time-of-day)

### Previous Failure (Standard Hyperparameters):
```
LR: 3e-5 → 5e-4 (warmup over 3 epochs)
λ_recon: 10.0, λ_sparse: 1e-5, λ_disen: 1e-5
Grad clip: 1.0

RESULT:
- Clip%: 100% throughout (even after warmup!)
- Logit std: 0.0000 (complete quantization)
- edges@0.5: 0 (all probabilities <0.5)
- SHD stuck at 13 (model predicts NO edges)
```

---

## ✅ NEW Extreme Hyperparameters

### Learning Rate: **10× LOWER**
```python
"learning_rate_init": 3e-6,  # Start at 0.000003 (was 3e-5)
"learning_rate_max": 5e-5,   # Max 0.00005 (was 5e-4)
"warmup_epochs": 5,          # Longer warmup (was 3)
```

**Rationale:**
- UCI has much higher gradient magnitudes than synthetic data
- Lower LR prevents early gradient explosion
- Longer warmup (5 epochs) allows smoother adaptation

**Expected Impact:**
- Epoch 1 clip%: 40-60% (was 100%)
- Epoch 5 clip%: <10% (was 100%)
- Gradients stabilize by epoch 10

---

### Loss Weights: **100× MORE EXTREME**
```python
"lambda_recon": 100.0,   # 10× stronger (was 10.0)
"lambda_sparse": 1e-6,   # 10× weaker (was 1e-5)
"lambda_acyclic": 1e-7,  # 10× weaker (was 3e-6)
"lambda_disen": 1e-6,    # 10× weaker (was 1e-5)
```

**Rationale:**
- Previous run: Recon=100%, Sparse=0%, Disen=0% (numerical underflow)
- Regularizers were being zeroed out, providing NO guidance
- Need to keep ALL losses in same numerical range

**Expected Impact:**
- Recon: 60-80% of total loss (dominant signal)
- Sparse: 10-20% (provides structure guidance)
- Disen: 5-10% (provides latent separation)
- Acyclic: 2-5% (gentle DAG constraint)

---

### Gradient Clipping: **2× MORE AGGRESSIVE**
```python
"grad_clip_norm": 0.5  # Was 1.0
```

**Rationale:**
- Even with low LR, some gradients may spike
- Aggressive clipping prevents outliers from destabilizing training
- 0.5 is very conservative but safe for real-world data

**Expected Impact:**
- Clip% starts lower (40% vs 100%)
- Clip% drops faster (reaches <5% by epoch 10)

---

## 📊 Expected Training Behavior

### Early Epochs (1-5): Warm-Up Phase
```
Epoch 1: Loss ~1000-5000 | Clip: 50-70% | LR: 3e-6
Epoch 2: Loss ~500-1000  | Clip: 30-50% | LR: 1e-5
Epoch 3: Loss ~200-500   | Clip: 20-30% | LR: 2e-5
Epoch 4: Loss ~100-200   | Clip: 10-20% | LR: 3e-5
Epoch 5: Loss ~50-100    | Clip: 5-10%  | LR: 5e-5
```

### Mid Training (5-20): Learning Phase
```
Loss breakdown should stabilize:
- Recon: 60-70% (dominant)
- Sparse: 15-20%
- Disen: 10-15%
- Acyclic: 3-5%

F1 should improve: 0.1 → 0.2 → 0.3 → 0.4+
SHD should drop: 50 → 30 → 20 → 15
Logit std should increase: 0.05 → 0.10 → 0.20
```

### Late Training (20+): Convergence
```
F1 plateau: ~0.5-0.6 (realistic for UCI complexity)
SHD plateau: ~10-15 (still learning structure)
Early stopping: ~epoch 40-60 (patience=15)
```

---

## 🎯 Success Criteria (Realistic for UCI)

### Minimum (Training Works):
- ✅ Clip% <50% by epoch 3
- ✅ Clip% <10% by epoch 10
- ✅ Logit std >0.1 by epoch 10
- ✅ edges@tuned >0 by epoch 5
- ✅ F1 >0.2 by epoch 20

### Good (Publishable):
- ✅ Clip% <5% by epoch 10
- ✅ F1 >0.4 by epoch 30
- ✅ SHD <20 by epoch 30
- ✅ Recon% >60% throughout
- ✅ Logit std >0.2 by epoch 20

### Excellent (Strong Result):
- ✅ F1 >0.5
- ✅ SHD <15
- ✅ Edges@tuned ≈ edges@0.5 (consistent)
- ✅ Robust to corruption (future test)

---

## 🔍 Monitoring Checklist

### Every 2 Epochs (Evaluation):
```bash
tail -f artifacts/training_ultra_low_lr_full.log
```

**Look for:**
1. **Clip%** - Should decrease monotonically
2. **Loss breakdown** - Recon should be 60-80%
3. **Logit std** - Should be >0.05 and increasing
4. **edges@tuned vs edges@0.5** - Should be close (within 3-5)
5. **F1** - Should improve every 5-10 epochs

### Red Flags:
- ❌ Clip% still >50% after epoch 10 → LR still too high
- ❌ Logit std <0.05 after epoch 15 → Quantization persisting
- ❌ Recon% <50% → Loss weights need more rebalancing
- ❌ edges@tuned=0 while edges@topk=15 → Threshold tuning broken
- ❌ F1 not improving for 20 epochs → Learning stalled

---

## 🛠️ Emergency Adjustments

### If Clip% Still >50% After Epoch 10:
```python
# Further reduce LR
"learning_rate_init": 1e-6,  # 3× lower
"learning_rate_max": 2e-5,   # 2.5× lower
"grad_clip_norm": 0.3,       # More aggressive
```

### If Recon% <50%:
```python
# Make recon EVEN MORE dominant
"lambda_recon": 500.0,   # 5× stronger
"lambda_sparse": 5e-7,   # 2× weaker
"lambda_disen": 5e-7,    # 2× weaker
```

### If Logit Std <0.1 After Epoch 20:
```python
# Change sparsification method
# In configs/model.yaml:
sparsify_method: "sigmoid"  # Instead of "topk"
```

### If F1 Not Improving:
```python
# Increase model capacity
# In configs/model.yaml:
latent_dim: 32    # Was 16
hidden_dim: 64    # Was 32
```

---

## 📝 Comparison: Synthetic vs UCI Hyperparameters

| Parameter | Synthetic (Easy) | UCI (Real-World) | Ratio |
|-----------|------------------|------------------|-------|
| LR init | 3e-5 | 3e-6 | **10× lower** |
| LR max | 5e-4 | 5e-5 | **10× lower** |
| Warmup | 3 epochs | 5 epochs | **1.7× longer** |
| λ_recon | 10.0 | 100.0 | **10× stronger** |
| λ_sparse | 1e-5 | 1e-6 | **10× weaker** |
| λ_disen | 1e-5 | 1e-6 | **10× weaker** |
| Grad clip | 1.0 | 0.5 | **2× tighter** |

**Key Insight:** Real-world data requires **10-100× more conservative** hyperparameters than synthetic benchmarks!

---

## 🎓 Lessons Learned

### Why Standard Hyperparameters Failed:
1. **Gradient scale mismatch:** UCI features have wider range → larger gradients
2. **Loss scale imbalance:** Regularizers underflowed to zero with standard weights
3. **Hot start:** Even 3e-5 LR was too high for UCI's gradient landscape
4. **Insufficient warm-up:** 3 epochs not enough for real-world complexity

### Why Extreme Hyperparameters Should Work:
1. **Ultra-low LR:** Prevents gradient explosion from step 1
2. **Extreme loss rebalancing:** Keeps all losses in numerical range (1-100)
3. **Long warm-up:** 5 epochs allows gradual stabilization
4. **Aggressive clipping:** Safety net for outlier batches

---

## 🚀 Current Training Status

**Command:**
```bash
python3 scripts/train_full_model.py
```

**Log:**
```bash
tail -f artifacts/training_ultra_low_lr_full.log
```

**Expected completion:** 5-7 minutes (UCI is slower than synthetic)

**Check progress:**
```bash
grep "⭐ BEST" artifacts/training_ultra_low_lr_full.log
```

---

**Generated:** October 25, 2025 23:35  
**Training:** In progress with extreme hyperparameters for UCI  
**Next:** Monitor for clip% <50% by epoch 3, F1 >0.2 by epoch 20
