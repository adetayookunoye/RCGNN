# Quick Reference: Training Modes

## TL;DR

```bash
# For TESTING (60 seconds)
make train

# For PRODUCTION (30-45 minutes)  
make train-full
```

---

## What's the Difference?

### 📊 Configuration Comparison

| Parameter | Quick | Full |
|-----------|-------|------|
| **Epochs** | 8 | 100 |
| **Batch Size** | 8 | 8 |
| **Learning Rate** | 0.001 | 0.001 |
| **Early Stopping** | None | Yes (patience=10) |
| **Eval Frequency** | Every epoch | Every 5 epochs |
| **Expected Runtime** | ~60 sec | ~30-45 min |
| **Expected SHD** | ~25-30 | ~15-20 |

### 📁 Which Script?

| Mode | Script | Config |
|------|--------|--------|
| Quick | `run_full_pipeline.py` | Built-in (8 epochs) |
| Full | `scripts/train_full_model.py` | Built-in (100 epochs) |

---

## When to Use Each?

### ✅ Use `make train` (Quick) When:
- [ ] Testing code changes
- [ ] Verifying pipeline works
- [ ] Quick demos
- [ ] CI/CD integration
- [ ] Sanity checks
- [ ] Rapid iteration
- [ ] Debugging issues

### ✅ Use `make train-full` (Full) When:
- [ ] Need production results
- [ ] Publishing/papers
- [ ] Final benchmarking
- [ ] Reproducible research
- [ ] Model selection
- [ ] Serious evaluation
- [ ] Comparing with baselines

---

## Step-by-Step: Quick Testing

```bash
# 1. Quick training (60 seconds)
make train

# 2. View results
make results

# Expected output: SHD ~25-30 (good for testing)
```

---

## Step-by-Step: Full Production

```bash
# 1. Full training (30-45 minutes)
make train-full

# 2. Optimize threshold
make analyze

# 3. Compare with baselines
make baseline

# 4. View final results
make results

# Expected output: SHD ~15-20 (publication-ready)
```

---

## Key Facts

🔹 **Same Dataset**: Both use full UCI Air Quality (9,448 samples)

🔹 **Same Model**: No changes to model architecture

🔹 **Same Seed**: Random seed 1337 for reproducibility

🔹 **Different Epochs**: Only difference is 8 vs 100 epochs

🔹 **Different Quality**: Quick is underfit, Full is well-optimized

🔹 **Different Time**: Quick is instant, Full is thorough

---

## Example Workflows

### Development Loop
```bash
# Make changes → Test quickly → Repeat
make train         # 60 sec
make results       
# Edit code...
make train         # 60 sec
make results
```

### Final Results Pipeline
```bash
make train-full    # 30-45 min
make analyze       
make baseline
make results
```

### Compare Both
```bash
# First: Quick version to verify it works
make train         # 60 sec
# ↓ Check SHD ~25-30

# Then: Full version for real results
make train-full    # 30-45 min
# ↓ Check SHD ~15-20 (better!)
```

---

## Checking Progress

### Quick Mode
```bash
# Real-time output:
# Epoch 1/8: Loss 4.2134, F1 0.4523, SHD 35.0
# Epoch 2/8: Loss 3.8291, F1 0.5234, SHD 28.0
# ...
# ✅ Training complete!  (takes 60 seconds)
```

### Full Mode
```bash
# Real-time output:
# Epoch  1/100: Loss 4.2134 | Val F1: 0.4523 | Val SHD: 35.0
# Epoch  2/100: Loss 3.8291 | Val F1: 0.5234 | Val SHD: 28.0
# ...
# Epoch 25/100: Loss 0.1234 | Val F1: 0.8901 | Val SHD: 18.0 ⭐ NEW BEST
# ...
# ⏹️  Early stopping triggered after 45 epochs
# ✅ TRAINING COMPLETE  (takes 30-45 minutes)
```

---

## Troubleshooting

**Q: Why only 60 seconds?**
A: That's the quick mode (8 epochs). Use `make train-full` for full training.

**Q: Can I interrupt and resume?**
A: Currently no. If interrupted, just restart - it will overwrite.

**Q: How do I know which one finished?**
A: Quick says "Training complete!" | Full says "TRAINING COMPLETE"

**Q: Can I change epochs?**
A: 
- Quick: Edit `run_full_pipeline.py` line 40
- Full: Edit `scripts/train_full_model.py` line 62

**Q: Is the code the same?**
A: Yes, same model. Only different epoch counts.

**Q: Do they use different data?**
A: No, same full UCI Air dataset.

---

## All Makefile Training Commands

```bash
make train              # Quick: 8 epochs, ~60 sec
make train-full        # Full: 100 epochs, ~30-45 min
make train-verbose     # Quick: with detailed output
make full-pipeline     # train + analyze + baseline
make results           # Display results
```

---

## Summary

| Need | Command | Time | Quality |
|------|---------|------|---------|
| Test code | `make train` | 60 sec | SHD ~25-30 |
| Final results | `make train-full` | 30-45 min | SHD ~15-20 |

**Choose based on your needs!** 🚀
