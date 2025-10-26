# Training Options: Quick vs Full

## Overview

The RC-GNN project now supports **two training modes** to accommodate different use cases:

### 1. Quick Training: `make train` (~60 seconds)
**Purpose**: Testing, CI/CD, demos, rapid iteration
**Configuration**:
- **Epochs**: 8 (reduced from 100+)
- **Batch size**: 8
- **Dataset**: Full UCI Air Quality (9,448 samples)
- **Runtime**: ~60 seconds on CPU
- **Expected SHD**: ~25-30 (less optimized)
- **Use case**: 
  - Verify pipeline works
  - Integration testing
  - Quick demos
  - Algorithm exploration
  - Sanity checks before long training runs

**Command**:
```bash
make train
# or
python3 run_full_pipeline.py
```

**Output**:
- Model checkpoint: `artifacts/checkpoints/rcgnn_best.pt`
- Adjacency matrix: `artifacts/adjacency/A_mean.npy`
- Training metrics: `artifacts/training_metrics.json`

---

### 2. Full Training: `make train-full` (~30+ minutes)
**Purpose**: Production-quality results, research, publications
**Configuration**:
- **Epochs**: 100+ (with early stopping if no improvement)
- **Batch size**: 8
- **Dataset**: Full UCI Air Quality (9,448 samples, all samples processed)
- **Runtime**: 30-45 minutes on CPU
- **Expected SHD**: ~15-20 (optimized)
- **Early stopping**: Patience=10 (stops if no improvement for 10 evaluation cycles)
- **Use case**:
  - Final results for papers/reports
  - Production models
  - Comprehensive benchmarking
  - Reproducible research

**Command**:
```bash
make train-full
# or
python3 scripts/train_full_model.py
```

**Output**:
- Model checkpoint: `artifacts/checkpoints/rcgnn_best.pt`
- Adjacency matrix: `artifacts/adjacency/A_mean.npy`
- Training metrics: `artifacts/training_metrics_full.json`
- Training history: `artifacts/training_history_full.json`

---

## Configuration Details

### Quick Training (8 epochs)
Located in: `run_full_pipeline.py` (lines ~35-40)
```yaml
epochs: 8
batch_size: 8
learning_rate: 0.001
weight_decay: 1e-5
device: "cpu"
```

### Full Training (100 epochs)
Located in: `scripts/train_full_model.py` (lines ~60-80)
```yaml
epochs: 100
batch_size: 8
learning_rate: 0.001
weight_decay: 1e-5
device: "cpu"
patience: 10  # Early stopping
eval_frequency: 5  # Evaluate every 5 epochs
```

---

## Why Two Modes?

| Aspect | Quick (8 epochs) | Full (100 epochs) |
|--------|------------------|------------------|
| **Time** | 60 seconds | 30-45 minutes |
| **Purpose** | Testing/Demo | Production/Research |
| **SHD Quality** | ~25-30 | ~15-20 |
| **Memory** | Minimal | Minimal (CPU-friendly) |
| **Use When** | Developing code | Final results needed |
| **Patience for bugs** | High | High |

---

## Example Workflows

### Quick Iteration (Testing)
```bash
# Make a code change, verify it works
make train              # ~60 seconds
make analyze            # Quick threshold optimization
make results            # See results
```

### Full Pipeline (Production)
```bash
# Start fresh full training
make clean              # Remove old artifacts
make train-full         # ~30-45 minutes
make analyze            # Threshold optimization
make baseline           # Compare with baselines
make results            # Final results
```

### Compare Both
```bash
# Run quick version first
make train              # ~60 seconds - see if everything works
make results            # Check artifacts

# Then run full version for final results
make train-full         # ~30-45 minutes
make results            # See improved SHD metrics
```

---

## Important Notes

### ⚠️ When to Use Quick Training
- ✅ Testing code changes
- ✅ Verifying pipeline works
- ✅ CI/CD integration
- ✅ Rapid prototyping
- ✅ Debugging issues
- ❌ NOT for publication results

### ⚠️ When to Use Full Training
- ✅ Final published results
- ✅ Reproducible research
- ✅ Benchmarking comparisons
- ✅ Performance evaluation
- ✅ Model selection

### 🔄 Early Stopping in Full Training
- Monitors validation SHD every 5 epochs
- Stops training if no improvement for 10 evaluation cycles
- Saves best checkpoint automatically
- May complete in <100 epochs if convergence is early

### 📊 Dataset Size
Both modes use the **SAME dataset**: Full UCI Air Quality (9,448 samples)
- The difference is in the number of epochs
- Quick mode: Underfit (less training)
- Full mode: Well-optimized (thorough training)

---

## Monitoring Training

### Quick Training
```bash
make train
# Output: One line per epoch showing loss and SHD
```

### Full Training
```bash
make train-full
# Output: More detailed logging including:
# - Train loss
# - Validation F1 and SHD
# - Patience counter
# - New best model markers (⭐)
# - Early stopping notification
```

### View Training History
After `make train-full`:
```bash
cat artifacts/training_history_full.json
# Shows: epoch losses, validation metrics, timing
```

---

## Troubleshooting

### Q: Why is `make train` so fast now?
**A**: It only runs 8 epochs (quick mode) instead of full training. Use `make train-full` for production results.

### Q: Can I run both modes?
**A**: Yes! Quick mode (~60 sec) is great for verification, then run `make train-full` for final results.

### Q: How do I know if training is done?
```bash
# For quick training:
# Look for: ✅ Training complete!

# For full training:
# Look for: ⏹️  Early stopping triggered (or all 100 epochs finished)
# Then: ✅ TRAINING COMPLETE
```

### Q: Can I resume training?
**A**: Currently no (not implemented). If interrupted, restart training and it will overwrite the checkpoint.

### Q: How can I adjust epochs?
- **Quick**: Edit `run_full_pipeline.py` line ~40
- **Full**: Edit `scripts/train_full_model.py` line ~62

---

## Makefile Targets Summary

```bash
make train              # Quick: 8 epochs, ~60 sec
make train-full        # Full: 100 epochs, ~30-45 min
make train-verbose     # Quick: with detailed output
make analyze           # Threshold optimization
make baseline          # Compare with baselines
make full-pipeline     # train → analyze → baseline
make results           # Display all results
make clean             # Remove artifacts
```

---

## Expected Performance

After each training mode:

**Quick Training (8 epochs)**
```
Best validation SHD: ~25-30
Training time: ~60 seconds
Model quality: Underfit (for testing only)
```

**Full Training (100 epochs with early stopping)**
```
Best validation SHD: ~15-20
Training time: ~30-45 minutes  
Model quality: Well-optimized (production-ready)
```

---

## Summary

✅ **`make train`** = Quick testing (60 seconds)  
✅ **`make train-full`** = Production quality (30-45 minutes)  
✅ **Same dataset** = Full UCI Air dataset in both cases  
✅ **Easy switching** = Choose which fits your need  

Use `make train` for development, `make train-full` for final results! 🚀
