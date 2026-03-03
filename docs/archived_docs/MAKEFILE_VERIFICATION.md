# Makefile Verification & Testing Guide

## ✅ Configuration Fixed

Both `make train-synth` and `make train-air` now correctly use `scripts/train_rcgnn.py` with different config files.

## Current Configuration Status

### All Config Files Present ✅

```bash
configs/data.yaml        # Synthetic dataset (created)
configs/data_uci.yaml    # UCI Air Quality (existing)
configs/model.yaml       # Model architecture (existing)
configs/train.yaml       # Training hyperparameters (existing)
```

### Training Targets Corrected ✅

**Synthetic Training:**
```makefile
train-synth: data-synth-small
    scripts/train_rcgnn.py \
        configs/data.yaml \          # 10 features, synth_small
        configs/model.yaml \
        configs/train.yaml
```

**UCI Air Training:**
```makefile
train-air: data-air
    scripts/train_rcgnn.py \
        configs/data_uci.yaml \      # 13 features, uci_air
        configs/model.yaml \
        configs/train.yaml
```

## Quick Verification Test

### Step 1: Check Project Status
```bash
make status
```
**Expected Output:**
- Shows checklist of data, configs, scripts, artifacts
- All config files should show ✅

### Step 2: Verify Config Files
```bash
# Check synthetic config
cat configs/data.yaml

# Check UCI Air config
cat configs/data_uci.yaml
```

**Expected:**
- `data.yaml`: dataset="synth_small", root="data/interim/synth_small"
- `data_uci.yaml`: dataset="uci_air", root="data"

### Step 3: Test Data Generation (Quick)
```bash
# Generate small synthetic dataset (~1-2 min)
make data-synth-small
```

**Expected Output:**
```
✅ Synthetic dataset generated!
📊 Data shape: (1000, 100, 10)
```

**Verify:**
```bash
ls -lh data/interim/synth_small/
# Should see: X.npy, M.npy, A_true.npy, e.npy, S.npy, meta.json
```

### Step 4: Test Training (CPU Quick Test)

**Option A: Synthetic (faster, ~5-10 min on CPU)**
```bash
make train-synth
```

**Option B: UCI Air (longer, ~15-20 min on CPU)**
```bash
make train-air
```

**Expected Output:**
```
🚀 Training RC-GNN on synthetic dataset...
Loading configs/data.yaml...
Epoch 1/100: loss=X.XXX
...
✅ Training complete! Adjacency at artifacts/adjacency/A_mean.npy
```

**Verify:**
```bash
ls -lh artifacts/adjacency/
# Synthetic: A_mean.npy
# UCI Air:   A_mean_air.npy
```

### Step 5: Test Validation
```bash
# Validate synthetic results
make validate-synth

# Validate UCI Air results
make validate-air

# Or validate both
make validate-all
```

**Expected Output:**
```
✅ Validation complete for synth!
   - SHD: XX
   - Precision: X.XX
   - Calibration: X.XX
```

## Common Issues & Solutions

### Issue 1: ModuleNotFoundError
```
ModuleNotFoundError: No module named 'src'
```

**Solution:**
```bash
# Ensure path_helper.py is imported in train_rcgnn.py
# Or set PYTHONPATH:
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
make train-synth
```

### Issue 2: Config File Not Found
```
FileNotFoundError: configs/data.yaml not found
```

**Solution:**
```bash
# Verify file exists
ls -lh configs/data.yaml

# If missing, recreate:
cat > configs/data.yaml << 'EOF'
dataset: "synth_small"
window_len: 100
window_stride: 1
features: 10
regime:
  mode: "provided"
split:
  regime_train: 0.6
  regime_val: 0.2
  regime_test: 0.2
paths:
  root: "data/interim/synth_small"
EOF
```

### Issue 3: CUDA Out of Memory (if using GPU)
```
RuntimeError: CUDA out of memory
```

**Solution:**
```bash
# Edit configs/train.yaml to use CPU:
sed -i 's/device: "cuda"/device: "cpu"/' configs/train.yaml

# Or reduce batch size:
sed -i 's/batch_size: 64/batch_size: 16/' configs/train.yaml
```

### Issue 4: Dataset Not Found
```
FileNotFoundError: data/interim/synth_small/X.npy not found
```

**Solution:**
```bash
# Generate the dataset first
make data-synth-small

# Or generate all datasets
make data
```

## Full Pipeline Test (30-60 min)

Run the complete workflow from scratch:

```bash
# Clean everything
make clean-all

# Generate all datasets + train + validate
make all
```

**This will:**
1. Generate synthetic small dataset (~1-2 min)
2. Generate synthetic nonlinear dataset (~2-3 min)
3. Download & prepare UCI Air data (~5-10 min)
4. Train on synthetic (~10-15 min)
5. Train on UCI Air (~15-20 min)
6. Validate both (~5-10 min)
7. Generate summary reports (~1-2 min)

**Success Indicators:**
- ✅ All datasets in `data/interim/*/`
- ✅ Checkpoints in `artifacts/checkpoints/`
- ✅ Adjacency matrices in `artifacts/adjacency/`
- ✅ Validation results in `artifacts/validation/`
- ✅ Summary report at `artifacts/results/summary.txt`

## Key Differences: Synthetic vs UCI Air

| Aspect | Synthetic (`data.yaml`) | UCI Air (`data_uci.yaml`) |
|--------|------------------------|---------------------------|
| **Dataset** | `synth_small` | `uci_air` |
| **Features** | 10 | 13 |
| **Window Length** | 100 timesteps | 24 hours |
| **Data Root** | `data/interim/synth_small` | `data` |
| **Training Script** | ✅ `train_rcgnn.py` | ✅ `train_rcgnn.py` |
| **Adjacency Output** | `A_mean.npy` | `A_mean_air.npy` |
| **Ground Truth** | ✅ `A_true.npy` | ❌ (not available) |
| **Validation** | Full metrics (SHD, precision, etc.) | Partial metrics |

## Recommended Testing Order

1. **Quick Smoke Test (5 min):**
   ```bash
   make status
   make data-synth-small
   head -20 artifacts/logs/synth_small.log
   ```

2. **Training Test (10 min):**
   ```bash
   make train-synth
   # Check for errors in training output
   ```

3. **Validation Test (5 min):**
   ```bash
   make validate-synth
   cat artifacts/validation/synth_validation_advanced.log
   ```

4. **Full Pipeline Test (30-60 min):**
   ```bash
   make all
   make summary
   cat artifacts/results/summary.txt
   ```

## Next Steps After Verification

Once everything works:

1. **Adjust hyperparameters** in `configs/train.yaml`:
   - Increase epochs for better convergence
   - Tune learning rate
   - Adjust batch size based on your hardware

2. **Add new datasets** by creating new config files:
   - Copy `configs/data_uci.yaml`
   - Modify dataset name, features, paths
   - Add corresponding Makefile target

3. **Run paper experiments**:
   ```bash
   make paper    # Complete pipeline for publication
   ```

4. **Generate LaTeX tables**:
   ```bash
   make tables
   # Output: artifacts/results/results_table.tex
   ```

## Troubleshooting Checklist

- [ ] All config files exist (`ls configs/*.yaml`)
- [ ] Python environment activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Data generated (`make data-synth-small`)
- [ ] Path helper in scripts (`import path_helper`)
- [ ] Sufficient disk space (`df -h`)
- [ ] Sufficient memory (`free -h`)

## Getting Help

**Check logs:**
```bash
tail -50 artifacts/logs/synth_small.log
tail -50 artifacts/logs/uci_air.log
```

**Check artifacts:**
```bash
tree artifacts/
```

**Inspect config:**
```bash
cat configs/data.yaml
cat configs/model.yaml
cat configs/train.yaml
```

---

**Last Updated:** October 26, 2024  
**Status:** Makefile corrected, all configs in place ✅
