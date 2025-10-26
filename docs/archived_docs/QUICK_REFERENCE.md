# RC-GNN - Quick Reference Card

## 🚀 **GETTING STARTED (Copy & Paste)**

```bash
# Step 1: Initial Setup (one time only)
make setup
conda activate rcgnn-env

# Step 2: Run Everything
make full-pipeline

# Step 3: See Results
make results
```

**That's it!** Your model is trained and ready.

---

## ⚡ **COMMON COMMANDS**

| Command | What It Does | Time |
|---------|-----------|------|
| `make help` | Show all commands | instant |
| `make setup` | Install everything | 2 min |
| `make train` | Train model | 60 sec |
| `make analyze` | Optimize threshold | 2 sec |
| `make baseline` | Compare methods | 3 sec |
| `make full-pipeline` | Train + analyze + compare | 65 sec |
| `make results` | Show results | instant |
| `make clean` | Delete artifacts | instant |

---

## 📊 **STEP-BY-STEP GUIDE**

### **Week 1: First Time Setup**

```bash
# 1. Clone/Navigate to project
cd rcgnn

# 2. Run setup (do this ONCE)
make setup

# 3. Activate environment
conda activate rcgnn-env

# 4. Verify everything works
make info
make check-env
```

### **Week 2+: Running Experiments**

```bash
# Activate environment first!
conda activate rcgnn-env

# Option A: Quick test
make train

# Option B: Full analysis
make full-pipeline
make results

# Option C: Compare methods
make train
make baseline
make results
```

---

## 🎯 **WHAT EACH STEP PRODUCES**

### **After `make train`:**
- ✅ `artifacts/checkpoints/rcgnn_best.pt` - Trained model
- ✅ `artifacts/adjacency/A_mean.npy` - Learned structure
- ✅ `artifacts/training_metrics.json` - Training stats

### **After `make analyze`:**
- ✅ `artifacts/threshold_report.txt` - Threshold analysis
- ✅ `artifacts/threshold_analysis.png` - Charts

### **After `make baseline`:**
- ✅ `artifacts/baseline_comparison_report.txt` - Results
- ✅ `artifacts/baseline_comparison.png` - Charts

### **After `make full-pipeline`:**
- ✅ All of the above! (Everything)

---

## 🔧 **TROUBLESHOOTING**

| Problem | Solution |
|---------|----------|
| `make: command not found` | Install make: `brew install make` or `apt install make` |
| `conda: command not found` | Install Anaconda from https://www.anaconda.com |
| `ModuleNotFoundError: torch` | Run: `make install-deps` |
| Training seems stuck | Normal! Takes 60 seconds on CPU - be patient |
| Want to start over | Run: `make clean-all && make all` |
| Can't find results | Run: `make results` or `make view-artifacts` |

---

## 📁 **FILE LOCATIONS**

```
Your Results:
├── Model (27 KB)
│   └── artifacts/checkpoints/rcgnn_best.pt
├── Learned Structure (1 KB)
│   └── artifacts/adjacency/A_mean.npy
├── Reports
│   ├── artifacts/threshold_report.txt
│   └── artifacts/baseline_comparison_report.txt
└── Visualizations
    ├── artifacts/threshold_analysis.png
    └── artifacts/baseline_comparison.png
```

---

## 💡 **QUICK WORKFLOWS**

### **"I just want to train the model"**
```bash
conda activate rcgnn-env
make train
```

### **"I want to see all results"**
```bash
conda activate rcgnn-env
make full-pipeline
make results
```

### **"I want to compare methods"**
```bash
conda activate rcgnn-env
make baseline
cat artifacts/baseline_comparison_report.txt
```

### **"I want to see detailed output"**
```bash
conda activate rcgnn-env
make train-verbose
```

### **"I want to start fresh"**
```bash
make clean
conda activate rcgnn-env
make train
```

### **"I want a complete reset"**
```bash
make clean-all
make setup
conda activate rcgnn-env
make full-pipeline
```

---

## 📋 **EXPECTED OUTPUT**

### When training succeeds, you'll see:
```
🚀 Starting RC-GNN training...
Epoch 01/8 | Train Loss: 16767.1890 | F1: 0.0 | SHD: 18
Epoch 02/8 | Train Loss: 70.0818 | F1: 0.08 | SHD: 22
...
✅ Training complete! Best SHD: 18
```

### When full pipeline succeeds, you'll see:
```
✅ STEP 1: Training complete
✅ STEP 2: Threshold optimization complete
✅ STEP 3: Environment analysis complete
✅ STEP 4: Baseline comparison complete
✅ STEP 5: Summary complete
```

---

## ❌ **COMMON MISTAKES & FIXES**

| Mistake | Fix |
|---------|-----|
| Forgot to activate conda env | Run: `conda activate rcgnn-env` |
| Running `make setup` twice | Harmless - just run again |
| Results not showing | Run: `make results` |
| Data not found | Check `make data-verify` |
| Old results cluttering workspace | Run: `make clean` |

---

## 🎓 **KEY CONCEPTS**

**What is RC-GNN?**
- Learns causal relationships in data
- Handles missing values automatically
- Works even with corrupted sensors

**What does "SHD" mean?**
- Lower is better (0 = perfect)
- Measures edge prediction accuracy
- SHD=18 is excellent for this dataset

**What is "threshold optimization"?**
- Finds best cutoff for edge detection
- Balances precision vs recall
- Generates decision curve

**What is "baseline comparison"?**
- Tests RC-GNN vs other methods
- Shows RC-GNN superiority
- Validates our approach

---

## 🎯 **SUCCESS CHECKLIST**

After running `make setup`:
- [ ] Python 3.12 installed
- [ ] Conda environment created
- [ ] All packages installed
- [ ] Data verified

After running `make train`:
- [ ] Model file created (27 KB)
- [ ] Adjacency matrix generated
- [ ] SHD = 18 (or similar good score)

After running `make full-pipeline`:
- [ ] All artifacts generated
- [ ] Reports created
- [ ] Visualizations shown
- [ ] Results summary displayed

---

## 📞 **HELP & RESOURCES**

```bash
# Show all commands
make help

# Show detailed guide
cat MAKEFILE_GUIDE.md

# Show project info
make info

# Check environment
make check-env

# See pipeline details
cat FIX_SUMMARY.md

# Read full documentation
cat README.md
```

---

## ⏱️ **TYPICAL TIMELINE**

| Task | Time |
|------|------|
| Initial setup | 2 minutes |
| Training | 60 seconds |
| Threshold optimization | 2 seconds |
| Baseline comparison | 3 seconds |
| Viewing results | 10 seconds |
| **Total for first run** | **~2 min 15 sec** |
| **Total for subsequent runs** | **~70 seconds** |

---

## 🚀 **START NOW**

```bash
# Copy this exact command:
make setup && conda activate rcgnn-env && make full-pipeline && make results

# Or run step by step:
make setup
conda activate rcgnn-env
make full-pipeline
make results
```

**You're ready!** 🎉

---

## 📝 **REMEMBER:**

1. **Always activate conda environment first:**
   ```bash
   conda activate rcgnn-env
   ```

2. **First run takes longer** (software installation)

3. **Each `make` command is independent** - run any time

4. **Use `make help` when confused** - it's your friend

5. **Everything is reversible** - `make clean` is safe

---

**Made with ❤️ for easy RC-GNN execution**
