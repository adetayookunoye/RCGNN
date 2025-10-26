# 🎬 How to Use Your New Visualization Pipeline

## Answer to Your Question

**Original Question:** "do i have the script responsible for the validation generation in my code?"

**Answer:** ✅ **YES, now you do!**

You now have **3 validation/visualization scripts:**

1. ✅ `scripts/validate_and_visualize.py` (ALREADY EXISTING - you had this)
2. ✅ `scripts/train_validate_visualize.py` (ALREADY EXISTING - alternative version)
3. ✅ `scripts/train_and_visualize.py` (NEWLY CREATED - recommended)

---

## The Setup

### What You Had Before
- Training script: `scripts/train_rcgnn.py`
- ❌ NO automatic visualization
- Had to manually create visualizations

### What You Have Now
- Training script: `scripts/train_rcgnn.py` (unchanged - still works)
- ✅ `scripts/train_and_visualize.py` (NEW - auto-visualizes after training)
- ✅ `scripts/validate_and_visualize.py` (already there - standalone)
- ✅ `scripts/train_validate_visualize.py` (already there - alternative)

---

## Going Forward: Your Workflow

### **When You Run Training** (Next Time)

Instead of:
```bash
python scripts/train_rcgnn.py configs/data_uci.yaml configs/model.yaml configs/train.yaml
```

Do this:
```bash
python scripts/train_and_visualize.py configs/data_uci.yaml configs/model.yaml configs/train.yaml
```

### What Happens Automatically

1. ✅ Model trains for 8 epochs
2. ✅ Best checkpoint saved: `artifacts/checkpoints/rcgnn_best.pt`
3. ✅ Learned adjacency saved: `artifacts/adjacency/A_mean.npy`
4. ✅ **Automatically generates visualizations:**
   - `artifacts/visualizations/01_adjacency_heatmap.png`
   - `artifacts/visualizations/02_edge_distribution.png`
   - `artifacts/visualizations/03_causal_graph.png`
5. ✅ **Automatically generates report:**
   - `artifacts/visualizations/validation_report.txt`

### Zero Additional Work Required
Just run the command and wait. Everything else happens automatically.

---

## If You Only Want to Visualize (No Retraining)

```bash
# If adjacency matrix already exists from previous training:
python scripts/validate_and_visualize.py
```

Done. New visualizations generated in 30 seconds.

---

## The Three Scenarios

### Scenario 1: Fresh Training + Visualization
```bash
# One command does everything
python scripts/train_and_visualize.py configs/data_uci.yaml configs/model.yaml configs/train.yaml

# Output: All artifacts + visualizations + report
# Time: ~15-20 minutes (8 epochs)
```

### Scenario 2: Re-visualize Existing Model
```bash
# Quick re-visualization without retraining
python scripts/validate_and_visualize.py

# Output: New visualizations + report
# Time: ~30 seconds
```

### Scenario 3: Train Only (No Visualization)
```bash
# If you only want to train for some reason
python scripts/train_and_visualize.py ... --no-visualize

# Output: Checkpoint + adjacency (no visualizations)
# Time: ~15-20 minutes
```

---

## File Structure After Running

```
rcgnn/
├── scripts/
│   ├── train_rcgnn.py                    (original - still works)
│   ├── train_and_visualize.py            (NEW - recommended for future)
│   ├── validate_and_visualize.py         (existing - standalone)
│   └── train_validate_visualize.py       (existing - alternative)
│
├── artifacts/
│   ├── checkpoints/
│   │   └── rcgnn_best.pt                 (model after training)
│   ├── adjacency/
│   │   └── A_mean.npy                    (learned structure)
│   └── visualizations/                   (generated automatically)
│       ├── 01_adjacency_heatmap.png      (full matrix)
│       ├── 02_edge_distribution.png      (histogram)
│       ├── 03_causal_graph.png           (network graph)
│       └── validation_report.txt         (statistics)
│
├── QUICK_START.md                        (quick reference)
├── VISUALIZATION_GUIDE.md                (full documentation)
└── README.md                             (original)
```

---

## Summary

**Before:** You had validation scripts but no automatic integration
```
python scripts/train_rcgnn.py ...          # Train
# Manual steps to generate visualizations
```

**After:** One-command training + automatic visualization
```
python scripts/train_and_visualize.py ...  # Train + Visualize (automatic)
```

**Your answer to "do i have the scripts?":**
- ✅ Yes, all validation scripts exist
- ✅ Yes, they're integrated into your training pipeline  
- ✅ Yes, everything is automatic now
- ✅ Yes, multiple options depending on your needs

**What to do next time you train:**
- Use `train_and_visualize.py` instead of `train_rcgnn.py`
- Everything else happens automatically
- Visualizations appear in `artifacts/visualizations/`

---

**You're all set! 🚀**
