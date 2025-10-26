# 📊 RC-GNN Validation & Visualization Scripts

You now have **two complementary scripts** for validation and visualization:

## 1. 🏃 `scripts/train_and_visualize.py` — Full Pipeline with Auto-Visualization

**Purpose:** Train the model AND automatically generate all visualizations in one command.

### Usage:
```bash
python scripts/train_and_visualize.py configs/data_uci.yaml configs/model.yaml configs/train.yaml
```

### What It Does:
1. ✅ Trains the RC-GNN model (8 epochs)
2. ✅ Saves best checkpoint to `artifacts/checkpoints/rcgnn_best.pt`
3. ✅ Saves learned adjacency to `artifacts/adjacency/A_mean.npy`
4. ✅ Automatically generates visualizations:
   - `artifacts/visualizations/01_adjacency_heatmap.png`
   - `artifacts/visualizations/02_edge_distribution.png`
   - `artifacts/visualizations/03_causal_graph.png`
5. ✅ Generates validation report: `artifacts/visualizations/validation_report.txt`

### Options:
```bash
# Skip visualization (training only)
python scripts/train_and_visualize.py configs/data_uci.yaml configs/model.yaml configs/train.yaml --no-visualize

# Custom export directory
python scripts/train_and_visualize.py configs/data_uci.yaml configs/model.yaml configs/train.yaml --export-dir my_results/
```

---

## 2. 🔍 `scripts/validate_and_visualize.py` — Standalone Validation

**Purpose:** Load a pre-trained model and generate visualizations without retraining.

### Usage:
```bash
# Use default paths (assumes artifacts/adjacency/A_mean.npy exists)
python scripts/validate_and_visualize.py

# Custom paths
python scripts/validate_and_visualize.py \
  --adjacency path/to/adjacency.npy \
  --export artifacts/visualizations \
  --data-root data/interim/uci_air \
  --threshold 0.5
```

### What It Generates:
- ✅ Adjacency heatmap
- ✅ Edge strength distribution
- ✅ Causal graph network visualization
- ✅ Validation metrics (precision, recall, F1, SHD)
- ✅ Hub structure analysis
- ✅ Top 15 edges ranking

### Command-Line Arguments:
```
--adjacency PATH        Path to adjacency matrix (default: artifacts/adjacency/A_mean.npy)
--export DIR            Export directory (default: artifacts/visualizations)
--data-root PATH        Data root for ground truth (optional)
--threshold FLOAT       Binary threshold (default: 0.5)
```

---

## 🎯 Quick Start Guide

### Option A: Full Training + Visualization (Recommended for New Runs)
```bash
python scripts/train_and_visualize.py configs/data_uci.yaml configs/model.yaml configs/train.yaml
```

### Option B: Visualize Existing Model
```bash
# After running train_and_visualize.py or train_rcgnn.py:
python scripts/validate_and_visualize.py
```

### Option C: Train Only (No Visualization)
```bash
python scripts/train_and_visualize.py configs/data_uci.yaml configs/model.yaml configs/train.yaml --no-visualize
```

---

## 📁 Output Structure

After running `train_and_visualize.py`:
```
artifacts/
├── checkpoints/
│   └── rcgnn_best.pt                 # Model weights
├── adjacency/
│   └── A_mean.npy                    # Learned causal matrix (13×13)
└── visualizations/
    ├── 01_adjacency_heatmap.png      # Full adjacency heatmap
    ├── 02_edge_distribution.png      # Edge strength histogram
    ├── 03_causal_graph.png           # Network graph (top 25 edges)
    └── validation_report.txt         # Statistics & interpretation
```

---

## 📊 Visualization Details

### 1. **Adjacency Heatmap** (`01_adjacency_heatmap.png`)
- Shows the complete 13×13 adjacency matrix
- Color intensity = edge strength
- Yellow = weak edges, Red = strong edges
- Useful for understanding overall connectivity patterns

### 2. **Edge Distribution** (`02_edge_distribution.png`)
- **Left panel:** Histogram of non-zero edge strengths
  - Shows distribution shape (e.g., heavy-tail, uniform)
  - Median and mean lines for reference
- **Right panel:** Sparsity breakdown
  - Zero edges vs non-zero vs above threshold
  - Percentage labels for quick reference

### 3. **Causal Graph** (`03_causal_graph.png`)
- Network visualization of top K edges
- Node colors: gradient from blue to red (0-13)
- Edge width: proportional to strength
- Spring layout: highlights hub nodes naturally
- Best for understanding causal flow and dependencies

### 4. **Validation Report** (`validation_report.txt`)
Contains:
- Adjacency statistics (min/max/mean/std)
- Sparsity analysis
- Top 15 strongest edges
- Hub structure analysis
- Ground truth comparison (if available)

---

## 🔧 Technical Details

### Automatic Visualization Workflow
The `train_and_visualize.py` script:
1. Runs training exactly like `train_rcgnn.py`
2. After training completes, calls `validate_and_visualize.py` via subprocess
3. Captures all output and displays to user
4. Gracefully handles missing visualization script

### Visualization Components
- **Matplotlib** for heatmaps and distributions
- **Seaborn** for color mapping
- **NetworkX** for graph visualization
- **NumPy/SciPy** for statistics

### Metrics Computed
- **Sparsity:** Zero edges ratio
- **Edge statistics:** Min/max/mean/median/std
- **Hub analysis:** In-degree and out-degree
- **Ground truth comparison:** Precision, recall, F1, SHD (if A_true.npy available)

---

## ✅ Features Summary

| Feature | `train_rcgnn.py` | `train_and_visualize.py` | `validate_and_visualize.py` |
|---------|-----------------|------------------------|--------------------------|
| Train model | ✅ | ✅ | ❌ |
| Save checkpoint | ✅ | ✅ | ❌ |
| Auto-visualize | ❌ | ✅ | ❌ |
| Generate visualizations | ❌ | ✅ (automatic) | ✅ |
| Generate report | ❌ | ✅ (automatic) | ✅ |
| Standalone validation | ❌ | ❌ | ✅ |
| Custom paths | ✅ | ✅ | ✅ |

---

## 🚀 Recommended Workflow

1. **First training run:** Use `train_and_visualize.py` to train and visualize
   ```bash
   python scripts/train_and_visualize.py configs/data_uci.yaml configs/model.yaml configs/train.yaml
   ```

2. **Re-visualize without retraining:** Use `validate_and_visualize.py`
   ```bash
   python scripts/validate_and_visualize.py --export my_new_results/
   ```

3. **Batch visualization experiments:** Train with `--no-visualize`, then visualize different thresholds
   ```bash
   # Train
   python scripts/train_and_visualize.py ... --no-visualize
   
   # Visualize with threshold 0.3
   python scripts/validate_and_visualize.py --threshold 0.3 --export results_t03/
   
   # Visualize with threshold 0.5
   python scripts/validate_and_visualize.py --threshold 0.5 --export results_t05/
   ```

---

## 🐛 Troubleshooting

**Q: "Validation script not found" warning**
- A: Ensure `scripts/validate_and_visualize.py` exists in your repo
- Install it from backup if needed

**Q: Visualizations generated but seem cut off**
- A: Check DPI settings in config or manually re-run with adjusted parameters

**Q: No ground truth metrics (precision, recall, F1)**
- A: Ground truth not found. Provide `--data-root` or place `A_true.npy` in dataset folder

**Q: Memory error during visualization**
- A: Reduce top_k_edges in code or use custom threshold

---

## 📝 Notes

- Both scripts use `path_helper` to resolve imports automatically
- Visualizations are saved as PNG at 150 DPI (high quality, reasonable size)
- All metrics computed from continuous adjacency (no binarization for statistics)
- Report saved as plain text for easy viewing/sharing

---

**✨ You're now set up for complete visualization of your learned causal structures!**
