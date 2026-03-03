# RC-GNN Makefile Quick Reference Card

## 🚀 Most Common Commands

```bash
# See all commands
make help

# Check what's ready
make status

# Install everything
make install

# Generate data
make data                    # All datasets
make data-synth-small        # Synthetic only
make data-air                # UCI Air only

# Train models
make train-synth             # Train on synthetic
make train-air               # Train on UCI Air
make train-all               # Train both

# Validate (publication-ready)
make validate-synth-advanced # Synthetic
make validate-air-advanced   # UCI Air  
make validate-all            # Both

# Results
make results                 # Summary
make results-paper           # LaTeX table

# Complete pipelines
make all                     # Everything
make pipeline-synth          # Synthetic only
make pipeline-air            # UCI Air only
make pipeline-paper          # Paper results
```

---

## 📊 Workflow Examples

### First Time Setup
```bash
make install
make data
make status
```

### Quick Test
```bash
make data-synth-small
make train-quick        # 5 epochs
make validate-synth
```

### Train & Validate Synthetic
```bash
make pipeline-synth
# Output: artifacts/validation_synth_advanced/
```

### Train & Validate UCI Air
```bash
make pipeline-air
# Output: artifacts/validation_air_advanced/
```

### Paper Submission
```bash
make clean-all          # Start fresh
make all                # Complete pipeline
make results            # Review
make results-paper      # LaTeX table
```

---

## 🔧 Utilities

```bash
make check-env          # Verify environment
make version            # Show versions
make docs               # List documentation
make test               # Run tests
make test-quick         # Smoke tests
make compare-baselines  # RC-GNN vs others
make clean              # Clean artifacts
make clean-all          # Clean everything
```

---

## 📁 Key Outputs

After running `make all`:

```
artifacts/
├── adjacency/
│   ├── A_mean.npy              # Synthetic adjacency
│   └── A_mean_air.npy          # UCI Air adjacency
├── validation_synth_advanced/
│   ├── calibration_curve.png
│   ├── score_distribution.png
│   └── metrics.json
├── validation_air_advanced/
│   ├── calibration_curve.png
│   ├── score_distribution.png
│   └── metrics.json            ← Main results
└── baseline_comparison/
    └── comparison_4panel.png
```

---

## ⏱️ Time Estimates

| Command | Time |
|---------|------|
| `make install` | 1-2 min |
| `make data` | 2-3 min |
| `make train-synth` | 5-10 min |
| `make train-air` | 10-15 min |
| `make validate-all` | 1-2 min |
| `make all` | 30-60 min |

---

## 🎯 For Paper Submission

**Complete workflow:**
```bash
# 1. Fresh start
make clean-all

# 2. Generate everything
make all

# 3. View results
make results

# 4. Get LaTeX table
make results-paper

# 5. Get baseline comparison
make compare-baselines
```

**Files for paper:**
- `artifacts/validation_air_advanced/metrics.json` → Table 1
- `artifacts/validation_air_advanced/calibration_curve.png` → Figure 1
- `artifacts/baseline_comparison/comparison_4panel.png` → Figure 2

---

## 🐛 Troubleshooting

**If training fails:**
```bash
make check-env       # Verify packages
make data-inspect    # Check data exists
```

**If validation fails:**
```bash
make status          # Check models trained
ls artifacts/adjacency/
```

**Start over:**
```bash
make clean-all       # Clean everything
make all             # Rebuild
```

---

## 💡 Tips

1. **Always check status first:**
   ```bash
   make status
   ```

2. **Use quick mode for testing:**
   ```bash
   make train-quick
   ```

3. **Save results after validation:**
   ```bash
   cp -r artifacts results_backup_$(date +%Y%m%d)
   ```

4. **Generate specific datasets:**
   ```bash
   make data-synth-small    # Just synthetic
   make data-air            # Just UCI Air
   ```

---

## 📚 Documentation

```bash
make docs               # List all guides
```

Available guides:
- `MAKEFILE_GUIDE.md` - This guide (detailed)
- `VALIDATION_INDEX.md` - Complete validation guide
- `VALIDATION_QUICK_REF.md` - Validation quick ref
- `VALIDATION_SUMMARY.md` - Results summary

---

## ✅ Checklist for Paper

Before submission, run:

```bash
□ make clean-all
□ make all
□ make results
□ make results-paper
□ make compare-baselines
□ Check artifacts/validation_air_advanced/metrics.json
□ Copy figures to paper/figures/
□ Copy LaTeX table to paper
```

---

**Print this card and keep it handy! 📋**
