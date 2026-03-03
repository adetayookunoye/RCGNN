# RC-GNN Makefile Guide

Complete guide for using the Makefile to streamline your RC-GNN workflow.

## Quick Start

```bash
# See all available commands
make help

# Complete pipeline (recommended for first run)
make all

# Or step by step:
make install          # Install dependencies
make data             # Generate all datasets
make train-synth      # Train on synthetic data
make train-air        # Train on UCI Air data
make validate-all     # Run advanced validation
make results          # Show results summary
```

---

## Installation & Setup

### Install Dependencies
```bash
make install
```
Installs all required Python packages from `requirements.txt`.

### Create Conda Environment (Optional)
```bash
make create-env        # Create environment
conda activate rcgnn-env
make install           # Install packages in environment
```

### Complete Setup
```bash
make setup            # Create environment + install dependencies
```

### Check Environment
```bash
make check-env        # Verify packages
make version          # Show version info
```

---

## Data Processing

### Generate All Datasets
```bash
make data
```
Creates:
- Synthetic small (linear)
- Synthetic nonlinear (MLP)  
- UCI Air Quality

### Individual Datasets

**Synthetic Small**
```bash
make data-synth-small
```
Output: `data/interim/synth_small/`

**Synthetic Nonlinear**
```bash
make data-synth-nonlinear
```
Output: `data/interim/synth_nonlinear/`

**UCI Air Quality**
```bash
make data-air
```
Output: `data/interim/uci_air/`

### Inspect Data
```bash
make data-inspect
```

---

## Training

### Train on Synthetic
```bash
make train-synth
```
Output: `artifacts/adjacency/A_mean.npy`

### Train on UCI Air
```bash
make train-air
```
Output: `artifacts/adjacency/A_mean_air.npy`

### Train Both
```bash
make train-all
```

### Quick Test
```bash
make train-quick     # 5 epochs only
```

---

## Validation

### Advanced Validation (Publication-Ready) ⭐

**Synthetic:**
```bash
make validate-synth-advanced
```

**UCI Air:**
```bash
make validate-air-advanced
```

**Both:**
```bash
make validate-all
```

### Basic Validation

```bash
make validate-synth    # Basic synthetic
make validate-air      # Basic UCI Air
```

---

## Results & Reports

### Show Results
```bash
make results
```

Example output:
```
📊 SYNTHETIC DATA:
  AUPRC: 0.3456
  F1: 0.4123
  SHD: 8

🌍 UCI AIR QUALITY:
  AUPRC: 0.1397 (+67.7% vs chance)
  F1: 0.2857
  Orientation: 100.0%
```

### LaTeX Table for Paper
```bash
make results-paper
```

---

## Complete Pipelines

### Full Pipeline
```bash
make all
```
Runs everything (~30-60 min)

### Synthetic Only
```bash
make pipeline-synth
```
(~10-15 min)

### UCI Air Only
```bash
make pipeline-air
```
(~15-20 min)

### Paper Results
```bash
make pipeline-paper
```

---

## Utilities

### Project Status
```bash
make status
```

### Baseline Comparison
```bash
make compare-baselines
```

### Testing
```bash
make test          # All tests
make test-quick    # Smoke tests only
```

### Cleanup
```bash
make clean         # Clean artifacts (keep data)
make clean-data    # Clean datasets (asks confirmation)
make clean-all     # Clean everything
```

---

## Common Workflows

### 1. First Time
```bash
make install
make data
make status
```

### 2. Quick Experiment
```bash
make data-synth-small
make train-quick
make validate-synth
```

### 3. Paper Submission
```bash
make clean-all
make all
make results-paper
```

---

## Summary of Key Commands

| Command | Purpose | Time |
|---------|---------|------|
| `make help` | Show commands | 1s |
| `make install` | Install deps | 1-2min |
| `make data` | Generate data | 2-3min |
| `make train-synth` | Train synthetic | 5-10min |
| `make train-air` | Train UCI Air | 10-15min |
| `make validate-all` | Validate both | 1-2min |
| `make results` | Show summary | 1s |
| `make all` | Complete pipeline | 30-60min |

---

## Quick Reference

```bash
# Setup
make install && make data

# Train
make train-synth        # Synthetic
make train-air          # Real data

# Validate  
make validate-all       # Advanced validation

# Results
make results            # Summary
make results-paper      # LaTeX

# Complete
make all               # Everything
```

---

**Happy training! 🚀**
