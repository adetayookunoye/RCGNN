# RC-GNN Makefile - Quick Start

## Installation

```bash
make install
```

## Complete Pipeline

```bash
make all
```

This runs:
1. Install dependencies
2. Generate all datasets (synthetic + UCI Air)
3. Train models on both datasets  
4. Run advanced validation
5. Compare with baselines
6. Show results summary

**Time:** 30-60 minutes

## Step-by-Step

```bash
# 1. Install
make install

# 2. Generate data
make data                    # All datasets
# OR
make data-synth-small        # Synthetic only
make data-air                # UCI Air only

# 3. Train
make train-synth             # Train on synthetic
make train-air               # Train on UCI Air

# 4. Validate (publication-ready)
make validate-synth-advanced # Synthetic validation
make validate-air-advanced   # UCI Air validation

# 5. View results
make results
```

## Quick Commands

```bash
make help          # Show all commands
make status        # Check project status
make clean         # Clean artifacts
make test          # Run tests
```

## For Paper Submission

```bash
make pipeline-paper    # Generate all paper results
make results-paper     # Get LaTeX table
```

## Key Outputs

After `make all`:

- **Adjacency matrices:** `artifacts/adjacency/A_mean*.npy`
- **Validation results:** `artifacts/validation_*_advanced/metrics.json`
- **Figures:** `artifacts/validation_*_advanced/*.png`
- **Baseline comparison:** `artifacts/baseline_comparison/`

## Documentation

- `MAKEFILE_GUIDE.md` - Complete guide
- `MAKEFILE_CHEATSHEET.md` - Quick reference card
- Run `make help` for all commands

## Troubleshooting

```bash
make check-env     # Verify environment
make version       # Show package versions
make status        # Check what's ready
```

---

**See `MAKEFILE_GUIDE.md` for detailed documentation.**
