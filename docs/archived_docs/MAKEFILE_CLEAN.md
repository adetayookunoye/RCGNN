# Clean Makefile - Simple & Silent

Your Makefile is now **clean and minimal** with just commands - no help text, no output unless needed.

## Available Commands

```bash
make setup              # Setup environment + dependencies + data
make train              # Quick training (~60 seconds)
make train-full        # Full training (~30+ minutes)
make analyze           # Threshold optimization
make baseline          # Baseline comparison
make full-pipeline     # train → analyze → baseline
make results           # Show results summary
make clean             # Remove artifacts
make clean-all         # Remove artifacts + conda env
make create-env        # Create conda environment
make install-deps      # Install Python dependencies
make data-prepare      # Prepare data directory
```

## That's It!

Just commands. No fluff. No help text.

- `make train` - runs training
- `make train-full` - runs full training
- `make analyze` - runs analysis
- etc.

Clean and simple. ✅
