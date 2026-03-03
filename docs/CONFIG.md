# RC-GNN Configuration Reference

## Core Hyperparameter Surface (12 Knobs)

The following 12 hyperparameters define the **official ablation surface** for RC-GNN experiments.
Reviewers can verify reproducibility by checking that experiments vary only these parameters.

| Knob | Type | Default | Description |
|------|------|---------|-------------|
| `seed` | int | 1337 | Random seed for reproducibility |
| `epochs` | int | 100 | Training epochs |
| `batch_size` | int | 32 | Batch size |
| `lr` | float | 1e-3 | Learning rate |
| `latent_dim` | int | 32 | Latent space dimension for encoders |
| `hidden_dim` | int | 64 | Hidden layer dimension |
| `lambda_sparse` | float | 0.01 | L1 sparsity penalty on adjacency |
| `lambda_acyclic` | float | 1.0 | DAG acyclicity constraint weight |
| `temperature_init` | float | 1.0 | Initial Gumbel-softmax temperature |
| `temperature_final` | float | 0.1 | Final Gumbel-softmax temperature |
| `k_calibration_mode` | str | "oracle" | TopK calibration: "oracle" or "fixed" |
| `missingness_strategy` | str | "mnar" | Missingness model: "mnar" or "mcar" |

## Config File Format

Configurations are stored in YAML format under `configs/`:

```yaml
# configs/train.yaml
training:
  seed: 1337
  epochs: 100
  batch_size: 32
  lr: 0.001
  device: "cpu"

model:
  latent_dim: 32
  hidden_dim: 64

loss:
  lambda_sparse: 0.01
  lambda_acyclic: 1.0

structure:
  temperature_init: 1.0
  temperature_final: 0.1

calibration:
  k_mode: "oracle"  # "oracle" uses ground truth edge count, "fixed" uses target_edges

missingness:
  strategy: "mnar"  # "mnar" for true missing-not-at-random, "mcar" ignores missingness
```

## Ablation Matrix

For the paper, we run a full factorial on 3 key dimensions:

1. **Corruption Level**: clean, moderate, compound_full
2. **Missingness**: mcar, mnar
3. **K Selection**: oracle, fixed

This produces 12 configurations (3 × 2 × 2), each run with 3 seeds.

## Provenance Logging

Every run generates a `run_manifest.json` with:

```json
{
  "git": {"commit": "abc123", "branch": "main", "dirty": false},
  "python_env": {"python_version": "3.10.0", "packages": {...}},
  "cuda": {"available": false},
  "dataset_checksums": {"X.npy": "md5:...", "M.npy": "md5:..."},
  "config_validation": {"warnings": [], "errors": []},
  "timestamp": "2024-01-15T10:30:00Z"
}
```

See `src/utils/run_manifest.py` for implementation.

## Validation Rules

The config validation catches common mistakes:

| Check | Severity | Description |
|-------|----------|-------------|
| `lambda_sparse > 0.5` | WARNING | May over-sparsify graph |
| `lambda_acyclic < 0.1` | ERROR | DAG constraint too weak |
| `epochs < 10` | WARNING | May underfit |
| `lr > 0.1` | WARNING | Learning rate too high |
| `temperature_init < temperature_final` | ERROR | Temperature should decrease |
| `batch_size > 256` | WARNING | May cause memory issues |

## Running with Specific Configs

```bash
# Standard training
python scripts/train_rcgnn.py configs/data.yaml configs/model.yaml configs/train.yaml

# Quick debug run (5 epochs)
python scripts/train_rcgnn.py configs/data.yaml configs/model.yaml configs/tmp_train_quick.yaml

# Full ablation sweep
python scripts/run_ablation_sweep.py --config-template configs/train.yaml --out artifacts/ablation/
```

## Ground Truth Usage Policy

**CRITICAL**: Ground truth (`A_true.npy`) is NEVER used during training.

- Training uses only: `X.npy` (data), `M.npy` (mask), `e.npy` (regime)
- `A_true.npy` is used ONLY for:
  1. Post-hoc evaluation metrics (SHD, F1)
  2. Validation-set K calibration (if `k_mode=oracle`)
  
- The oracle audit test (`tests/test_oracle_audit.py`) verifies this constraint.

## Adding New Hyperparameters

When adding a new hyperparameter:

1. Add to `CORE_CONFIG_KNOBS` in `src/utils/run_manifest.py` if ablation-worthy
2. Add validation rule in `validate_config()` 
3. Document default and range in this file
4. Update golden tests if it affects metric computation
