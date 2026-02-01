# RC-GNN: Robust Causal Graph Neural Networks

**Robust causal structure learning from time-series data with compound sensor corruptions**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Architecture](#architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Datasets](#datasets)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [License](#license)

---

## Overview

RC-GNN is a graph neural network framework for learning causal structures from time-series data corrupted by multiple sensor failure modes. The model combines:

- **Disentangled encoders** for separating signal from corruption factors
- **Causal graph learning** with per-environment adjacency deltas
- **MNAR missingness modeling** with inverse probability weighting
- **Student-t robust likelihood** for outlier resistance
- **Differentiable sparsification** (top-k, sparsemax, entmax, Gumbel)
- **Causal prior regularization** for intervention and mechanism invariance

### Key Contributions

1. **Robust to compound corruptions**: Missing data (MCAR/MNAR), sensor drift, measurement noise
2. **Environment-aware structure learning**: Learns base graph with environment-specific deltas
3. **MNAR-aware reconstruction**: Selection model with stabilized inverse probability weighting
4. **Causal priors**: Intervention, orientation, necessity, and mechanism invariance losses
5. **Calibrated evaluation**: Sensitivity analysis and threshold calibration protocol

---

## Key Results

### UCI Air Quality Dataset (13 variables, 13 true edges)

Evaluation across multiple corruption scenarios using calibrated Top-K sparsification:

| Corruption | RC-GNN SHD | RC-GNN F1 | NOTEARS-Lite SHD | NOTEARS-Lite F1 |
|------------|------------|-----------|------------------|-----------------|
| **clean_full** | 0 | 1.00 | 12 | 0.62 |
| **compound_full** | 9 | 0.92 | 25 | 0.19 |
| **compound_mnar_bias** | 0 | 1.00 | 21 | 0.32 |
| **mcar_20** | 0 | 1.00 | - | - |
| **mcar_40** | 0 | 1.00 | - | - |
| **extreme (40% missing)** | 35 | 0.43 | - | - |

---

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/adetayookunoye/rcgnn.git
cd rcgnn

# Install dependencies
pip install -r requirements.txt

# Or use conda
conda env create -f environment.yml
conda activate rcgnn-env
```

### 2. Generate Synthetic Data

```bash
python scripts/synth_bench.py --d 15 --edges 30 --n_envs 3 --output data/interim/synth_small
```

### 3. Train Model

```bash
python scripts/train_rcgnn_unified.py \
    --data_dir data/interim/uci_air_c/compound_full \
    --output_dir artifacts/experiment_1 \
    --epochs 100
```

### 4. Evaluate Results

```bash
python scripts/comprehensive_evaluation.py \
    --artifacts-dir artifacts \
    --data-dir data/interim \
    --output artifacts/evaluation_report.json
```

---

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.9+
- NumPy, SciPy, Pandas
- scikit-learn
- Matplotlib, Seaborn
- NetworkX

### Method 1: pip

```bash
pip install -r requirements.txt
```

### Method 2: conda

```bash
conda env create -f environment.yml
conda activate rcgnn-env
```

---

## Architecture

### Model Components

RC-GNN consists of four main components:

```
Input: X (observed data), M (missingness mask), e (environment labels)
                           |
                           v
           +-------------------------------+
           |     Disentangled Encoder      |
           |   X -> (z_signal, z_corrupt)  |
           +-------------------------------+
                           |
                           v
           +-------------------------------+
           |    Causal Graph Learner       |
           |  A = sigmoid(W_adj + delta_e) |
           +-------------------------------+
                           |
                           v
           +-------------------------------+
           |   Heteroscedastic Decoder     |
           |   (z, A) -> (mu, sigma, nu)   |
           +-------------------------------+
                           |
                           v
           +-------------------------------+
           |     MNAR Missingness Head     |
           |      P(M | X*, e) -> IPW      |
           +-------------------------------+
```

### Disentangled Encoder

Separates signal from corruption factors:

```python
z_signal = E_signal(X)    # Causal signal
z_corrupt = E_corrupt(X)  # Corruption factors (noise, drift)
```

### Causal Graph Learner

Learns adjacency with per-environment deltas:

```python
A_base = sigmoid(W_adj / tau)           # Shared base graph
A_env = sigmoid((W_adj + delta_e) / tau) # Environment-specific
```

### Loss Function

The complete objective combines six loss terms:

$$\mathcal{L} = \lambda_r \mathcal{L}_{\text{recon}} + \lambda_s \|W\|_1 + \lambda_a h(A) + \lambda_d \mathcal{L}_{\text{HSIC}} + \lambda_{\text{inv}} \mathcal{L}_{\text{inv}} + \lambda_c \mathcal{L}_{\text{causal}}$$

| Component | Description |
|-----------|-------------|
| Reconstruction | Student-t NLL with IPW for MNAR |
| Sparsity | L1 on logits W (not A) |
| Acyclicity | NOTEARS constraint h(A) |
| Disentanglement | HSIC between z_signal and z_corrupt |
| Invariance | IRM-style structure stability |
| Causal Prior | Intervention/mechanism invariance |

---

## Training

### Unified Training Script

The main training script consolidates all features:

```bash
# Basic training
python scripts/train_rcgnn_unified.py \
    --data_dir data/interim/uci_air_c/compound_full \
    --epochs 100

# Multi-GPU with DDP
torchrun --nproc_per_node=4 scripts/train_rcgnn_unified.py \
    --ddp \
    --data_dir data/interim/uci_air_c/compound_full

# With GroupDRO for worst-case robustness
python scripts/train_rcgnn_unified.py \
    --data_dir data/interim/uci_air_c/compound_full \
    --use_groupdro
```

### Training Features

- Multi-GPU DDP support with single-GPU/CPU fallback
- GroupDRO for worst-case robustness across regimes
- 3-stage training: discovery, pruning, refinement
- Gradient stability with aggressive clipping and LR scheduling
- Causal diagnostics (correlation vs causation detection)

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 100 | Training epochs |
| `batch_size` | 32 | Batch size |
| `lr` | 5e-4 | Learning rate |
| `lambda_recon` | 1.0 | Reconstruction weight |
| `lambda_sparse` | 1e-4 | Sparsity weight (on logits) |
| `lambda_acyclic` | 0.05 | Acyclicity weight |
| `lambda_hsic` | 0.1 | Disentanglement weight |
| `target_edges` | 13 | Target number of edges |

### Loss Weight Scheduling

Regularizers use delayed warm-up schedules:

- **Acyclicity**: Delayed until 50% of training, then ramped
- **Sparsity**: Gradual increase from 30% of training
- **Budget penalty**: Asymmetric (stronger for under-shooting)

---

## Evaluation

### Comprehensive Evaluation

The evaluation script computes ground truth metrics, disentanglement quality, and baseline comparisons:

```bash
python scripts/comprehensive_evaluation.py \
    --artifacts-dir artifacts \
    --data-dir data/interim \
    --output artifacts/evaluation_report.json
```

### Metrics

| Metric | Description |
|--------|-------------|
| **SHD** | Structural Hamming Distance (lower is better) |
| **Skeleton F1** | F1 on undirected edges |
| **Directed F1** | F1 on directed edges |
| **Disentanglement** | HSIC between signal and corrupt latents |

### Calibration Protocol

The evaluation uses a threshold calibration protocol:

1. Select validation corruption (e.g., compound_full)
2. Sweep K (number of edges) from 5 to 50
3. Select K maximizing F1 on validation set
4. Apply same K to all test corruptions
5. Report sensitivity curve for robustness

### Baseline Methods

Seven methods are compared at equal sparsity:

- **RC-GNN** (this work)
- **NOTEARS** and **NOTEARS-Lite**
- **Granger causality**
- **PCMCI+**
- **PC Algorithm**
- **Correlation baseline**

---

## Datasets

### UCI Air Quality

The UCI Air Quality dataset (13 variables, 13 ground truth causal edges) with various corruption profiles:

| Dataset | Corruption Type | Missing % | Environments |
|---------|-----------------|-----------|--------------|
| clean_full | None | 0% | 1 |
| compound_full | Noise + MNAR + bias | 25% | 3 |
| compound_mnar_bias | MNAR + bias | 25% | 1 |
| mcar_20 | MCAR | 20% | 2 |
| mcar_40 | MCAR | 40% | 1 |
| extreme | All corruptions | 40% | 5 |
| mnar_structural | MNAR (structural) | 25% | 1 |

### Data Format

Each dataset directory contains:

```
dataset_name/
├── X_train.npy    # (N, T, d) observed data
├── M_train.npy    # (N, T, d) missingness mask (1=observed)
├── e_train.npy    # (N,) environment labels
├── A_true.npy     # (d, d) ground truth adjacency
├── X_val.npy
├── M_val.npy
├── e_val.npy
└── config.json    # Dataset metadata
```

### Generate Synthetic Data

```bash
python scripts/synth_bench.py \
    --d 15 \
    --edges 30 \
    --n_envs 3 \
    --missing_type mcar \
    --missing_rate 0.2 \
    --output data/interim/synth_custom
```

---

## Configuration

### Model Configuration (`configs/model.yaml`)

```yaml
encoder:
  hidden_dim: 64
  latent_dim: 32

structure:
  n_envs: 3
  temperature:
    init: 1.5
    final: 0.5
  sparsify:
    method: "topk"
    k: 13

loss:
  lambda_hsic: 0.1
  lambda_inv: 0.1
```

### Training Configuration (`configs/train.yaml`)

```yaml
epochs: 100
batch_size: 32
learning_rate: 5e-4
gradient_clip: 1.0

loss:
  lambda_recon: 1.0
  lambda_sparse: 1e-4
  lambda_acyclic: 0.05

device: "cpu"
seed: 1337
```

---

## Project Structure

```
rcgnn/
├── README.md
├── requirements.txt
├── environment.yml
├── Makefile
│
├── configs/                # Configuration files
│   ├── data.yaml
│   ├── model.yaml
│   └── train.yaml
│
├── data/interim/           # Datasets
│   ├── uci_air/
│   └── uci_air_c/          # Corrupted variants
│
├── scripts/                # Executable scripts
│   ├── train_rcgnn_unified.py
│   ├── comprehensive_evaluation.py
│   ├── synth_bench.py
│   └── run_baselines.py
│
├── src/                    # Source code
│   ├── models/
│   │   ├── rcgnn.py        # Main model
│   │   ├── encoders.py
│   │   ├── structure.py
│   │   ├── causal_priors.py
│   │   ├── invariance.py
│   │   └── disentanglement.py
│   └── training/
│       ├── loop.py
│       ├── optim.py
│       ├── metrics.py
│       └── baselines.py
│
├── tests/                  # Unit tests
│
└── artifacts/              # Outputs
    ├── checkpoints/
    └── evaluation_report.json
```

---

## Citation

If you use RC-GNN in your research, please cite:

```bibtex
@article{rcgnn2026,
  title={RC-GNN: Robust Causal Graph Neural Networks under Compound Sensor Corruptions},
  author={Okunoye, Adetayo},
  journal={SPIE Medical Imaging},
  year={2026}
}
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

