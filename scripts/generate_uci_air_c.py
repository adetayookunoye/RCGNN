#!/usr/bin/env python3
"""
Generate UCI-Air-C (Corrupted) Benchmark Suite

Creates synthetically corrupted variants of the UCI Air Quality dataset
with known ground-truth DAG for proper robustness evaluation.

Corruption Grid (reviewer-approved):
┌─────────────────┬─────────────────────────────────┐
│ Corruption Type │ Levels                          │
├─────────────────┼─────────────────────────────────┤
│ Gaussian noise  │ σ ∈ {0.1, 0.3, 0.5}             │
│ Missingness     │ MCAR: 20%, 30%, 40%             │
│ MNAR            │ structural (parent-dependent)   │
│ Sensor bias     │ additive, multiplicative        │
│ Regime shift    │ env ∈ {0, 1, 2}                 │
└─────────────────┴─────────────────────────────────┘

Each sample gets MULTIPLE corruptions simultaneously = compound corruption.

Output structure:
    data/interim/uci_air_c/
    ├── clean/                    # Reference (no corruption)
    ├── mild/                     # Low corruption  
    ├── moderate/                 # Medium corruption
    ├── severe/                   # High corruption
    ├── extreme/                  # Very high corruption
    ├── noise_0.1/                # Single: noise only
    ├── noise_0.3/
    ├── noise_0.5/
    ├── mcar_20/                  # Single: MCAR only
    ├── mcar_30/
    ├── mcar_40/
    ├── mnar_structural/          # Single: structural MNAR
    ├── bias_additive/            # Single: additive bias
    ├── bias_multiplicative/      # Single: multiplicative bias
    ├── compound_mcar_noise/      # Compound: MCAR + noise
    ├── compound_mnar_bias/       # Compound: MNAR + bias
    ├── compound_full/            # Compound: all corruptions
    └── regimes_3/                # Multi-regime (3 environments)
"""

import sys
import os

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Import corruption utilities
from src.rc_gnn.corruption import (
    generate_mcar_mask,
    generate_mar_mask,
    generate_mnar_mask,
    generate_structural_mnar_mask,
    generate_regime_shifts,
    add_gaussian_noise,
    add_outliers,
    add_drift,
    add_constant_bias,
    add_scaling_bias,
    add_nonlinear_bias,
    CompoundCorruptionGenerator,
)


def load_clean_data(data_root: str) -> Dict[str, np.ndarray]:
    """Load clean UCI Air data."""
    X = np.load(os.path.join(data_root, "X.npy"))
    M = np.load(os.path.join(data_root, "M.npy"))
    A_true = np.load(os.path.join(data_root, "A_true.npy"))
    
    # Load metadata if exists
    meta_path = os.path.join(data_root, "A_true_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    else:
        meta = {"num_edges": int(A_true.sum())}
    
    return {
        "X": X,
        "M": M,
        "A_true": A_true,
        "meta": meta,
    }


def save_corrupted_data(
    output_dir: str,
    X: np.ndarray,
    M: np.ndarray,
    e: np.ndarray,
    A_true: np.ndarray,
    corruption_config: Dict,
    meta: Dict,
):
    """Save corrupted dataset."""
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, "X.npy"), X)
    np.save(os.path.join(output_dir, "M.npy"), M)
    np.save(os.path.join(output_dir, "e.npy"), e)
    np.save(os.path.join(output_dir, "A_true.npy"), A_true)
    
    # Save corruption config
    config = {
        **meta,
        "corruption": corruption_config,
        "X_shape": list(X.shape),
        "missing_rate_actual": float(1 - M.mean()),
        "n_regimes": len(np.unique(e)),
    }
    
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"  Saved to {output_dir}")
    print(f"    X: {X.shape}, Missing: {1-M.mean():.1%}, Regimes: {len(np.unique(e))}")


def apply_corruption(
    X: torch.Tensor,
    A_true: torch.Tensor,
    config: Dict,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply compound corruption based on config.
    
    Args:
        X: Clean data [B, T, d]
        A_true: Ground truth adjacency [d, d]
        config: Corruption configuration
        seed: Random seed
        
    Returns:
        X_corrupt: Corrupted data
        M: Missingness mask (1=observed)
        e: Regime indicators
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    B, T, d = X.shape
    X_out = X.clone()
    
    # Default: no corruption
    M = torch.ones_like(X)
    e = torch.zeros(B, dtype=torch.long)
    
    # 1. Apply regime shifts (if specified)
    if config.get("n_regimes", 1) > 1:
        X_out, e = generate_regime_shifts(
            X_out,
            n_regimes=config["n_regimes"],
            shift_type=config.get("regime_shift_type", "intervention"),
            intervention_strength=config.get("intervention_strength", 1.0),
            seed=seed,
        )
    
    # 2. Apply bias (before noise)
    if config.get("bias_type") == "additive":
        X_out = add_constant_bias(X_out, config.get("bias_magnitude", 0.5))
    elif config.get("bias_type") == "multiplicative":
        X_out = add_scaling_bias(X_out, config.get("scale_range", (0.8, 1.2)))
    elif config.get("bias_type") == "nonlinear":
        X_out = add_nonlinear_bias(X_out, config.get("poly_degree", 2), config.get("bias_magnitude", 0.1))
    
    # 3. Apply noise
    if config.get("noise_scale", 0) > 0:
        X_out = add_gaussian_noise(
            X_out,
            scale=config["noise_scale"],
            heteroscedastic=config.get("heteroscedastic", False),
        )
    
    if config.get("outlier_rate", 0) > 0:
        X_out = add_outliers(
            X_out,
            outlier_rate=config["outlier_rate"],
            outlier_scale=config.get("outlier_scale", 5.0),
        )
    
    if config.get("drift_rate", 0) > 0:
        X_out = add_drift(X_out, drift_rate=config["drift_rate"])
    
    # 4. Apply missingness
    missing_mechanism = config.get("missing_mechanism", "none")
    missing_rate = config.get("missing_rate", 0)
    
    if missing_mechanism == "mcar" and missing_rate > 0:
        M = generate_mcar_mask(X.shape, missing_rate, X.device)
    elif missing_mechanism == "mar" and missing_rate > 0:
        M = generate_mar_mask(X_out, missing_rate)
    elif missing_mechanism == "mnar" and missing_rate > 0:
        M = generate_mnar_mask(X_out, missing_rate, config.get("mnar_type", "self"))
    elif missing_mechanism == "structural_mnar" and missing_rate > 0:
        M = generate_structural_mnar_mask(X_out, A_true, missing_rate)
    
    # Apply mask (set missing to 0)
    X_out = X_out * M
    
    return X_out, M, e


# ============================================================================
# CORRUPTION CONFIGURATIONS
# ============================================================================

# Single corruption ablations
SINGLE_CORRUPTIONS = {
    "clean": {},
    
    # Noise levels
    "noise_0.1": {"noise_scale": 0.1},
    "noise_0.3": {"noise_scale": 0.3},
    "noise_0.5": {"noise_scale": 0.5},
    
    # MCAR levels
    "mcar_20": {"missing_mechanism": "mcar", "missing_rate": 0.2},
    "mcar_30": {"missing_mechanism": "mcar", "missing_rate": 0.3},
    "mcar_40": {"missing_mechanism": "mcar", "missing_rate": 0.4},
    
    # MNAR variants
    "mnar_self": {"missing_mechanism": "mnar", "mnar_type": "self", "missing_rate": 0.2},
    "mnar_threshold": {"missing_mechanism": "mnar", "mnar_type": "threshold", "missing_rate": 0.2},
    "mnar_structural": {"missing_mechanism": "structural_mnar", "missing_rate": 0.2},
    
    # Bias variants
    "bias_additive": {"bias_type": "additive", "bias_magnitude": 0.5},
    "bias_multiplicative": {"bias_type": "multiplicative", "scale_range": (0.7, 1.3)},
    
    # Regime shifts
    "regimes_3": {"n_regimes": 3, "regime_shift_type": "intervention"},
    "regimes_5": {"n_regimes": 5, "regime_shift_type": "distribution"},
}

# Compound corruptions (multiple simultaneous)
COMPOUND_CORRUPTIONS = {
    # Preset severity levels
    "mild": {
        "noise_scale": 0.1,
        "missing_mechanism": "mcar",
        "missing_rate": 0.1,
        "n_regimes": 1,
    },
    "moderate": {
        "noise_scale": 0.2,
        "heteroscedastic": True,
        "missing_mechanism": "mar",
        "missing_rate": 0.2,
        "bias_type": "additive",
        "bias_magnitude": 0.2,
        "n_regimes": 2,
    },
    "severe": {
        "noise_scale": 0.3,
        "heteroscedastic": True,
        "outlier_rate": 0.05,
        "drift_rate": 0.02,
        "missing_mechanism": "mnar",
        "mnar_type": "self",
        "missing_rate": 0.3,
        "bias_type": "multiplicative",
        "scale_range": (0.7, 1.3),
        "n_regimes": 3,
    },
    "extreme": {
        "noise_scale": 0.5,
        "heteroscedastic": True,
        "outlier_rate": 0.1,
        "outlier_scale": 10.0,
        "drift_rate": 0.05,
        "missing_mechanism": "structural_mnar",
        "missing_rate": 0.4,
        "bias_type": "nonlinear",
        "bias_magnitude": 0.2,
        "poly_degree": 3,
        "n_regimes": 5,
    },
    
    # Specific compound combinations
    "compound_mcar_noise": {
        "noise_scale": 0.3,
        "missing_mechanism": "mcar",
        "missing_rate": 0.3,
    },
    "compound_mnar_bias": {
        "missing_mechanism": "structural_mnar",
        "missing_rate": 0.25,
        "bias_type": "multiplicative",
        "scale_range": (0.8, 1.2),
    },
    "compound_mnar_noise_bias": {
        "noise_scale": 0.2,
        "missing_mechanism": "mnar",
        "mnar_type": "self",
        "missing_rate": 0.2,
        "bias_type": "additive",
        "bias_magnitude": 0.3,
    },
    "compound_full": {
        "noise_scale": 0.3,
        "heteroscedastic": True,
        "outlier_rate": 0.03,
        "drift_rate": 0.01,
        "missing_mechanism": "structural_mnar",
        "missing_rate": 0.25,
        "bias_type": "multiplicative",
        "scale_range": (0.85, 1.15),
        "n_regimes": 3,
    },
    
    # Sensor failure simulation
    "sensor_failure": {
        "noise_scale": 0.1,
        "heteroscedastic": True,
        "missing_mechanism": "mnar",
        "mnar_type": "threshold",  # Sensor fails at high readings
        "missing_rate": 0.25,
        "bias_type": "multiplicative",  # Calibration drift
        "scale_range": (0.9, 1.1),
        "n_regimes": 2,
    },
}


def main():
    parser = argparse.ArgumentParser(description="Generate UCI-Air-C benchmark suite")
    parser.add_argument("--input", default="data/interim/uci_air",
                        help="Path to clean UCI Air data")
    parser.add_argument("--output", default="data/interim/uci_air_c",
                        help="Output directory for corrupted datasets")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Specific configs to generate (default: all)")
    parser.add_argument("--list", action="store_true",
                        help="List available configurations")
    args = parser.parse_args()
    
    # Combine all configurations
    ALL_CONFIGS = {**SINGLE_CORRUPTIONS, **COMPOUND_CORRUPTIONS}
    
    if args.list:
        print("\n=== Available Corruption Configurations ===\n")
        print("Single corruptions:")
        for name in SINGLE_CORRUPTIONS:
            print(f"  - {name}")
        print("\nCompound corruptions:")
        for name in COMPOUND_CORRUPTIONS:
            print(f"  - {name}")
        return
    
    print("=" * 60)
    print("UCI-Air-C Benchmark Generator")
    print("=" * 60)
    
    # Load clean data
    print(f"\nLoading clean data from {args.input}...")
    data = load_clean_data(args.input)
    
    X_clean = torch.from_numpy(data["X"]).float()
    A_true = torch.from_numpy(data["A_true"]).float()
    
    print(f"  X shape: {X_clean.shape}")
    print(f"  Ground truth edges: {int(A_true.sum())}")
    
    # Determine which configs to generate
    if args.configs:
        configs_to_gen = {k: ALL_CONFIGS[k] for k in args.configs if k in ALL_CONFIGS}
    else:
        configs_to_gen = ALL_CONFIGS
    
    print(f"\nGenerating {len(configs_to_gen)} corrupted variants...")
    print("-" * 60)
    
    # Generate each variant
    for name, config in configs_to_gen.items():
        print(f"\n[{name}]")
        print(f"  Config: {config}")
        
        # Apply corruption
        X_corrupt, M, e = apply_corruption(
            X_clean, A_true, config, seed=args.seed
        )
        
        # Save
        output_dir = os.path.join(args.output, name)
        save_corrupted_data(
            output_dir,
            X_corrupt.numpy(),
            M.numpy(),
            e.numpy(),
            data["A_true"],
            config,
            data["meta"],
        )
    
    print("\n" + "=" * 60)
    print("BENCHMARK GENERATION COMPLETE")
    print("=" * 60)
    print(f"\nOutput: {args.output}/")
    print(f"Total variants: {len(configs_to_gen)}")
    print("\nTo train on corrupted data:")
    print(f"  python scripts/train_rcgnn_v3.py --data_root {args.output}/severe ...")
    print("\nTo run baselines on same corruptions:")
    print(f"  python scripts/run_baselines.py --data_root {args.output}/severe ...")


if __name__ == "__main__":
    main()
