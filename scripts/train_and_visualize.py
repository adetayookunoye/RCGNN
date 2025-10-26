#!/usr/bin/env python3
"""
Training script with automatic visualization generation.

This script:
1. Trains the RC-GNN model
2. Automatically generates visualizations of the learned causal structure
3. Creates a comprehensive validation report

Usage:
  python scripts/train_and_visualize.py configs/data.yaml configs/model.yaml configs/train.yaml
"""

import argparse
import yaml
import os
import sys
import torch
import numpy as np
import subprocess
from pathlib import Path

import path_helper  # noqa: F401  # adds project root to sys.path

from torch.utils.data import DataLoader
from src.dataio.loaders import load_synth
from src.models.rcgnn import RCGNN
from src.training.optim import make_optimizer
from src.training.loop import train_epoch, eval_epoch


def main():
    """Main training pipeline with visualization."""
    parser = argparse.ArgumentParser(
        description="Train RC-GNN and automatically generate visualizations"
    )
    parser.add_argument("data_cfg", help="Data configuration file")
    parser.add_argument("model_cfg", help="Model configuration file")
    parser.add_argument("train_cfg", help="Training configuration file")
    parser.add_argument("--no-visualize", action="store_true",
                       help="Skip visualization generation after training")
    parser.add_argument("--export-dir", default="artifacts/visualizations",
                       help="Directory for exported visualizations")
    
    args = parser.parse_args()

    # Load configurations
    with open(args.data_cfg) as f:
        dc = yaml.safe_load(f)
    with open(args.model_cfg) as f:
        mc = yaml.safe_load(f)
    with open(args.train_cfg) as f:
        tc = yaml.safe_load(f)

    # Setup dataset paths
    dataset_dir = dc.get("dataset_dir", dc.get("dataset", "synth_small"))
    root = os.path.join(dc["paths"]["root"], "interim", dataset_dir)
    train_ds = load_synth(root, "train", seed=tc["seed"])
    val_ds = load_synth(root, "val", seed=tc["seed"] + 1)

    train_ld = DataLoader(train_ds, batch_size=tc["batch_size"], shuffle=True)
    val_ld = DataLoader(val_ds, batch_size=1, shuffle=False)

    d = train_ds.X.shape[-1]

    # Initialize model
    model = RCGNN(d, mc)
    device = tc["device"]
    model.to(device)

    opt = make_optimizer(model.parameters(), tc)
    best_shd = 1e9

    # Load ground truth if available
    try:
        A_true = np.load(os.path.join(root, "A_true.npy"))
    except FileNotFoundError:
        A_true = None
        print("⚠️  Warning: A_true.npy not found; SHD metric will not be computed")

    # Training loop
    print("=" * 80)
    print("🚀 TRAINING RC-GNN")
    print("=" * 80)
    print(f"Dataset: {dataset_dir}")
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
    print(f"Features: {d} | Epochs: {tc['epochs']} | Device: {device}")
    print("=" * 80)

    for ep in range(tc["epochs"]):
        out = train_epoch(
            model, train_ld, opt,
            inv_weight=mc["loss"]["invariance"]["lambda_inv"],
            device=device
        )
        ev = eval_epoch(model, val_ld, A_true=A_true, thr=0.5, device=device)

        shd_val = ev.get("shd", "-")
        print(
            f"Epoch {ep:03d} | Loss: {out['loss']:10.4f} | "
            f"L_rec: {out['L_rec']:10.4f} | L_acy: {out['L_acy']:10.4f} | "
            f"SHD: {shd_val}"
        )

        if ev.get("shd", 1e9) < best_shd:
            best_shd = ev["shd"]
            os.makedirs("artifacts/checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "artifacts/checkpoints/rcgnn_best.pt")
            os.makedirs("artifacts/adjacency", exist_ok=True)
            np.save("artifacts/adjacency/A_mean.npy", ev["A_mean"])

    print("=" * 80)
    print(f"✅ Training complete! Best SHD: {best_shd}")
    print("=" * 80)

    # Generate visualizations
    if not args.no_visualize:
        print("\n📊 Generating visualizations...")
        os.makedirs(args.export_dir, exist_ok=True)

        # Call the standalone validation script
        script_dir = Path(__file__).parent
        validate_script = script_dir / "validate_and_visualize.py"

        if validate_script.exists():
            cmd = [
                sys.executable,
                str(validate_script),
                "--adjacency", "artifacts/adjacency/A_mean.npy",
                "--export", args.export_dir,
                "--data-root", root,
                "--threshold", "0.5"
            ]
            try:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                print(result.stdout)
                if result.stderr:
                    print("⚠️  Warnings during visualization:")
                    print(result.stderr)
            except subprocess.CalledProcessError as e:
                print(f"⚠️  Visualization generation failed:")
                print(e.stdout)
                print(e.stderr)
        else:
            print(f"⚠️  Visualization script not found at {validate_script}")
            print("   Skipping automatic visualization.")

    print("\n✅ All done!")
    print(f"   Model checkpoint: artifacts/checkpoints/rcgnn_best.pt")
    print(f"   Adjacency matrix: artifacts/adjacency/A_mean.npy")
    print(f"   Visualizations: {args.export_dir}/")


if __name__ == "__main__":
    main()
