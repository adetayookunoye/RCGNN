#!/usr/bin/env python3
"""
Test script to verify the empty graph collapse fix.
Runs training on h1_easy benchmark with warm-up schedules.
Uses the correct RCGNN interface and working training loop pattern.
"""

import argparse
import yaml
import torch
import numpy as np
import math
import os
from pathlib import Path
from torch.utils.data import DataLoader

# Add project root to path 
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import path_helper # noqa: F401
from src.dataio.loaders import load_synth
from src.models.rcgnn import RCGNN
from src.training.optim import make_optimizer
from src.training.loop import train_epoch, eval_epoch


def main():
    parser = argparse.ArgumentParser(description="Test empty graph collapse fix")
    parser.add_argument("--dataset", default="h1_easy", help="Dataset name")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("\n" + "="*60)
    print("RC-GNN Empty Graph Collapse Fix - Test")
    print("="*60)

    # [1/5] Load configs
    print("\n[1/5] Loading configurations...")
    
    # Use fixed configs by default
    model_cfg_path = "configs/model_fixed.yaml"
    train_cfg_path = "configs/train_fixed_v2.yaml" # Use v2 (safer schedule)
    
    with open(model_cfg_path) as f:
        model_cfg = yaml.safe_load(f)
    with open(train_cfg_path) as f:
        train_cfg = yaml.safe_load(f)

    # [2/5] Load data
    print(f"[2/5] Loading data ({args.dataset})...")
    
    # Build path to benchmark directory
    # Handle both synthetic (synth_corrupted_*) and real-world (uci_air) datasets
    if args.dataset == "uci_air":
        data_root = Path("data") / "interim" / "uci_air"
    else:
        data_root = Path("data") / "interim" / f"synth_corrupted_{args.dataset}"
    
    if not data_root.exists():
        # Fallback to synth_small if benchmark doesn't exist
        data_root = Path("data") / "interim" / "synth_small"
        print(f" Dataset dir not found, using fallback: {data_root}")
    
    train_ds = load_synth(str(data_root), split="train", seed=args.seed)
    val_ds = load_synth(str(data_root), split="val", seed=args.seed + 1)
    
    print(f" [OK] Loaded {len(train_ds)} train, {len(val_ds)} val samples")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    # Get data dimension from first batch
    d = train_ds.X.shape[-1]
    print(f" Data dimension: d={d}")

    # [3/5] Create model with CORRECT interface
    print(f"[3/5] Creating RC-GNN model...")
    model = RCGNN(
        d=d,
        latent_dim=model_cfg.get("latent_dim", 16),
        hidden_dim=model_cfg.get("hidden_dim", 32),
        n_envs=model_cfg.get("n_envs", 1),
        sparsify_method=model_cfg.get("sparsify", {}).get("method", "topk"),
        topk_ratio=model_cfg.get("sparsify", {}).get("topk_ratio", 0.1),
        device=args.device
    )
    print(f" [OK] Model created (d={d}, latent_dim={model_cfg.get('latent_dim', 16)})")

    # Optimizer
    optimizer = make_optimizer(model, lr=args.lr, weight_decay=train_cfg.get("weight_decay", 0.0))

    # Load ground truth adjacency if available
    A_true_path = data_root / "A_true.npy"
    A_true = None
    if A_true_path.exists():
        A_true = np.load(A_true_path)
        mask_no_diag = np.ones_like(A_true, dtype=bool)
        np.fill_diagonal(mask_no_diag, False)
        offdiag_nnz = int((A_true[mask_no_diag] != 0).sum())
        print(f" [OK] Ground truth adjacency loaded: shape={A_true.shape}, edges={offdiag_nnz}")
    else:
        print(f" [WARN] No ground truth adjacency found at {A_true_path}")

    # [4/5] Training loop with warm-up schedules
    print(f"\n[4/5] Training for {args.epochs} epochs...")
    
    # Setup warm-up schedule (cosine ramp)
    def cos_ramp(t, T):
        """Cosine ramp from 0 to 1 over T timesteps."""
        x = max(0.0, min(1.0, t / max(1, T)))
        return 0.5 * (1.0 - math.cos(math.pi * x))

    # Warm-up parameters
    T_sup_on = int(train_cfg.get("sup_warmup_epochs", 10)) # Supervised only epochs 0-10
    T_acy_on = int(train_cfg.get("acy_warmup_epochs", 40)) # Acyclic dormant epochs 0-40
    T_acy_ramp = int(train_cfg.get("acy_ramp_epochs", 20)) # Then ramp over 20 epochs
    T_sparse_on = int(train_cfg.get("sparse_warmup_epochs", 50)) # Sparsity dormant epochs 0-50
    T_sparse_ramp = int(train_cfg.get("sparse_ramp_epochs", 20)) # Then ramp over 20 epochs
    
    base_lambda_sup = float(train_cfg.get("lambda_supervised_max", 0.01))
    base_lambda_acy = float(train_cfg.get("acy_max", 0.05))
    base_lambda_sparse = float(train_cfg.get("sparse_max", 1e-5))

    # Create output directory
    output_dir = Path("artifacts")
    output_dir.mkdir(exist_ok=True, parents=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True, parents=True)
    (output_dir / "adjacency").mkdir(exist_ok=True, parents=True)

    best_shd = 1e9
    epoch_metrics = []

    for epoch in range(args.epochs):
        # Dynamic weights with warm-up schedules
        
        # (1) Supervised loss: Active only epochs 0-10, then OFF
        if epoch <= T_sup_on:
            lam_sup = base_lambda_sup * cos_ramp(epoch, T_sup_on)
        else:
            lam_sup = 0.0 # TURN OFF after warm-up
        
        # (2) Acyclic loss: Dormant 0-40, then ramp 40-60
        if epoch < T_acy_on:
            lam_acy = 0.0
        else:
            # Ramp from 0 to base_lambda_acy over T_acy_ramp epochs
            progress = min(1.0, (epoch - T_acy_on) / max(1, T_acy_ramp))
            lam_acy = base_lambda_acy * 0.5 * (1.0 - math.cos(math.pi * progress))
        
        # (3) Sparsity loss: Dormant 0-50, then ramp 50-70
        if epoch < T_sparse_on:
            lam_sparse = 0.0
        else:
            # Ramp from 0 to base_lambda_sparse over T_sparse_ramp epochs
            progress = min(1.0, (epoch - T_sparse_on) / max(1, T_sparse_ramp))
            lam_sparse = base_lambda_sparse * 0.5 * (1.0 - math.cos(math.pi * progress))

        # Loss kwargs for training
        loss_kwargs = {
            "lambda_recon": train_cfg.get("lambda_recon", 1.0),
            "lambda_sparse": lam_sparse,
            "lambda_acyclic": lam_acy,
            "lambda_disen": train_cfg.get("lambda_disen", 0.01),
            "target_sparsity": train_cfg.get("target_sparsity", 0.1),
            "lambda_supervised": lam_sup,
            "A_true": A_true,
            "lambda_inv": train_cfg.get("lambda_inv", 0.0),
            "invariance_loss_fn": None,
        }

        # Train epoch
        out = train_epoch(model, train_loader, optimizer, device=args.device, **loss_kwargs)

        # Eval epoch
        ev = eval_epoch(model, val_loader, A_true=A_true, threshold=0.5, device=args.device)

        # Format SHD for printing
        shd_str = f"{ev.get('shd', float('inf')):.1f}" if ev.get('shd') is not None else 'N/A'

        # Print summary
        if epoch % max(1, args.epochs // 20) == 0 or epoch == args.epochs - 1:
            print(f"Epoch {epoch:3d}/{args.epochs} | loss {out['loss']:7.4f} | "
                  f"recon {out.get('recon', 0):6.4f} | acy {out.get('acyclic', 0):6.4f} | "
                  f"SHD {shd_str:>5s} | λ_sup {lam_sup:.4f} | λ_acy {lam_acy:.5f} | λ_sparse {lam_sparse:.2e}")

        epoch_metrics.append({
            "epoch": epoch,
            "loss": out["loss"],
            "recon": out.get("recon", 0),
            "acyclic": out.get("acyclic", 0),
            "shd": ev.get("shd", 1e9),
            "lambda_sup": lam_sup,
            "lambda_acy": lam_acy,
            "lambda_sparse": lam_sparse,
        })

        # Save best model ONLY when validation SHD improves
        if ev.get("shd", 1e9) < best_shd:
            best_shd = ev["shd"]
            torch.save(model.state_dict(), output_dir / "checkpoints" / "rcgnn_best.pt")
            if "A_mean" in ev:
                np.save(output_dir / "adjacency" / "A_mean.npy", ev["A_mean"])
            if epoch % max(1, args.epochs // 20) == 0 or epoch == args.epochs - 1:
                print(f" [OK] Checkpoint saved (SHD={best_shd:.1f})")

    # [5/5] Final results
    print(f"\n[5/5] Training complete!")
    print(f" Best SHD: {best_shd:.1f}")
    print(f" Results saved to: {output_dir}")
    print("\n" + "="*60)
    print("Fix verification:")
    if best_shd < 10:
        print(f" [DONE] SUCCESS! SHD={best_shd:.1f} (improved from 30.0)")
    elif best_shd < 20:
        print(f" [WARN] PARTIAL - SHD={best_shd:.1f} (some improvement)")
    else:
        print(f" [FAIL] FAILED - SHD={best_shd:.1f} (no improvement)")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
