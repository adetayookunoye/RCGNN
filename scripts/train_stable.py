#!/usr/bin/env python3
"""
Stable RC-GNN training with gradient/loss fixes and comprehensive logging.

Key improvements:
1. Loss consistency: all components use mean reduction
2. Gradient taming: aggressive clipping, LR scheduling, weight decay
3. Input standardization: zero-mean/unit-variance normalization
4. Loss rebalancing: sweep lambda values, log edge count
5. Validation protocol: optimal threshold selection, edge count tracking
6. Health metrics: per-epoch logging of gradients, learning rate, density

Expected: Stable gradients, no F1=0 collapse, proper SHD convergence

Run: python3 scripts/train_stable.py
"""

import os
import sys
import torch
import numpy as np
import yaml
import json
from pathlib import Path
from torch.utils.data import DataLoader
from datetime import datetime
import time

# Get project root
project_root = Path(__file__).parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

# Import RC-GNN components
from src.dataio.loaders import load_synth
from src.models.rcgnn import RCGNN
from src.training.optim import make_optimizer, compute_total_loss
from src.training.loop import eval_epoch

print("\n" + "=" * 100)
print("🚀 STABLE RC-GNN TRAINING WITH GRADIENT FIXES")
print("=" * 100)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 100)

# ============================================================================
# 1. LOAD CONFIGURATION
# ============================================================================
print("\n📋 Loading configuration files...")
with open("configs/data_uci.yaml") as f:
    dc = yaml.safe_load(f)
with open("configs/model.yaml") as f:
    mc = yaml.safe_load(f)

# STABLE CONFIG: tuned for gradient stability
tc = {
    # Training
    "epochs": 100,
    "batch_size": 8,
    
    # Optimizer (TAMED)
    "learning_rate": 0.0005,  # REDUCED: 0.001 → 0.0005 for stability
    "weight_decay": 1e-4,      # INCREASED: 1e-5 → 1e-4 (more regularization)
    
    # Gradient control (AGGRESSIVE)
    "grad_clip_norm": 1.0,     # REDUCED: 10 → 1.0 (tighter clipping)
    "grad_clip_ratio_log": True,  # Log how often we clip
    
    # LR Scheduling (NEW)
    "use_scheduler": True,
    "scheduler_type": "plateau",  # ReduceLROnPlateau
    "scheduler_factor": 0.5,
    "scheduler_patience": 3,
    "scheduler_cooldown": 1,
    
    # Device
    "device": "cpu",
    "seed": 1337,
    
    # Loss weights (REBALANCED - see sweep guidance below)
    # ⚠️ IF EDGES → 0: reduce lambda_sparse/acyclic, raise lambda_recon
    # ⚠️ IF F1 STAYS 0: check edge threshold (may need tuning)
    "lambda_recon": 10.0,       # Reconstruction (PRIMARY - raised to encourage structure learning)
    "lambda_sparse": 0.0001,    # Sparsity (was 0.001, REDUCED to prevent over-sparsification)
    "lambda_acyclic": 0.00001,  # Acyclicity (was 0.0001, REDUCED to allow edge learning)
    "lambda_disen": 0.0001,     # Disentanglement (REDUCED for stability)
    "target_sparsity": 0.1,
    
    # Early stopping & validation
    "patience": 15,  # INCREASED: 10 → 15 (allow more exploration)
    "eval_frequency": 2,  # Evaluate every 2 epochs (more frequent)
    
    # Threshold tuning (NEW)
    "auto_tune_threshold": True,  # Find best threshold on val set each epoch
    "threshold_grid": np.linspace(0, 1, 21),  # 21 thresholds to try
    
    "verbose": True
}

print(f"✅ Configuration loaded")
print(f"   - Training epochs: {tc['epochs']}")
print(f"   - Learning rate: {tc['learning_rate']} (with ReduceLROnPlateau)")
print(f"   - Gradient clip: {tc['grad_clip_norm']}")
print(f"   - Loss weights (recon/sparse/acyclic/disen): "
      f"{tc['lambda_recon']}/{tc['lambda_sparse']}/{tc['lambda_acyclic']}/{tc['lambda_disen']}")
print(f"   - Auto threshold tuning: {tc['auto_tune_threshold']}")
print(f"   ⚠️  Note: High recon weight to encourage edge learning; minimal sparsity/acyclic penalties")

# ============================================================================
# 2. LOAD & STANDARDIZE DATA
# ============================================================================
print(f"\n📊 Loading UCI Air Quality dataset...")
dataset_dir = dc.get("dataset", "uci_air")
root = os.path.join(dc["paths"]["root"], "interim", dataset_dir)

train_ds = load_synth(root, "train", seed=tc["seed"])
val_ds = load_synth(root, "val", seed=tc["seed"] + 1)

# ⚠️ INPUT STANDARDIZATION (crucial for gradient stability)
X_train = train_ds.X  # shape: [N, T, d]
X_mean = X_train.mean(axis=(0, 1), keepdims=True)
X_std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8

print(f"   - Raw X range: [{X_train.min():.3f}, {X_train.max():.3f}]")
print(f"   - Standardizing X (zero-mean, unit-var per feature)...")

# In-place standardization
train_ds.X = (train_ds.X - X_mean) / X_std
val_ds.X = (val_ds.X - X_mean) / X_std

print(f"   - Standardized X range: [{train_ds.X.min():.3f}, {train_ds.X.max():.3f}]")

train_ld = DataLoader(train_ds, batch_size=tc["batch_size"], shuffle=True)
val_ld = DataLoader(val_ds, batch_size=1, shuffle=False)

d = train_ds.X.shape[-1]
n_train = len(train_ds)
n_val = len(val_ds)

print(f"✅ Dataset loaded & standardized:")
print(f"   - Features: {d}")
print(f"   - Train samples: {n_train}")
print(f"   - Val samples: {n_val}")
print(f"   - Batch size: {tc['batch_size']}")

# ============================================================================
# 3. INITIALIZE MODEL
# ============================================================================
print(f"\n🏗️  Initializing RC-GNN model...")
device = tc["device"]

# Set seed for reproducibility
torch.manual_seed(tc["seed"])
np.random.seed(tc["seed"])

model = RCGNN(
    d=d,
    latent_dim=mc.get("latent_dim", 16),
    hidden_dim=mc.get("hidden_dim", 32),
    n_envs=mc.get("n_envs", 1),
    sparsify_method=mc.get("sparsify_method", "topk"),
    device=device
)
print(f"✅ Model initialized on {device}")

# ============================================================================
# 4. SETUP OPTIMIZER & SCHEDULER
# ============================================================================
print(f"\n⚙️  Setting up optimizer...")

opt = make_optimizer(
    model,
    lr=tc["learning_rate"],
    weight_decay=tc["weight_decay"]
)
print(f"✅ Optimizer: Adam")
print(f"   - LR: {tc['learning_rate']}")
print(f"   - Weight decay: {tc['weight_decay']}")

# Learning rate scheduler (NEW)
if tc["use_scheduler"]:
    if tc["scheduler_type"] == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode='min',
            factor=tc["scheduler_factor"],
            patience=tc["scheduler_patience"],
            cooldown=tc["scheduler_cooldown"],
            verbose=True
        )
        print(f"✅ Scheduler: ReduceLROnPlateau")
        print(f"   - Factor: {tc['scheduler_factor']}")
        print(f"   - Patience: {tc['scheduler_patience']}")
    else:
        scheduler = None
else:
    scheduler = None

# ============================================================================
# 5. LOAD GROUND TRUTH
# ============================================================================
print(f"\n📋 Loading ground truth adjacency...")
A_true_tensor = None
A_true = None
n_true_edges = 0

try:
    A_true = np.load(os.path.join(root, "A_true.npy"))
    A_true_tensor = torch.FloatTensor(A_true)
    n_true_edges = (A_true > 0).sum()
    print(f"✅ Ground truth loaded: {n_true_edges} edges in DAG")
except FileNotFoundError:
    print("⚠️  A_true.npy not found - will compute SHD but can't track vs ground truth")

# ============================================================================
# 6. TRAINING LOOP WITH HEALTH LOGGING
# ============================================================================
print("\n" + "=" * 100)
print("[TRAINING] Starting stable training loop with health monitoring")
print("=" * 100 + "\n")

# Start timer
start_time = time.time()

# History tracking
best_shd = 1e9
best_adjacency = None
best_threshold = 0.5
patience_counter = 0
clip_ratio_history = []
lr_history = []
edge_count_history = []

# Comprehensive health metrics logged per epoch
training_log = []

for ep in range(tc["epochs"]):
    epoch_start = time.time()
    
    # Training epoch
    model.train()
    total_loss = 0.0
    total_loss_components = {"recon": 0.0, "sparse": 0.0, "acyclic": 0.0, "disen": 0.0}
    n_batches = 0
    n_clipped = 0  # Counter for gradient clipping events
    
    for batch_idx, batch in enumerate(train_ld):
        X = batch["X"].to(device).float()  # Ensure float
        M = batch.get("M", None)
        if M is not None:
            M = M.to(device).float()
        e = batch.get("e", None)
        if e is not None:
            e = e.to(device)
        
        # Forward pass
        opt.zero_grad()
        output = model(X, M, e)
        
        # Compute loss (all components already use .mean())
        loss, comps = compute_total_loss(
            output, X, M,
            lambda_recon=tc["lambda_recon"],
            lambda_sparse=tc["lambda_sparse"],
            lambda_acyclic=tc["lambda_acyclic"],
            lambda_disen=tc["lambda_disen"],
            target_sparsity=tc["target_sparsity"]
        )
        
        # Accumulate component losses (as scalars)
        for key in total_loss_components:
            if key in comps:
                val = comps[key]
                total_loss_components[key] += (val.item() if isinstance(val, torch.Tensor) else val)
        
        total_loss += loss.item()
        
        # Backward
        loss.backward()
        
        # Gradient clipping with tracking
        grad_norm_before = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 
            max_norm=tc["grad_clip_norm"]
        )
        
        # Check if clipped (clip happens if returned norm > max_norm, roughly)
        if grad_norm_before > tc["grad_clip_norm"]:
            n_clipped += 1
        
        # Optimizer step
        opt.step()
        
        n_batches += 1
        
        # Sparse logging (every 1/3 of batches)
        if batch_idx % max(1, len(train_ld) // 3) == 0 and batch_idx > 0:
            print(f"  [Batch {batch_idx:4d}/{len(train_ld)}] "
                  f"Loss: {loss.item():8.4f} | "
                  f"Grad Norm: {grad_norm_before:8.4f} | "
                  f"Clipped: {'✓' if n_clipped > 0 else '·'}")
    
    # Epoch-level statistics
    avg_train_loss = total_loss / n_batches
    for key in total_loss_components:
        total_loss_components[key] /= n_batches
    
    clip_ratio = n_clipped / n_batches
    clip_ratio_history.append(clip_ratio)
    
    current_lr = opt.param_groups[0]["lr"]
    lr_history.append(current_lr)
    
    # ========================================================================
    # VALIDATION & THRESHOLD TUNING
    # ========================================================================
    val_metrics = {"shd": 1e9, "f1": 0.0, "A_mean": None, "edge_count": 0}
    
    if (ep + 1) % tc.get("eval_frequency", 2) == 0 or ep == tc["epochs"] - 1:
        model.eval()
        with torch.no_grad():
            # Try multiple thresholds, pick best F1 on validation
            if tc["auto_tune_threshold"]:
                best_val_f1 = 0.0
                best_val_threshold = 0.5
                
                for threshold in tc["threshold_grid"]:
                    metrics = eval_epoch(
                        model, val_ld, A_true_tensor, 
                        device=device, 
                        threshold=threshold
                    )
                    if metrics["f1"] > best_val_f1:
                        best_val_f1 = metrics["f1"]
                        best_val_threshold = threshold
                        val_metrics = metrics.copy()
                
                best_threshold = best_val_threshold
                print(f"   🎯 Threshold tuned: {best_threshold:.3f} → F1={best_val_f1:.4f}")
            else:
                val_metrics = eval_epoch(
                    model, val_ld, A_true_tensor, 
                    device=device, 
                    threshold=0.5
                )
    
    shd_val = val_metrics.get("shd", 1e9)
    f1_val = val_metrics.get("f1", 0.0)
    edge_count = val_metrics.get("edge_count", 0)
    
    edge_count_history.append(edge_count)
    
    # ========================================================================
    # LEARNING RATE SCHEDULER
    # ========================================================================
    if scheduler is not None and (ep + 1) % tc["eval_frequency"] == 0:
        scheduler.step(shd_val)
    
    # ========================================================================
    # BEST MODEL TRACKING
    # ========================================================================
    if A_true is not None and shd_val < best_shd:
        best_shd = shd_val
        best_adjacency = val_metrics.get("A_mean", None)
        if best_adjacency is not None:
            best_adjacency = best_adjacency.copy()
        patience_counter = 0
        
        # Save checkpoint
        os.makedirs("artifacts/checkpoints", exist_ok=True)
        torch.save(model.state_dict(), "artifacts/checkpoints/rcgnn_best.pt")
        os.makedirs("artifacts/adjacency", exist_ok=True)
        if best_adjacency is not None:
            np.save("artifacts/adjacency/A_mean.npy", best_adjacency)
        
        status = (f"\nEpoch {ep+1:3d}/{tc['epochs']} | "
                 f"Loss: {avg_train_loss:8.4f} | "
                 f"Val F1: {f1_val:6.4f} | Val SHD: {shd_val:6.1f} | "
                 f"Edges: {edge_count:3.0f}/{n_true_edges if n_true_edges else '?'} "
                 f"| Grad Clip: {clip_ratio:.1%} | LR: {current_lr:.2e} | "
                 f"⭐ NEW BEST")
        print(status)
    else:
        patience_counter += 1
        status = (f"Epoch {ep+1:3d}/{tc['epochs']} | "
                 f"Loss: {avg_train_loss:8.4f} | "
                 f"Val F1: {f1_val:6.4f} | Val SHD: {shd_val:6.1f} | "
                 f"Edges: {edge_count:3.0f}/{n_true_edges if n_true_edges else '?'} | "
                 f"Grad Clip: {clip_ratio:.1%} | LR: {current_lr:.2e} | "
                 f"Patience: {patience_counter}/{tc.get('patience', 15)}")
        print(status)
    
    # Log health metrics
    epoch_log = {
        "epoch": ep + 1,
        "train_loss": float(avg_train_loss),
        "train_loss_components": {k: float(v) for k, v in total_loss_components.items()},
        "val_f1": float(f1_val),
        "val_shd": float(shd_val),
        "edge_count": int(edge_count),
        "grad_clip_ratio": float(clip_ratio),
        "learning_rate": float(current_lr),
        "best_threshold": float(best_threshold),
        "epoch_time": time.time() - epoch_start
    }
    training_log.append(epoch_log)
    
    # ========================================================================
    # EARLY STOPPING
    # ========================================================================
    if patience_counter >= tc.get("patience", 15):
        print(f"\n⏹️  Early stopping triggered after {ep+1} epochs "
              f"(no improvement for {tc.get('patience', 15)} evaluations)")
        break

# ============================================================================
# 7. SAVE RESULTS & HEALTH METRICS
# ============================================================================
total_time = time.time() - start_time
print("\n" + "=" * 100)
print("✅ TRAINING COMPLETE")
print("=" * 100)
print(f"Total training time: {total_time/60:.2f} minutes ({total_time:.1f} seconds)")
print(f"Total epochs: {ep+1}/{tc['epochs']}")
print(f"Best validation SHD: {best_shd:.1f}")
print(f"Best threshold: {best_threshold:.3f}")
print(f"Final edge count: {edge_count:.0f}")
print(f"Avg epoch time: {np.mean([log['epoch_time'] for log in training_log]):.2f} seconds")

# Save comprehensive training log (NEW)
os.makedirs("artifacts", exist_ok=True)
log_file = "artifacts/training_log_stable.json"
with open(log_file, "w") as f:
    json.dump({
        "config": {k: (v.tolist() if isinstance(v, np.ndarray) else v) 
                  for k, v in tc.items()},
        "epochs": training_log,
        "total_time": float(total_time),
        "best_shd": float(best_shd),
        "best_threshold": float(best_threshold),
        "final_epoch": int(ep + 1),
        "data_stats": {
            "X_mean": float(X_mean.mean().item()),
            "X_std": float(X_std.mean().item()),
            "n_train_samples": int(n_train),
            "n_val_samples": int(n_val),
            "n_features": int(d),
            "n_true_edges": int(n_true_edges)
        }
    }, f, indent=2)
print(f"\n📊 Comprehensive training log saved to: {log_file}")

# Save summary
summary_file = "artifacts/training_summary_stable.json"
with open(summary_file, "w") as f:
    json.dump({
        "total_time": float(total_time),
        "epochs": int(ep + 1),
        "best_shd": float(best_shd),
        "best_f1": float(max([log["val_f1"] for log in training_log]) if training_log else 0.0),
        "best_threshold": float(best_threshold),
        "final_edge_count": int(edge_count),
        "true_edge_count": int(n_true_edges),
        "avg_grad_clip_ratio": float(np.mean(clip_ratio_history)),
        "final_learning_rate": float(current_lr)
    }, f, indent=2)
print(f"✅ Summary saved to: {summary_file}")

# Verify artifacts
print(f"\n📁 Generated artifacts:")
for path in ["artifacts/checkpoints/rcgnn_best.pt", "artifacts/adjacency/A_mean.npy"]:
    if os.path.exists(path):
        size = os.path.getsize(path) / (1024*1024)
        print(f"   ✅ {path} ({size:.1f} MB)")
    else:
        print(f"   ❌ {path} not found")

print(f"\n📝 Health metrics available in: {log_file}")
print(f"   - Per-epoch: loss components, val F1/SHD, edge count, grad clip ratio, LR")
print(f"   - Summary: best metrics, gradient behavior, convergence info")
print(f"\n✅ Stable training pipeline complete!")
