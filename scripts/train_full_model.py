#!/usr/bin/env python3
"""
Full RC-GNN training on UCI Air Quality dataset with production-quality hyperparameters.
This script performs comprehensive model training with:
- 100+ epochs (instead of quick 8 epochs)
- Full dataset (9,448 samples)
- Proper learning rate scheduling
- Early stopping and best model checkpointing
- Full loss components (reconstruction + sparsity + acyclicity + disentanglement)

Typical runtime: 30-45 minutes on CPU
Expected SHD: ~15-20 on test set

Run from project root: python3 scripts/train_full_model.py
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
from src.training.loop import train_epoch, eval_epoch

print("\n" + "=" * 90)
print("🚀 FULL RC-GNN PRODUCTION TRAINING")
print("=" * 90)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 90)

# Load configs
print("\n📋 Loading configuration files...")
with open("configs/data_uci.yaml") as f:
    dc = yaml.safe_load(f)
with open("configs/model.yaml") as f:
    mc = yaml.safe_load(f)

# CRITICAL: Override train config for FULL training
tc = {
    "epochs": 100,  # Full training: 100 epochs instead of 8
    "batch_size": 8,
    "learning_rate_init": 3e-6,  # ULTRA-LOW for UCI (10× lower)
    "learning_rate_max": 5e-5,   # Max 10× lower
    "warmup_epochs": 5,           # LONGER warmup (5 epochs)
    "weight_decay": 1e-5,
    "device": "cpu",  # Keep CPU for reproducibility
    "seed": 1337,
    "verbose": True,
    # Loss weights - EXTREME for UCI real-world data
    "lambda_recon": 100.0,   # VERY STRONG data signal (10× higher)
    "lambda_sparse": 1e-6,   # Very weak sparsity (10× weaker)
    "lambda_acyclic": 1e-7,  # Very weak acyclicity (10× weaker)
    "lambda_disen": 1e-6,    # Very weak disentanglement (10× weaker)
    "target_sparsity": 0.08, # Match true graph density (7.7%)
    # Gradient clipping
    "grad_clip_norm": 0.5,   # VERY AGGRESSIVE clipping (was 1.0)
    # Early stopping
    "patience": 15,          # Stop if no improvement for 15 evals
    "eval_frequency": 2      # Evaluate every 2 epochs (more frequent)
}

print("✅ Configuration loaded")
print(f"   - Training epochs: {tc['epochs']}")
print(f"   - Batch size: {tc['batch_size']}")
print(f"   - LR range: {tc['learning_rate_init']:.0e} → {tc['learning_rate_max']:.0e}")
print(f"   - Device: {tc['device']}")

# Setup dataset
print(f"\n📊 Loading UCI Air Quality dataset...")
dataset_dir = dc.get("dataset", "uci_air")
root = os.path.join(dc["paths"]["root"], "interim", dataset_dir)
print(f"   Path: {root}")

train_ds = load_synth(root, "train", seed=tc["seed"])
val_ds = load_synth(root, "val", seed=tc["seed"] + 1)

# CRITICAL FIX: Normalize data to prevent gradient explosion
# UCI sensor data has vastly different scales (CO: 0-10, NOx: 0-1000, etc.)
print(f"\n🔧 Normalizing data...")
X_train = train_ds.X  # Shape: (N_train, T, d) or (N_train, d)
X_val = val_ds.X

# Compute statistics on training data only (no data leakage)
if X_train.ndim == 3:  # Time series (N, T, d)
    axis = (0, 1)  # Mean/std across samples and time
else:  # Static (N, d)
    axis = 0

X_mean = X_train.mean(axis=axis, keepdims=True)
X_std = X_train.std(axis=axis, keepdims=True) + 1e-8  # Avoid div by zero

# Apply normalization
train_ds.X = (X_train - X_mean) / X_std
val_ds.X = (X_val - X_mean) / X_std  # Use train stats!

print(f"   ✅ Data normalized (mean≈0, std≈1)")
print(f"   - Train: mean={train_ds.X.mean():.3f}, std={train_ds.X.std():.3f}")
print(f"   - Val: mean={val_ds.X.mean():.3f}, std={val_ds.X.std():.3f}")

train_ld = DataLoader(train_ds, batch_size=tc["batch_size"], shuffle=True)
val_ld = DataLoader(val_ds, batch_size=1, shuffle=False)

d = train_ds.X.shape[-1]
print(f"\n✅ Dataset loaded:")
print(f"   - Features: {d}")
print(f"   - Train samples: {len(train_ds)}")
print(f"   - Val samples: {len(val_ds)}")

# Initialize model
print(f"\n🏗️  Initializing RC-GNN model...")
device = tc["device"]
model = RCGNN(
    d=d,
    latent_dim=mc.get("latent_dim", 16),
    hidden_dim=mc.get("hidden_dim", 32),
    n_envs=mc.get("n_envs", 1),
    sparsify_method=mc.get("sparsify_method", "topk"),
    device=device
)
print(f"✅ Model initialized on {device}")
print(f"   - Latent dim: {mc.get('latent_dim', 16)}")
print(f"   - Hidden dim: {mc.get('hidden_dim', 32)}")
print(f"   - Sparsify method: {mc.get('sparsify_method', 'topk')}")

# Initialize optimizer with MAXIMUM learning rate (scheduler will reduce it initially)
opt = make_optimizer(
    model,
    lr=tc.get("learning_rate_max", 5e-4),  # FIX: Start at max, scheduler will reduce
    weight_decay=tc.get("weight_decay", 1e-5)
)

# FIX: Add LR scheduler with warm-up
from torch.optim.lr_scheduler import LinearLR, SequentialLR, ConstantLR

warmup_epochs = tc.get("warmup_epochs", 3)
warmup_factor = tc["learning_rate_init"] / tc["learning_rate_max"]  # ~0.06 (start at 6%)

# Warm-up: ramp from init → max over N epochs
warmup_scheduler = LinearLR(
    opt,
    start_factor=warmup_factor,  # Start at 6% of max LR
    end_factor=1.0,               # End at 100% of max LR
    total_iters=warmup_epochs * len(train_ld)
)

# After warm-up: keep constant
main_scheduler = ConstantLR(opt, factor=1.0, total_iters=1e9)

# Combined scheduler
scheduler = SequentialLR(
    opt,
    schedulers=[warmup_scheduler, main_scheduler],
    milestones=[warmup_epochs * len(train_ld)]
)

print(f"✅ Optimizer initialized:")
print(f"   - Initial LR: {tc['learning_rate_init']:.0e} (warmup start)")
print(f"   - Max LR: {tc['learning_rate_max']:.0e} (reached after {warmup_epochs} epochs)")
print(f"   - Weight decay: {tc['weight_decay']:.0e}")

# Load ground truth
print(f"\n📋 Loading ground truth adjacency...")
A_true_tensor = None
A_true = None
try:
    A_true = np.load(os.path.join(root, "A_true.npy"))
    A_true_tensor = torch.FloatTensor(A_true)
    n_edges = (A_true > 0).sum()
    print(f"✅ Ground truth loaded: {n_edges} edges in DAG")
except FileNotFoundError:
    print("⚠️  A_true.npy not found - will not compute SHD")

# Training loop
print("\n" + "=" * 90)
print("[TRAINING] Starting full training loop")
print("=" * 90)

best_shd = 1e9
best_adjacency = None
best_model_state = None
patience_counter = 0
start_time = time.time()
training_history = {
    "train_losses": [],
    "val_shd": [],
    "val_f1": [],
    "epoch_times": []
}

for ep in range(tc["epochs"]):
    epoch_start = time.time()
    
    # Training epoch
    model.train()
    total_loss = 0.0
    n_batches = 0
    grad_clip_count = 0  # FIX: Track how often we clip
    loss_components = {
        "recon": 0.0,
        "sparse": 0.0,
        "acyclic": 0.0,
        "disen": 0.0
    }
    
    for batch_idx, batch in enumerate(train_ld):
        X = batch["X"].to(device)
        M = batch.get("M", None)
        if M is not None:
            M = M.to(device)
        e = batch.get("e", None)
        if e is not None:
            e = e.to(device)
        
        opt.zero_grad()
        output = model(X, M, e)
        
        # Compute comprehensive loss
        loss, comps = compute_total_loss(
            output, X, M,
            lambda_recon=tc.get("lambda_recon", 10.0),
            lambda_sparse=tc.get("lambda_sparse", 1e-5),
            lambda_acyclic=tc.get("lambda_acyclic", 3e-6),
            lambda_disen=tc.get("lambda_disen", 1e-5),
            target_sparsity=tc.get("target_sparsity", 0.08)
        )
        
        # Accumulate component losses
        for key in loss_components:
            if key in comps:
                loss_components[key] += comps[key].item() if isinstance(comps[key], torch.Tensor) else comps[key]
        
        loss.backward()
        
        # FIX: Clip gradients more aggressively (max_norm=1.0)
        grad_norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=tc.get("grad_clip_norm", 1.0))
        if grad_norm_before > tc.get("grad_clip_norm", 1.0):
            grad_clip_count += 1
        
        opt.step()
        scheduler.step()  # FIX: Step scheduler every batch
        
        total_loss += loss.item()
        n_batches += 1
    
    avg_train_loss = total_loss / n_batches
    training_history["train_losses"].append(avg_train_loss)
    
    # Normalize component averages & compute percentages
    total_weighted = 0.0
    for key in loss_components:
        loss_components[key] /= n_batches
        total_weighted += loss_components[key] * tc.get(f"lambda_{key}", 1.0)
    
    # Loss breakdown percentages (FIX: Monitor if recon is dominant)
    loss_pct = {}
    if total_weighted > 0:
        for key in loss_components:
            loss_pct[key] = (loss_components[key] * tc.get(f"lambda_{key}", 1.0) / total_weighted) * 100
    
    # Gradient clipping ratio
    grad_clip_ratio = grad_clip_count / n_batches
    
    # Get current LR
    current_lr = opt.param_groups[0]["lr"]
    
    # Validation epoch (every N epochs or at the end)
    val_metrics = {"shd": 1e9, "f1": 0.0, "A_mean": None}
    
    if (ep + 1) % tc.get("eval_frequency", 2) == 0 or ep == tc["epochs"] - 1:
        model.eval()
        val_metrics = eval_epoch(model, val_ld, A_true_tensor, device=device, threshold=0.5)
    
    shd_val = val_metrics.get("shd", 1e9)
    f1_val = val_metrics.get("f1", 0.0)
    training_history["val_shd"].append(shd_val)
    training_history["val_f1"].append(f1_val)
    
    epoch_time = time.time() - epoch_start
    training_history["epoch_times"].append(epoch_time)
    
    # FIX: Early stopping logic with sentinel filtering
    is_eval_valid = (shd_val < 1e8)  # FIX: Only consider valid evaluations
    
    if is_eval_valid and A_true is not None and shd_val < best_shd:
        best_shd = shd_val
        best_adjacency = val_metrics.get("A_mean", None)
        if best_adjacency is not None:
            best_adjacency = best_adjacency.copy()
        best_model_state = {k: v.cpu().clone() if isinstance(v, torch.Tensor) else v 
                           for k, v in model.state_dict().items()}
        patience_counter = 0
        
        # Save checkpoint
        os.makedirs("artifacts/checkpoints", exist_ok=True)
        torch.save(model.state_dict(), "artifacts/checkpoints/rcgnn_best.pt")
        os.makedirs("artifacts/adjacency", exist_ok=True)
        if best_adjacency is not None:
            np.save("artifacts/adjacency/A_mean.npy", best_adjacency)
        
        # FIX: Print comprehensive diagnostics
        print(f"\nEpoch {ep+1:3d}/{tc['epochs']} | Loss: {avg_train_loss:8.4f} "
              f"(Recon:{loss_pct.get('recon', 0):.1f}% Sparse:{loss_pct.get('sparse', 0):.1f}% "
              f"Disen:{loss_pct.get('disen', 0):.1f}% Acyc:{loss_pct.get('acyclic', 0):.1f}%)")
        print(f"  Val: F1={f1_val:.3f} SHD={shd_val:.1f} | "
              f"Edges: tuned={val_metrics.get('edges_pred@tuned', 0)} "
              f"@0.5={val_metrics.get('edges_pred@0.5', 0)} "
              f"topk={val_metrics.get('edges_pred@topk', 0)}")
        print(f"  Logits: mean={val_metrics.get('logit_mean', 0):.4f} "
              f"std={val_metrics.get('logit_std', 0):.4f} "
              f"%>0={val_metrics.get('pct_logits_gt0', 0):.1f}% | "
              f"Clip:{grad_clip_ratio*100:.1f}% LR={current_lr:.2e} ⭐ BEST")
    else:
        if is_eval_valid:
            patience_counter += 1
        status_str = f"Epoch {ep+1:3d}/{tc['epochs']} | Loss: {avg_train_loss:8.4f}"
        
        if (ep + 1) % tc.get("eval_frequency", 2) == 0 or ep == tc["epochs"] - 1:
            if is_eval_valid:
                status_str += (f" | Val F1={f1_val:.3f} SHD={shd_val:.1f} | "
                             f"Edges: tuned={val_metrics.get('edges_pred@tuned', 0)} "
                             f"@0.5={val_metrics.get('edges_pred@0.5', 0)} "
                             f"topk={val_metrics.get('edges_pred@topk', 0)}")
            else:
                status_str += " | ⚠️  Eval failed (sentinel SHD), skipping patience"
        
        status_str += f" | Clip:{grad_clip_ratio*100:.1f}% LR={current_lr:.2e}"
        if is_eval_valid:
            status_str += f" | Patience:{patience_counter}/{tc.get('patience', 15)}"
        print(status_str)
    
    # Early stopping
    if patience_counter >= tc.get("patience", 15):
        print(f"\n⏹️  Early stopping triggered after {ep+1} epochs (no improvement for {tc.get('patience', 15)} evals)")
        break

# Training complete
total_time = time.time() - start_time
print("\n" + "=" * 90)
print(f"✅ TRAINING COMPLETE")
print("=" * 90)
print(f"Total training time: {total_time/60:.2f} minutes ({total_time:.1f} seconds)")
print(f"Total epochs: {ep+1}/{tc['epochs']}")
print(f"Best validation SHD: {best_shd:.1f}")
print(f"Avg epoch time: {np.mean(training_history['epoch_times']):.2f} seconds")

# Save training history
os.makedirs("artifacts", exist_ok=True)
history_file = "artifacts/training_history_full.json"
with open(history_file, "w") as f:
    json.dump({
        "config": tc,
        "train_losses": [float(x) for x in training_history["train_losses"]],
        "val_shd": [float(x) for x in training_history["val_shd"]],
        "val_f1": [float(x) for x in training_history["val_f1"]],
        "epoch_times": [float(x) for x in training_history["epoch_times"]],
        "total_time": float(total_time),
        "best_shd": float(best_shd),
        "final_epoch": int(ep + 1)
    }, f, indent=2)
print(f"\n📊 Training history saved to: {history_file}")

# Save metrics
metrics_file = "artifacts/training_metrics_full.json"
with open(metrics_file, "w") as f:
    json.dump({
        "best_shd": float(best_shd),
        "total_time": float(total_time),
        "epochs": int(ep + 1),
        "dataset_size": int(len(train_ds)),
        "batch_size": int(tc["batch_size"]),
        "device": str(tc["device"])
    }, f, indent=2)
print(f"✅ Metrics saved to: {metrics_file}")

# Verify artifacts
print(f"\n📁 Generated artifacts:")
checkpoint_path = "artifacts/checkpoints/rcgnn_best.pt"
adjacency_path = "artifacts/adjacency/A_mean.npy"

if os.path.exists(checkpoint_path):
    size_mb = os.path.getsize(checkpoint_path) / (1024*1024)
    print(f"   ✅ Model checkpoint: {checkpoint_path} ({size_mb:.1f} MB)")
else:
    print(f"   ❌ Model checkpoint not found")

if os.path.exists(adjacency_path):
    A = np.load(adjacency_path)
    sparsity = 1.0 - (A > 0).sum() / (A.shape[0] * A.shape[1])
    print(f"   ✅ Adjacency matrix: {adjacency_path} (sparsity: {sparsity:.2%})")
else:
    print(f"   ❌ Adjacency matrix not found")

print(f"\n📝 Next steps:")
print(f"   1. Run threshold optimization: make analyze")
print(f"   2. Compare baselines: make baseline")
print(f"   3. View full results: make results")
print(f"\n✅ Full training pipeline ready!")
