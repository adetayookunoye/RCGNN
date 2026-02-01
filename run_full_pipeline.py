#!/usr/bin/env python3
"""
Complete training + analysis pipeline for RC-GNN on UCI Air Quality dataset.
Run this script from the project root to execute full reproducibility pipeline.
"""

import os
import sys
import torch
import numpy as np
import yaml
from pathlib import Path
from torch.utils.data import DataLoader
import subprocess
import json
import time

# Get project root
project_root = Path(__file__).parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

# Import RC-GNN components
from src.dataio.loaders import load_synth
from src.models.rcgnn import RCGNN
from src.training.optim import make_optimizer, compute_total_loss
from src.training.loop import train_epoch, eval_epoch

print("\n" + "=" * 80)
print(" FULL RC-GNN TRAINING + ANALYSIS PIPELINE")
print("=" * 80)

# Load configs
with open("configs/data_uci.yaml") as f:
    dc = yaml.safe_load(f)
with open("configs/model.yaml") as f:
    mc = yaml.safe_load(f)
with open("configs/train.yaml") as f:
    tc = yaml.safe_load(f)

# Setup dataset
dataset_dir = dc.get("dataset", "uci_air")
root = os.path.join(dc["paths"]["root"], "interim", dataset_dir)

print(f"\n Loading UCI Air Quality dataset...")
print(f" Path: {root}")

train_ds = load_synth(root, "train", seed=tc["seed"])
val_ds = load_synth(root, "val", seed=tc["seed"] + 1)

train_ld = DataLoader(train_ds, batch_size=tc["batch_size"], shuffle=True)
val_ld = DataLoader(val_ds, batch_size=1, shuffle=False)

d = train_ds.X.shape[-1]
print(f"[DONE] Dataset: {d} features, {len(train_ds)} train, {len(val_ds)} val samples")

# Initialize model
print(f"\n Initializing RC-GNN model...")
device = tc["device"]
model = RCGNN(
    d=d,
    latent_dim=mc.get("latent_dim", 16),
    hidden_dim=mc.get("hidden_dim", 32),
    n_envs=mc.get("n_envs", 1),
    sparsify_method=mc.get("sparsify_method", "topk"),
    device=device
)
print(f"[DONE] Model on {device}")

opt = make_optimizer(
    model,
    lr=tc.get("learning_rate", 0.001),
    weight_decay=tc.get("weight_decay", 1e-5)
)
best_shd = 1e9
best_adjacency = None

# Load ground truth
print(f"\n Loading ground truth...")
A_true_tensor = None
try:
    A_true = np.load(os.path.join(root, "A_true.npy"))
    A_true_tensor = torch.FloatTensor(A_true)
    print(f"[DONE] Ground truth: {(A_true > 0).sum()} edges")
except FileNotFoundError:
    A_true = None
    print("[WARN] A_true.npy not found")

# STEP 1: TRAINING
print("\n" + "=" * 80)
print("[STEP 1/5] TRAINING RC-GNN MODEL")
print("=" * 80)

start_time = time.time()

for ep in range(tc["epochs"]):
    # Training epoch
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for batch in train_ld:
        X = batch["X"].to(device)
        M = batch.get("M", None)
        if M is not None:
            M = M.to(device)
        e = batch.get("e", None)
        if e is not None:
            e = e.to(device)
        
        opt.zero_grad()
        output = model(X, M, e)
        
        # Compute loss
        loss, comps = compute_total_loss(
            output, X, M,
            lambda_recon=tc.get("lambda_recon", 1.0),
            lambda_sparse=tc.get("lambda_sparse", 0.01),
            lambda_acyclic=tc.get("lambda_acyclic", 0.1),
            lambda_disen=tc.get("lambda_disen", 0.01),
            target_sparsity=tc.get("target_sparsity", 0.1)
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    avg_train_loss = total_loss / n_batches
    
    # Validation epoch
    model.eval()
    val_metrics = eval_epoch(model, val_ld, A_true_tensor, device=device, threshold=0.5)
    
    shd_val = val_metrics.get("shd", "-")
    f1_val = val_metrics.get("f1", "-")
    
    print(f"Epoch {ep+1:02d}/{tc['epochs']} | Train Loss: {avg_train_loss:.4f} | F1: {f1_val} | SHD: {shd_val}")
    
    # Check for best model
    if A_true is not None and val_metrics.get("shd", 1e9) < best_shd:
        best_shd = val_metrics["shd"]
        best_adjacency = val_metrics["A_mean"].copy()
        os.makedirs("artifacts/checkpoints", exist_ok=True)
        torch.save(model.state_dict(), "artifacts/checkpoints/rcgnn_best.pt")
        os.makedirs("artifacts/adjacency", exist_ok=True)
        np.save("artifacts/adjacency/A_mean.npy", best_adjacency)
        print(f" * New best SHD: {best_shd:.1f}")

elapsed = time.time() - start_time
print(f"\n[DONE] Training complete! Best SHD: {best_shd if best_shd != 1e9 else 'N/A'}")
print(f" Time: {elapsed:.1f}s")
print(f" Checkpoint: artifacts/checkpoints/rcgnn_best.pt")
print(f" Adjacency: artifacts/adjacency/A_mean.npy")

# Save final metrics
os.makedirs("artifacts", exist_ok=True)
with open("artifacts/training_metrics.json", "w") as f:
    json.dump({
        "best_shd": float(best_shd) if best_shd != 1e9 else None,
        "training_time": elapsed,
        "epochs": tc["epochs"],
        "dataset": "uci_air",
        "n_samples": len(train_ds) + len(val_ds),
    }, f, indent=2)

# STEP 2: THRESHOLD OPTIMIZATION
print("\n" + "=" * 80)
print("[STEP 2/5] OPTIMIZING BINARY THRESHOLD")
print("=" * 80)

try:
    subprocess.run([sys.executable, "scripts/optimize_threshold.py",
                   "--adjacency", "artifacts/adjacency/A_mean.npy",
                   "--data-root", root,
                   "--export", "artifacts"],
                   check=False, timeout=60)
    print("[DONE] Threshold optimization complete!")
except Exception as e:
    print(f"[WARN] Threshold optimization skipped: {e}")

# STEP 3: ENVIRONMENT STRUCTURE ANALYSIS
print("\n" + "=" * 80)
print("[STEP 3/5] ANALYZING PER-ENVIRONMENT STRUCTURES")
print("=" * 80)

try:
    subprocess.run([sys.executable, "scripts/visualize_environment_structure.py",
                   "--checkpoint", "artifacts/checkpoints/rcgnn_best.pt",
                   "--config-data", "configs/data_uci.yaml",
                   "--config-model", "configs/model.yaml",
                   "--export", "artifacts"],
                   check=False, timeout=60)
    print("[DONE] Environment analysis complete!")
except Exception as e:
    print(f"[WARN] Environment analysis skipped: {e}")

# STEP 4: BASELINE COMPARISON
print("\n" + "=" * 80)
print("[STEP 4/5] COMPARING AGAINST BASELINE METHODS")
print("=" * 80)

try:
    subprocess.run([sys.executable, "scripts/compare_baselines.py",
                   "--data-root", root,
                   "--adjacency", "artifacts/adjacency/A_mean.npy",
                   "--export", "artifacts"],
                   check=False, timeout=60)
    print("[DONE] Baseline comparison complete!")
except Exception as e:
    print(f"[WARN] Baseline comparison skipped: {e}")

# STEP 6: RUN ORIGINAL TRAINING SCRIPT (if available)
print("\n" + "=" * 80)
print("[STEP 6/5] RUNNING ORIGINAL TRAINING SCRIPT")
print("=" * 80)

original_script_run = False
try:
    # Try to run train_and_visualize.py with PYTHONPATH
    result = subprocess.run(
        [sys.executable, "-c", f"""
import sys
sys.path.insert(0, '.')
sys.argv = ['scripts/train_and_visualize.py', 'configs/data_uci.yaml', 'configs/model.yaml', 'configs/train.yaml']
exec(open('scripts/train_and_visualize.py').read())
"""],
        check=False, timeout=300, capture_output=True, text=True
    )
    if result.returncode == 0:
        print("[DONE] Original training script completed!")
        original_script_run = True
        # Save original script outputs
        if os.path.exists("artifacts/checkpoints/rcgnn_best.pt"):
            os.rename("artifacts/checkpoints/rcgnn_best.pt", "artifacts/checkpoints/rcgnn_best_original.pt")
            print(" Saved original checkpoint as rcgnn_best_original.pt")
    else:
        print(f"[WARN] Original script error: {result.stderr[:200]}")
except Exception as e:
    print(f"[WARN] Original script skipped: {e}")

# STEP 5: SUMMARY
print("\n" + "=" * 80)
print("[STEP 5/5] GENERATING FINAL SUMMARY")
print("=" * 80)

summary = {
    "title": "RC-GNN Full Training Pipeline - UCI Air Quality",
    "dataset": {
        "name": "UCI Air Quality",
        "n_samples": len(train_ds) + len(val_ds),
        "n_features": d,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
    },
    "model": {
        "type": "RCGNN",
        "latent_dim": mc.get("latent_dim", 16),
        "hidden_dim": mc.get("hidden_dim", 32),
        "sparsify_method": mc.get("sparsify_method", "topk"),
    },
    "training": {
        "epochs": tc["epochs"],
        "batch_size": tc["batch_size"],
        "learning_rate": tc.get("learning_rate", 0.001),
        "device": device,
        "time_seconds": elapsed,
        "original_script_run": original_script_run,
    },
    "results": {
        "best_shd": float(best_shd) if best_shd != 1e9 else None,
        "adjacency_path": "artifacts/adjacency/A_mean.npy",
        "checkpoint_path": "artifacts/checkpoints/rcgnn_best.pt",
        "original_checkpoint_path": "artifacts/checkpoints/rcgnn_best_original.pt" if original_script_run else None,
    },
    "outputs": [
        "artifacts/checkpoints/rcgnn_best.pt",
        "artifacts/adjacency/A_mean.npy",
        "artifacts/training_metrics.json",
    ]
}

# Add optional outputs if they exist
for fname in ["threshold_analysis.json", "environment_analysis.json", "baseline_comparison.json"]:
    if os.path.exists(f"artifacts/{fname}"):
        summary["outputs"].append(f"artifacts/{fname}")

# Add original checkpoint if it exists
if original_script_run and os.path.exists("artifacts/checkpoints/rcgnn_best_original.pt"):
    summary["outputs"].append("artifacts/checkpoints/rcgnn_best_original.pt")

with open("artifacts/pipeline_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n Summary saved to: artifacts/pipeline_summary.json")

print("\n" + "=" * 80)
print("[DONE] FULL PIPELINE COMPLETE!")
print("=" * 80)
print("\nGenerated outputs:")
for path in summary["outputs"]:
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024**2)
        print(f" [OK] {path} ({size_mb:.2f} MB)")
    else:
        print(f" - {path} (not generated)")

print(f"\nTotal time: {elapsed:.1f}s")
print("\nNext steps:")
print("1. Review artifacts/pipeline_summary.json for results")
print("2. Check artifacts/training_metrics.json for detailed metrics")
print("3. View visualizations in artifacts/ directory")
print("4. Compare RC-GNN vs baselines")
