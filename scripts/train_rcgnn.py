import argparse, yaml, os, torch
from pathlib import Path

import path_helper  # noqa: F401  # adds project root to sys.path
from torch.utils.data import DataLoader
from src.dataio.loaders import load_synth
from src.models.rcgnn import RCGNN
from src.training.optim import make_optimizer
from src.training.loop import train_epoch, eval_epoch
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_cfg")
    parser.add_argument("model_cfg")
    parser.add_argument("train_cfg")
    parser.add_argument("--adj-output", default="artifacts/adjacency/A_mean.npy", 
                        help="Path to save learned adjacency matrix")
    args = parser.parse_args()

    with open(args.data_cfg) as f: dc = yaml.safe_load(f)
    with open(args.model_cfg) as f: mc = yaml.safe_load(f)
    with open(args.train_cfg) as f: tc = yaml.safe_load(f)

    root = dc["paths"]["root"]  # Already contains full path like "data/interim/synth_small"
    print(f"📂 Loading data from: {root}")
    
    train_ds = load_synth(root, "train", seed=tc["seed"])
    val_ds   = load_synth(root, "val", seed=tc["seed"]+1)
    
    print(f"✅ Data loaded: {len(train_ds)} train, {len(val_ds)} val samples")

    train_ld = DataLoader(train_ds, batch_size=tc["batch_size"], shuffle=True)
    val_ld   = DataLoader(val_ds, batch_size=1, shuffle=False)

    d = train_ds.X.shape[-1]
    device = tc["device"]
    
    # Extract model parameters from config
    model = RCGNN(
        d=d,
        latent_dim=mc.get("latent_dim", 16),
        hidden_dim=mc.get("hidden_dim", 32),
        n_envs=mc.get("n_envs", 1),
        sparsify_method=mc.get("sparsify_method", "topk"),
        topk_ratio=mc.get("topk_ratio", 0.1),
        device=device
    )
    model.to(device)

    opt = make_optimizer(model, lr=tc["learning_rate"], weight_decay=tc["weight_decay"])
    
    # Load ground truth adjacency if available
    A_true_path = Path(root) / "A_true.npy"
    A_true = None
    if A_true_path.exists():
        A_true = np.load(A_true_path)
        print(f"✅ Ground truth adjacency loaded: {A_true.shape}")
    else:
        print("⚠️  No ground truth adjacency (A_true.npy) - SHD will not be computed")
    
    best_shd = 1e9

    for ep in range(tc["epochs"]):
        loss_kwargs = {
            "lambda_recon": tc.get("lambda_recon", 1.0),
            "lambda_sparse": tc.get("lambda_sparse", 0.01),
            "lambda_acyclic": tc.get("lambda_acyclic", 0.1),
            "lambda_disen": tc.get("lambda_disen", 0.01),
            "target_sparsity": tc.get("target_sparsity", 0.1),
        }
        out = train_epoch(model, train_ld, opt, device=device, **loss_kwargs)
        ev = eval_epoch(model, val_ld, A_true=A_true, threshold=0.5, device=device)
        
        # Print epoch summary
        shd_str = f"{ev.get('shd', float('inf')):.1f}" if ev.get('shd') is not None else 'N/A'
        print(f"Epoch {ep:03d} | loss {out['loss']:.4f} | recon {out.get('recon', 0):.4f} | "
              f"acy {out.get('acyclic', 0):.4f} | SHD {shd_str}")
        
        if ev.get("shd", 1e9) < best_shd:
            best_shd = ev["shd"]
            os.makedirs("artifacts/checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "artifacts/checkpoints/rcgnn_best.pt")
            # Save adjacency to configured path
            adj_path = Path(args.adj_output)
            adj_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(adj_path, ev["A_mean"])
            print(f"✅ Saved best adjacency to {adj_path}")
    print("Best SHD:", best_shd)

if __name__ == "__main__":
    main()
