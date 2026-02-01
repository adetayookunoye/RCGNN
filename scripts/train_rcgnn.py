import argparse, yaml, os, torch
from pathlib import Path

import path_helper # noqa: F401 # adds project root to sys.path
from torch.utils.data import DataLoader
from src.dataio.loaders import load_synth
from src.models.rcgnn import RCGNN
from src.models.invariance import IRMStructureInvariance
from src.training.optim import make_optimizer, disentanglement_loss
from src.training.loop import train_epoch, eval_epoch
import numpy as np
import math

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_cfg")
    parser.add_argument("model_cfg")
    parser.add_argument("train_cfg")
    parser.add_argument("--adj-output", default="artifacts/adjacency/A_mean.npy", 
                        help="Path to save learned adjacency matrix")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs (takes precedence over train config)")
    args = parser.parse_args()

    with open(args.data_cfg) as f: dc = yaml.safe_load(f)
    with open(args.model_cfg) as f: mc = yaml.safe_load(f)
    with open(args.train_cfg) as f: tc = yaml.safe_load(f)
    # Optional override from CLI
    if args.epochs is not None:
        tc["epochs"] = int(args.epochs)

    root = dc["paths"]["root"] # Already contains full path like "data/interim/synth_small" or dataset dir
    print(f" Loading data from: {root}")

    # Optional normalization for real datasets (e.g., UCI Air)
    normalize = dc.get("normalize", False)
    
    train_ds = load_synth(root, "train", seed=tc["seed"])
    val_ds = load_synth(root, "val", seed=tc["seed"]+1)

    # Apply simple per-feature standardization using train stats
    if normalize:
        with torch.no_grad():
            # Compute mean/std over N and T for each feature d
            Xt = train_ds.X # [N,T,d]
            mean = Xt.mean(dim=(0,1), keepdim=True)
            std = Xt.std(dim=(0,1), keepdim=True).clamp_min(1e-6)
            train_ds.X.sub_(mean).div_(std)
            val_ds.X.sub_(mean).div_(std)
    
    print(f"[DONE] Data loaded: {len(train_ds)} train, {len(val_ds)} val samples")

    train_ld = DataLoader(train_ds, batch_size=tc["batch_size"], shuffle=True)
    val_ld = DataLoader(val_ds, batch_size=1, shuffle=False)

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

    # Initialize invariance loss module if lambda_inv > 0
    lambda_inv = mc.get("loss", {}).get("invariance", {}).get("lambda_inv", 0.0)
    n_envs = mc.get("loss", {}).get("invariance", {}).get("n_envs", 1)
    invariance_loss_fn = None
    if lambda_inv > 0:
        invariance_loss_fn = IRMStructureInvariance(n_features=d, n_envs=n_envs, gamma=0.1)
        invariance_loss_fn.to(device)
        print(f" Initialized invariance loss with lambda_inv={lambda_inv}, n_envs={n_envs}")

    opt = make_optimizer(model, lr=tc["learning_rate"], weight_decay=tc["weight_decay"])
    
    # Load ground truth adjacency if available
    A_true_path = Path(root) / "A_true.npy"
    A_true = None
    if A_true_path.exists():
        A_true = np.load(A_true_path)
        # Sanity check: off-diagonal nonzeros
        mask_no_diag = np.ones_like(A_true, dtype=bool)
        np.fill_diagonal(mask_no_diag, False)
        offdiag_nnz = int((A_true[mask_no_diag] != 0).sum())
        print(f"[DONE] Ground truth adjacency loaded: {A_true.shape}, offdiag_nnz={offdiag_nnz}")
    else:
        print("[WARN] No ground truth adjacency (A_true.npy) - SHD will not be computed")
    
    best_shd = 1e9

    # Schedules for supervised/acyclic weights
    def cos_ramp(t, T):
        x = max(0.0, min(1.0, t / max(1, T)))
        return 0.5 * (1.0 - math.cos(math.pi * x))

    T_sup_on = int(tc.get("supervised_warmup_epochs", 10))
    T_acy_on = int(tc.get("acyclic_warmup_epochs", 20))

    base_lambda_sup = float(tc.get("lambda_supervised", 0.0))
    base_lambda_acy = float(tc.get("lambda_acyclic", 0.1))

    for ep in range(tc["epochs"]):
        # Dynamic weights (warm-up then ramp up)
        lam_sup = base_lambda_sup * cos_ramp(ep, T_sup_on)
        lam_acy = base_lambda_acy * cos_ramp(ep, T_acy_on)

        loss_kwargs = {
            "lambda_recon": tc.get("lambda_recon", 1.0),
            "lambda_sparse": tc.get("lambda_sparse", 0.01),
            "lambda_acyclic": lam_acy,
            "lambda_disen": tc.get("lambda_disen", 0.01),
            "target_sparsity": tc.get("target_sparsity", 0.1),
            "lambda_supervised": lam_sup,
            "A_true": A_true,
            "lambda_inv": lambda_inv,
            "invariance_loss_fn": invariance_loss_fn,
        }
        out = train_epoch(model, train_ld, opt, device=device, **loss_kwargs)
        ev = eval_epoch(model, val_ld, A_true=A_true, threshold=0.5, device=device)
        
        # Print epoch summary
        shd_str = f"{ev.get('shd', float('inf')):.1f}" if ev.get('shd') is not None else 'N/A'
        print(f"Epoch {ep:03d} | loss {out['loss']:.4f} | recon {out.get('recon', 0):.4f} | "
              f"acy {out.get('acyclic', 0):.4f} | SHD {shd_str} | λ_sup {lam_sup:.3f} | λ_acy {lam_acy:.3f}")

        # Naive SHD debug on off-diagonal to catch metric wiring issues
        if A_true is not None and 'A_mean' in ev:
            thr_use = ev.get('best_thr', 0.5) if 'best_thr' in ev else 0.5
            A_pred_bin = (ev['A_mean'] >= thr_use).astype(int)
            if A_pred_bin.shape == A_true.shape:
                off = np.ones_like(A_true, dtype=bool)
                np.fill_diagonal(off, False)
                A_true_b = (A_true > 0).astype(int)
                shd_naive = int((A_true_b[off] ^ A_pred_bin[off]).sum())
                print(f" [DEBUG] SHD_naive(offdiag)={shd_naive}")
        
        if ev.get("shd", 1e9) < best_shd:
            best_shd = ev["shd"]
            os.makedirs("artifacts/checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "artifacts/checkpoints/rcgnn_best.pt")
            # Save adjacency to configured path
            adj_path = Path(args.adj_output)
            adj_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(adj_path, ev["A_mean"])
            print(f"[DONE] Saved best adjacency to {adj_path}")
    print("Best SHD:", best_shd)

if __name__ == "__main__":
    main()
