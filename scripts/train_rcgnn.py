import argparse, yaml, os, torch

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
    args = parser.parse_args()

    with open(args.data_cfg) as f: dc = yaml.safe_load(f)
    with open(args.model_cfg) as f: mc = yaml.safe_load(f)
    with open(args.train_cfg) as f: tc = yaml.safe_load(f)

    root = os.path.join(dc["paths"]["root"], "interim", "synth_small")
    train_ds = load_synth(root, "train", seed=tc["seed"])
    val_ds   = load_synth(root, "val", seed=tc["seed"]+1)

    train_ld = DataLoader(train_ds, batch_size=tc["batch_size"], shuffle=True)
    val_ld   = DataLoader(val_ds, batch_size=1, shuffle=False)

    d = train_ds.X.shape[-1]
    model = RCGNN(d, mc)
    device = tc["device"]
    model.to(device)

    opt = make_optimizer(model.parameters(), tc)
    best_shd = 1e9
    A_true = np.load(os.path.join(root, "A_true.npy"))

    for ep in range(tc["epochs"]):
        out = train_epoch(model, train_ld, opt, inv_weight=mc["loss"]["invariance"]["lambda_inv"], device=device)
        ev = eval_epoch(model, val_ld, A_true=A_true, thr=0.5, device=device)
        print(f"Epoch {ep:03d} | loss {out['loss']:.4f} | L_rec {out['L_rec']:.4f} | L_acy {out['L_acy']:.4f} | SHD {ev.get('shd','-')}")
        if ev.get("shd", 1e9) < best_shd:
            best_shd = ev["shd"]
            os.makedirs("artifacts/checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "artifacts/checkpoints/rcgnn_best.pt")
            os.makedirs("artifacts/adjacency", exist_ok=True)
            np.save("artifacts/adjacency/A_mean.npy", ev["A_mean"])
    print("Best SHD:", best_shd)

if __name__ == "__main__":
    main()
