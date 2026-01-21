
import argparse, yaml, os, numpy as np

import path_helper  # noqa: F401  # adds project root to sys.path
from src.training.baselines import notears_lite

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="notears_lite")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f: dc = yaml.safe_load(f)
    root = dc["paths"]["root"]
    X = np.load(os.path.join(root, "X.npy"))  # [N,T,d]
    A_true = np.load(os.path.join(root, "A_true.npy"))
    if args.method=="notears_lite":
        A_hat = notears_lite(X)
    else:
        raise ValueError("Unknown baseline")
    shd = int((A_hat != A_true).sum())
    os.makedirs("reports/tables", exist_ok=True)
    np.save("reports/tables/baseline_adj.npy", A_hat)
    print("Baseline:", args.method, "| SHD:", shd)

if __name__ == "__main__":
    main()
