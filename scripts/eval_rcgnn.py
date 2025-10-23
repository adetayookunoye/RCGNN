
import argparse, yaml, os, numpy as np, matplotlib.pyplot as plt

import path_helper  # noqa: F401  # adds project root to sys.path
from src.training.metrics import pairwise_l1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_cfg")
    parser.add_argument("--export", default="reports/figs/")
    args = parser.parse_args()
    os.makedirs(args.export, exist_ok=True)

    A_mean = np.load("artifacts/adjacency/A_mean.npy")
    plt.imshow(A_mean, aspect='auto')
    plt.title("Adjacency (mean)")
    plt.colorbar()
    plt.savefig(os.path.join(args.export, "adjacency_mean.png"), dpi=150)
    plt.close()
    print("Saved adjacency_mean.png")

if __name__ == "__main__":
    main()
