
import argparse
import os

import yaml

import path_helper  # noqa: F401  # adds project root to sys.path

from src.dataio.synth import build_synth

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_cfg")
    parser.add_argument("model_cfg")
    parser.add_argument("train_cfg")
    args = parser.parse_args()
    with open(args.data_cfg) as f: dc = yaml.safe_load(f)
    out_root = os.path.join(dc["paths"]["root"], "interim", "synth_small")
    build_synth(out_root, d=dc["features"], T=512, regimes=2, Tw=dc["window_len"], Ts=dc["window_stride"], seed=0)
    print("Wrote synthetic data to", out_root)

if __name__ == "__main__":
    main()
