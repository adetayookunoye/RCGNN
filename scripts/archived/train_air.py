#!/usr/bin/env python3
"""
Convenience wrapper to train RC-GNN on the UCI Air dataset using the standard
train_rcgnn.py entrypoint and the UCI config.

Usage:
  python scripts/train_air.py [--epochs N] [--adj-output PATH]

Defaults:
  --adj-output artifacts/adjacency/A_mean_air.npy
  uses configs/data_uci.yaml, configs/model.yaml, configs/train.yaml
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Ensure project root on sys.path for any relative imports inside train_rcgnn
import path_helper  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs (defaults to configs/train.yaml)")
    parser.add_argument("--adj-output", type=str, default=str(ROOT / "artifacts/adjacency/A_mean_air.npy"),
                        help="Where to save the learned adjacency")
    args = parser.parse_args()

    cmd = [
        sys.executable,
        str(ROOT / "scripts/train_rcgnn.py"),
        str(ROOT / "configs/data_uci.yaml"),
        str(ROOT / "configs/model.yaml"),
        str(ROOT / "configs/train.yaml"),
        "--adj-output", args.adj_output,
    ]
    if args.epochs is not None:
        cmd += ["--epochs", str(args.epochs)]

    env = os.environ.copy()
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
