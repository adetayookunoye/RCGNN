
import os, yaml, numpy as np, subprocess, sys, time, pathlib

def run(cmd):
    print("RUN:", cmd)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(r.stdout); print(r.stderr)
    assert r.returncode==0, f"Failed: {cmd}"

def test_pipeline_smoke():
    assert os.path.exists("configs/data.yaml")
    run("python scripts/synth_bench.py configs/data.yaml configs/model.yaml configs/train.yaml")
    run("python scripts/train_rcgnn.py configs/data.yaml configs/model.yaml configs/train.yaml")
    assert os.path.exists("artifacts/adjacency/A_mean.npy")
    run("python scripts/eval_rcgnn.py configs/eval.yaml --export reports/figs/")
    assert os.path.exists("reports/figs/adjacency_mean.png")
