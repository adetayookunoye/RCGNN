
import os, numpy as np

def test_artifacts_exist_after_train():
    assert os.path.exists("artifacts/checkpoints/rcgnn_best.pt")
    assert os.path.exists("artifacts/adjacency/A_mean.npy")
