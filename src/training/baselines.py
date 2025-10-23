
import numpy as np

def notears_lite(Xw):
    # simple correlation-threshold baseline approximating structure (undirected)
    # Xw: [N,T,d] windows
    X = Xw.mean(axis=1)  # average over time
    d = X.shape[1]
    C = np.corrcoef(X, rowvar=False)
    A = np.abs(C) - np.eye(d)
    thr = np.quantile(A, 0.9)
    A_bin = (A>=thr).astype(int)
    return A_bin
