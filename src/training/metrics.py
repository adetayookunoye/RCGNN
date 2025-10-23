
import numpy as np
def shd(A_hat, A_true):
    return int(np.sum(A_hat != A_true))

def jaccard_edges(A1, A2):
    e1 = (A1>0).astype(int).flatten()
    e2 = (A2>0).astype(int).flatten()
    inter = (e1 & e2).sum()
    union = ((e1 + e2) > 0).sum()
    return inter / max(1, union)

def pairwise_l1(As):
    As = [a.astype(float) for a in As]
    n = len(As)
    s = 0.0; c = 0
    for i in range(n):
        for j in range(i+1, n):
            s += np.abs(As[i]-As[j]).mean()
            c += 1
    return s / max(1,c)
