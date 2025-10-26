import torch, math

def notears_acyclicity(A):
    # A is [d,d] in [0,1]
    expm = torch.matrix_exp(A*A)
    return torch.trace(expm) - A.shape[0]

def topk_per_column(A, k):
    # Create a mask matrix (d x d) selecting top-k entries per column.
    d = A.shape[0]
    if k >= d:
        return A
    mask_mat = torch.zeros_like(A)
    for j in range(d):
        col = A[:, j]
        vals, idx = torch.topk(col, k)
        mask = torch.zeros_like(col)
        mask[idx] = 1.0
        mask_mat[:, j] = mask
    return A * mask_mat

def threshold_adj(A, thr=0.5):
    return (A>=thr).float().cpu().numpy()
