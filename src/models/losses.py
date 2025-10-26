import torch
from .utils import notears_acyclicity

def masked_l1(x, y, m, unc=None):
    # if unc provided, weight errors by predicted uncertainty (smaller unc -> higher weight)
    e = torch.abs((x - y) * m)
    if unc is None:
        return torch.mean(e)
    else:
        w = 1.0 / (unc + 1e-6)
        return torch.mean(e * w)

def _rbf_kernel(x, sigma=None):
    # x: [n, d] or [d], compute pairwise RBF kernel matrix
    if x.dim()==1:
        x = x.unsqueeze(0)
    dists = torch.cdist(x, x, p=2)**2
    if sigma is None:
        # median heuristic
        sigma = torch.sqrt(torch.median(dists[dists>0]))
        if torch.isnan(sigma) or sigma<=0:
            sigma = torch.tensor(1.0, device=x.device)
    K = torch.exp(-dists / (2 * (sigma**2 + 1e-6)))
    return K

def hsic_xy(x, y):
    # kernel HSIC biased estimator using RBF kernels
    # x,y: 1D or 2D tensors with samples along dim 0
    if x.dim()==1: x = x.unsqueeze(1)
    if y.dim()==1: y = y.unsqueeze(1)
    n = x.shape[0]
    K = _rbf_kernel(x)
    L = _rbf_kernel(y)
    H = torch.eye(n, device=x.device) - (1.0/n) * torch.ones((n,n), device=x.device)
    KH = K @ H
    HL = H @ L
    hsic = torch.trace(KH @ HL) / ((n-1)**2 + 1e-6)
    return hsic

def adj_variance(As):
    # As: list of [d,d] tensors
    Astack = torch.stack(As, dim=0)
    return torch.mean(torch.var(Astack, dim=0))
