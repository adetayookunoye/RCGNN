#!/usr/bin/env python3
"""Quick comparison: RC-GNN skeleton vs PC skeleton on h1_easy."""
import numpy as np, sys
sys.path.insert(0, '.')
from src.training.baselines import pc_algorithm, impute_with_mask

X = np.load('data/interim/table2a/h1_easy/seed_0/X.npy')
M = np.load('data/interim/table2a/h1_easy/seed_0/M.npy')
A_true = np.load('data/interim/table2a/h1_easy/seed_0/A_true.npy')
d = 15
K = int(A_true.sum())
print(f'X={X.shape}, M={M.shape}, A_true edges={K}')
miss_rate = 1 - M.mean()
print(f'Missing rate: {miss_rate:.2%}')

# --- PC at multiple alphas ---
for alpha in [0.01, 0.05, 0.10, 0.20]:
    A_pc = pc_algorithm(X, M, alpha=alpha)
    skel_pc = np.maximum((A_pc>0).astype(float), (A_pc>0).astype(float).T)
    skel_true = np.maximum((A_true>0).astype(float), (A_true>0).astype(float).T)
    upper = np.triu(np.ones((d,d)), k=1)
    tp_s = int(((skel_pc*upper>0) & (skel_true*upper>0)).sum())
    fp_s = int(((skel_pc*upper>0) & (skel_true*upper==0)).sum())
    fn_s = int(((skel_pc*upper==0) & (skel_true*upper>0)).sum())
    n_edges_pc = int((skel_pc*upper).sum())
    p_s = tp_s/(tp_s+fp_s) if (tp_s+fp_s) else 0
    r_s = tp_s/(tp_s+fn_s) if (tp_s+fn_s) else 0
    f1_s = 2*p_s*r_s/(p_s+r_s) if (p_s+r_s) else 0

    # Directed
    tp_d = int(((A_pc>0)&(A_true>0)).sum())
    fp_d = int(((A_pc>0)&(A_true==0)).sum())
    fn_d = int(((A_pc==0)&(A_true>0)).sum())
    p_d = tp_d/(tp_d+fp_d) if (tp_d+fp_d) else 0
    r_d = tp_d/(tp_d+fn_d) if (tp_d+fn_d) else 0
    f1_d = 2*p_d*r_d/(p_d+r_d) if (p_d+r_d) else 0

    # PC + variance orientation
    X_flat = X.reshape(-1, d)
    var_x = np.var(X_flat, axis=0)
    rows, cols = np.where(np.triu(skel_pc, k=1) > 0)
    A_pc_var = np.zeros((d,d))
    for i,j in zip(rows, cols):
        if var_x[j] > var_x[i]:
            A_pc_var[i,j] = 1
        else:
            A_pc_var[j,i] = 1
    tp_pv = int(((A_pc_var>0)&(A_true>0)).sum())
    fp_pv = int(((A_pc_var>0)&(A_true==0)).sum())
    fn_pv = int(((A_pc_var==0)&(A_true>0)).sum())
    p_pv = tp_pv/(tp_pv+fp_pv) if (tp_pv+fp_pv) else 0
    r_pv = tp_pv/(tp_pv+fn_pv) if (tp_pv+fn_pv) else 0
    f1_pv = 2*p_pv*r_pv/(p_pv+r_pv) if (p_pv+r_pv) else 0

    print(f'  PC(alpha={alpha:.2f}): {n_edges_pc} edges, Skel-F1={f1_s:.4f} (TP={tp_s}), '
          f'Dir-F1={f1_d:.4f}, PC+Var-F1={f1_pv:.4f}')

# --- NOTEARS ---
from src.training.baselines import notears_linear
A_nt = notears_linear(X, M)
skel_nt = np.maximum((A_nt>0).astype(float), (A_nt>0).astype(float).T)
tp_nts = int(((skel_nt*upper>0) & (skel_true*upper>0)).sum())
fp_nts = int(((skel_nt*upper>0) & (skel_true*upper==0)).sum())
n_nt = int((skel_nt*upper).sum())
p_nts = tp_nts/(tp_nts+fp_nts) if (tp_nts+fp_nts) else 0
r_nts = tp_nts/(tp_nts+int(((skel_nt*upper==0)&(skel_true*upper>0)).sum())) if tp_nts else 0
f1_nts = 2*p_nts*r_nts/(p_nts+r_nts) if (p_nts+r_nts) else 0
tp_ntd = int(((A_nt>0)&(A_true>0)).sum())
fp_ntd = int(((A_nt>0)&(A_true==0)).sum())
p_ntd = tp_ntd/(tp_ntd+fp_ntd) if (tp_ntd+fp_ntd) else 0
r_ntd = tp_ntd/(tp_ntd+int(((A_nt==0)&(A_true>0)).sum())) if tp_ntd else 0
f1_ntd = 2*p_ntd*r_ntd/(p_ntd+r_ntd) if (p_ntd+r_ntd) else 0
print(f'  NOTEARS: {n_nt} edges, Skel-F1={f1_nts:.4f} (TP={tp_nts}), Dir-F1={f1_ntd:.4f}')

# --- Correlation ---
from src.training.baselines import correlation_scores
A_corr = correlation_scores(X, M)
# Top-K by correlation
A_corr_flat = A_corr.copy()
np.fill_diagonal(A_corr_flat, 0)
A_corr_sym = np.maximum(A_corr_flat, A_corr_flat.T)
rows_u, cols_u = np.triu_indices(d, k=1)
pair_vals = A_corr_sym[rows_u, cols_u]
topk_idx = np.argsort(pair_vals)[::-1][:K]
skel_corr = np.zeros((d,d))
for idx in topk_idx:
    i,j = rows_u[idx], cols_u[idx]
    skel_corr[i,j] = skel_corr[j,i] = 1
tp_cs = int(((skel_corr*upper>0)&(skel_true*upper>0)).sum())
fp_cs = int(((skel_corr*upper>0)&(skel_true*upper==0)).sum())
p_cs = tp_cs/(tp_cs+fp_cs) if (tp_cs+fp_cs) else 0
r_cs = tp_cs/(tp_cs+int(((skel_corr*upper==0)&(skel_true*upper>0)).sum())) if tp_cs else 0
f1_cs = 2*p_cs*r_cs/(p_cs+r_cs) if (p_cs+r_cs) else 0
# Corr + variance
A_cv = np.zeros((d,d))
var_x = np.var(X.reshape(-1,d), axis=0)
for idx in topk_idx:
    i,j = rows_u[idx], cols_u[idx]
    if var_x[j] > var_x[i]:
        A_cv[i,j] = 1
    else:
        A_cv[j,i] = 1
tp_cv = int(((A_cv>0)&(A_true>0)).sum())
fp_cv = int(((A_cv>0)&(A_true==0)).sum())
p_cv = tp_cv/(tp_cv+fp_cv) if (tp_cv+fp_cv) else 0
r_cv = tp_cv/(tp_cv+int(((A_cv==0)&(A_true>0)).sum())) if tp_cv else 0
f1_cv = 2*p_cv*r_cv/(p_cv+r_cv) if (p_cv+r_cv) else 0
print(f'  Corr-TopK: {K} edges, Skel-F1={f1_cs:.4f} (TP={tp_cs}), Corr+Var-F1={f1_cv:.4f}')

print()
print('=== FINAL COMPARISON ===')
print(f'RC-GNN skeleton     : Skel-F1=0.5333 (TP=16/30)')
print(f'RC-GNN+var orient   : TopK-F1=0.4333, OrientAcc=81.25%')
print(f'RC-GNN native       : TopK-F1=0.3000')
