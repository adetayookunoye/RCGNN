#!/usr/bin/env python3
"""Extract metrics from all completed Table 2 task artifacts."""
import numpy as np
import os
import sys

sys.path.insert(0, '.')

def compute_metrics(A_pred, A_true):
    d = A_true.shape[0]
    K = int(A_true.sum())
    
    # Top-K threshold on A_pred
    A_flat = np.abs(A_pred).copy()
    np.fill_diagonal(A_flat, 0)
    topk_idx = np.argsort(A_flat.flatten())[::-1][:K]
    A_topk = np.zeros_like(A_flat)
    for idx in topk_idx:
        i, j = idx // d, idx % d
        A_topk[i, j] = 1.0
    
    A_t = (A_true > 0).astype(float)
    
    # Directed TopK-F1
    tp_d = float(((A_topk > 0) & (A_t > 0)).sum())
    fp_d = float(((A_topk > 0) & (A_t == 0)).sum())
    fn_d = float(((A_topk == 0) & (A_t > 0)).sum())
    prec_d = tp_d / (tp_d + fp_d) if (tp_d + fp_d) else 0
    rec_d = tp_d / (tp_d + fn_d) if (tp_d + fn_d) else 0
    f1_d = 2 * prec_d * rec_d / (prec_d + rec_d) if (prec_d + rec_d) else 0
    
    # Skeleton F1
    sp = np.maximum(A_topk, A_topk.T)
    st = np.maximum(A_t, A_t.T)
    upper = np.triu(np.ones((d, d)), k=1)
    tp_s = float(((sp * upper > 0) & (st * upper > 0)).sum())
    fp_s = float(((sp * upper > 0) & (st * upper == 0)).sum())
    fn_s = float(((sp * upper == 0) & (st * upper > 0)).sum())
    p_s = tp_s / (tp_s + fp_s) if (tp_s + fp_s) else 0
    r_s = tp_s / (tp_s + fn_s) if (tp_s + fn_s) else 0
    f1_s = 2 * p_s * r_s / (p_s + r_s) if (p_s + r_s) else 0
    
    # SHD
    shd = int(np.sum(np.abs(A_topk - A_t)))
    
    # Bidir
    bidir = 0
    for i in range(d):
        for j in range(i + 1, d):
            if A_topk[i, j] > 0 and A_topk[j, i] > 0:
                bidir += 1
    
    # AUROC
    try:
        from sklearn.metrics import roc_auc_score
        auroc = roc_auc_score(A_t.flatten(), np.abs(A_pred).flatten())
    except:
        auroc = 0.5
    
    return {
        'TopK_F1': round(f1_d, 4), 'Skel_F1': round(f1_s, 4),
        'SHD': shd, 'AUROC': round(auroc, 4), 'bidir': bidir,
        'TP_dir': int(tp_d), 'TP_skel': int(tp_s),
        'n_pred': int(A_topk.sum()), 'K': K, 'd': d,
    }


# Read tasks.tsv
tasks_file = 'artifacts/table2/tasks.tsv'
with open(tasks_file) as f:
    lines = f.readlines()

print(f"{'Config':15s} {'Seed':>4s} | {'TopK-F1':>7s} {'Skel-F1':>7s} {'SHD':>4s} {'AUROC':>6s} {'bidir':>5s} | {'TP_d':>4s}/{'':<3s} {'TP_s':>4s} {'d':>3s}")
print("-" * 85)

from collections import defaultdict
agg = defaultdict(list)

for line in lines[1:]:
    parts = line.strip().split('\t')
    table, config, seed, data_dir, art_dir = parts
    
    A_path = os.path.join(art_dir, 'A_best_score.npy')
    A_true_path = os.path.join(data_dir, 'A_true.npy')
    
    if not os.path.exists(A_path):
        break  # Stop at first missing (sequential)
    
    A_pred = np.load(A_path)
    A_true = np.load(A_true_path)
    m = compute_metrics(A_pred, A_true)
    
    print(f"{config:15s} s{seed:>3s} | {m['TopK_F1']:7.4f} {m['Skel_F1']:7.4f} {m['SHD']:4d} {m['AUROC']:6.4f} {m['bidir']:5d} | {m['TP_dir']:4d}/{m['K']:<3d} {m['TP_skel']:4d} {m['d']:3d}")
    
    agg[config].append(m)

print("-" * 85)
print()
print(f"{'Config':15s} | {'TopK-F1':>15s} {'Skel-F1':>15s} {'SHD':>10s} {'AUROC':>15s} {'bidir':>8s} | n")
print("-" * 95)

for config in agg:
    vals = agg[config]
    n = len(vals)
    f1s = [v['TopK_F1'] for v in vals]
    skels = [v['Skel_F1'] for v in vals]
    shds = [v['SHD'] for v in vals]
    aurocs = [v['AUROC'] for v in vals]
    bidirs = [v['bidir'] for v in vals]
    
    print(f"{config:15s} | {np.mean(f1s):6.4f}±{np.std(f1s):.4f} {np.mean(skels):6.4f}±{np.std(skels):.4f} {np.mean(shds):5.1f}±{np.std(shds):.1f} {np.mean(aurocs):6.4f}±{np.std(aurocs):.4f} {np.mean(bidirs):5.1f}±{np.std(bidirs):.1f} | {n}")
