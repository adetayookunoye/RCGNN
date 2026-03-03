#!/usr/bin/env python3
"""
Comprehensive metrics: RC-GNN vs ALL baselines on completed Table 2 tasks.
Runs PC (multiple alphas), NOTEARS, Correlation, and variance-oriented versions.
"""
import numpy as np
import os, sys, json
sys.path.insert(0, '.')

from src.training.baselines import pc_algorithm, notears_linear, correlation_scores, impute_with_mask

# ============================================================================
# METRIC HELPERS
# ============================================================================

def topk_binarize(A, K):
    """Binarize A by keeping top-K entries."""
    d = A.shape[0]
    A_flat = np.abs(A).copy()
    np.fill_diagonal(A_flat, 0)
    topk_idx = np.argsort(A_flat.flatten())[::-1][:K]
    A_bin = np.zeros_like(A_flat)
    for idx in topk_idx:
        A_bin[idx // d, idx % d] = 1.0
    return A_bin

def skel_from_directed(A_bin):
    """Symmetric skeleton from directed binary."""
    return np.maximum(A_bin, A_bin.T)

def directed_f1(A_pred_bin, A_true_bin):
    tp = float(((A_pred_bin > 0) & (A_true_bin > 0)).sum())
    fp = float(((A_pred_bin > 0) & (A_true_bin == 0)).sum())
    fn = float(((A_pred_bin == 0) & (A_true_bin > 0)).sum())
    p = tp / (tp + fp) if (tp + fp) else 0
    r = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * p * r / (p + r) if (p + r) else 0
    return f1, p, r, int(tp)

def skeleton_f1(A_pred_bin, A_true_bin):
    d = A_pred_bin.shape[0]
    sp = skel_from_directed(A_pred_bin)
    st = skel_from_directed(A_true_bin)
    upper = np.triu(np.ones((d, d)), k=1)
    tp = float(((sp * upper > 0) & (st * upper > 0)).sum())
    fp = float(((sp * upper > 0) & (st * upper == 0)).sum())
    fn = float(((sp * upper == 0) & (st * upper > 0)).sum())
    p = tp / (tp + fp) if (tp + fp) else 0
    r = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * p * r / (p + r) if (p + r) else 0
    return f1, int(tp)

def compute_shd(A_pred_bin, A_true_bin):
    return int(np.sum(np.abs(A_pred_bin - A_true_bin)))

def compute_auroc(A_pred_cont, A_true_bin):
    try:
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(A_true_bin.flatten(), np.abs(A_pred_cont).flatten())
    except:
        return 0.5

def count_bidir(A_bin):
    d = A_bin.shape[0]
    c = 0
    for i in range(d):
        for j in range(i+1, d):
            if A_bin[i,j] > 0 and A_bin[j,i] > 0:
                c += 1
    return c

def orient_by_variance(X, M, skel):
    """Orient skeleton edges by variance ordering."""
    d = X.shape[-1]
    X_flat = X.reshape(-1, d)
    if M is not None:
        M_flat = M.reshape(-1, d)
        # masked variance
        n_obs = np.maximum(M_flat.sum(axis=0), 1)
        mean_x = (X_flat * M_flat).sum(axis=0) / n_obs
        var_x = ((X_flat - mean_x)**2 * M_flat).sum(axis=0) / n_obs
    else:
        var_x = np.var(X_flat, axis=0)
    
    A_dir = np.zeros((d, d))
    rows, cols = np.where(np.triu(skel, k=1) > 0)
    for i, j in zip(rows, cols):
        if var_x[j] > var_x[i]:
            A_dir[i, j] = 1
        else:
            A_dir[j, i] = 1
    return A_dir

def full_metrics(A_pred_bin, A_pred_cont, A_true_bin, K):
    f1, prec, rec, tp_d = directed_f1(A_pred_bin, A_true_bin)
    sf1, tp_s = skeleton_f1(A_pred_bin, A_true_bin)
    shd = compute_shd(A_pred_bin, A_true_bin)
    auroc = compute_auroc(A_pred_cont, A_true_bin)
    bidir = count_bidir(A_pred_bin)
    return {
        'TopK_F1': round(f1, 4), 'Skel_F1': round(sf1, 4),
        'SHD': shd, 'AUROC': round(auroc, 4), 'bidir': bidir,
        'TP_dir': tp_d, 'TP_skel': tp_s, 'Prec': round(prec, 4), 'Rec': round(rec, 4),
    }

# ============================================================================
# BASELINE RUNNERS
# ============================================================================

def run_baselines(X, M, A_true, K):
    d = A_true.shape[0]
    A_true_bin = (A_true > 0).astype(float)
    results = {}
    
    # --- PC at alpha=0.05 ---
    try:
        A_pc = pc_algorithm(X, M, alpha=0.05)
        A_pc_bin = (A_pc > 0).astype(float)
        np.fill_diagonal(A_pc_bin, 0)
        results['PC(0.05)'] = full_metrics(A_pc_bin, A_pc, A_true_bin, K)
        # PC skeleton + variance
        skel_pc = skel_from_directed(A_pc_bin)
        A_pcv = orient_by_variance(X, M, skel_pc)
        results['PC(0.05)+Var'] = full_metrics(A_pcv, A_pcv, A_true_bin, K)
    except Exception as e:
        results['PC(0.05)'] = {'TopK_F1': 0, 'Skel_F1': 0, 'SHD': K*2, 'AUROC': 0.5, 'bidir': 0, 'TP_dir': 0, 'TP_skel': 0, 'Prec': 0, 'Rec': 0, 'error': str(e)}
    
    # --- PC at alpha=0.10 ---
    try:
        A_pc10 = pc_algorithm(X, M, alpha=0.10)
        A_pc10_bin = (A_pc10 > 0).astype(float)
        np.fill_diagonal(A_pc10_bin, 0)
        results['PC(0.10)'] = full_metrics(A_pc10_bin, A_pc10, A_true_bin, K)
        skel_pc10 = skel_from_directed(A_pc10_bin)
        A_pc10v = orient_by_variance(X, M, skel_pc10)
        results['PC(0.10)+Var'] = full_metrics(A_pc10v, A_pc10v, A_true_bin, K)
    except Exception as e:
        results['PC(0.10)'] = {'TopK_F1': 0, 'Skel_F1': 0, 'SHD': K*2, 'AUROC': 0.5, 'bidir': 0, 'TP_dir': 0, 'TP_skel': 0, 'Prec': 0, 'Rec': 0, 'error': str(e)}
    
    # --- NOTEARS ---
    try:
        A_nt = notears_linear(X, M)
        A_nt_bin = (np.abs(A_nt) > 0.3).astype(float)
        np.fill_diagonal(A_nt_bin, 0)
        results['NOTEARS'] = full_metrics(A_nt_bin, A_nt, A_true_bin, K)
    except Exception as e:
        results['NOTEARS'] = {'TopK_F1': 0, 'Skel_F1': 0, 'SHD': K*2, 'AUROC': 0.5, 'bidir': 0, 'TP_dir': 0, 'TP_skel': 0, 'Prec': 0, 'Rec': 0, 'error': str(e)}
    
    # --- Correlation top-K ---
    try:
        A_corr = correlation_scores(X, M)
        A_corr_topk = topk_binarize(A_corr, K)
        results['Corr-TopK'] = full_metrics(A_corr_topk, A_corr, A_true_bin, K)
        # Corr skeleton + variance
        skel_corr = skel_from_directed(A_corr_topk)
        A_cv = orient_by_variance(X, M, skel_corr)
        results['Corr+Var'] = full_metrics(A_cv, A_corr, A_true_bin, K)
    except Exception as e:
        results['Corr-TopK'] = {'TopK_F1': 0, 'Skel_F1': 0, 'SHD': K*2, 'AUROC': 0.5, 'bidir': 0, 'TP_dir': 0, 'TP_skel': 0, 'Prec': 0, 'Rec': 0, 'error': str(e)}
    
    return results

# ============================================================================
# MAIN
# ============================================================================

tasks_file = 'artifacts/table2/tasks.tsv'
with open(tasks_file) as f:
    lines = f.readlines()

from collections import defaultdict
all_results = {}  # config -> seed -> {method -> metrics}
agg = defaultdict(lambda: defaultdict(list))  # config -> method -> [metrics]

completed = 0
for line in lines[1:]:
    parts = line.strip().split('\t')
    table, config, seed, data_dir, art_dir = parts
    
    A_path = os.path.join(art_dir, 'A_best_score.npy')
    A_true_path = os.path.join(data_dir, 'A_true.npy')
    X_path = os.path.join(data_dir, 'X.npy')
    M_path = os.path.join(data_dir, 'M.npy')
    
    if not os.path.exists(A_path):
        continue
    
    completed += 1
    A_pred = np.load(A_path)
    A_true = np.load(A_true_path)
    X = np.load(X_path)
    M = np.load(M_path)
    
    d = A_true.shape[0]
    K = int(A_true.sum())
    A_true_bin = (A_true > 0).astype(float)
    
    seed_results = {}
    
    # RC-GNN native (top-K)
    A_rcgnn_topk = topk_binarize(A_pred, K)
    seed_results['RCGNN'] = full_metrics(A_rcgnn_topk, A_pred, A_true_bin, K)
    
    # RC-GNN skeleton + variance orientation
    skel_rcgnn = skel_from_directed(A_rcgnn_topk)
    A_rcgnn_var = orient_by_variance(X, M, skel_rcgnn)
    seed_results['RCGNN+Var'] = full_metrics(A_rcgnn_var, A_pred, A_true_bin, K)
    
    # All baselines
    bl = run_baselines(X, M, A_true, K)
    seed_results.update(bl)
    
    key = f"{config}/s{seed}"
    all_results[key] = seed_results
    
    for method, m in seed_results.items():
        agg[config][method].append(m)
    
    print(f"[{completed}] {config}/s{seed}: d={d} K={K} RCGNN_F1={seed_results['RCGNN']['TopK_F1']:.4f} RCGNN+Var={seed_results['RCGNN+Var']['TopK_F1']:.4f}")

print(f"\n{'='*120}")
print(f" COMPLETED: {completed} tasks")
print(f"{'='*120}")

# --- Per-seed detail table ---
methods_order = ['RCGNN', 'RCGNN+Var', 'PC(0.05)', 'PC(0.05)+Var', 'PC(0.10)', 'PC(0.10)+Var', 'NOTEARS', 'Corr-TopK', 'Corr+Var']

print(f"\n{'='*120}")
print(f" PER-SEED DETAIL: TopK-F1 / Skel-F1 / SHD / AUROC")
print(f"{'='*120}")
header = f"{'Task':20s}"
for m in methods_order:
    header += f" | {m:>14s}"
print(header)
print("-" * len(header))

for key in sorted(all_results.keys()):
    row = f"{key:20s}"
    for m in methods_order:
        if m in all_results[key]:
            v = all_results[key][m]
            row += f" | {v['TopK_F1']:5.3f}/{v['Skel_F1']:5.3f}"
        else:
            row += f" |       —      "
    print(row)

# --- Aggregated table ---
print(f"\n{'='*140}")
print(f" AGGREGATED (mean±std)")
print(f"{'='*140}")

for config in sorted(agg.keys()):
    n = len(agg[config].get('RCGNN', []))
    print(f"\n  {config} ({n} seeds)")
    print(f"  {'Method':16s} | {'TopK-F1':>15s} | {'Skel-F1':>15s} | {'SHD':>10s} | {'AUROC':>15s} | {'bidir':>8s} | {'TP_dir':>6s} | {'TP_skel':>6s}")
    print(f"  {'-'*110}")
    
    for method in methods_order:
        if method not in agg[config] or len(agg[config][method]) == 0:
            continue
        vals = agg[config][method]
        f1s = [v['TopK_F1'] for v in vals]
        skels = [v['Skel_F1'] for v in vals]
        shds = [v['SHD'] for v in vals]
        aurocs = [v['AUROC'] for v in vals]
        bidirs = [v['bidir'] for v in vals]
        tp_ds = [v['TP_dir'] for v in vals]
        tp_ss = [v['TP_skel'] for v in vals]
        
        print(f"  {method:16s} | {np.mean(f1s):.4f}±{np.std(f1s):.4f} | {np.mean(skels):.4f}±{np.std(skels):.4f} | {np.mean(shds):5.1f}±{np.std(shds):4.1f} | {np.mean(aurocs):.4f}±{np.std(aurocs):.4f} | {np.mean(bidirs):5.1f}±{np.std(bidirs):3.1f} | {np.mean(tp_ds):5.1f} | {np.mean(tp_ss):5.1f}")

# Save JSON
out = {'per_seed': all_results, 'aggregated': {}}
for config in agg:
    out['aggregated'][config] = {}
    for method in agg[config]:
        vals = agg[config][method]
        out['aggregated'][config][method] = {
            'TopK_F1_mean': round(np.mean([v['TopK_F1'] for v in vals]), 4),
            'TopK_F1_std': round(np.std([v['TopK_F1'] for v in vals]), 4),
            'Skel_F1_mean': round(np.mean([v['Skel_F1'] for v in vals]), 4),
            'SHD_mean': round(np.mean([v['SHD'] for v in vals]), 1),
            'AUROC_mean': round(np.mean([v['AUROC'] for v in vals]), 4),
            'n_seeds': len(vals),
        }

with open('artifacts/table2/baseline_comparison_partial.json', 'w') as f:
    json.dump(out, f, indent=2, default=float)
print(f"\nSaved: artifacts/table2/baseline_comparison_partial.json")
