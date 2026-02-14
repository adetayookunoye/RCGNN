#!/usr/bin/env python3
"""Comprehensive analysis of Table 2 results: RC-GNN vs all baselines."""
import json, glob, statistics, sys, os

def avg(vals):
    return statistics.mean(vals) if vals else 0.0

def std(vals):
    return statistics.stdev(vals) if len(vals) > 1 else 0.0

def safe_float(v):
    """Handle NaN values."""
    if v != v:  # NaN check
        return 0.0
    return float(v)

# Determine root
root = os.environ.get("RCGNN_ROOT", "/scratch/aoo29179/rcgnn")

# Collect all evaluation.json files
results = {}

# Table 2A
for path in sorted(glob.glob(f"{root}/artifacts/table2a/*/seed_*/evaluation.json")):
    parts = path.split("/")
    config = parts[-3]
    seed = parts[-2]
    with open(path) as f:
        d = json.load(f)
    key = f"2A/{config}"
    if key not in results:
        results[key] = {"seeds": [], "d": d["metadata"]["d"]}
    results[key]["seeds"].append(d)

# Table 2B (SEM)
for path in sorted(glob.glob(f"{root}/artifacts/sem_table2/*/seed_*/evaluation.json")):
    parts = path.split("/")
    config = parts[-3]
    seed = parts[-2]
    with open(path) as f:
        d = json.load(f)
    key = f"2B/{config}"
    if key not in results:
        results[key] = {"seeds": [], "d": d["metadata"]["d"]}
    results[key]["seeds"].append(d)

# Table 2C
for path in sorted(glob.glob(f"{root}/artifacts/table2c/*/seed_*/evaluation.json")):
    parts = path.split("/")
    config = parts[-3]
    seed = parts[-2]
    with open(path) as f:
        d = json.load(f)
    key = f"2C/{config}"
    if key not in results:
        results[key] = {"seeds": [], "d": d["metadata"]["d"]}
    results[key]["seeds"].append(d)

if not results:
    print("ERROR: No evaluation.json files found.")
    sys.exit(1)

W = 130

# ============================================================================
# PART 1: Detailed per-config breakdown
# ============================================================================
print("=" * W)
print("COMPREHENSIVE TABLE 2 RESULTS: RC-GNN vs BASELINES")
print("=" * W)

for config_name, data in results.items():
    seeds = data["seeds"]
    d = data["d"]
    n = len(seeds)

    # Collect all baseline names
    baseline_names = []
    for s in seeds:
        for bn in s.get("baselines", {}):
            if bn not in baseline_names:
                baseline_names.append(bn)

    print(f"\n{'─' * W}")
    print(f"{config_name} (d={d}, {n} seeds)")
    print(f"{'─' * W}")
    header = f"{'Method':<20} {'SHD':>10}  {'Skel-F1':>12}  {'Dir-F1':>12}  {'Skel-Prec':>12}  {'Skel-Rec':>12}  {'AUROC':>10}  {'Edges':>7}"
    print(header)
    print("-" * len(header))

    # RC-GNN
    shds = [s["rc_gnn"]["topk"]["SHD"] for s in seeds]
    skf1 = [s["rc_gnn"]["topk"]["Skeleton_F1"] for s in seeds]
    df1  = [s["rc_gnn"]["topk"]["Directed_F1"] for s in seeds]
    sp   = [s["rc_gnn"]["topk"]["Skeleton_Precision"] for s in seeds]
    sr   = [s["rc_gnn"]["topk"]["Skeleton_Recall"] for s in seeds]
    aurocs = [safe_float(s["rc_gnn"]["topk"].get("AUROC", 0)) for s in seeds]
    edges  = [s["rc_gnn"]["topk"].get("Edges", 0) for s in seeds]
    print(f"{'RC-GNN (TopK)':<20} {avg(shds):>5.1f}±{std(shds):<4.1f} {avg(skf1):>7.3f}±{std(skf1):.3f}  {avg(df1):>7.3f}±{std(df1):.3f}  {avg(sp):>7.3f}±{std(sp):.3f}  {avg(sr):>7.3f}±{std(sr):.3f}  {avg(aurocs):>7.3f}   {avg(edges):>5.1f}")

    # Baselines
    for bn in baseline_names:
        shds = []; skf1 = []; df1 = []; sp = []; sr = []; aurocs = []; edges = []
        for s in seeds:
            b = s.get("baselines", {}).get(bn)
            if b:
                shds.append(b["SHD"])
                skf1.append(b["Skeleton_F1"])
                df1.append(b["Directed_F1"])
                sp.append(b["Skeleton_Precision"])
                sr.append(b["Skeleton_Recall"])
                aurocs.append(safe_float(b.get("AUROC", 0)))
                edges.append(b.get("Edges", 0))
        if shds:
            print(f"{bn:<20} {avg(shds):>5.1f}±{std(shds):<4.1f} {avg(skf1):>7.3f}±{std(skf1):.3f}  {avg(df1):>7.3f}±{std(df1):.3f}  {avg(sp):>7.3f}±{std(sp):.3f}  {avg(sr):>7.3f}±{std(sr):.3f}  {avg(aurocs):>7.3f}   {avg(edges):>5.1f}")

# ============================================================================
# PART 2: Summary table — Mean Skel-F1 per config, all methods
# ============================================================================
all_baseline_names = []
for data in results.values():
    for s in data["seeds"]:
        for bn in s.get("baselines", {}):
            if bn not in all_baseline_names:
                all_baseline_names.append(bn)

print(f"\n\n{'=' * W}")
print("SUMMARY TABLE: Mean Skeleton-F1 (±std) across seeds")
print(f"{'=' * W}")

col_w = 14
header = f"{'Config':<22} {'d':>3} {'n':>2}"
method_order = ["RC-GNN"] + all_baseline_names
for m in method_order:
    header += f" {m:>{col_w}}"
print(header)
print("-" * len(header))

for config_name, data in results.items():
    seeds = data["seeds"]
    d = data["d"]
    n = len(seeds)
    row = f"{config_name:<22} {d:>3} {n:>2}"

    # RC-GNN
    vals = [s["rc_gnn"]["topk"]["Skeleton_F1"] for s in seeds]
    row += f" {avg(vals):>{col_w}.3f}"

    for m in all_baseline_names:
        vals = []
        for s in seeds:
            b = s.get("baselines", {}).get(m)
            if b:
                vals.append(b["Skeleton_F1"])
        if vals:
            row += f" {avg(vals):>{col_w}.3f}"
        else:
            row += f" {'—':>{col_w}}"
    print(row)

# ============================================================================
# PART 3: Summary table — Mean Directed-F1 per config
# ============================================================================
print(f"\n\n{'=' * W}")
print("SUMMARY TABLE: Mean Directed-F1 (±std) across seeds")
print(f"{'=' * W}")

header = f"{'Config':<22} {'d':>3} {'n':>2}"
for m in method_order:
    header += f" {m:>{col_w}}"
print(header)
print("-" * len(header))

for config_name, data in results.items():
    seeds = data["seeds"]
    d = data["d"]
    n = len(seeds)
    row = f"{config_name:<22} {d:>3} {n:>2}"

    vals = [s["rc_gnn"]["topk"]["Directed_F1"] for s in seeds]
    row += f" {avg(vals):>{col_w}.3f}"

    for m in all_baseline_names:
        vals = []
        for s in seeds:
            b = s.get("baselines", {}).get(m)
            if b:
                vals.append(b["Directed_F1"])
        if vals:
            row += f" {avg(vals):>{col_w}.3f}"
        else:
            row += f" {'—':>{col_w}}"
    print(row)

# ============================================================================
# PART 4: Summary table — Mean SHD per config
# ============================================================================
print(f"\n\n{'=' * W}")
print("SUMMARY TABLE: Mean SHD (↓ better) across seeds")
print(f"{'=' * W}")

header = f"{'Config':<22} {'d':>3} {'n':>2}"
for m in method_order:
    header += f" {m:>{col_w}}"
print(header)
print("-" * len(header))

for config_name, data in results.items():
    seeds = data["seeds"]
    d = data["d"]
    n = len(seeds)
    row = f"{config_name:<22} {d:>3} {n:>2}"

    vals = [s["rc_gnn"]["topk"]["SHD"] for s in seeds]
    row += f" {avg(vals):>{col_w}.1f}"

    for m in all_baseline_names:
        vals = []
        for s in seeds:
            b = s.get("baselines", {}).get(m)
            if b:
                vals.append(b["SHD"])
        if vals:
            row += f" {avg(vals):>{col_w}.1f}"
        else:
            row += f" {'—':>{col_w}}"
    print(row)

# ============================================================================
# PART 5: Win/Loss count
# ============================================================================
print(f"\n\n{'=' * W}")
print("WIN/LOSS ANALYSIS: How often RC-GNN beats each baseline (by Skel-F1)")
print(f"{'=' * W}")

for bn in all_baseline_names:
    wins = 0; losses = 0; ties = 0; total = 0
    for config_name, data in results.items():
        for s in data["seeds"]:
            rcgnn_f1 = s["rc_gnn"]["topk"]["Skeleton_F1"]
            b = s.get("baselines", {}).get(bn)
            if b:
                total += 1
                bf1 = b["Skeleton_F1"]
                if rcgnn_f1 > bf1 + 0.001:
                    wins += 1
                elif bf1 > rcgnn_f1 + 0.001:
                    losses += 1
                else:
                    ties += 1
    pct = wins / total * 100 if total else 0
    print(f"  vs {bn:<18}: RC-GNN wins {wins:>3}/{total} ({pct:>5.1f}%), loses {losses:>3}/{total}, ties {ties:>3}/{total}")

# ============================================================================
# PART 6: Overall aggregates
# ============================================================================
print(f"\n\n{'=' * W}")
print("OVERALL AGGREGATE (mean across ALL configs and seeds)")
print(f"{'=' * W}")

all_rcgnn_skf1 = []
all_rcgnn_df1 = []
all_rcgnn_shd = []
baseline_agg = {}

for config_name, data in results.items():
    for s in data["seeds"]:
        all_rcgnn_skf1.append(s["rc_gnn"]["topk"]["Skeleton_F1"])
        all_rcgnn_df1.append(s["rc_gnn"]["topk"]["Directed_F1"])
        all_rcgnn_shd.append(s["rc_gnn"]["topk"]["SHD"])
        for bn, b in s.get("baselines", {}).items():
            if bn not in baseline_agg:
                baseline_agg[bn] = {"skf1": [], "df1": [], "shd": []}
            baseline_agg[bn]["skf1"].append(b["Skeleton_F1"])
            baseline_agg[bn]["df1"].append(b["Directed_F1"])
            baseline_agg[bn]["shd"].append(b["SHD"])

print(f"{'Method':<20} {'Mean Skel-F1':>14} {'Mean Dir-F1':>14} {'Mean SHD':>12} {'N':>5}")
print("-" * 70)
print(f"{'RC-GNN (TopK)':<20} {avg(all_rcgnn_skf1):>11.3f}    {avg(all_rcgnn_df1):>11.3f}    {avg(all_rcgnn_shd):>9.1f}  {len(all_rcgnn_skf1):>5}")

for bn in all_baseline_names:
    if bn in baseline_agg:
        ba = baseline_agg[bn]
        print(f"{bn:<20} {avg(ba['skf1']):>11.3f}    {avg(ba['df1']):>11.3f}    {avg(ba['shd']):>9.1f}  {len(ba['skf1']):>5}")

# ============================================================================
# PART 7: AUROC / AUPRC (threshold-free, oracle-K-free)
# ============================================================================
print(f"\n\n{'=' * W}")
print("THRESHOLD-FREE METRICS: AUROC / AUPRC (no K needed)")
print(f"{'=' * W}")

header = f"{'Config':<22} {'d':>3}  {'RC-GNN AUROC':>14} {'RC-GNN AUPRC':>14}"
score_baselines_avail = []
for bn in all_baseline_names:
    # Check if any seed has AUROC for this baseline
    has_auroc = False
    for data in results.values():
        for s in data["seeds"]:
            b = s.get("baselines", {}).get(bn, {})
            if safe_float(b.get("AUROC", 0)) > 0:
                has_auroc = True
                break
        if has_auroc:
            break
    if has_auroc:
        score_baselines_avail.append(bn)
        header += f" {bn + ' AUROC':>16}"
print(header)
print("-" * len(header))

for config_name, data in results.items():
    seeds = data["seeds"]
    d = data["d"]
    rc_aurocs = [safe_float(s["rc_gnn"].get("auroc", s["rc_gnn"]["topk"].get("AUROC", 0))) for s in seeds]
    rc_auprcs = [safe_float(s["rc_gnn"].get("auprc", s["rc_gnn"]["topk"].get("AUPRC", 0))) for s in seeds]
    row = f"{config_name:<22} {d:>3}  {avg(rc_aurocs):>10.3f}±{std(rc_aurocs):.3f} {avg(rc_auprcs):>10.3f}±{std(rc_auprcs):.3f}"
    for bn in score_baselines_avail:
        vals = [safe_float(s.get("baselines", {}).get(bn, {}).get("AUROC", 0)) for s in seeds if s.get("baselines", {}).get(bn)]
        if vals:
            row += f" {avg(vals):>12.3f}±{std(vals):.3f}"
        else:
            row += f" {'—':>16}"
    print(row)

# ============================================================================
# PART 8: K-ROBUSTNESS (misspecified K performance)
# ============================================================================
print(f"\n\n{'=' * W}")
print("K-ROBUSTNESS: RC-GNN Skel-F1 under misspecified K (K*ratio)")
print("  Shows performance is NOT dependent on oracle K")
print(f"{'=' * W}")

header = f"{'Config':<22} {'d':>3}  {'K×0.5':>10} {'K×0.8':>10} {'K×1.0★':>10} {'K×1.2':>10} {'K×1.5':>10}"
print(header)
print("-" * len(header))

for config_name, data in results.items():
    seeds = data["seeds"]
    d = data["d"]
    row = f"{config_name:<22} {d:>3} "
    for ratio_key in ["K_0.5", "K_0.8", "K_1.0", "K_1.2", "K_1.5"]:
        vals = []
        for s in seeds:
            kr = s.get("rc_gnn", {}).get("k_robustness", {}).get(ratio_key, {})
            if kr:
                vals.append(kr.get("Skeleton_F1", 0))
        if vals:
            row += f" {avg(vals):>10.3f}"
        else:
            row += f" {'—':>10}"
    print(row)

print(f"\n{'=' * W}")
print("END OF COMPREHENSIVE ANALYSIS")
print(f"{'=' * W}")
