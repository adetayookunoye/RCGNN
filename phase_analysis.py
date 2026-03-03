#!/usr/bin/env python3
"""Comprehensive Phase 1 & Phase 2 analysis for RC-GNN pipeline."""
import json
import os
from pathlib import Path

def safe_mean(vals):
    return sum(vals) / len(vals) if vals else 0.0

def safe_std(vals):
    if len(vals) < 2:
        return 0.0
    m = safe_mean(vals)
    return (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5

def main():
    print("=" * 90)
    print("         COMPREHENSIVE PHASE 1 & PHASE 2 ANALYSIS — RC-GNN V9.2.12")
    print("=" * 90)

    # ═══════════════════════════════════════════════════════════════
    # PHASE 1: UCI TRAINING
    # ═══════════════════════════════════════════════════════════════
    print()
    print("━" * 90)
    print("  PHASE 1: UCI TRAINING  (12 corruption configs × 5 seeds = 60 runs)")
    print("━" * 90)

    uci_root = Path("artifacts/uci_multiseed")
    uci_configs = {}

    if uci_root.exists():
        for cfg_dir in sorted(uci_root.iterdir()):
            if not cfg_dir.is_dir():
                continue
            info = {"train": 0, "eval": 0, "temporal": 0, "f1s": [], "shds": [],
                    "dir_f1s": [], "skel_f1s": [], "baselines": {}}
            for sd in sorted(cfg_dir.iterdir()):
                if not sd.is_dir():
                    continue
                # Training
                hp = sd / "training_history.json"
                if hp.exists():
                    info["train"] += 1
                    with open(hp) as f:
                        hist = json.load(f)
                    if isinstance(hist, list) and len(hist) > 0:
                        best_f1 = max(h.get("topk_f1", 0) for h in hist)
                        best_shd = min(h.get("topk_shd", 9999) for h in hist)
                        info["f1s"].append(best_f1)
                        info["shds"].append(best_shd)

                # Evaluation
                ep = sd / "evaluation.json"
                if ep.exists():
                    info["eval"] += 1
                    with open(ep) as f:
                        ev = json.load(f)
                    # RC-GNN metrics from evaluation
                    rc = ev.get("rc_gnn", {}).get("topk", {})
                    if "Skeleton_F1" in rc:
                        info["skel_f1s"].append(rc["Skeleton_F1"])
                    if "Directed_F1" in rc:
                        info["dir_f1s"].append(rc["Directed_F1"])
                    # Baselines
                    bl = ev.get("baselines", {})
                    bl_lower = {k.lower(): k for k in bl}
                    has_temporal = "granger" in bl_lower or "pcmci+" in bl_lower or "pcmci_plus" in bl_lower
                    if has_temporal:
                        info["temporal"] += 1
                    for bname, bmetrics in bl.items():
                        if bname not in info["baselines"]:
                            info["baselines"][bname] = {"skel": [], "dir": [], "shd": []}
                        if isinstance(bmetrics, dict):
                            if "Skeleton_F1" in bmetrics:
                                info["baselines"][bname]["skel"].append(bmetrics["Skeleton_F1"])
                            if "Directed_F1" in bmetrics:
                                info["baselines"][bname]["dir"].append(bmetrics["Directed_F1"])
                            if "SHD" in bmetrics:
                                info["baselines"][bname]["shd"].append(bmetrics["SHD"])

            uci_configs[cfg_dir.name] = info

    total_trained = sum(v["train"] for v in uci_configs.values())
    total_evaled = sum(v["eval"] for v in uci_configs.values())
    total_temporal = sum(v["temporal"] for v in uci_configs.values())

    print(f"\n  Training:    {total_trained}/60 complete")
    print(f"  Evaluation:  {total_evaled}/60 complete")
    print(f"  Temporal:    {total_temporal}/60 (Granger/PCMCI+)")
    print()

    # Per-config table
    print(f"  {'Config':<30s}  {'Train':>5s}  {'Eval':>5s}  {'Best F1 (train)':>16s}  {'Skel F1 (eval)':>16s}  {'Dir F1 (eval)':>16s}")
    print(f"  {'─'*30}  {'─'*5}  {'─'*5}  {'─'*16}  {'─'*16}  {'─'*16}")

    for cfg_name in sorted(uci_configs.keys()):
        info = uci_configs[cfg_name]
        tr = f"{info['train']}/5"
        ev = f"{info['eval']}/5"
        if info["f1s"]:
            bf1 = f"{safe_mean(info['f1s']):.4f}±{safe_std(info['f1s']):.4f}"
        else:
            bf1 = "—"
        if info["skel_f1s"]:
            sf1 = f"{safe_mean(info['skel_f1s']):.4f}±{safe_std(info['skel_f1s']):.4f}"
        else:
            sf1 = "—"
        if info["dir_f1s"]:
            df1 = f"{safe_mean(info['dir_f1s']):.4f}±{safe_std(info['dir_f1s']):.4f}"
        else:
            df1 = "—"
        print(f"  {cfg_name:<30s}  {tr:>5s}  {ev:>5s}  {bf1:>16s}  {sf1:>16s}  {df1:>16s}")

    # UCI baseline comparison for configs with evals
    print()
    print("  UCI Baseline Comparison (across evaluated configs):")
    all_bl_names = set()
    for info in uci_configs.values():
        all_bl_names.update(info["baselines"].keys())

    if all_bl_names:
        # Aggregate per method
        method_agg = {}
        for bname in sorted(all_bl_names):
            all_skel = []
            all_dir = []
            for info in uci_configs.values():
                if bname in info["baselines"]:
                    all_skel.extend(info["baselines"][bname]["skel"])
                    all_dir.extend(info["baselines"][bname]["dir"])
            method_agg[bname] = (safe_mean(all_skel), safe_mean(all_dir), len(all_skel))

        # Also aggregate RC-GNN from eval
        rc_all_skel = []
        rc_all_dir = []
        for info in uci_configs.values():
            rc_all_skel.extend(info["skel_f1s"])
            rc_all_dir.extend(info["dir_f1s"])
        method_agg["RC-GNN"] = (safe_mean(rc_all_skel), safe_mean(rc_all_dir), len(rc_all_skel))

        print(f"  {'Method':<20s}  {'Avg Skel F1':>12s}  {'Avg Dir F1':>12s}  {'# evals':>8s}")
        print(f"  {'─'*20}  {'─'*12}  {'─'*12}  {'─'*8}")
        for method in sorted(method_agg.keys(), key=lambda x: -method_agg[x][1]):
            s, d, n = method_agg[method]
            tag = " ★" if method == "RC-GNN" else ""
            print(f"  {method:<20s}  {s:>12.4f}  {d:>12.4f}  {n:>8d}{tag}")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 2: SEM TABLE 2
    # ═══════════════════════════════════════════════════════════════
    print()
    print("━" * 90)
    print("  PHASE 2: SEM TABLE 2  (15 configs × 5 seeds = 75 runs)")
    print("━" * 90)

    # SEM results live in TWO directory layouts
    sem_dirs = {
        "sem_table2": Path("artifacts/sem_table2"),
        "table2a": Path("artifacts/table2a"),
        "table2c": Path("artifacts/table2c"),
    }

    sem_configs = {}  # config_name -> {source, skel_f1s, dir_f1s, shds, baselines, ...}

    for source, root in sem_dirs.items():
        if not root.exists():
            continue
        for cfg_dir in sorted(root.iterdir()):
            if not cfg_dir.is_dir():
                continue
            cfg_name = cfg_dir.name
            info = {"source": source, "trained": 0, "evaluated": 0,
                    "skel_f1s": [], "dir_f1s": [], "shds": [],
                    "baselines": {}}

            for sd in sorted(cfg_dir.iterdir()):
                if not sd.is_dir():
                    continue
                # Check training artifacts
                if (sd / "A_best.npy").exists() or (sd / "training_history.json").exists() or (sd / "best_model.pt").exists():
                    info["trained"] += 1

                # Evaluation
                ep = sd / "evaluation.json"
                if ep.exists():
                    info["evaluated"] += 1
                    with open(ep) as f:
                        ev = json.load(f)
                    rc = ev.get("rc_gnn", {}).get("topk", {})
                    if "Skeleton_F1" in rc:
                        info["skel_f1s"].append(rc["Skeleton_F1"])
                    if "Directed_F1" in rc:
                        info["dir_f1s"].append(rc["Directed_F1"])
                    if "SHD" in rc:
                        info["shds"].append(rc["SHD"])

                    for bname, bmetrics in ev.get("baselines", {}).items():
                        if bname not in info["baselines"]:
                            info["baselines"][bname] = {"skel": [], "dir": [], "shd": []}
                        if isinstance(bmetrics, dict):
                            if "Skeleton_F1" in bmetrics:
                                info["baselines"][bname]["skel"].append(bmetrics["Skeleton_F1"])
                            if "Directed_F1" in bmetrics:
                                info["baselines"][bname]["dir"].append(bmetrics["Directed_F1"])
                            if "SHD" in bmetrics:
                                info["baselines"][bname]["shd"].append(bmetrics["SHD"])

            sem_configs[f"{source}/{cfg_name}"] = info

    total_sem_trained = sum(v["trained"] for v in sem_configs.values())
    total_sem_evaled = sum(v["evaluated"] for v in sem_configs.values())
    print(f"\n  Configs found: {len(sem_configs)}")
    print(f"  Total trained: {total_sem_trained}")
    print(f"  Total evaluated: {total_sem_evaled}")
    print()

    # Group by paper table
    table2a_configs = []  # d=13 ER/SF configs
    table2b_configs = []  # d=20 ER/SF configs
    table2c_configs = []  # d=50 or compound
    hypothesis_configs = []  # h1/h2/h3

    for key, info in sorted(sem_configs.items()):
        source, cfg = key.split("/", 1)
        if source == "table2a":
            hypothesis_configs.append((key, info))
        elif source == "table2c":
            table2c_configs.append((key, info))
        elif "d13" in cfg or "sf_d13" in cfg:
            table2a_configs.append((key, info))
        elif "d20" in cfg or "sf_d20" in cfg:
            table2b_configs.append((key, info))
        elif "d50" in cfg:
            table2c_configs.append((key, info))
        else:
            table2a_configs.append((key, info))

    def print_sem_group(title, items):
        if not items:
            return
        print(f"\n  ┌─ {title}")
        print(f"  │ {'Config':<40s}  {'Tr':>3s}  {'Ev':>3s}  {'Skel F1':>14s}  {'Dir F1':>14s}  {'SHD':>10s}")
        print(f"  │ {'─'*40}  {'─'*3}  {'─'*3}  {'─'*14}  {'─'*14}  {'─'*10}")

        wins_skel = 0
        wins_dir = 0

        for key, info in items:
            cfg = key.split("/", 1)[1]
            tr = str(info["trained"])
            ev = str(info["evaluated"])
            sf = f"{safe_mean(info['skel_f1s']):.4f}±{safe_std(info['skel_f1s']):.4f}" if info["skel_f1s"] else "—"
            df = f"{safe_mean(info['dir_f1s']):.4f}±{safe_std(info['dir_f1s']):.4f}" if info["dir_f1s"] else "—"
            sh = f"{safe_mean(info['shds']):.1f}" if info["shds"] else "—"
            print(f"  │ {cfg:<40s}  {tr:>3s}  {ev:>3s}  {sf:>14s}  {df:>14s}  {sh:>10s}")

            # Check wins vs baselines
            rc_skel = safe_mean(info["skel_f1s"]) if info["skel_f1s"] else -1
            rc_dir = safe_mean(info["dir_f1s"]) if info["dir_f1s"] else -1
            best_bl_skel = 0
            best_bl_dir = 0
            for bname, bm in info["baselines"].items():
                if bm["skel"]:
                    best_bl_skel = max(best_bl_skel, safe_mean(bm["skel"]))
                if bm["dir"]:
                    best_bl_dir = max(best_bl_dir, safe_mean(bm["dir"]))
            if rc_skel > best_bl_skel and rc_skel > 0:
                wins_skel += 1
            if rc_dir > best_bl_dir and rc_dir > 0:
                wins_dir += 1

        evaluated_items = [i for i in items if i[1]["evaluated"] > 0]
        print(f"  │")
        print(f"  │ RC-GNN wins: Skel F1 {wins_skel}/{len(evaluated_items)}, Dir F1 {wins_dir}/{len(evaluated_items)}")
        print(f"  └{'─'*88}")

    print_sem_group("Table 2A — ER/SF d=13 configs", table2a_configs)
    print_sem_group("Table 2B — ER/SF d=20 configs", table2b_configs)
    print_sem_group("Table 2C — d=50 / compound configs", table2c_configs)
    print_sem_group("Hypothesis configs (h1/h2/h3)", hypothesis_configs)

    # Cross-method comparison for SEM
    print()
    print("  SEM Cross-Method Ranking (all evaluated configs):")
    all_sem_bl = set()
    for info in sem_configs.values():
        all_sem_bl.update(info["baselines"].keys())

    sem_method_agg = {}
    # RC-GNN
    rc_s = [v for info in sem_configs.values() for v in info["skel_f1s"]]
    rc_d = [v for info in sem_configs.values() for v in info["dir_f1s"]]
    rc_shd = [v for info in sem_configs.values() for v in info["shds"]]
    sem_method_agg["RC-GNN"] = (safe_mean(rc_s), safe_mean(rc_d), safe_mean(rc_shd), len(rc_s))

    for bname in sorted(all_sem_bl):
        s = [v for info in sem_configs.values() for v in info["baselines"].get(bname, {}).get("skel", [])]
        d = [v for info in sem_configs.values() for v in info["baselines"].get(bname, {}).get("dir", [])]
        shd = [v for info in sem_configs.values() for v in info["baselines"].get(bname, {}).get("shd", [])]
        sem_method_agg[bname] = (safe_mean(s), safe_mean(d), safe_mean(shd), len(s))

    print(f"  {'Method':<20s}  {'Avg Skel F1':>12s}  {'Avg Dir F1':>12s}  {'Avg SHD':>10s}  {'# evals':>8s}")
    print(f"  {'─'*20}  {'─'*12}  {'─'*12}  {'─'*10}  {'─'*8}")
    for method in sorted(sem_method_agg.keys(), key=lambda x: -sem_method_agg[x][1]):
        s, d, shd, n = sem_method_agg[method]
        tag = " ★" if method == "RC-GNN" else ""
        print(f"  {method:<20s}  {s:>12.4f}  {d:>12.4f}  {shd:>10.1f}  {n:>8d}{tag}")

    # ═══════════════════════════════════════════════════════════════
    # ISSUES & GAPS
    # ═══════════════════════════════════════════════════════════════
    print()
    print("━" * 90)
    print("  ISSUES & GAPS")
    print("━" * 90)

    # er_d50_mlp
    print()
    for key, info in sorted(sem_configs.items()):
        if info["trained"] > 0 and info["evaluated"] == 0:
            print(f"  ⚠  {key}: trained={info['trained']} but evaluated=0 — needs evaluation re-run")

    # UCI missing temporal
    uci_missing_temporal = []
    for cfg_name, info in sorted(uci_configs.items()):
        if info["eval"] > 0 and info["temporal"] < info["eval"]:
            uci_missing_temporal.append(f"{cfg_name} ({info['temporal']}/{info['eval']})")
    if uci_missing_temporal:
        print(f"  ⚠  UCI configs missing temporal baselines:")
        for m in uci_missing_temporal:
            print(f"       {m}")

    # Missing UCI evals
    uci_no_eval = [c for c, i in uci_configs.items() if i["eval"] == 0]
    if uci_no_eval:
        print(f"  ⚠  UCI configs with NO evaluation:")
        for c in uci_no_eval:
            print(f"       {c}")

    # ═══════════════════════════════════════════════════════════════
    # PIPELINE PROGRESS
    # ═══════════════════════════════════════════════════════════════
    print()
    print("━" * 90)
    print("  PIPELINE PROGRESS SUMMARY")
    print("━" * 90)

    # Check ablation
    abl_root = Path("artifacts/ablation")
    abl_count = 0
    if abl_root.exists():
        for cfg_dir in abl_root.rglob("training_history.json"):
            abl_count += 1

    # Check Phase 3 tasks
    tasks_file = Path("artifacts/ablation/tasks.tsv")
    total_abl_tasks = 0
    if tasks_file.exists():
        with open(tasks_file) as f:
            lines = f.readlines()
            total_abl_tasks = len(lines) - 1  # header

    print(f"""
  Phase 0 (Data):          ✅ Complete (12 corruption datasets)
  Phase 1 (UCI Train):     ✅ {total_trained}/60 complete
  Phase 1 (UCI Eval):      {'✅' if total_evaled == 60 else '⚠️'} {total_evaled}/60 (temporal: {total_temporal}/60)
  Phase 2 (SEM Train):     ✅ {total_sem_trained} trained
  Phase 2 (SEM Eval):      {'✅' if total_sem_evaled >= 70 else '⚠️'} {total_sem_evaled} evaluated
  Phase 3 (Ablation):      🔄 {abl_count}/{total_abl_tasks if total_abl_tasks else '99'} complete
  Phase 4 (Robustness):    ⏳ Pending
  Phase 5 (Plots/LaTeX):   ⏳ Pending
""")

    print("=" * 90)


if __name__ == "__main__":
    main()
