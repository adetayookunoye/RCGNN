#!/usr/bin/env python3
"""
Aggregate Table 2 evaluation results into summary tables.

Reads every evaluation.json under artifacts/table2a, artifacts/sem_table2, 
artifacts/table2c and produces:
  - artifacts/table2/table2_all_runs.csv (one row per method per dataset per seed)
  - artifacts/table2/table2_summary_meanstd.csv (mean/std across seeds)
  - artifacts/table2/table2A.tex, table2B.tex, table2C.tex (LaTeX tables)

Usage:
    python scripts/aggregate_table2.py
    python scripts/aggregate_table2.py --artifacts_root artifacts --out_dir artifacts/table2
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

import pandas as pd

# ============================================================================
# Expected experiments (must match generate_table2_tasks.py)
# ============================================================================
EXPECTED_EXPERIMENTS = {
    "2A": [
        ("h1_easy", list(range(5))),
        ("h1_medium", list(range(5))),
        ("h1_hard", list(range(5))),
        ("h2_multi_env", list(range(5))),
        ("h2_stability", list(range(5))),
        ("h3_policy", list(range(5))),
    ],
    "2B": [
        ("er_d13_lin", list(range(5))),
        ("er_d13_mlp", list(range(5))),
        ("er_d20_lin", list(range(5))),
        ("er_d20_mlp", list(range(5))),
        ("er_d50_mlp", list(range(5))),
        ("sf_d13_mlp", list(range(5))),
        ("sf_d20_mlp", list(range(5))),
        ("sf_d13_hard", list(range(5))),
    ],
    "2C": [
        ("compound_sem_medium", list(range(5))),
    ],
}


def find_eval_files(art_root: Path):
    """Find all evaluation.json files recursively."""
    return list(art_root.rglob("evaluation.json"))


def check_missing_experiments(art_root: Path):
    """
    Check for missing experiments and print warnings.
    Returns dict with {table: [(config, seed), ...]} of missing experiments.
    """
    missing = {"2A": [], "2B": [], "2C": []}
    
    # Table 2A: artifacts/table2a/{config}/seed_{N}/evaluation.json
    for config, seeds in EXPECTED_EXPERIMENTS["2A"]:
        for seed in seeds:
            eval_path = art_root / "table2a" / config / f"seed_{seed}" / "evaluation.json"
            if not eval_path.exists():
                missing["2A"].append((config, seed))
    
    # Table 2B: artifacts/sem_table2/{config}/seed_{N}/evaluation.json
    for config, seeds in EXPECTED_EXPERIMENTS["2B"]:
        for seed in seeds:
            eval_path = art_root / "sem_table2" / config / f"seed_{seed}" / "evaluation.json"
            if not eval_path.exists():
                missing["2B"].append((config, seed))
    
    # Table 2C: artifacts/table2c/{config}/seed_{N}/evaluation.json
    for config, seeds in EXPECTED_EXPERIMENTS["2C"]:
        for seed in seeds:
            eval_path = art_root / "table2c" / config / f"seed_{seed}" / "evaluation.json"
            if not eval_path.exists():
                missing["2C"].append((config, seed))
    
    return missing


def parse_one(eval_path: Path):
    """Parse one evaluation.json and return list of rows."""
    art_dir = eval_path.parent
    meta_path = art_dir / "run_meta.json"
    
    if not meta_path.exists():
        print(f"[WARN] Missing run_meta.json for {eval_path}")
        return []

    try:
        meta = json.loads(meta_path.read_text())
        table = meta.get("table", "unknown")
        config = meta.get("config", "unknown")
        seed = int(meta.get("seed", -1))
    except Exception as e:
        print(f"[WARN] Failed to parse {meta_path}: {e}")
        return []

    try:
        results = json.loads(eval_path.read_text())
    except Exception as e:
        print(f"[WARN] Failed to parse {eval_path}: {e}")
        return []

    rows = []
    
    # Method 1: Look for baseline_comparison (list of dicts)
    comp = results.get("baseline_comparison", [])
    if isinstance(comp, list):
        for entry in comp:
            if not isinstance(entry, dict):
                continue
            row = {
                "table": table,
                "config": config,
                "seed": seed,
                "method": entry.get("Method", entry.get("method", "Unknown")),
                "SHD": entry.get("SHD", entry.get("shd", None)),
                "Skeleton_F1": entry.get("Skeleton_F1", entry.get("skeleton_f1", None)),
                "Directed_F1": entry.get("Directed_F1", entry.get("directed_f1", entry.get("f1", None))),
                "AUROC": entry.get("AUROC", entry.get("auroc", None)),
                "AUPRC": entry.get("AUPRC", entry.get("auprc", None)),
                "Precision": entry.get("Precision", entry.get("precision", None)),
                "Recall": entry.get("Recall", entry.get("recall", None)),
            }
            rows.append(row)
    
    # Method 2: Look for rcgnn_metrics directly (if no baseline_comparison)
    if not rows and "rcgnn_metrics" in results:
        rcgnn = results["rcgnn_metrics"]
        if isinstance(rcgnn, dict):
            row = {
                "table": table,
                "config": config,
                "seed": seed,
                "method": "RC-GNN",
                "SHD": rcgnn.get("SHD", rcgnn.get("shd", None)),
                "Skeleton_F1": rcgnn.get("Skeleton_F1", rcgnn.get("skeleton_f1", None)),
                "Directed_F1": rcgnn.get("Directed_F1", rcgnn.get("directed_f1", rcgnn.get("f1", None))),
                "AUROC": rcgnn.get("AUROC", rcgnn.get("auroc", None)),
                "AUPRC": rcgnn.get("AUPRC", rcgnn.get("auprc", None)),
                "Precision": rcgnn.get("Precision", rcgnn.get("precision", None)),
                "Recall": rcgnn.get("Recall", rcgnn.get("recall", None)),
            }
            rows.append(row)
    
    # Method 3: Look for "baselines" dict and "rc_gnn" dict (new format)
    if not rows and ("baselines" in results or "rc_gnn" in results):
        # 3a. Process baselines
        baselines = results.get("baselines", {})
        if isinstance(baselines, dict):
            for method_name, metrics in baselines.items():
                if not isinstance(metrics, dict): continue
                row = {
                    "table": table,
                    "config": config,
                    "seed": seed,
                    "method": method_name,
                    "SHD": metrics.get("SHD"),
                    "Skeleton_F1": metrics.get("Skeleton_F1"),
                    "Directed_F1": metrics.get("Directed_F1"),
                    "AUROC": metrics.get("AUROC"),
                    "AUPRC": metrics.get("AUPRC"),
                    "Precision": metrics.get("Directed_Precision"), 
                    "Recall": metrics.get("Directed_Recall"),     
                }
                rows.append(row)
        
        # 3b. Process RC-GNN
        rc_gnn = results.get("rc_gnn", {})
        # RC-GNN might have "topk" sub-key containing the metrics
        if "topk" in rc_gnn:
            metrics = rc_gnn["topk"]
        else:
            metrics = rc_gnn
            
        if isinstance(metrics, dict) and metrics:
             row = {
                "table": table,
                "config": config,
                "seed": seed,
                "method": "RC-GNN",
                "SHD": metrics.get("SHD"),
                "Skeleton_F1": metrics.get("Skeleton_F1"),
                "Directed_F1": metrics.get("Directed_F1"),
                "AUROC": metrics.get("AUROC"),
                "AUPRC": metrics.get("AUPRC"),
                "Precision": metrics.get("Directed_Precision"),
                "Recall": metrics.get("Directed_Recall"),
            }
             rows.append(row)

    return rows


def format_mean_std(mean_val, std_val, precision=2):
    """Format as 'mean ± std'."""
    if pd.isna(mean_val):
        return "—"
    if pd.isna(std_val) or std_val == 0:
        return f"{mean_val:.{precision}f}"
    return f"{mean_val:.{precision}f} ± {std_val:.{precision}f}"


def to_latex_table(df, out_path: Path, caption: str, label: str = None):
    """Write DataFrame to LaTeX table."""
    if label is None:
        label = f"tab:{out_path.stem}"
    
    # Format for LaTeX
    latex = df.to_latex(
        index=False, 
        escape=True, 
        caption=caption, 
        label=label,
        column_format="l" * len(df.columns),
        float_format="%.3f",
    )
    
    # Add some styling
    latex = latex.replace("\\begin{table}", "\\begin{table}[htbp]")
    latex = latex.replace("\\centering", "\\centering\n\\small")
    
    out_path.write_text(latex)


def make_compact_table(df, table_id, out_dir: Path):
    """Create a compact LaTeX table with mean±std formatting."""
    sub = df[df["table"] == table_id].copy()
    if sub.empty:
        print(f"[WARN] No data for Table {table_id}")
        return
    
    num_cols = ["SHD", "Skeleton_F1", "Directed_F1", "AUROC"]
    
    # Pivot: rows = config, columns = method, values = metric
    # For compactness, show one metric (e.g., Directed_F1)
    agg = sub.groupby(["config", "method"]).agg({
        "Directed_F1": ["mean", "std"],
        "SHD": ["mean", "std"],
    }).reset_index()
    
    # Flatten column names
    agg.columns = [
        "_".join([str(c) for c in col if c]).rstrip("_") 
        for col in agg.columns.values
    ]
    
    # Format mean±std
    agg["F1"] = agg.apply(
        lambda r: format_mean_std(r["Directed_F1_mean"], r["Directed_F1_std"]), 
        axis=1
    )
    agg["SHD"] = agg.apply(
        lambda r: format_mean_std(r["SHD_mean"], r["SHD_std"], precision=1), 
        axis=1
    )
    
    # Pivot to wide format
    pivot = agg.pivot(index="config", columns="method", values="F1").reset_index()
    
    tex_path = out_dir / f"table{table_id}.tex"
    caption = {
        "2A": "Table 2A: Hypothesis Benchmarks (H1/H2/H3) — Directed F1 (mean ± std over 5 seeds)",
        "2B": "Table 2B: SEM Benchmark Grid — Directed F1 (mean ± std over 5 seeds)",
        "2C": "Table 2C: Causal Validity Ablation — Directed F1 (mean ± std over 5 seeds)",
    }.get(table_id, f"Table {table_id} Results")
    
    to_latex_table(pivot, tex_path, caption, label=f"tab:table{table_id}")
    return tex_path


def main():
    ap = argparse.ArgumentParser(
        description="Aggregate Table 2 evaluation results"
    )
    ap.add_argument("--artifacts_root", type=str, default="artifacts",
                    help="Root directory for artifacts")
    ap.add_argument("--out_dir", type=str, default="artifacts/table2",
                    help="Output directory for aggregated results")
    args = ap.parse_args()

    art_root = Path(args.artifacts_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" AGGREGATING TABLE 2 RESULTS")
    print("=" * 70)
    print(f"  Artifacts root: {art_root}")
    print(f"  Output dir:     {out_dir}")
    print()

    # Find all evaluation files
    eval_files = (
        find_eval_files(art_root / "table2a") +
        find_eval_files(art_root / "sem_table2") +
        find_eval_files(art_root / "table2c")
    )

    print(f"Found {len(eval_files)} evaluation.json files:")
    print(f"  table2a:    {len(find_eval_files(art_root / 'table2a'))}")
    print(f"  sem_table2: {len(find_eval_files(art_root / 'sem_table2'))}")
    print(f"  table2c:    {len(find_eval_files(art_root / 'table2c'))}")
    print()

    # ========================================================================
    # Check for missing experiments
    # ========================================================================
    missing = check_missing_experiments(art_root)
    total_missing = sum(len(v) for v in missing.values())
    total_expected = sum(
        sum(len(seeds) for _, seeds in configs) 
        for configs in EXPECTED_EXPERIMENTS.values()
    )
    
    if total_missing > 0:
        print("=" * 70)
        print(f" WARNING: {total_missing}/{total_expected} EXPERIMENTS MISSING")
        print("=" * 70)
        for table_id, items in missing.items():
            if items:
                print(f"\n  Table {table_id} ({len(items)} missing):")
                # Group by config
                by_config = {}
                for config, seed in items:
                    by_config.setdefault(config, []).append(seed)
                for config, seeds in by_config.items():
                    if len(seeds) == 5:
                        print(f"    - {config}: ALL seeds missing")
                    else:
                        print(f"    - {config}: seeds {seeds} missing")
        print()
        print("  These experiments may have failed or not yet run.")
        print("  Continuing with available data...")
        print()
    else:
        print(f"[OK] All {total_expected} expected experiments found.")
        print()

    # Parse all results
    all_rows = []
    for p in eval_files:
        rows = parse_one(p)
        all_rows.extend(rows)
        if rows:
            print(f"  [OK] {p.parent.name}: {len(rows)} methods")

    if not all_rows:
        print("\n[WARN] No evaluation.json files found or parsed successfully.")
        print("  Expected structure: artifacts/{table2a,sem_table2,table2c}/{config}/seed_N/evaluation.json")
        print()
        
        # Write missing experiments report even when nothing completed
        if total_missing > 0:
            missing_report_path = out_dir / "missing_experiments.txt"
            with open(missing_report_path, "w") as f:
                f.write(f"Missing Experiments Report\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n")
                f.write(f"Total missing: {total_missing}/{total_expected}\n\n")
                for table_id, items in missing.items():
                    if items:
                        f.write(f"Table {table_id}:\n")
                        for config, seed in items:
                            f.write(f"  - {config} seed {seed}\n")
                        f.write("\n")
            print(f"[OK] {missing_report_path}")
        
        print("\n" + "=" * 70)
        print(" NO DATA TO AGGREGATE - RUN EXPERIMENTS FIRST")
        print("=" * 70)
        raise SystemExit(0)  # exit gracefully with success code

    df = pd.DataFrame(all_rows)
    print(f"\nParsed {len(df)} total rows ({df['method'].nunique()} methods)")

    # ========================================================================
    # Output 1: All runs (one row per method per dataset per seed)
    # ========================================================================
    all_csv = out_dir / "table2_all_runs.csv"
    df.to_csv(all_csv, index=False)
    print(f"\n[OK] {all_csv}")

    # ========================================================================
    # Output 2: Summary with mean/std across seeds
    # ========================================================================
    num_cols = ["SHD", "Skeleton_F1", "Directed_F1", "AUROC", "AUPRC", "Precision", "Recall"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Compute mean and std
    available_cols = [c for c in num_cols if c in df.columns]
    summary = (
        df.groupby(["table", "config", "method"], as_index=False)[available_cols]
        .agg(["mean", "std"])
    )
    
    # Flatten multi-level columns
    summary.columns = [
        "_".join([str(c) for c in col if c]).rstrip("_") 
        for col in summary.columns.values
    ]
    
    summary_csv = out_dir / "table2_summary_meanstd.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"[OK] {summary_csv}")

    # ========================================================================
    # Output 3: LaTeX tables (one per table section)
    # ========================================================================
    for table_id in ["2A", "2B", "2C"]:
        tex_path = make_compact_table(df, table_id, out_dir)
        if tex_path:
            print(f"[OK] {tex_path}")

    # ========================================================================
    # Output 4: Missing experiments report (if any)
    # ========================================================================
    if total_missing > 0:
        missing_report_path = out_dir / "missing_experiments.txt"
        with open(missing_report_path, "w") as f:
            f.write(f"Missing Experiments Report\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Total missing: {total_missing}/{total_expected}\n\n")
            for table_id, items in missing.items():
                if items:
                    f.write(f"Table {table_id}:\n")
                    for config, seed in items:
                        f.write(f"  - {config} seed {seed}\n")
                    f.write("\n")
        print(f"[OK] {missing_report_path}")

    # ========================================================================
    # Output 5: Quick summary statistics
    # ========================================================================
    print("\n" + "=" * 70)
    print(" QUICK SUMMARY")
    print("=" * 70)
    
    for table_id in ["2A", "2B", "2C"]:
        sub = df[df["table"] == table_id]
        if sub.empty:
            continue
        
        rcgnn = sub[sub["method"].str.contains("RC-GNN", case=False, na=False)]
        if not rcgnn.empty and "Directed_F1" in rcgnn.columns:
            mean_f1 = rcgnn["Directed_F1"].mean()
            std_f1 = rcgnn["Directed_F1"].std()
            print(f"  Table {table_id} RC-GNN: F1 = {mean_f1:.3f} ± {std_f1:.3f}")
    
    print()
    print("=" * 70)
    print(f" AGGREGATION COMPLETE")
    print(f" Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
