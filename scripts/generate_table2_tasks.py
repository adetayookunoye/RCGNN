#!/usr/bin/env python3
"""
Generate task manifest for Table 2 experiments (SPIE 2026 paper).

Creates a single tasks.tsv that enumerates all (table, config, seed, data_dir, artifact_dir).
This file can be used by SLURM array jobs or sequential runners.

Usage:
    python scripts/generate_table2_tasks.py
    python scripts/generate_table2_tasks.py --seeds 0,1,2,3,4 --out artifacts/table2/tasks.tsv

Output: TSV file with columns:
    table    config    seed    data_dir    artifact_dir

Total tasks: 30 (2A) + 40 (2B) + 5 (2C) = 75
"""

import argparse
from pathlib import Path

# ============================================================================
# Table 2A: H1/H2/H3 Hypothesis Benchmarks (6 benchmarks × 5 seeds = 30)
# ============================================================================
HYP_BENCHES = [
    "h1_easy",       # H1: structural accuracy under low missingness
    "h1_medium",     # H1: moderate missingness, diverse corruption
    "h1_hard",       # H1: high missingness (40-60%), all corruption types
    "h2_multi_env",  # H2: stability across multiple environments
    "h2_stability",  # H2: invariance loss effectiveness
    "h3_policy",     # H3: policy-relevant pathway recovery
]

# ============================================================================
# Table 2B: SEM Benchmark Grid (8 configs × 5 seeds = 40)
# ============================================================================
TABLE2_SEM_CONFIGS = [
    "er_d13_lin",   # ER d=13, Linear, Medium corruption
    "er_d13_mlp",   # ER d=13, MLP, Medium corruption
    "er_d20_lin",   # ER d=20, Linear, Medium corruption
    "er_d20_mlp",   # ER d=20, MLP, Medium corruption
    "er_d50_mlp",   # ER d=50, MLP, Medium corruption
    "sf_d13_mlp",   # SF d=13, MLP, Medium corruption
    "sf_d20_mlp",   # SF d=20, MLP, Medium corruption
    "sf_d13_hard",  # SF d=13, MLP, Hard corruption (40% MNAR)
]

# ============================================================================
# Table 2C: Causal Validity Ablation (1 benchmark × 5 seeds = 5)
# ============================================================================
CAUSAL_VALIDITY_BENCHES = [
    "compound_sem_medium",  # Known DAG by construction, MNAR + drift + bias
]


def main():
    ap = argparse.ArgumentParser(
        description="Generate task manifest for Table 2 experiments"
    )
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4",
                    help="Comma-separated seeds (default: 0,1,2,3,4)")
    ap.add_argument("--data_root", type=str, default="data/interim",
                    help="Root directory for datasets")
    ap.add_argument("--art_root", type=str, default="artifacts",
                    help="Root directory for artifacts")
    ap.add_argument("--out", type=str, default="artifacts/table2/tasks.tsv",
                    help="Output TSV file path")
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip() != ""]
    data_root = Path(args.data_root)
    art_root = Path(args.art_root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    # ========================================================================
    # Table 2A: H1/H2/H3 Hypothesis Benchmarks
    # ========================================================================
    for b in HYP_BENCHES:
        for seed in seeds:
            data_dir = data_root / "table2a" / b / f"seed_{seed}"
            art_dir = art_root / "table2a" / b / f"seed_{seed}"
            rows.append(("2A", b, seed, str(data_dir), str(art_dir)))

    # ========================================================================
    # Table 2B: SEM Benchmark Grid
    # ========================================================================
    for cfg in TABLE2_SEM_CONFIGS:
        for seed in seeds:
            data_dir = data_root / "sem_table2" / cfg / f"seed_{seed}"
            art_dir = art_root / "sem_table2" / cfg / f"seed_{seed}"
            rows.append(("2B", cfg, seed, str(data_dir), str(art_dir)))

    # ========================================================================
    # Table 2C: Causal Validity Ablation
    # ========================================================================
    for b in CAUSAL_VALIDITY_BENCHES:
        for seed in seeds:
            data_dir = data_root / "table2c" / b / f"seed_{seed}"
            art_dir = art_root / "table2c" / b / f"seed_{seed}"
            rows.append(("2C", b, seed, str(data_dir), str(art_dir)))

    # Write TSV
    with out_path.open("w") as f:
        f.write("table\tconfig\tseed\tdata_dir\tartifact_dir\n")
        for r in rows:
            f.write("\t".join(map(str, r)) + "\n")

    print(f"[OK] Wrote {len(rows)} tasks → {out_path}")
    print()
    print("Breakdown:")
    print(f"  2A (H1/H2/H3):        {len(HYP_BENCHES)} benchmarks × {len(seeds)} seeds = {len(HYP_BENCHES) * len(seeds)}")
    print(f"  2B (SEM grid):        {len(TABLE2_SEM_CONFIGS)} configs × {len(seeds)} seeds = {len(TABLE2_SEM_CONFIGS) * len(seeds)}")
    print(f"  2C (Causal validity): {len(CAUSAL_VALIDITY_BENCHES)} benchmark × {len(seeds)} seeds = {len(CAUSAL_VALIDITY_BENCHES) * len(seeds)}")
    print(f"  ─────────────────────────────────────────────")
    print(f"  TOTAL:                {len(rows)} tasks")
    print()
    print("Usage with SLURM array:")
    print(f"  #SBATCH --array=1-{len(rows)}")
    print(f"  TASK=$(sed -n \"${{SLURM_ARRAY_TASK_ID}}p\" {out_path} | tail -1)")
    print()


if __name__ == "__main__":
    main()
