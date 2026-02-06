#!/usr/bin/env python3
"""
Run ONE Table 2 task (generate → train → evaluate).

This script processes a single row from tasks.tsv:
  1. Ensures dataset exists (generates if missing)
  2. Trains RC-GNN
  3. Runs comprehensive evaluation
  4. Writes run_meta.json and evaluation.json

Usage:
    # Direct invocation:
    python scripts/run_one_table2_task.py \\
        --table 2A --config h1_easy --seed 0 \\
        --data_dir data/interim/table2a/h1_easy/seed_0 \\
        --artifact_dir artifacts/table2a/h1_easy/seed_0

    # From SLURM array job (see slurm/run_table2_array.sh):
    TASK=$(sed -n "${SLURM_ARRAY_TASK_ID}p" artifacts/table2/tasks.tsv)
    python scripts/run_one_table2_task.py ... (parsed from TASK)

Outputs:
    {artifact_dir}/run_meta.json     - Task metadata
    {artifact_dir}/evaluation.json   - Evaluation results
    {artifact_dir}/rcgnn_best.pt     - Best model checkpoint
    {artifact_dir}/A_mean.npy        - Learned adjacency
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def sh(cmd, cwd=None):
    """Run a shell command and return the exit code."""
    print(f"\n$ {' '.join(cmd)}")
    sys.stdout.flush()
    p = subprocess.run(cmd, cwd=cwd)
    return p.returncode


def ensure_dir(p: Path):
    """Create directory if it doesn't exist."""
    p.mkdir(parents=True, exist_ok=True)


def dataset_ready(data_dir: Path) -> bool:
    """Check if dataset has been generated."""
    # All generators output A_true.npy as the ground truth DAG
    # Table 2A also outputs X_train.npy, Table 2B/2C output X.npy
    required = ["A_true.npy"]
    return all((data_dir / n).exists() for n in required)


def generate_dataset(table, config, seed, data_dir: Path):
    """Generate dataset using the appropriate script."""
    ensure_dir(data_dir)

    if table == "2A":
        # H1/H2/H3 hypothesis benchmarks
        # synth_corruption_benchmark.py --benchmark {h1_easy,...}
        return sh([
            "python", "scripts/synth_corruption_benchmark.py",
            "--benchmark", config,
            "--seed", str(seed),
            "--output", str(data_dir),
        ])
    elif table == "2B":
        # SEM benchmark grid
        # synth_corruption_benchmark.py --table2 --config {er_d13_lin,...}
        # Note: --seeds expects comma-separated, but we pass single seed
        return sh([
            "python", "scripts/synth_corruption_benchmark.py",
            "--table2",
            "--config", config,
            "--seeds", str(seed),
            "--output", str(data_dir.parent.parent),  # Output to sem_table2/, script adds config/seed_N
        ])
    elif table == "2C":
        # Causal validity ablation (known DAG by construction)
        # run_synthetic_benchmark.py --benchmark compound_sem_medium
        return sh([
            "python", "scripts/run_synthetic_benchmark.py",
            "--benchmark", config,
            "--seed", str(seed),
            "--output", str(data_dir),
        ])
    else:
        raise ValueError(f"Unknown table: {table}")


def train_rcgnn(seed, data_dir: Path, art_dir: Path):
    """Train RC-GNN on the dataset."""
    ensure_dir(art_dir)

    # Try --data_dir first (unified script), then fallback to --data_root
    cmd1 = [
        "python", "scripts/train_rcgnn_unified.py",
        "--data_dir", str(data_dir),
        "--output_dir", str(art_dir),
        "--seed", str(seed),
        "--device", "cuda",
    ]
    rc = sh(cmd1)
    if rc == 0:
        return 0

    # Fallback: try --data_root
    print("[INFO] Retrying with --data_root flag...")
    cmd2 = [
        "python", "scripts/train_rcgnn_unified.py",
        "--data_root", str(data_dir),
        "--output_dir", str(art_dir),
        "--seed", str(seed),
        "--device", "cuda",
    ]
    return sh(cmd2)


def eval_all(data_dir: Path, art_dir: Path):
    """Run comprehensive evaluation."""
    ensure_dir(art_dir)
    out_json = art_dir / "evaluation.json"
    
    # Try different argument combinations for comprehensive_evaluation.py
    cmd = [
        "python", "scripts/comprehensive_evaluation.py",
        "--artifacts-dir", str(art_dir),
        "--data-dir", str(data_dir),
        "--output", str(out_json),
    ]
    rc = sh(cmd)
    return rc


def main():
    ap = argparse.ArgumentParser(
        description="Run ONE Table 2 task (generate → train → evaluate)"
    )
    ap.add_argument("--table", required=True, choices=["2A", "2B", "2C"],
                    help="Table section (2A=H1/H2/H3, 2B=SEM grid, 2C=validity)")
    ap.add_argument("--config", required=True,
                    help="Benchmark or config name")
    ap.add_argument("--seed", type=int, required=True,
                    help="Random seed")
    ap.add_argument("--data_dir", required=True,
                    help="Dataset directory")
    ap.add_argument("--artifact_dir", required=True,
                    help="Artifact output directory")
    ap.add_argument("--skip_generate", action="store_true",
                    help="Skip dataset generation (assume exists)")
    ap.add_argument("--skip_train", action="store_true",
                    help="Skip training (run eval only)")
    ap.add_argument("--skip_eval", action="store_true",
                    help="Skip evaluation (train only)")
    args = ap.parse_args()

    # Change to repo root
    repo = Path(__file__).resolve().parent.parent
    os.chdir(repo)

    data_dir = Path(args.data_dir)
    art_dir = Path(args.artifact_dir)
    ensure_dir(data_dir)
    ensure_dir(art_dir)

    # Write run metadata
    run_meta = {
        "table": args.table,
        "config": args.config,
        "seed": args.seed,
        "data_dir": str(data_dir),
        "artifact_dir": str(art_dir),
        "started_at": datetime.now().isoformat(),
        "status": "running",
    }
    meta_path = art_dir / "run_meta.json"
    meta_path.write_text(json.dumps(run_meta, indent=2))

    print("=" * 70)
    print(f" TABLE 2 TASK: {args.table} / {args.config} / seed={args.seed}")
    print("=" * 70)
    print(f"  Data dir:     {data_dir}")
    print(f"  Artifact dir: {art_dir}")
    print("=" * 70)

    try:
        # ====================================================================
        # Stage 1: Dataset Generation
        # ====================================================================
        if not args.skip_generate:
            if not dataset_ready(data_dir):
                print("\n[STAGE 1/3] Generating dataset...")
                rc = generate_dataset(args.table, args.config, args.seed, data_dir)
                if rc != 0:
                    run_meta["status"] = "failed_generation"
                    run_meta["error"] = f"Dataset generation failed (rc={rc})"
                    meta_path.write_text(json.dumps(run_meta, indent=2))
                    raise SystemExit(f"[FAIL] Dataset generation failed (rc={rc})")
                print("[OK] Dataset generated")
            else:
                print(f"\n[STAGE 1/3] Dataset already exists: {data_dir}")
        else:
            print("\n[STAGE 1/3] Skipping dataset generation (--skip_generate)")

        # ====================================================================
        # Stage 2: Training
        # ====================================================================
        if not args.skip_train:
            print("\n[STAGE 2/3] Training RC-GNN...")
            rc = train_rcgnn(args.seed, data_dir, art_dir)
            if rc != 0:
                run_meta["status"] = "failed_training"
                run_meta["error"] = f"Training failed (rc={rc})"
                meta_path.write_text(json.dumps(run_meta, indent=2))
                raise SystemExit(f"[FAIL] Training failed (rc={rc})")
            print("[OK] Training complete")
        else:
            print("\n[STAGE 2/3] Skipping training (--skip_train)")

        # ====================================================================
        # Stage 3: Evaluation
        # ====================================================================
        if not args.skip_eval:
            print("\n[STAGE 3/3] Running evaluation...")
            rc = eval_all(data_dir, art_dir)
            if rc != 0:
                print(f"[WARN] Evaluation returned rc={rc} (check logs)")
                run_meta["eval_warning"] = f"Evaluation returned rc={rc}"
            else:
                print("[OK] Evaluation complete")
        else:
            print("\n[STAGE 3/3] Skipping evaluation (--skip_eval)")

        # Success
        run_meta["status"] = "completed"
        run_meta["completed_at"] = datetime.now().isoformat()
        meta_path.write_text(json.dumps(run_meta, indent=2))

        print("\n" + "=" * 70)
        print(f"[DONE] {args.table} / {args.config} / seed={args.seed}")
        print(f"  Data:      {data_dir}")
        print(f"  Artifacts: {art_dir}")
        print("=" * 70)

    except Exception as e:
        run_meta["status"] = "failed"
        run_meta["error"] = str(e)
        run_meta["failed_at"] = datetime.now().isoformat()
        meta_path.write_text(json.dumps(run_meta, indent=2))
        raise


if __name__ == "__main__":
    main()
