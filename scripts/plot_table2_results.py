#!/usr/bin/env python3
"""
Plot Table 2 benchmark results from table2_all_runs.csv.

Generates publication-quality figures:
  1. Grouped bar charts per table (2A, 2B, 2C) for each metric
  2. Overall win-count heatmap (methods × metrics)
  3. RC-GNN vs best-baseline advantage gap
  4. Training loss curves (from training_history.json)
  5. F1 convergence across epochs
  6. Adjacency heatmaps (learned vs ground truth)
  7. Radar chart (mean metrics across all configs)

Usage:
    python scripts/plot_table2_results.py
    python scripts/plot_table2_results.py --csv artifacts/table2/table2_all_runs.csv
    python scripts/plot_table2_results.py --artifacts_root artifacts --out_dir artifacts/table2/plots
"""

import argparse
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================================
# CONFIG
# ============================================================================

METHOD_ORDER = ["RC-GNN", "Correlation", "PC", "GES", "NOTEARS", "NOTEARS-MLP", "GOLEM"]
METHOD_COLORS = {
    "RC-GNN":       "#e41a1c",   # red (ours)
    "Correlation":  "#377eb8",
    "PC":           "#4daf4a",
    "GES":          "#984ea3",
    "NOTEARS":      "#ff7f00",
    "NOTEARS-MLP":  "#a65628",
    "GOLEM":        "#999999",
}
METRICS_LOWER_BETTER = ["SHD"]

TABLE_LABELS = {
    "2A": "Table 3 — Hypothesis Benchmarks (H1/H2/H3)",
    "2B": "Table 3 — SEM Benchmark Grid",
    "2C": "Table 3 — Causal Validity Ablation",
}

CONFIG_DISPLAY = {
    # 2A
    "h1_easy": "H1-Easy", "h1_medium": "H1-Med", "h1_hard": "H1-Hard",
    "h2_multi_env": "H2-Env", "h2_stability": "H2-Stab", "h3_policy": "H3-Policy",
    # 2B
    "er_d13_lin": "ER-13-Lin", "er_d13_mlp": "ER-13-MLP",
    "er_d20_lin": "ER-20-Lin", "er_d20_mlp": "ER-20-MLP",
    "er_d50_mlp": "ER-50-MLP",
    "sf_d13_mlp": "SF-13-MLP", "sf_d20_mlp": "SF-20-MLP",
    "sf_d13_hard": "SF-13-Hard",
    # 2C
    "compound_sem_medium": "Compound-SEM",
}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_and_deduplicate(csv_path: str) -> pd.DataFrame:
    """Load CSV and drop duplicate rows (keep last per group)."""
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset=["table", "config", "seed", "method"], keep="last")
    df["AUROC"] = df["AUROC"].fillna(0.5)
    df["AUPRC"] = df["AUPRC"].fillna(0.0)
    return df


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std per (table, config, method)."""
    metrics = ["SHD", "Skeleton_F1", "Directed_F1", "AUROC", "AUPRC", "Precision", "Recall"]
    present = [m for m in metrics if m in df.columns]
    agg = df.groupby(["table", "config", "method"])[present].agg(["mean", "std"]).reset_index()
    agg.columns = ["_".join(c).rstrip("_") for c in agg.columns]
    return agg


# ============================================================================
# PLOT 1: Grouped bar charts per (table, metric)
# ============================================================================

def plot_grouped_bars(df: pd.DataFrame, out_dir: Path):
    """One figure per (table, metric) with grouped bars + error bars."""
    summary = compute_summary(df)

    for table_name in sorted(df["table"].unique()):
        tdf = summary[summary["table"] == table_name]
        configs = sorted(
            tdf["config"].unique(),
            key=lambda c: list(CONFIG_DISPLAY.keys()).index(c)
            if c in CONFIG_DISPLAY else 999,
        )

        for metric in ["Directed_F1", "Skeleton_F1", "AUROC", "SHD"]:
            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"
            if mean_col not in tdf.columns:
                continue

            methods = [m for m in METHOD_ORDER if m in tdf["method"].values]
            n_methods = len(methods)
            n_configs = len(configs)

            fig, ax = plt.subplots(figsize=(max(8, n_configs * 1.4), 5))
            bar_width = 0.8 / max(n_methods, 1)
            x = np.arange(n_configs)

            for i, method in enumerate(methods):
                mdf = tdf[tdf["method"] == method].set_index("config")
                vals = [mdf.loc[c, mean_col] if c in mdf.index else 0 for c in configs]
                errs = [mdf.loc[c, std_col] if c in mdf.index else 0 for c in configs]
                offset = (i - n_methods / 2 + 0.5) * bar_width
                ax.bar(
                    x + offset, vals, bar_width * 0.9,
                    yerr=errs, capsize=2,
                    label=method,
                    color=METHOD_COLORS.get(method, "#cccccc"),
                    edgecolor="white", linewidth=0.5,
                    alpha=1.0 if method == "RC-GNN" else 0.7,
                )

            ax.set_xticks(x)
            ax.set_xticklabels(
                [CONFIG_DISPLAY.get(c, c) for c in configs],
                rotation=30, ha="right", fontsize=9,
            )
            ax.set_ylabel(metric.replace("_", " "), fontsize=11)
            ax.set_title(
                f"{TABLE_LABELS.get(table_name, table_name)} — {metric.replace('_', ' ')}",
                fontsize=12, fontweight="bold",
            )
            ax.legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.8)
            ax.grid(axis="y", alpha=0.3)
            if metric in METRICS_LOWER_BETTER:
                ax.set_ylim(bottom=0)
            else:
                ax.set_ylim(0, 1.05)

            plt.tight_layout()
            fname = f"{table_name}_{metric}_bars.png"
            fig.savefig(out_dir / fname, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"  ✓ {fname}")


# ============================================================================
# PLOT 2: Win-count heatmap
# ============================================================================

def plot_win_heatmap(df: pd.DataFrame, out_dir: Path):
    """Heatmap: methods × metrics, cell = # configs where method is best."""
    summary = compute_summary(df)
    metrics = ["Directed_F1", "Skeleton_F1", "AUROC", "SHD"]
    methods = [m for m in METHOD_ORDER if m in summary["method"].values]

    win_matrix = np.zeros((len(methods), len(metrics)), dtype=int)

    for j, metric in enumerate(metrics):
        mean_col = f"{metric}_mean"
        higher = metric not in METRICS_LOWER_BETTER
        if mean_col not in summary.columns:
            continue

        for _, group in summary.groupby(["table", "config"]):
            if higher:
                best_val = group[mean_col].max()
            else:
                best_val = group[mean_col].min()
            winners = group[group[mean_col] == best_val]["method"].values
            for w in winners:
                if w in methods:
                    win_matrix[methods.index(w), j] += 1

    fig, ax = plt.subplots(figsize=(6, max(4, len(methods) * 0.6)))
    im = ax.imshow(win_matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m.replace("_", "\n") for m in metrics], fontsize=9)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=9)

    for i in range(len(methods)):
        for j in range(len(metrics)):
            color = "white" if win_matrix[i, j] > win_matrix.max() * 0.6 else "black"
            ax.text(j, i, str(win_matrix[i, j]), ha="center", va="center",
                    fontsize=11, fontweight="bold", color=color)

    n_configs = len(summary.groupby(["table", "config"]))
    ax.set_title(f"Win Count per Metric (of {n_configs} configs)", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    fig.savefig(out_dir / "win_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ win_heatmap.png")


# ============================================================================
# PLOT 3: RC-GNN vs best-baseline advantage
# ============================================================================

def plot_rcgnn_gap(df: pd.DataFrame, out_dir: Path):
    """Bar chart: RC-GNN margin over best baseline per config."""
    summary = compute_summary(df)

    for metric in ["Directed_F1", "AUROC"]:
        mean_col = f"{metric}_mean"
        higher = metric not in METRICS_LOWER_BETTER
        if mean_col not in summary.columns:
            continue

        gaps, labels = [], []
        for (table, config), group in summary.groupby(["table", "config"]):
            rcgnn_row = group[group["method"] == "RC-GNN"]
            baseline_rows = group[group["method"] != "RC-GNN"]
            if rcgnn_row.empty or baseline_rows.empty:
                continue
            rcgnn_val = rcgnn_row[mean_col].values[0]
            best_base = baseline_rows[mean_col].max() if higher else baseline_rows[mean_col].min()
            gap = (rcgnn_val - best_base) if higher else (best_base - rcgnn_val)
            gaps.append(gap)
            labels.append(CONFIG_DISPLAY.get(config, config))

        if not gaps:
            continue

        fig, ax = plt.subplots(figsize=(max(8, len(gaps) * 0.8), 4))
        colors = ["#2ca02c" if g > 0 else "#d62728" for g in gaps]
        ax.bar(range(len(gaps)), gaps, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(gaps)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel(f"RC-GNN − Best Baseline ({metric.replace('_', ' ')})", fontsize=10)
        ax.set_title(f"RC-GNN Advantage: {metric.replace('_', ' ')}", fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        avg_gap = np.mean(gaps)
        ax.axhline(avg_gap, color="blue", ls="--", lw=1, alpha=0.6,
                    label=f"Mean gap: {avg_gap:+.3f}")
        ax.legend(fontsize=9)
        plt.tight_layout()
        fname = f"rcgnn_gap_{metric}.png"
        fig.savefig(out_dir / fname, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ {fname}")


# ============================================================================
# PLOT 4: Training loss curves
# ============================================================================

def _find_histories(artifacts_root: Path, max_count: int = 6):
    """Locate training_history.json files under table2a/table2c artifact dirs."""
    found = []
    for sub in ["table2a", "table2c"]:
        d = artifacts_root / sub
        if not d.exists():
            continue
        for hist in sorted(d.glob("*/seed_0/training_history.json")):
            found.append(hist)
            if len(found) >= max_count:
                return found
    return found


def plot_training_curves(artifacts_root: Path, out_dir: Path):
    """Log-scale loss curves from training_history.json files."""
    histories = _find_histories(artifacts_root)
    if not histories:
        print("  (no training_history.json found — skipping training curves)")
        return

    n = len(histories)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for idx, hist_path in enumerate(histories):
        config_name = hist_path.parent.parent.name
        with open(hist_path) as f:
            history = json.load(f)

        epochs = [e.get("epoch", i) for i, e in enumerate(history)]
        ax = axes_flat[idx]

        for key, label, color in [
            ("loss",    "Total",   "#e41a1c"),
            ("L_recon", "L_recon", "#377eb8"),
            ("L_inv",   "L_inv",   "#4daf4a"),
            ("L_hsic",  "L_hsic",  "#984ea3"),
        ]:
            vals = [e.get(key) for e in history]
            if vals[0] is not None:
                ax.plot(epochs, vals, label=label, lw=1.5 if key == "loss" else 1,
                        alpha=1.0 if key == "loss" else 0.7, color=color)

        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("Loss", fontsize=9)
        ax.set_title(CONFIG_DISPLAY.get(config_name, config_name), fontsize=10, fontweight="bold")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.3)
        ax.set_yscale("log")

    for idx in range(len(histories), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Training Loss Curves (seed 0)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ training_curves.png")


# ============================================================================
# PLOT 5: F1 convergence across epochs
# ============================================================================

def plot_f1_convergence(artifacts_root: Path, out_dir: Path):
    """TopK-F1 and Skeleton-F1 across training."""
    histories = _find_histories(artifacts_root)
    if not histories:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for hist_path in histories:
        config_name = hist_path.parent.parent.name
        with open(hist_path) as f:
            history = json.load(f)
        epochs = [e.get("epoch", i) for i, e in enumerate(history)]
        label = CONFIG_DISPLAY.get(config_name, config_name)

        for ax, key in zip(axes, ["topk_f1", "skeleton_f1"]):
            vals = [e.get(key) for e in history]
            if vals[0] is not None:
                ax.plot(epochs, vals, label=label, lw=1.2, alpha=0.8)

    for ax, title in zip(axes, ["TopK F1", "Skeleton F1"]):
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("F1 Score", fontsize=10)
        ax.set_title(f"{title} Convergence", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    fig.savefig(out_dir / "f1_convergence.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ f1_convergence.png")


# ============================================================================
# PLOT 6: Adjacency heatmaps (learned vs ground truth)
# ============================================================================

def plot_adjacency_heatmaps(artifacts_root: Path, out_dir: Path):
    """Learned A vs A_true for representative configs."""
    candidates = ["h1_easy", "h1_hard", "h2_multi_env"]
    found = []
    for config in candidates:
        a_pred = artifacts_root / f"table2a/{config}/seed_0/A_best.npy"
        a_true = Path(f"data/interim/table2a/{config}/seed_0/A_true.npy")
        if a_pred.exists() and a_true.exists():
            found.append((config, a_pred, a_true))

    if not found:
        print("  (no adjacency matrices found — skipping heatmaps)")
        return

    fig, axes = plt.subplots(len(found), 3, figsize=(14, 4 * len(found)), squeeze=False)
    for row, (config, pred_path, true_path) in enumerate(found):
        A_pred = np.abs(np.load(pred_path))
        A_true = np.load(true_path).astype(float)
        diff = A_pred - A_true

        for col, (mat, title, cmap, vlims) in enumerate([
            (A_true, f"{CONFIG_DISPLAY.get(config, config)}: Ground Truth", "Blues", (0, 1)),
            (A_pred, f"{CONFIG_DISPLAY.get(config, config)}: RC-GNN Learned", "Reds", (0, 1)),
            (diff,   f"{CONFIG_DISPLAY.get(config, config)}: Difference",     "RdBu_r", (-1, 1)),
        ]):
            im = axes[row, col].imshow(mat, cmap=cmap, aspect="equal",
                                       vmin=vlims[0], vmax=vlims[1])
            axes[row, col].set_title(title, fontsize=9)
            plt.colorbar(im, ax=axes[row, col], shrink=0.7)

    fig.suptitle("Adjacency Matrix Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "adjacency_heatmaps.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ adjacency_heatmaps.png")


# ============================================================================
# PLOT 7: Radar chart (mean metrics per method)
# ============================================================================

def plot_radar(df: pd.DataFrame, out_dir: Path):
    """Spider chart of average metric values per method."""
    metrics = ["Directed_F1", "Skeleton_F1", "AUROC", "Precision", "Recall"]
    present = [m for m in metrics if m in df.columns]
    methods = [m for m in METHOD_ORDER if m in df["method"].values]
    means = df.groupby("method")[present].mean()

    angles = np.linspace(0, 2 * np.pi, len(present), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for method in methods:
        if method not in means.index:
            continue
        vals = means.loc[method].values.tolist()
        vals += vals[:1]
        ax.plot(angles, vals, lw=2 if method == "RC-GNN" else 1,
                label=method, color=METHOD_COLORS.get(method, "#cccccc"),
                alpha=1.0 if method == "RC-GNN" else 0.6)
        if method == "RC-GNN":
            ax.fill(angles, vals, alpha=0.15, color=METHOD_COLORS["RC-GNN"])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace("_", "\n") for m in present], fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title("Method Comparison (Mean Across All Configs)",
                 fontsize=12, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "radar_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ radar_comparison.png")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Plot Table 2 benchmark results (7 figure types)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv", type=str, default="artifacts/table2/table2_all_runs.csv",
                        help="Path to table2_all_runs.csv")
    parser.add_argument("--artifacts_root", type=str, default="artifacts",
                        help="Root artifacts directory (for training histories)")
    parser.add_argument("--out_dir", type=str, default="artifacts/table2/plots",
                        help="Output directory for figures")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts_root = Path(args.artifacts_root)

    print("=" * 60)
    print(" Table 2 Results — Publication Plotting")
    print("=" * 60)

    # Load data
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"\n[ERROR] CSV not found: {csv_path}")
        print("  Run aggregate_table2.py first, or specify --csv path.")
        sys.exit(1)

    print(f"\nLoading: {csv_path}")
    df = load_and_deduplicate(str(csv_path))
    print(f"  {len(df)} rows | {df['method'].nunique()} methods | "
          f"{df['config'].nunique()} configs | {df['seed'].nunique()} seeds")

    # Generate all plots
    print("\n1. Grouped bar charts...")
    plot_grouped_bars(df, out_dir)

    print("\n2. Win-count heatmap...")
    plot_win_heatmap(df, out_dir)

    print("\n3. RC-GNN advantage gap...")
    plot_rcgnn_gap(df, out_dir)

    print("\n4. Training loss curves...")
    plot_training_curves(artifacts_root, out_dir)

    print("\n5. F1 convergence plots...")
    plot_f1_convergence(artifacts_root, out_dir)

    print("\n6. Adjacency heatmaps...")
    plot_adjacency_heatmaps(artifacts_root, out_dir)

    print("\n7. Radar comparison chart...")
    plot_radar(df, out_dir)

    print(f"\n{'=' * 60}")
    n_plots = len(list(out_dir.glob("*.png")))
    print(f" Done — {n_plots} figures saved to {out_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
