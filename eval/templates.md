# Paper Templates: Figures & Tables

Publication-ready templates for main paper figures and tables.

---

## 1. Main Results Table (Table 1)

**Purpose:** Central result showing RC-GNN superiority on both ID and LODO protocols.

### LaTeX Template

```latex
\begin{table}[h]
\centering
\caption{Main Results: RC-GNN vs. Baselines on Causal Discovery}
\label{tab:main_results}
\begin{tabular}{l|ccc|ccc}
\toprule
& \multicolumn{3}{c|}{In-Distribution (ID)} & \multicolumn{3}{c}{Leave-One-Regime-Out (LODO)} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
Method & Directed F1 & Skeleton F1 & SHD & Directed F1 & Skeleton F1 & SHD \\
\midrule
RC-GNN (ours) & \textbf{0.92 ± 0.03} & \textbf{1.00 ± 0.00} & \textbf{1.0 ± 0.2} & \textbf{0.86 ± 0.06} & \textbf{0.98 ± 0.02} & \textbf{1.8 ± 0.5} \\
NOTEARS-nonlin & 0.68 ± 0.07 & 0.75 ± 0.09 & 7.0 ± 1.0 & 0.52 ± 0.12 & 0.61 ± 0.11 & 9.2 ± 1.5 \\
PCMCI+ & 0.52 ± 0.10 & 0.60 ± 0.11 & 12.0 ± 2.0 & 0.41 ± 0.14 & 0.48 ± 0.13 & 13.5 ± 2.5 \\
Granger Causality & 0.42 ± 0.06 & 0.58 ± 0.09 & 15.0 ± 2.5 & 0.35 ± 0.10 & 0.45 ± 0.12 & 17.0 ± 3.0 \\
Correlation & 0.31 ± 0.05 & 0.50 ± 0.08 & 19.0 ± 2.0 & 0.28 ± 0.08 & 0.42 ± 0.10 & 20.5 ± 2.5 \\
\bottomrule
\end{tabular}
\\[0.5em]
{\small Results are mean ± std over 5 random seeds. Best results bolded. LODO: train on 4 regimes, test on held-out regime. SHD: Structural Hamming Distance (lower is better).}
\end{table}
```

### Python Code to Generate

```python
def generate_table1(results_dict, output_file='table1.tex'):
 """
 Generate Table 1 from aggregated results.

 Args:
 results_dict: {method: {'id': {...}, 'lodo': {...}}}
 """
 import pandas as pd

 rows = []
 for method in ['RC-GNN', 'NOTEARS', 'PCMCI+', 'Granger', 'Correlation']:
 row = {'Method': method}

 # ID metrics
 id_results = results_dict[method]['id']
 row['ID Dir F1'] = f"{id_results['directed_f1_mean']:.2f} ± {id_results['directed_f1_std']:.2f}"
 row['ID Skel F1'] = f"{id_results['skeleton_f1_mean']:.2f} ± {id_results['skeleton_f1_std']:.2f}"
 row['ID SHD'] = f"{id_results['shd_mean']:.1f} ± {id_results['shd_std']:.1f}"

 # LODO metrics
 lodo_results = results_dict[method]['lodo']
 row['LODO Dir F1'] = f"{lodo_results['directed_f1_mean']:.2f} ± {lodo_results['directed_f1_std']:.2f}"
 row['LODO Skel F1'] = f"{lodo_results['skeleton_f1_mean']:.2f} ± {lodo_results['skeleton_f1_std']:.2f}"
 row['LODO SHD'] = f"{lodo_results['shd_mean']:.1f} ± {lodo_results['shd_std']:.1f}"

 rows.append(row)

 df = pd.DataFrame(rows)

 # Save as LaTeX
 latex_table = df.to_latex(index=False)
 with open(output_file, 'w') as f:
 f.write(latex_table)

 print(f"Table saved to {output_file}")
```

---

## 2. Ablation Study Table (Table 2)

**Purpose:** Show component contributions to RC-GNN performance.

### LaTeX Template

```latex
\begin{table}[h]
\centering
\caption{Ablation Study: Component Contributions}
\label{tab:ablations}
\begin{tabular}{l|cccc}
\toprule
Ablation & Directed F1 ↓ & Skeleton F1 ↓ & SHD ↑ & Remarks \\
\midrule
Full RC-GNN & 0.92 ± 0.03 & 1.00 ± 0.00 & 1.0 ± 0.2 & Baseline \\
\quad w/o 3-stage schedule & 0.84 ± 0.05 & 0.96 ± 0.02 & 2.1 ± 0.4 & -8\% \\
\quad w/o signal encoder & 0.80 ± 0.08 & 0.91 ± 0.05 & 2.8 ± 0.6 & -12\% \\
\quad w/o direction phase & 0.68 ± 0.10 & 1.00 ± 0.00 & 5.0 ± 1.2 & -24\% \\
\quad w/o DAG penalty & 0.75 ± 0.12 & 0.87 ± 0.09 & 4.2 ± 1.1 & -17\% \\
\quad w/o multi-regime & 0.62 ± 0.15 & 0.78 ± 0.12 & 6.1 ± 1.8 & -30\% \\
\quad w/o GroupDRO & 0.86 ± 0.09 & 0.97 ± 0.03 & 2.0 ± 0.8 & -6\% \\
\quad w/o MNAR model & 0.82 ± 0.08 & 0.94 ± 0.05 & 2.5 ± 0.7 & -10\% \\
\bottomrule
\end{tabular}
\\[0.5em]
{\small Ablations listed in order of impact. Each ablation run for 5 seeds. \% denotes drop in Directed F1 from full model.}
\end{table}
```

---

## 3. Robustness Stress Test Table (Table 3)

**Purpose:** Demonstrate RC-GNN robustness to realistic perturbations.

### LaTeX Template

```latex
\begin{table}[h]
\centering
\caption{Robustness Evaluation: Performance Under Data Perturbations}
\label{tab:robustness}
\begin{tabular}{l|cc|cc}
\toprule
& \multicolumn{2}{c|}{Directed F1} & \multicolumn{2}{c}{Skeleton F1} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Stress Test & Condition & Score & Condition & Score \\
\midrule
\multirow{3}{*}{Missingness} & 20\% & 0.87 ± 0.05 & 20\% & 0.98 ± 0.01 \\
& 40\% & 0.80 ± 0.06 & 40\% & 0.95 ± 0.02 \\
& 60\% & 0.65 ± 0.08 & 60\% & 0.85 ± 0.05 \\
\midrule
\multirow{3}{*}{Corruption} & MNAR only & 0.85 ± 0.07 & MNAR only & 0.97 ± 0.02 \\
& Bias only & 0.88 ± 0.06 & Bias only & 0.98 ± 0.01 \\
& MNAR+Bias & 0.80 ± 0.08 & MNAR+Bias & 0.94 ± 0.03 \\
\midrule
\multirow{2}{*}{Cross-Regime} & ID (trained) & 0.92 ± 0.03 & ID (trained) & 1.00 ± 0.00 \\
& OOD (held-out) & 0.86 ± 0.06 & OOD (held-out) & 0.98 ± 0.02 \\
\bottomrule
\end{tabular}
\\[0.5em]
{\small All results mean ± std over 5 seeds. OOD: performance on regime not seen during training (leave-one-regime-out).}
\end{table}
```

---

## 4. Precision-Recall Curve (Figure 1A)

**Purpose:** Show threshold-free performance via PR curve.

### Python Code

```python
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc

def plot_pr_curve(A_true, A_pred_scores, method_name='RC-GNN',
 output_file='figure1a_pr_curve.png'):
 """
 Plot precision-recall curve for causal discovery.

 Args:
 A_true: [d, d] ground truth adjacency (binary)
 A_pred_scores: [d, d] predicted edge scores (continuous)
 method_name: str, method name for legend
 """
 # Flatten to compute PR curve
 y_true = A_true.flatten()
 y_scores = A_pred_scores.flatten()

 precision, recall, _ = precision_recall_curve(y_true, y_scores)
 auc_pr = auc(recall, precision)

 # Plot
 fig, ax = plt.subplots(figsize=(8, 6))
 ax.plot(recall, precision, linewidth=2.5, label=f'{method_name} (AUPRC={auc_pr:.3f})', color='#1f77b4')

 # Baseline (random classifier)
 baseline_precision = np.sum(A_true) / A_true.size
 ax.axhline(y=baseline_precision, color='gray', linestyle='--', label='Random', alpha=0.7)

 ax.set_xlabel('Recall', fontsize=12)
 ax.set_ylabel('Precision', fontsize=12)
 ax.set_title('Precision-Recall Curve for Causal Discovery', fontsize=14)
 ax.legend(fontsize=11, loc='best')
 ax.grid(True, alpha=0.3)
 ax.set_xlim([0, 1])
 ax.set_ylim([0, 1])

 plt.tight_layout()
 plt.savefig(output_file, dpi=300, bbox_inches='tight')
 print(f"Figure saved to {output_file}")

 return auc_pr
```

---

## 5. ROC Curve (Figure 1B)

**Purpose:** Show ROC-AUC performance.

### Python Code

```python
from sklearn.metrics import roc_curve, auc

def plot_roc_curve(A_true, A_pred_scores, method_name='RC-GNN',
 output_file='figure1b_roc_curve.png'):
 """
 Plot ROC curve for causal discovery.
 """
 y_true = A_true.flatten()
 y_scores = A_pred_scores.flatten()

 fpr, tpr, _ = roc_curve(y_true, y_scores)
 auc_roc = auc(fpr, tpr)

 fig, ax = plt.subplots(figsize=(8, 6))
 ax.plot(fpr, tpr, linewidth=2.5, label=f'{method_name} (ROC-AUC={auc_roc:.3f})', color='#1f77b4')
 ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random (AUC=0.5)')

 ax.set_xlabel('False Positive Rate', fontsize=12)
 ax.set_ylabel('True Positive Rate', fontsize=12)
 ax.set_title('ROC Curve for Causal Discovery', fontsize=14)
 ax.legend(fontsize=11, loc='lower right')
 ax.grid(True, alpha=0.3)
 ax.set_xlim([0, 1])
 ax.set_ylim([0, 1])

 plt.tight_layout()
 plt.savefig(output_file, dpi=300, bbox_inches='tight')
 print(f"Figure saved to {output_file}")

 return auc_roc
```

---

## 6. F1 vs Edge Count Curve (Figure 2)

**Purpose:** Show F1 performance across different sparsity thresholds.

### Python Code

```python
def plot_f1_vs_k_curve(A_true, A_pred_scores, output_file='figure2_f1_vs_k.png'):
 """
 Plot F1 score vs. number of edges kept (K).
 Threshold at different quantiles: {0.5E, 0.75E, E, 1.5E, 2.0E}
 where E = number of true edges.
 """
 from sklearn.metrics import f1_score

 n_edges_true = np.sum(A_true)
 k_values = [0.5 * n_edges_true, 0.75 * n_edges_true,
 n_edges_true, 1.5 * n_edges_true, 2.0 * n_edges_true]
 k_values = [int(k) for k in k_values if 0 < k < A_true.size]

 f1_scores = []
 for k in k_values:
 threshold = np.partition(A_pred_scores.flatten(), -k)[-k]
 A_pred_binary = (A_pred_scores >= threshold).astype(int)
 f1 = f1_score(A_true.flatten(), A_pred_binary)
 f1_scores.append(f1)

 fig, ax = plt.subplots(figsize=(10, 6))
 k_labels = ['0.5E', '0.75E', 'E', '1.5E', '2.0E'][:len(k_values)]

 ax.plot(k_labels, f1_scores, marker='o', markersize=8, linewidth=2.5,
 color='#1f77b4', label='Directed F1')

 # Mark optimal K
 best_idx = np.argmax(f1_scores)
 ax.plot(k_labels[best_idx], f1_scores[best_idx], marker='*',
 markersize=20, color='red', label=f'Optimal (K={k_values[best_idx]})')

 ax.set_xlabel('Edge Count Threshold (K)', fontsize=12)
 ax.set_ylabel('Directed F1 Score', fontsize=12)
 ax.set_title('F1 Score vs. Sparsity Level', fontsize=14)
 ax.legend(fontsize=11)
 ax.grid(True, alpha=0.3)
 ax.set_ylim([0, 1.05])

 plt.tight_layout()
 plt.savefig(output_file, dpi=300, bbox_inches='tight')
 print(f"Figure saved to {output_file}")
```

---

## 7. Robustness Curve: Missingness Impact (Figure 3)

**Purpose:** Show graceful degradation under increasing missingness.

### Python Code

```python
def plot_robustness_missingness(results_dict, output_file='figure3_robustness_missingness.png'):
 """
 Plot directed F1 vs. missingness rate for RC-GNN vs. baselines.

 Args:
 results_dict: {'rc_gnn': [...], 'notears': [...], ...}
 each containing (missingness_rate, f1_mean, f1_std)
 """
 fig, ax = plt.subplots(figsize=(10, 6))

 colors = {'RC-GNN': '#1f77b4', 'NOTEARS': '#ff7f0e', 'Correlation': '#2ca02c'}

 for method, data in results_dict.items():
 miss_rates = [x[0] for x in data]
 f1_means = [x[1] for x in data]
 f1_stds = [x[2] for x in data]

 ax.errorbar(miss_rates, f1_means, yerr=f1_stds, marker='o',
 label=method, linewidth=2.5, markersize=8, color=colors.get(method))

 ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
 ax.set_xlabel('Missingness Rate (%)', fontsize=12)
 ax.set_ylabel('Directed F1 Score', fontsize=12)
 ax.set_title('Robustness to Missing Data', fontsize=14)
 ax.legend(fontsize=11, loc='lower left')
 ax.grid(True, alpha=0.3)
 ax.set_xlim([-5, 85])
 ax.set_ylim([0, 1.05])

 plt.tight_layout()
 plt.savefig(output_file, dpi=300, bbox_inches='tight')
 print(f"Figure saved to {output_file}")
```

---

## 8. Environment Structure Heatmap (Figure 4)

**Purpose:** Visualize learned per-environment adjacency deltas.

### Python Code

```python
import seaborn as sns

def plot_environment_deltas(A_base, A_deltas_by_env, env_names=None,
 output_file='figure4_env_deltas.png'):
 """
 Plot base adjacency and per-environment deltas.

 Args:
 A_base: [d, d] base adjacency
 A_deltas_by_env: {env_id: [d, d] delta adjacency}
 env_names: list of environment names
 """
 n_envs = len(A_deltas_by_env)
 fig, axes = plt.subplots(1, n_envs + 1, figsize=(5 * (n_envs + 1), 5))

 # Plot base
 sns.heatmap(A_base, ax=axes[0], cmap='YlOrRd', cbar=True,
 square=True, vmin=0, vmax=1)
 axes[0].set_title('Base Adjacency $A_{base}$', fontsize=12)

 # Plot deltas
 for env_id, A_delta in A_deltas_by_env.items():
 ax = axes[env_id + 1]
 env_label = env_names[env_id] if env_names else f'Env {env_id}'
 sns.heatmap(A_delta, ax=ax, cmap='RdBu_r', cbar=True,
 square=True, vmin=-0.5, vmax=0.5)
 ax.set_title(f'Delta: {env_label}', fontsize=12)

 plt.tight_layout()
 plt.savefig(output_file, dpi=300, bbox_inches='tight')
 print(f"Figure saved to {output_file}")
```

---

## 9. Comparative Bar Chart: Methods (Figure 5)

**Purpose:** Direct comparison of all methods across metrics.

### Python Code

```python
def plot_method_comparison(results_df, metric='directed_f1',
 output_file='figure5_method_comparison.png'):
 """
 Bar chart comparing methods.

 Args:
 results_df: DataFrame with columns [method, metric_mean, metric_std, dataset]
 """
 fig, ax = plt.subplots(figsize=(12, 6))

 methods = results_df['method'].unique()
 x = np.arange(len(methods))
 width = 0.35

 for i, dataset in enumerate(results_df['dataset'].unique()):
 subset = results_df[results_df['dataset'] == dataset]
 means = subset[f'{metric}_mean'].values
 stds = subset[f'{metric}_std'].values

 ax.bar(x + i * width, means, width, label=dataset,
 yerr=stds, capsize=5, alpha=0.8)

 ax.set_ylabel('Score', fontsize=12)
 ax.set_title(f'{metric.replace("_", " ").title()} Comparison Across Methods', fontsize=14)
 ax.set_xticks(x + width / 2)
 ax.set_xticklabels(methods, rotation=45, ha='right')
 ax.legend(fontsize=11)
 ax.grid(True, alpha=0.3, axis='y')
 ax.set_ylim([0, 1.05])

 plt.tight_layout()
 plt.savefig(output_file, dpi=300, bbox_inches='tight')
 print(f"Figure saved to {output_file}")
```

---

## 10. Supplementary: Per-Seed Scatter Plot

**Purpose:** Show individual seed results for transparency.

### Python Code

```python
def plot_per_seed_scatter(results_list, methods=['RC-GNN', 'NOTEARS'],
 metric='directed_f1', output_file='figure_s1_per_seed.png'):
 """
 Scatter plot of individual seed results.
 """
 fig, ax = plt.subplots(figsize=(10, 6))

 for method in methods:
 scores = [r[metric] for r in results_list if r['method'] == method]
 seeds = range(len(scores))

 ax.scatter(seeds, scores, s=100, marker='o', alpha=0.7, label=method)

 ax.set_xlabel('Seed ID', fontsize=12)
 ax.set_ylabel(f'{metric.replace("_", " ").title()}', fontsize=12)
 ax.set_title('Per-Seed Results (Transparency)', fontsize=14)
 ax.legend(fontsize=11)
 ax.grid(True, alpha=0.3)
 ax.set_ylim([0, 1.05])

 plt.tight_layout()
 plt.savefig(output_file, dpi=300, bbox_inches='tight')
 print(f"Figure saved to {output_file}")
```

---

## 11. Figure Placement Guide (Paper Structure)

```
Main Paper:
├─ Figure 1: Precision-Recall & ROC Curves (§Results)
├─ Figure 2: F1 vs. K Curve (§Results)
├─ Figure 3: Robustness to Missingness (§Robustness)
├─ Figure 4: Environment Structure Heatmaps (§Method)
└─ Figure 5: Method Comparison Bar Chart (§Results)

Appendix:
├─ Figure S1: Per-Seed Scatter Plot (§Reproducibility)
├─ Figure S2: Ablation Impact Bar Chart (§Ablations)
├─ Figure S3: Corruption Mode Heatmap (§Robustness)
└─ Figure S4: OOD Regime Generalization (§Robustness)
```

---

## 12. Generation Script (Automated)

```python
# scripts/generate_paper_figures.py

def main():
 """Generate all paper figures and tables."""

 # Load aggregated results
 with open('artifacts/metrics/results_aggregated.json', 'r') as f:
 results = json.load(f)

 with open('artifacts/ablations/results_aggregated.json', 'r') as f:
 ablation_results = json.load(f)

 # Generate tables
 print("Generating tables...")
 generate_table1(results, 'reports/tables/table1.tex')
 generate_table_ablations(ablation_results, 'reports/tables/table2.tex')

 # Generate figures
 print("Generating figures...")
 plot_pr_curve(results['rc_gnn']['A_true'], results['rc_gnn']['A_pred'],
 output_file='reports/figs/figure1a.png')
 plot_roc_curve(results['rc_gnn']['A_true'], results['rc_gnn']['A_pred'],
 output_file='reports/figs/figure1b.png')

 print("[OK] All figures and tables generated")

if __name__ == '__main__':
 main()
```

---

**Usage:** Run `python scripts/generate_paper_figures.py` after training all methods to auto-generate publication-ready figures and tables.

