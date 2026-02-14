#!/usr/bin/env python3
"""
Generate Table 2 (LaTeX) for RC-GNN SPIE 2026 paper.

Reads evaluation_report_final.json and outputs the exact format used in the paper:
- Multi-row structure: Skeleton F1, Directed F1, SHD sections
- 7 methods as columns
- Selected corruption scenarios as rows

Usage:
    python scripts/generate_table2_latex.py [--json PATH] [--output PATH]
"""

import json
import argparse
from pathlib import Path
import numpy as np


def load_data(json_path: Path) -> dict:
    """Load evaluation results from JSON."""
    with open(json_path) as f:
        return json.load(f)


def get_method_results(baseline_data: list, corruption: str, method: str) -> dict:
    """Extract results for a specific corruption-method pair."""
    for r in baseline_data:
        if r['Corruption'] == corruption and r['Method'] == method:
            return r
    return None


def format_value(val: float, metric: str, is_best: bool = False) -> str:
    """Format a value for LaTeX output."""
    if metric in ['Skeleton_F1', 'Directed_F1']:
        # Format as .XX
        formatted = f".{int(val * 100):02d}" if val > 0 else ".00"
    elif metric == 'SHD':
        formatted = str(int(val))
    else:
        formatted = f"{val:.2f}"
    
    if is_best:
        return f"\\textbf{{{formatted}}}"
    return formatted


def generate_table2_latex(data: dict) -> str:
    """Generate Table 2 in LaTeX format matching the paper style."""
    
    baseline_data = data['baseline_comparison']
    
    # Method mapping (column headers)
    methods = [
        ('RC-GNN (sparse)', 'RC-GNN'),
        ('NOTEARS', 'NOTEARS'),
        ('NOTears-Lite', 'NOTEARS-L'),
        ('PC', 'PC'),
        ('PCMCI+', 'PCMCI+'),
        ('Granger', 'Granger'),
        ('Correlation', 'Corr.')
    ]
    
    # Corruption scenarios to include (selected for paper)
    skeleton_corruptions = [
        'clean_full', 'mcar_20', 'mcar_40', 'mnar_structural',
        'compound_full', 'compound_mnar_bias', 'extreme'
    ]
    
    directed_corruptions = skeleton_corruptions  # Same set
    
    shd_corruptions = ['clean_full', 'mcar_40', 'compound_full', 'extreme']
    
    # Build LaTeX
    lines = []
    
    # Table header
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Causal structure recovery across corruption scenarios. "
                 r"NOTEARS-L = NOTEARS-Lite. Skeleton F1 measures undirected edge recovery; "
                 r"Directed F1 measures edge orientation accuracy; SHD = Structural Hamming Distance "
                 r"(lower is better). Best results in bold. Ground truth: 13 directed edges.}")
    lines.append(r"\label{tab:results}")
    lines.append(r"\small")  # Smaller font for compactness
    lines.append(r"\begin{tabular}{@{}l" + "c" * len(methods) + r"@{}}")
    lines.append(r"\toprule")
    
    # Column headers
    header = "Dataset & " + " & ".join([m[1] for m in methods]) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")
    
    # ===== SKELETON F1 SECTION =====
    lines.append(r"\multicolumn{" + str(len(methods) + 1) + r"}{l}{\textit{Skeleton F1} $\uparrow$} \\")
    
    for corr in skeleton_corruptions:
        row_values = []
        for method_key, _ in methods:
            result = get_method_results(baseline_data, corr, method_key)
            if result:
                row_values.append(result['Skeleton_F1'])
            else:
                row_values.append(0.0)
        
        # Find best (max for F1)
        best_idx = np.argmax(row_values)
        
        # Format row
        corr_display = corr.replace('_', ' ')
        cells = [format_value(v, 'Skeleton_F1', i == best_idx) for i, v in enumerate(row_values)]
        row = f"{corr_display} & " + " & ".join(cells) + r" \\"
        lines.append(row)
    
    lines.append(r"\midrule")
    
    # ===== DIRECTED F1 SECTION =====
    lines.append(r"\multicolumn{" + str(len(methods) + 1) + r"}{l}{\textit{Directed F1} $\uparrow$} \\")
    
    for corr in directed_corruptions:
        row_values = []
        for method_key, _ in methods:
            result = get_method_results(baseline_data, corr, method_key)
            if result:
                row_values.append(result['Directed_F1'])
            else:
                row_values.append(0.0)
        
        # Find best (max for F1)
        best_idx = np.argmax(row_values)
        
        # Format row
        corr_display = corr.replace('_', ' ')
        cells = [format_value(v, 'Directed_F1', i == best_idx) for i, v in enumerate(row_values)]
        row = f"{corr_display} & " + " & ".join(cells) + r" \\"
        lines.append(row)
    
    lines.append(r"\midrule")
    
    # ===== SHD SECTION =====
    lines.append(r"\multicolumn{" + str(len(methods) + 1) + r"}{l}{\textit{SHD} $\downarrow$} \\")
    
    for corr in shd_corruptions:
        row_values = []
        for method_key, _ in methods:
            result = get_method_results(baseline_data, corr, method_key)
            if result:
                row_values.append(result['SHD'])
            else:
                row_values.append(999)  # Missing = bad
        
        # Find best (min for SHD)
        best_idx = np.argmin(row_values)
        
        # Format row
        corr_display = corr.replace('_', ' ')
        cells = [format_value(v, 'SHD', i == best_idx) for i, v in enumerate(row_values)]
        row = f"{corr_display} & " + " & ".join(cells) + r" \\"
        lines.append(row)
    
    lines.append(r"\midrule")
    
    # ===== MEAN SUMMARY =====
    # Compute means across ALL corruptions in baseline_comparison
    all_corruptions = list(set(r['Corruption'] for r in baseline_data))
    
    # Mean Skeleton F1
    mean_skel_f1 = []
    for method_key, _ in methods:
        vals = [r['Skeleton_F1'] for r in baseline_data if r['Method'] == method_key]
        mean_skel_f1.append(np.mean(vals) if vals else 0.0)
    
    best_skel = np.argmax(mean_skel_f1)
    cells = [format_value(v, 'Skeleton_F1', i == best_skel) for i, v in enumerate(mean_skel_f1)]
    lines.append(r"Mean Skel-F1 & " + " & ".join(cells) + r" \\")
    
    # Mean Directed F1
    mean_dir_f1 = []
    for method_key, _ in methods:
        vals = [r['Directed_F1'] for r in baseline_data if r['Method'] == method_key]
        mean_dir_f1.append(np.mean(vals) if vals else 0.0)
    
    best_dir = np.argmax(mean_dir_f1)
    cells = [format_value(v, 'Directed_F1', i == best_dir) for i, v in enumerate(mean_dir_f1)]
    lines.append(r"Mean Dir-F1 & " + " & ".join(cells) + r" \\")
    
    # Table footer
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    return "\n".join(lines)


def generate_markdown_table(data: dict) -> str:
    """Generate Table 2 in Markdown format for quick preview."""
    
    baseline_data = data['baseline_comparison']
    
    methods = [
        ('RC-GNN (sparse)', 'RC-GNN'),
        ('NOTEARS', 'NOTEARS'),
        ('NOTears-Lite', 'NOTEARS-L'),
        ('PC', 'PC'),
        ('PCMCI+', 'PCMCI+'),
        ('Granger', 'Granger'),
        ('Correlation', 'Corr.')
    ]
    
    corruptions = ['clean_full', 'mcar_20', 'mcar_40', 'mnar_structural',
                   'compound_full', 'compound_mnar_bias', 'extreme']
    
    lines = []
    lines.append("## Table 2: Causal Structure Recovery\n")
    
    # Header
    header = "| Dataset | " + " | ".join([m[1] for m in methods]) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(methods) + 1))
    
    # Skeleton F1
    lines.append("| **Skeleton F1 ↑** | | | | | | | |")
    for corr in corruptions:
        cells = [corr.replace('_', ' ')]
        for method_key, _ in methods:
            result = get_method_results(baseline_data, corr, method_key)
            val = result['Skeleton_F1'] if result else 0.0
            cells.append(f".{int(val * 100):02d}")
        lines.append("| " + " | ".join(cells) + " |")
    
    # Directed F1
    lines.append("| **Directed F1 ↑** | | | | | | | |")
    for corr in corruptions:
        cells = [corr.replace('_', ' ')]
        for method_key, _ in methods:
            result = get_method_results(baseline_data, corr, method_key)
            val = result['Directed_F1'] if result else 0.0
            cells.append(f".{int(val * 100):02d}")
        lines.append("| " + " | ".join(cells) + " |")
    
    # SHD
    lines.append("| **SHD ↓** | | | | | | | |")
    for corr in ['clean_full', 'mcar_40', 'compound_full', 'extreme']:
        cells = [corr.replace('_', ' ')]
        for method_key, _ in methods:
            result = get_method_results(baseline_data, corr, method_key)
            val = int(result['SHD']) if result else 999
            cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate Table 2 LaTeX from JSON")
    parser.add_argument('--json', type=Path, 
                        default=Path(__file__).parent.parent / "artifacts" / "evaluation_report_final.json",
                        help="Path to evaluation JSON file")
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).parent.parent / "paper" / "table2_generated.tex",
                        help="Output LaTeX file path")
    parser.add_argument('--markdown', action='store_true',
                        help="Also generate Markdown preview")
    args = parser.parse_args()
    
    print(f"Loading data from: {args.json}")
    data = load_data(args.json)
    
    # Generate LaTeX
    latex = generate_table2_latex(data)
    
    # Save LaTeX
    with open(args.output, 'w') as f:
        f.write(latex)
    print(f"LaTeX table saved to: {args.output}")
    
    # Print preview
    print("\n" + "=" * 60)
    print("LATEX OUTPUT:")
    print("=" * 60)
    print(latex)
    
    if args.markdown:
        md = generate_markdown_table(data)
        md_path = args.output.with_suffix('.md')
        with open(md_path, 'w') as f:
            f.write(md)
        print(f"\nMarkdown preview saved to: {md_path}")
        print("\n" + md)


if __name__ == "__main__":
    main()
