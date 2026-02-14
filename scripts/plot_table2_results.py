import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
from pathlib import Path

def parse_latex_table(filepath):
    """Parses a LaTeX table file into a pandas DataFrame."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    data = []
    methods = []
    parsing_data = False
    
    for line in lines:
        line = line.strip()
        if line.startswith(r'\toprule'):
            # Next line usually has headers
            continue
        elif line.startswith('config'): # Header line
            parts = line.split('&')
            methods = [p.strip().replace(r'\\', '') for p in parts[1:]]
            continue
        elif line.startswith(r'\midrule'):
            parsing_data = True
            continue
        elif line.startswith(r'\bottomrule'):
            break
        
        if parsing_data and '&' in line:
            parts = line.split('&')
            config = parts[0].strip().replace('\\_', '_')
            
            row = {'config': config}
            for i, part in enumerate(parts[1:]):
                part = part.strip().replace(r'\\', '')
                if '±' in part:
                    mean_str, std_str = part.split('±')
                    mean = float(mean_str)
                    std = float(std_str)
                else:
                    try:
                        mean = float(part)
                        std = 0.0
                    except ValueError:
                        mean = 0.0
                        std = 0.0
                        
                method_name = methods[i]
                row[method_name] = mean
                row[f'{method_name}_std'] = std
            data.append(row)
            
    return pd.DataFrame(data)

def plot_table(df, title, output_path):
    """Plots the DataFrame as a grouped bar chart."""
    df = df.set_index('config')
    
    # Extract method columns (excluding std columns)
    method_cols = [c for c in df.columns if not c.endswith('_std')]
    
    # Plotting
    n_configs = len(df)
    n_methods = len(method_cols)
    bar_width = 0.8 / n_methods
    indices = np.arange(n_configs)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, n_methods))
    
    for i, method in enumerate(method_cols):
        means = df[method]
        stds = df[f'{method}_std']
        
        # Highlight RC-GNN
        alpha = 1.0 if method == 'RC-GNN' else 0.7
        edgecolor = 'black' if method == 'RC-GNN' else None
        
        ax.bar(indices + i * bar_width, means, bar_width, 
               yerr=stds, label=method, capsize=5, alpha=alpha, edgecolor=edgecolor)

    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Directed F1 Score', fontsize=12)
    ax.set_title(title, fontsize=16)
    ax.set_xticks(indices + bar_width * (n_methods - 1) / 2)
    ax.set_xticklabels(df.index, rotation=45, ha='right')
    ax.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved plot to {output_path}")

def main():
    artifacts_dir = Path('artifacts')
    
    # Table 2A
    file_2a = artifacts_dir / 'table2A.tex'
    if file_2a.exists():
        df_2a = parse_latex_table(file_2a)
        print("Table 2A Data:")
        print(df_2a)
        plot_table(df_2a, 'Table 2A: Hypothesis Benchmarks (H1/H2/H3)', artifacts_dir / 'table2A_plot.png')
    
    # Table 2B
    file_2b = artifacts_dir / 'table2B.tex'
    if file_2b.exists():
        df_2b = parse_latex_table(file_2b)
        print("\nTable 2B Data:")
        print(df_2b)
        plot_table(df_2b, 'Table 2B: SEM Benchmark Grid', artifacts_dir / 'table2B_plot.png')

    # Table 2C
    file_2c = artifacts_dir / 'table2C.tex'
    if file_2c.exists():
        df_2c = parse_latex_table(file_2c)
        print("\nTable 2C Data:")
        print(df_2c)
        plot_table(df_2c, 'Table 2C: Causal Validity Ablation', artifacts_dir / 'table2C_plot.png')

if __name__ == "__main__":
    main()
