"""
Simple converter: CSV/Parquet panel -> NumPy dataset for RC-GNN

Usage examples:

# Single CSV containing multiple examples stacked with an `example_id` column:
python scripts/convert_to_numpy.py --input data/raw/panel.csv --out data/interim/my_real_dataset --time_col time --id_col example_id --features f1 f2 f3

# Directory of per-example CSVs (each file = one example, rows=time, cols=features):
python scripts/convert_to_numpy.py --input data/raw/episodes/ --out data/interim/my_real_dataset --per_file

The script writes: X.npy, M.npy, e.npy, S.npy and meta.json

For flexibility it supports CSV or Parquet and basic imputation of missingness mask.
"""

import argparse
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd


def read_table(path):
    path = Path(path)
    if path.suffix.lower() in ('.csv', '.tsv'):
        return pd.read_csv(path)
    elif path.suffix.lower() in ('.parquet', '.pq'):
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")


def per_file_mode(input_path, features):
    files = sorted([p for p in Path(input_path).iterdir() if p.suffix.lower() in ('.csv', '.tsv', '.parquet', '.pq')])
    X_list = []
    M_list = []
    e_list = []
    S_list = []
    for i, p in enumerate(files):
        df = read_table(p)
        # keep only requested features (if provided)
        if features:
            df = df[features]
        arr = df.values.astype(np.float32)
        mask = (~np.isnan(arr)).astype(np.float32)
        # replace nan by zero for storage (model uses M to know missingness)
        arr = np.nan_to_num(arr, nan=0.0)
        X_list.append(arr)
        M_list.append(mask)
        # default regime id 0; user can override later
        e_list.append(0)
        S_list.append(np.zeros(1, dtype=np.float32))
    # pad sequences to same length
    T = max(x.shape[0] for x in X_list)
    d = X_list[0].shape[1]
    N = len(X_list)
    X = np.zeros((N, T, d), dtype=np.float32)
    M = np.zeros_like(X)
    for i, arr in enumerate(X_list):
        X[i, :arr.shape[0], :] = arr
        M[i, :arr.shape[0], :] = M_list[i]
    e = np.array(e_list, dtype=np.int64)
    S = np.stack(S_list, axis=0)
    return X, M, e, S


def single_table_mode(input_path, id_col, time_col, features):
    df = read_table(input_path)
    if id_col is None:
        raise ValueError('single-table mode requires --id_col')
    ids = df[id_col].unique()
    X_list = []
    M_list = []
    e_list = []
    S_list = []
    for i, uid in enumerate(sorted(ids)):
        sub = df[df[id_col] == uid].sort_values(time_col)
        if features:
            sub = sub[features]
        arr = sub.values.astype(np.float32)
        mask = (~np.isnan(arr)).astype(np.float32)
        arr = np.nan_to_num(arr, nan=0.0)
        X_list.append(arr)
        M_list.append(mask)
        e_list.append(0)
        S_list.append(np.zeros(1, dtype=np.float32))
    # pad
    T = max(x.shape[0] for x in X_list)
    d = X_list[0].shape[1]
    N = len(X_list)
    X = np.zeros((N, T, d), dtype=np.float32)
    M = np.zeros_like(X)
    for i, arr in enumerate(X_list):
        X[i, :arr.shape[0], :] = arr
        M[i, :arr.shape[0], :] = M_list[i]
    e = np.array(e_list, dtype=np.int64)
    S = np.stack(S_list, axis=0)
    return X, M, e, S


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, help='Input CSV/Parquet file or directory')
    p.add_argument('--out', required=True, help='Output dataset folder (will be created)')
    p.add_argument('--per_file', action='store_true', help='Treat input as directory of per-example files')
    p.add_argument('--id_col', default='example_id', help='Column name for example id (single-table mode)')
    p.add_argument('--time_col', default='time', help='Column name for time ordering (single-table mode)')
    p.add_argument('--features', nargs='*', help='List of feature column names to keep (default: all numeric columns)')
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.per_file:
        X, M, e, S = per_file_mode(args.input, args.features)
    else:
        X, M, e, S = single_table_mode(args.input, args.id_col, args.time_col, args.features)

    np.save(out / 'X.npy', X)
    np.save(out / 'M.npy', M)
    np.save(out / 'e.npy', e)
    np.save(out / 'S.npy', S)

    meta = {'N': X.shape[0], 'T': X.shape[1], 'd': X.shape[2]}
    with open(out / 'meta.json', 'w') as f:
        json.dump(meta, f)

    print('Wrote dataset to', out)

if __name__ == '__main__':
    main()
