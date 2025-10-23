"""
Download the UCI Air Quality dataset (UCI archive or GitHub raw CSV) and convert it into
the NumPy dataset layout expected by RC-GNN: X.npy, M.npy, e.npy, S.npy, meta.json

Usage examples:

# Download from the UCI ML repo and convert with 24-hour windows:
python scripts/download_and_convert_uci.py --source uci --out data/interim/uci_air --window-length 24 --stride 1

# Download the raw CSV from GitHub and convert using month-based regimes:
python scripts/download_and_convert_uci.py --source github --out data/interim/uci_air --regime month

This script is intentionally simple and focused on reproducible preprocessing for RC-GNN.
"""

import argparse
import os
import io
import json
import zipfile
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import requests


UCI_ZIP_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip'
GITHUB_RAW_CSV = 'https://raw.githubusercontent.com/asharvi1/UCI-Air-Quality-Data/master/AirQualityUCI.csv'


def download_to_file(url, dst_path):
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(dst_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return dst_path


def extract_zip_and_find_csv(zip_path, dst_dir):
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(dst_dir)
        # find the first CSV-like file
        for nm in z.namelist():
            if nm.lower().endswith('.csv') or nm.lower().endswith('.txt'):
                return Path(dst_dir) / nm
    return None


def read_uci_csv(path):
    # UCI file uses semicolon separator and contains an extra trailing column
    df = pd.read_csv(path, sep=';', decimal=',')
    # drop completely empty columns (UCI has an extra empty column at the end)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    return df


def preprocess_df(df):
    # UCI uses -200 to signal missing values — convert to NaN
    df = df.replace(-200, np.nan)
    # Combine Date and Time if present into datetime
    if 'Date' in df.columns and 'Time' in df.columns:
        # UCI uses Date format dd/mm/yyyy and Time like HH.MM.SS; try explicit parsing first
        combined = df['Date'].astype(str).str.strip() + ' ' + df['Time'].astype(str).str.strip()
        dt = pd.to_datetime(combined, format='%d/%m/%Y %H.%M.%S', dayfirst=True, errors='coerce')
        # fallback to generic parse if explicit format fails for some rows
        if dt.isna().any():
            dt_fallback = pd.to_datetime(combined, dayfirst=True, errors='coerce')
            # prefer explicit parse where possible
            dt = dt.fillna(dt_fallback)
        df.insert(0, 'datetime', dt)
    elif 'Time' in df.columns:
        df.insert(0, 'datetime', pd.to_datetime(df['Time'], errors='coerce'))

    # drop columns that are completely non-numeric (except datetime)
    non_numeric = [c for c in df.columns if c != 'datetime' and not pd.api.types.is_numeric_dtype(df[c])]
    # keep numeric replacements only
    for c in non_numeric:
        # If column looks like 'Date' or 'Time' we've already handled
        if c in ('Date', 'Time'):
            df = df.drop(columns=[c])
        else:
            # try to coerce to numeric
            df[c] = pd.to_numeric(df[c], errors='coerce')

    return df


def select_feature_columns(df, features=None):
    # Exclude datetime column
    cols = [c for c in df.columns if c != 'datetime']
    if features:
        # validate provided features exist
        cols = [c for c in features if c in df.columns]
    else:
        # automatically choose numeric columns
        cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    return cols


def build_windows(df, feature_cols, T, stride=1, pad=False):
    # sort by datetime if present
    if 'datetime' in df.columns:
        df = df.sort_values('datetime')
    arr = df[feature_cols].to_numpy(dtype=np.float32)
    # mask: 1 if not nan
    mask = (~np.isnan(arr)).astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0).astype(np.float32)
    n_steps, d = arr.shape
    windows = []
    masks = []
    if n_steps == 0:
        return np.zeros((0, T, d), dtype=np.float32), np.zeros((0, T, d), dtype=np.float32)
    if n_steps < T and pad:
        # pad beginning with zeros
        pad_amt = T - n_steps
        padded = np.zeros((T, d), dtype=np.float32)
        padded_mask = np.zeros((T, d), dtype=np.float32)
        padded[pad_amt:, :] = arr
        padded_mask[pad_amt:, :] = mask
        return padded[np.newaxis, ...], padded_mask[np.newaxis, ...]
    for start in range(0, max(1, n_steps - T + 1), stride):
        win = arr[start:start+T]
        mwin = mask[start:start+T]
        if win.shape[0] != T:
            # skip incomplete window unless pad=True
            if pad:
                pad_amt = T - win.shape[0]
                pw = np.zeros((T, d), dtype=np.float32)
                pm = np.zeros((T, d), dtype=np.float32)
                pw[pad_amt:, :] = win
                pm[pad_amt:, :] = mwin
                win = pw; mwin = pm
            else:
                continue
        windows.append(win)
        masks.append(mwin)
    X = np.stack(windows, axis=0)
    M = np.stack(masks, axis=0)
    return X, M


def make_regimes(df, method='none'):
    # For single-station UCI dataset we can create regimes by month or season
    if method == 'none':
        return None
    if 'datetime' not in df.columns:
        return None
    dt = pd.to_datetime(df['datetime'], errors='coerce')
    if method == 'month':
        return dt.dt.month
    if method == 'season':
        # meteorological seasons by month
        m = dt.dt.month
        # Dec-Feb=0, Mar-May=1, Jun-Aug=2, Sep-Nov=3
        season = ((m % 12 + 3) // 3 - 1) % 4
        return season
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', choices=['uci', 'github', 'raw'], default='uci', help='Where to download the dataset from')
    p.add_argument('--raw-url', help='If source=raw, the direct CSV URL')
    p.add_argument('--out', required=True, help='Output directory to write dataset into')
    p.add_argument('--window-length', '-T', type=int, default=24, help='Window length in timesteps')
    p.add_argument('--stride', type=int, default=1, help='Window stride')
    p.add_argument('--features', nargs='*', help='Feature column names to keep (default: auto numeric columns)')
    p.add_argument('--regime', choices=['none', 'month', 'season'], default='none', help='How to construct regimes (e.npy)')
    p.add_argument('--pad', action='store_true', help='Pad short series to window length instead of dropping')
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        if args.source == 'uci':
            print('Downloading UCI archive...')
            zip_path = td / 'AirQualityUCI.zip'
            download_to_file(UCI_ZIP_URL, zip_path)
            csv_path = extract_zip_and_find_csv(zip_path, td)
            if csv_path is None:
                raise RuntimeError('Could not find CSV inside the UCI ZIP archive')
        elif args.source == 'github':
            print('Downloading raw CSV from GitHub...')
            csv_path = td / 'AirQualityUCI.csv'
            download_to_file(GITHUB_RAW_CSV, csv_path)
        else:
            if not args.raw_url:
                raise ValueError('When --source raw you must provide --raw-url')
            csv_path = td / 'raw.csv'
            download_to_file(args.raw_url, csv_path)

        print('Reading CSV...')
        df = read_uci_csv(csv_path)
        df = preprocess_df(df)
        feature_cols = select_feature_columns(df, args.features)
        if len(feature_cols) == 0:
            raise RuntimeError('No numeric feature columns found. Provide --features explicitly.')
        print('Using features:', feature_cols)

        X, M = build_windows(df, feature_cols, args.window_length, stride=args.stride, pad=args.pad)

        # Build regimes per-window if requested
        regi = None
        if args.regime != 'none':
            per_time = make_regimes(df, method=args.regime)
            if per_time is not None:
                # Fill any NaNs in the time-based regimes by forward/back-filling,
                # then fallback to 0 for any remaining missing values. This avoids
                # conversion errors when casting to int and when windows include
                # timestamps with missing parsed datetimes.
                per_time = per_time.fillna(method='ffill').fillna(method='bfill').fillna(0)
                try:
                    per_time = per_time.astype(int)
                except Exception:
                    # If for some reason casting fails, coerce via numpy
                    per_time = per_time.to_numpy()
                    per_time = np.nan_to_num(per_time, nan=0).astype(int)

                # For each window, take the regime of the last timestep in the window
                regimes = []
                n_steps = len(per_time)
                for start in range(0, max(1, n_steps - args.window_length + 1), args.stride):
                    end = start + args.window_length - 1
                    if end >= n_steps:
                        if args.pad:
                            # padded windows get regime of last available time
                            regimes.append(int(per_time[-1]))
                        else:
                            continue
                    else:
                        regimes.append(int(per_time[end]))
                regi = np.array(regimes, dtype=np.int64)

        if X.shape[0] == 0:
            print('No windows created; writing empty arrays')

        # Save outputs
        np.save(out / 'X.npy', X)
        np.save(out / 'M.npy', M)
        if regi is None:
            e = np.zeros((X.shape[0],), dtype=np.int64)
        else:
            e = regi
        np.save(out / 'e.npy', e)
        # S: placeholder site metadata — for UCI single-station we put zeros
        S = np.zeros((X.shape[0], 1), dtype=np.float32)
        np.save(out / 'S.npy', S)

        meta = {'N': int(X.shape[0]), 'T': int(X.shape[1]) if X.shape[0]>0 else args.window_length, 'd': int(X.shape[2]) if X.shape[0]>0 else len(feature_cols)}
        with open(out / 'meta.json', 'w') as f:
            json.dump(meta, f)

        print('Wrote dataset to', out)


if __name__ == '__main__':
    main()
"""
Download the UCI Air Quality dataset (UCI archive or GitHub raw CSV) and convert it into
the NumPy dataset layout expected by RC-GNN: X.npy, M.npy, e.npy, S.npy, meta.json

Usage examples:

# Download from the UCI ML repo and convert with 24-hour windows:
python scripts/download_and_convert_uci.py --source uci --out data/interim/uci_air --window-length 24 --stride 1

# Download the raw CSV from GitHub and convert using month-based regimes:
python scripts/download_and_convert_uci.py --source github --out data/interim/uci_air --regime month

This script is intentionally simple and focused on reproducible preprocessing for RC-GNN.
"""

import argparse
import os
import io
import json
import zipfile
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import requests


UCI_ZIP_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip'
GITHUB_RAW_CSV = 'https://raw.githubusercontent.com/asharvi1/UCI-Air-Quality-Data/master/AirQualityUCI.csv'


def download_to_file(url, dst_path):
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(dst_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return dst_path


def extract_zip_and_find_csv(zip_path, dst_dir):
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(dst_dir)
        # find the first CSV-like file
        for nm in z.namelist():
            if nm.lower().endswith('.csv') or nm.lower().endswith('.txt'):
                return Path(dst_dir) / nm
    return None


def read_uci_csv(path):
    # UCI file uses semicolon separator and contains an extra trailing column
    df = pd.read_csv(path, sep=';', decimal=',')
    # drop completely empty columns (UCI has an extra empty column at the end)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    return df


def preprocess_df(df):
    # UCI uses -200 to signal missing values — convert to NaN
    df = df.replace(-200, np.nan)
    # Combine Date and Time if present into datetime
    if 'Date' in df.columns and 'Time' in df.columns:
        # Date format in UCI: dd/mm/yyyy ; Time: HH.MM.SS
        try:
            dt = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True, errors='coerce')
            df.insert(0, 'datetime', dt)
        except Exception:
            # fallback: try default parsing
            df.insert(0, 'datetime', pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce'))
    elif 'Time' in df.columns:
        df.insert(0, 'datetime', pd.to_datetime(df['Time'], errors='coerce'))

    # drop columns that are completely non-numeric (except datetime)
    non_numeric = [c for c in df.columns if c != 'datetime' and not pd.api.types.is_numeric_dtype(df[c])]
    # keep numeric replacements only
    for c in non_numeric:
        # If column looks like 'Date' or 'Time' we've already handled
        if c in ('Date', 'Time'):
            df = df.drop(columns=[c])
        else:
            # try to coerce to numeric
            df[c] = pd.to_numeric(df[c], errors='coerce')

    return df


def select_feature_columns(df, features=None):
    # Exclude datetime column
    cols = [c for c in df.columns if c != 'datetime']
    if features:
        # validate provided features exist
        cols = [c for c in features if c in df.columns]
    else:
        # automatically choose numeric columns
        cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    return cols


def build_windows(df, feature_cols, T, stride=1, pad=False):
    # sort by datetime if present
    if 'datetime' in df.columns:
        df = df.sort_values('datetime')
    arr = df[feature_cols].to_numpy(dtype=np.float32)
    # mask: 1 if not nan
    mask = (~np.isnan(arr)).astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0).astype(np.float32)
    n_steps, d = arr.shape
    windows = []
    masks = []
    if n_steps == 0:
        return np.zeros((0, T, d), dtype=np.float32), np.zeros((0, T, d), dtype=np.float32)
    if n_steps < T and pad:
        # pad beginning with zeros
        pad_amt = T - n_steps
        padded = np.zeros((T, d), dtype=np.float32)
        padded_mask = np.zeros((T, d), dtype=np.float32)
        padded[pad_amt:, :] = arr
        padded_mask[pad_amt:, :] = mask
        return padded[np.newaxis, ...], padded_mask[np.newaxis, ...]
    for start in range(0, max(1, n_steps - T + 1), stride):
        win = arr[start:start+T]
        mwin = mask[start:start+T]
        if win.shape[0] != T:
            # skip incomplete window unless pad=True
            if pad:
                pad_amt = T - win.shape[0]
                pw = np.zeros((T, d), dtype=np.float32)
                pm = np.zeros((T, d), dtype=np.float32)
                pw[pad_amt:, :] = win
                pm[pad_amt:, :] = mwin
                win = pw; mwin = pm
            else:
                continue
        windows.append(win)
        masks.append(mwin)
    X = np.stack(windows, axis=0)
    M = np.stack(masks, axis=0)
    return X, M


def make_regimes(df, method='none'):
    # For single-station UCI dataset we can create regimes by month or season
    if method == 'none':
        return None
    if 'datetime' not in df.columns:
        return None
    dt = pd.to_datetime(df['datetime'], errors='coerce')
    if method == 'month':
        return dt.dt.month
    if method == 'season':
        # meteorological seasons by month
        m = dt.dt.month
        # Dec-Feb=0, Mar-May=1, Jun-Aug=2, Sep-Nov=3
        season = ((m % 12 + 3) // 3 - 1) % 4
        return season
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', choices=['uci', 'github', 'raw'], default='uci', help='Where to download the dataset from')
    p.add_argument('--raw-url', help='If source=raw, the direct CSV URL')
    p.add_argument('--out', required=True, help='Output directory to write dataset into')
    p.add_argument('--window-length', '-T', type=int, default=24, help='Window length in timesteps')
    p.add_argument('--stride', type=int, default=1, help='Window stride')
    p.add_argument('--features', nargs='*', help='Feature column names to keep (default: auto numeric columns)')
    p.add_argument('--regime', choices=['none', 'month', 'season'], default='none', help='How to construct regimes (e.npy)')
    p.add_argument('--pad', action='store_true', help='Pad short series to window length instead of dropping')
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        if args.source == 'uci':
            print('Downloading UCI archive...')
            zip_path = td / 'AirQualityUCI.zip'
            download_to_file(UCI_ZIP_URL, zip_path)
            csv_path = extract_zip_and_find_csv(zip_path, td)
            if csv_path is None:
                raise RuntimeError('Could not find CSV inside the UCI ZIP archive')
        elif args.source == 'github':
            print('Downloading raw CSV from GitHub...')
            csv_path = td / 'AirQualityUCI.csv'
            download_to_file(GITHUB_RAW_CSV, csv_path)
        else:
            if not args.raw_url:
                raise ValueError('When --source raw you must provide --raw-url')
            csv_path = td / 'raw.csv'
            download_to_file(args.raw_url, csv_path)

        print('Reading CSV...')
        df = read_uci_csv(csv_path)
        df = preprocess_df(df)
        feature_cols = select_feature_columns(df, args.features)
        if len(feature_cols) == 0:
            raise RuntimeError('No numeric feature columns found. Provide --features explicitly.')
        print('Using features:', feature_cols)

        X, M = build_windows(df, feature_cols, args.window_length, stride=args.stride, pad=args.pad)

        # Build regimes per-window if requested
        regi = None
        if args.regime != 'none':
            per_time = make_regimes(df, method=args.regime)
            if per_time is not None:
                # For each window, take the regime of the last timestep in the window
                regimes = []
                n_steps = len(per_time)
                for start in range(0, max(1, n_steps - args.window_length + 1), args.stride):
                    end = start + args.window_length - 1
                    if end >= n_steps:
                        if args.pad:
                            # padded windows get regime of last available time
                            regimes.append(int(per_time.iloc[-1]))
                        else:
                            continue
                    else:
                        regimes.append(int(per_time.iloc[end]))
                regi = np.array(regimes, dtype=np.int64)

        if X.shape[0] == 0:
            print('No windows created; writing empty arrays')

        # Save outputs
        np.save(out / 'X.npy', X)
        np.save(out / 'M.npy', M)
        if regi is None:
            e = np.zeros((X.shape[0],), dtype=np.int64)
        else:
            e = regi
        np.save(out / 'e.npy', e)
        # S: placeholder site metadata — for UCI single-station we put zeros
        S = np.zeros((X.shape[0], 1), dtype=np.float32)
        np.save(out / 'S.npy', S)

        meta = {'N': int(X.shape[0]), 'T': int(X.shape[1]) if X.shape[0]>0 else args.window_length, 'd': int(X.shape[2]) if X.shape[0]>0 else len(feature_cols)}
        with open(out / 'meta.json', 'w') as f:
            json.dump(meta, f)

        print('Wrote dataset to', out)


if __name__ == '__main__':
    main()
