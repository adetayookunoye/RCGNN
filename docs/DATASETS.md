# Datasets and data utilities

This project includes small utilities to prepare datasets for RC-GNN.

Scripts:

- `scripts/convert_to_numpy.py` — convert CSV/Parquet panel data (single-file with `example_id` or directory of per-example files) into the NumPy layout expected by the project (`X.npy`, `M.npy`, `e.npy`, `S.npy`).
- `scripts/download_and_convert_uci_safe.py` — downloader + converter for the UCI Air Quality dataset (downloads raw CSV/ZIP and converts into windows with masks and optional regime labels).

Typical usage:

1. Download and convert the UCI dataset (24-hour windows, monthly regimes):

```bash
python scripts/download_and_convert_uci_safe.py --source github --out data/interim/uci_air --window-length 24 --stride 1 --regime month
```

2. Convert custom CSV panel data:

```bash
python scripts/convert_to_numpy.py --input data/raw/panel.csv --out data/interim/my_dataset --id_col example_id --time_col timestamp --features f1 f2 f3
```

After running, point training to the dataset folder (the training script constructs `root = os.path.join(dc["paths"]["root"], "interim", "synth_small")` by default). You can either place your dataset under the expected path or call the training script after editing `scripts/train_rcgnn.py` to point to your dataset folder.
