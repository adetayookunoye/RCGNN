
# RC-GNN (Robust Causal Graph Neural Networks under Compound Sensor Corruptions)

This is a **working reference implementation** of RC-GNN, including:
- Synthetic benchmark generator with ground-truth graph
- Tri-latent encoders (signal/noise/bias)
- Structure learner with NOTEARS acyclicity penalty and sparsity
- Corruption-aware reconstruction
- Invariance penalty across regimes
- Training & evaluation scripts
- Simple baseline (correlation-threshold "NOTears-lite") for CPU-only use
- Unit tests and quickstart

> CPU smoke test runs in ~2–3 minutes.

## Quickstart (CPU)
```bash
# (Optional) python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) generate tiny synthetic benchmark
python scripts/synth_bench.py configs/data.yaml configs/model.yaml configs/train.yaml

# 2) train RC-GNN on synthetic (tiny run)
python scripts/train_rcgnn.py configs/data.yaml configs/model.yaml configs/train.yaml

# 3) evaluate & plot
python scripts/eval_rcgnn.py configs/eval.yaml --export reports/figs/

# 4) run a simple baseline
python scripts/run_baselines.py --method notears_lite --config configs/data.yaml
```

## Tests
```bash
pytest -q
```

Artifacts and figures will appear under `reports/` and `artifacts/`.
