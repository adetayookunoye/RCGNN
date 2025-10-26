# Copilot instructions for contributors and AI agents

Purpose: brief, actionable guidance so an AI agent can safely edit, extend, and run the RC-GNN codebase.

Key concepts
- This repo implements RC-GNN: a graph-structure learner + tri-latent encoders (signal/noise/bias). Main model sits in `src/models/rcgnn.py` and composes:
  - `src/models/encoders.py` — encoders for signal/noise/bias and batched imputer with uncertainty
  - `src/models/structure.py` — `StructureLearner` with environment-specific adjacency deltas, temperature annealing, and differentiable sparsification
  - `src/models/mechanisms.py`, `recon.py`, `losses.py` — mechanisms, reconstruction with uncertainty, and disentanglement losses
  - `src/models/missingness.py` — MNAR missingness modeling
  - `src/models/disentanglement.py` — improved latent space disentanglement via MINE/InfoNCE
- Data pipeline: synthetic datasets live under `data/interim/synth_small/` and are loaded by `src/dataio/loaders.py` via `load_synth(root, split, seed)` which returns a PyTorch Dataset producing dicts: {"X","M","e","S"}.
- Training/eval scripts:
  - `scripts/synth_bench.py` — generates tiny synthetic benchmarks (see README quickstart)
  - `scripts/train_rcgnn.py` — orchestrates training loop (DataLoader -> model -> optimizer -> save best checkpoint to `artifacts/checkpoints/rcgnn_best.pt` and adjacency to `artifacts/adjacency/A_mean.npy`)
  - `scripts/eval_rcgnn.py` — loads `A_mean.npy` and plots adjacency
  - `scripts/run_baselines.py` — simple baseline runner (e.g. `notears_lite`) that loads `X.npy` and `A_true.npy` from dataset root

Developer workflows & commands
- Quickstart (CPU): `pip install -r requirements.txt` then follow README.
- Run unit tests: `pytest -q` (tests live in `tests/` — small smoke tests are included).
- Train tiny CPU run for debugging: `python scripts/train_rcgnn.py configs/data.yaml configs/model.yaml configs/train.yaml`.
- Baseline run: `python scripts/run_baselines.py --method notears_lite --config configs/data.yaml`.

Conventions important for edits
- Path helper: scripts import `path_helper` at top (it mutates sys.path to include project root). When editing scripts, preserve this import line or ensure imports stay resolvable.
- Configs: YAML-based configs in `configs/` are authoritative; prefer adding/reading flags there instead of hardcoding values in scripts.
- Devices: `configs/train.yaml` sets `device: "cpu"` by default. Don't assume GPU unless config changed.
- Dataset splitting: `SynthDataset` splits by regime identifiers in `e` (regimes are shuffled then split), so changes to splitting semantics must respect `e` as regime index.
- Structure learning: `StructureLearner.forward` returns A and logits; per-environment deltas controlled by `structure.n_envs` in config. Base adjacency shared across envs with learned deltas per env.
- Sparsification: `structure.sparsify.method` selects between `topk`, `sparsemax`, `entmax`, or `gumbel_topk`. Each provides differentiable sparsity.
- Uncertainty handling: Imputer and reconstruction now provide uncertainty estimates. MNAR missingness model helps calibrate uncertainties.
- Batched operations: All components handle batched inputs `[B,T,d]` natively. Per-batch invariance losses computed efficiently.

Safe edit rules for AI agents
- Keep public script CLI signatures intact. Scripts are used in README quickstart; preserve args and config file ordering unless intentionally version-bumping.
- When modifying model checkpoints: follow existing artifact layout (`artifacts/checkpoints/rcgnn_best.pt`, `artifacts/adjacency/A_mean.npy`). Tests and eval expect those paths.
- Avoid adding heavy dependencies. This project targets CPU-comfortable runs; keep changes lightweight and prefer standard libs (numpy, torch) already pinned in `requirements.txt`.
- Preserve random seeds in `configs/train.yaml` for reproducible smoke tests (seed: 1337).

Examples of common edits
- To change temperature schedule in `StructureLearner`, edit `temperature()` in `src/models/structure.py` and ensure `StructureLearner.step` is registered as a buffer. Run `pytest -q` and a small `train_rcgnn` run to validate.
- To modify per-environment structure learning, adjust `structure.n_envs` in config and verify env_deltas initialization in `StructureLearner`.
- To tune disentanglement, adjust `loss.disentangle` config (method: "mine"/"infonce") and verify with disentanglement metrics during training.
- To adjust uncertainty modeling, check imputer and reconstruction uncertainty outputs along with MNAR missingness predictions.
- To add a new CLI flag, add it to the script `argparse` parsing and to the corresponding `configs/*.yaml` with a default. Update README quickstart only if user-facing behavior changes.

Files to inspect when debugging
- `src/models/rcgnn.py` — model composition and training losses (including invariance and disentanglement)
- `src/models/structure.py` — adjacency logits, temperature annealing, differentiable sparsification
- `src/models/encoders.py` — batched imputer with uncertainty, MNAR missingness integration
- `src/dataio/loaders.py` — dataset split logic by regime
- `scripts/train_rcgnn.py`, `scripts/eval_rcgnn.py`, `scripts/run_baselines.py` — runnable entrypoints
- `configs/*` — default configs for data/model/train/eval (check structure.n_envs and sparsify.method)

If you need to run tests or training but are unsure of environment
- Use CPU defaults in `configs/train.yaml`. For faster local iter, reduce `epochs` and `batch_size` in `configs/train.yaml`.

What I could not auto-discover
- Any non-checked-in dataset roots (configs refer to `paths.root` in YAML). When modifying dataset flow, confirm `configs/data.yaml` points to existing `data/interim/...` during runs.

If something is unclear, ask for which area to expand (training loop details, loss terms, or dataset format). Please review this draft and tell me which parts you want expanded or merged differently.