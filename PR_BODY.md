Title: Fixes and stability improvements — imputer uncertainty, sparsification, invariance, structure

Summary

This PR contains a collection of targeted fixes and small improvements to stabilize training, improve sparsification, and ensure invariance/temporal behaviors in RC-GNN. All changes are unit-tested and the full test suite passes locally (45 tests).

Key changes

- Imputer (TransformerImputer)
  - Ensure missing-value uncertainty is not reduced below model-predicted sigma.
  - Deterministic MC-dropout seeding for reproducible tests.
  - Prevent tiny variances for missing entries which caused exploding NLL gradients.

- Reconstruction (Recon)
  - MNAR scaling computed from ZB and amplified by clamped missingness.
  - Ensures total_unc >= base_unc.

- Sparsification
  - Robust `sparsemax` implementation with thresholding and renormalization.
  - `entmax15` delegates to sparsemax for stable behavior.
  - Configurable module-level threshold `set_sparse_eps(eps)`.

- Sparse attention / LowRankProjection
  - `LowRankProjection` now accepts both (d_in, rank) and (d_in, d_out, rank) calling conventions and preserves output shapes.

- Structure learner
  - Enforce monotonic non-increasing adjacency across lags so temporal prior has decreasing influence with lag.

- Invariance (IRM)
  - IRM gradient penalty now attempts to compute gradients w.r.t. `logits` directly so `logits.grad` is populated; falls back to previous approach if needed.

- Training loop
  - Gradient clipping (norm=1.0) applied before optimizer.step.
  - Gradient-norm logging added for debugging spikes (prints per-minibatch max/mean/top3 param grad norms).

Tests added

- `tests/test_invariance_grad_flow.py` — asserts gradients flow to logits
- `tests/test_low_rank_projection_shape.py` — validates LowRankProjection shapes
- `tests/test_regress_sparsification.py` — asserts sparsemax zero-ratio and entmax behavior
- `tests/test_regress_imputer.py` — ensures imputer returns finite sigma and sigma floor for missing entries

Checklist

- [ ] Review changes and confirm behavior
- [ ] Optionally run full training locally to validate on larger runs
- [ ] Merge after approvals

How to run tests locally

```bash
# from repo root
PYTHONPATH=. pytest -q
```

Notes

- The workspace is not a git repository here; I did not create commits. If you want, I can initialize a git repo here and make a commit with a suggested message (I will ask before doing that).
- I also created `CHANGES.md` with a concise per-file summary.

If you'd like me to create a local commit and/or produce a patch file (`git format-patch`), tell me which option you prefer (I can init a git repo and commit, or create a tarball/zip of the changes).