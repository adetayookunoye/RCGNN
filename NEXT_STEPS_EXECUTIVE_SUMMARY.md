# 🎯 RC-GNN Next Steps: Executive Summary

**Date**: October 27, 2025  
**Status**: Integration validated ✅ | Experiments pending 🚀

---

## Where You Are

✅ **Complete**:
- Architecture: Tri-latent encoder, structure learner, reconstruction module
- Training: All 6 loss components integrated and computing
- Core integration: End-to-end training validated on UCI Air (100+ epochs running)
- Codebase quality: Syntax validated, imports verified, no runtime crashes

⚠️ **Partially Complete**:
- Evaluation: Standard metrics (SHD, F1) working; stability metrics missing
- Experiments: No hypothesis tests (H1/H2/H3) yet run
- Baselines: Integration ready but no comparative results yet

❌ **Missing**:
- Masking-aware imputer (GRU-D) with uncertainty
- Edge-specific transformation networks
- Mechanism fitting loss
- Stability metrics (adjacency variance, Jaccard, policy consistency)
- Synthetic corruption benchmarks
- Experimental validation of 3 core hypotheses

---

## Where You Need to Go: 13-18 Days to Publication

### 🔴 **Week 1: Infrastructure** (Oct 29 - Nov 2)

**Must Complete**:

1. **GRU-D Imputer** (2 days)
   - Add masking-aware time-series imputation to encoders.py
   - Integrate into TriLatentEncoder before encoding
   - Return uncertainty estimates for reconstruction loss
   - Test on UCI Air with 50% simulated missingness

2. **Stability Metrics** (1 day)
   - Adjacency variance: $\mathrm{Var}_{e,e'}[\|A^{(e)} - A^{(e')}\|_F]$
   - Edge-set Jaccard: $J(E^{(e)}, E^{(e')})$
   - Policy consistency: Pathway stability across regimes
   - Modify eval_epoch to compute after each epoch

3. **Synthetic Benchmark** (1-2 days)
   - Extend synth_bench.py to create multi-environment datasets
   - Tunable parameters: missingness (40-60%), noise, drift
   - 4-5 environments with known $G^\star$
   - Save to data/interim/synth_corrupted/

**Why Critical**: 
- GRU-D handles missingness as described in paper (Section 4.1)
- Stability metrics validate paper's core innovation (structure invariance)
- Benchmarks enable H1/H2 hypothesis tests

---

### 🔴 **Week 2: Theoretical Validation** (Nov 3 - Nov 8)

**Must Complete**:

1. **Identifiability Test** (1-2 days)
   - Run RC-GNN on synthetic data with known $G^\star$
   - Vary environment diversity; measure SHD
   - Validate: SHD low despite corruption shifts
   - Demonstrates: Proposition 1 (identifiability under multi-environment corruption)

2. **Stability Bound Verification** (1-2 days)
   - Measure actual cross-environment adjacency variance
   - Compare against theoretical bound: $\frac{2}{\lambda_{\text{inv}}} \mathbb{E}[\mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{disent}}]$
   - Demonstrates: Proposition 2 (stability improvement guarantee)

**Why Critical**: 
- Paper makes strong theoretical claims; experiments validate them
- Reviewers will scrutinize whether propositions hold in practice
- Results here form foundation for H1/H2 experiments

---

### 🔴 **Week 3: Hypothesis Testing** (Nov 8 - Nov 13)

**Must Complete** (Parallelize with Week 2):

1. **H1: Structural Accuracy** (2-3 days)
   - Generate synthetic: 4-5 environments, 40-60% MCAR/MAR/MNAR, known $G^\star$
   - Compare: RC-GNN vs NOTEARS vs DCDI vs DECI vs MissDAG
   - Measure: SHD, Precision, Recall, F1
   - **Success**: RC-GNN SHD within 15% of oracle; baselines >40% SHD increase
   - **Deliverable**: experiments/h1_results_table.csv + comparison figure

2. **H2: Stability Improvement** (1-2 days)
   - Run RC-GNN with $\lambda_{\text{inv}} > 0$ vs $\lambda_{\text{inv}} = 0$
   - Measure: Cross-environment adjacency variance for both
   - **Success**: >60% variance reduction with invariance
   - **Deliverable**: experiments/h2_stability_reduction.pdf + numbers

3. **H3: Expert Agreement** (Parallel, 2-3 days)
   - Real dataset (UCI Air): Extract learned graphs
   - Domain expert evaluation: Policy-relevance scoring
   - **Success**: >80% expert agreement on RC-GNN
   - **Alternative**: Literature-based validation if expert unavailable
   - **Deliverable**: experiments/h3_expert_validation_report.pdf

**Why Critical**: 
- These three hypotheses ARE the paper's main contributions
- Each directly addresses a claim in the abstract
- Reviewers will demand evidence for each

---

### 🟡 **Week 4: Baselines & Ablations** (Nov 13 - Nov 16)

**Complete**:

1. **Baseline Integration** (1 day)
   - Ensure all methods (NOTEARS, DCDI, DECI, MissDAG) run on same datasets
   - Unified evaluation script
   - Statistical significance tests (t-tests, confidence intervals)

2. **Ablation Studies** (1 day)
   - RC-GNN$_{\setminus \text{inv}}$: Remove invariance
   - RC-GNN$_{\setminus \text{disent}}$: Single latent
   - RC-GNN$_{\setminus \text{recon}}$: Standard reconstruction
   - Report component contributions

**Why Important**:
- Demonstrates your innovations aren't just adding more losses
- Shows each component (invariance, disentanglement) helps

---

### 🟡 **Week 5: Results & Paper** (Nov 16 - Nov 22)

**Complete**:

1. **Generate Figures** (1-2 days)
   - SHD comparison curves (H1)
   - Stability improvement plots (H2)
   - Adjacency heatmaps + learned graphs
   - Robustness curves (performance vs corruption intensity)

2. **Write Results Section** (1-2 days)
   - Summarize H1/H2/H3 findings
   - Report: Mean ± std, statistical tests
   - Discuss surprising results
   - Connect back to paper claims

3. **Final Paper** (1 day)
   - Integrate results into draft
   - Proofread + polish
   - Address anticipated reviewer questions

---

## Your Specific Next Actions (This Week)

### 🎯 **Must Do by Friday (Oct 31)**

1. **Review Gap Analysis** (1 hour)
   - Read: `PAPER_CODE_GAP_ANALYSIS.md`
   - Understand: What's missing vs paper spec

2. **Plan GRU-D Imputer** (2 hours)
   - Read: Paper Section 4.1 (Eq. 3-5)
   - Read: Che et al. (GRU-D paper) or Yoon et al. (GAIN)
   - Decide: Full GRU-D or simplified time-aware imputation?
   - Create: `src/models/encoders.py` skeleton with interface

3. **Create Stability Metrics Script** (2 hours)
   - Add 3 functions to `src/training/metrics.py`:
     ```python
     def adjacency_variance(A_by_env):
     def edge_set_jaccard(E_by_env):
     def policy_consistency(A_by_env, policy_edges):
     ```
   - Test on dummy data to verify correctness

4. **Plan Synthetic Benchmark** (1 hour)
   - Design: Parameters for MCAR/MAR/MNAR generation
   - Sketch: extend_synth_bench() function signature
   - List: Dependencies needed (missing data library, etc.)

**Total Time**: ~6 hours  
**Outcome**: Ready to start implementation Monday

---

## Key Decisions to Make Now

| Decision | Options | Recommendation |
|----------|---------|-----------------|
| **Imputer** | Full GRU-D vs simplified vs pretrained? | Simplified (~2 layers) if time tight, full GRU-D if 3+ days |
| **Edge Networks** | Full $d \times d$ vs factorized? | Factorized (source + target embeddings) for efficiency |
| **Expert (H3)** | Real expert vs literature validation? | Literature-based if no expert available (faster) |
| **Baseline Suite** | All 6 methods or just 3 strongest? | Focus on NOTEARS, DCDI, DECI first; add MissDAG if time |
| **GPU/CPU** | Assume GPU or CPU-only? | CPU-only recommended (broader accessibility) |

---

## Success Criteria: How to Know You're Done

✅ **Core Results**:
- [ ] H1: RC-GNN SHD within 15% of oracle under 50% missingness; baselines degrade >40%
- [ ] H2: >60% variance reduction with invariance vs without
- [ ] H3: >80% expert/literature agreement on learned pathways

✅ **Paper Quality**:
- [ ] All 6 loss components explained and validated
- [ ] Propositions 1-2 empirically demonstrated
- [ ] Figures: SHD curves, stability plots, heatmaps
- [ ] Baselines compared with statistical significance
- [ ] Ablations show component contributions

✅ **Reproducibility**:
- [ ] All configs documented in `configs/`
- [ ] Scripts runnable with: `python scripts/train_rcgnn.py configs/...`
- [ ] Results reproducible (fixed seeds, no randomness in metrics)
- [ ] Benchmarks data in `data/interim/` with README

---

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|-----------|
| **GRU-D complex** | Week delay | Use simplified imputation first, iterate |
| **H1 fails** (RC-GNN not better) | Paper claim disproven | Debug: Check hyperparameters, loss weights, initialization |
| **Expert unavailable (H3)** | Missing validation | Use literature-based approach (air quality causal graphs) |
| **Baselines slow** | Timeline pressure | Pre-compute on small dataset first, parallelize on multiple machines |
| **Synthetic corruptions unrealistic** | Limited applicability | Validate against real UCI Air corruption patterns first |

---

## Final Checklist: What You Have vs What You Need

| Component | Have | Need | Effort |
|-----------|------|------|--------|
| Tri-latent encoder | ✅ | ✅ | 0h |
| Structure learner | ✅ | ✅ | 0h |
| Reconstruction module | ✅ | ✅ | 0h |
| Training loop | ✅ | ✅ | 0h |
| All 6 losses | ✅ | ✅ | 0h |
| **GRU-D imputer** | ❌ | ✅ | 8h |
| **Stability metrics** | ❌ | ✅ | 4h |
| **Synthetic benchmarks** | ⚠️ | ✅ | 6h |
| **H1 experiment** | ❌ | ✅ | 12h |
| **H2 experiment** | ❌ | ✅ | 8h |
| **H3 experiment** | ❌ | ✅ | 8h |
| **Baselines** | ⚠️ | ✅ | 6h |
| **Results writing** | ❌ | ✅ | 8h |
| **Total** | | | **60h = ~1.5 weeks** |

---

## Questions to Get You Started

1. **Imputer Complexity**: Do you want full GRU-D (more realistic but complex) or simplified masking-aware imputation (faster)?

2. **Expert Access**: Do you have access to an air quality expert for H3 validation, or should I prioritize literature-based approach?

3. **Computational Resources**: Any GPU clusters available, or should all experiments assume 8-core CPU?

4. **Paper Target**: Aiming for conference (NeurIPS, ICML deadline soon?) or journal?

5. **Baseline Priority**: Should we focus on NOTEARS/DCDI/DECI or include MissDAG and robust variants?

---

**You're on track.** The integration is validated. Now it's systematic execution of experiments to prove the paper's claims. 🚀

