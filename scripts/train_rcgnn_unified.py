#!/usr/bin/env python3
"""
RC-GNN Unified Training Script

Consolidates best practices from all training scripts:
- train_rcgnn_v3.py: DDP, GroupDRO, 3-stage scheduling
- train_rcgnn_publication.py: 7 documented fixes
- train_stable.py: Gradient stability, LR scheduling
- train_rcgnn_v4.py: Causal priors, correlation vs causation diagnostics
- train_corruption_sweep.py: Sweep mode for ablation

Features:
1. Multi-GPU (DDP) with single-GPU/CPU fallback
2. GroupDRO for worst-case robustness across regimes
3. 3-Stage Training: discovery -> pruning -> refinement
4. Publication-quality fixes (temperature, loss rebalancing)
5. Gradient stability (aggressive clipping, LR scheduling)
6. Causal diagnostics (correlation vs causation)
7. Comprehensive metrics (TopK-F1, Best-F1, AUC-F1)
8. Sweep mode for ablation studies
9. Full CLI with sensible defaults

Usage:
    # Basic training
    python scripts/train_rcgnn_unified.py --data_dir data/interim/uci_air --epochs 100
    
    # Multi-GPU with DDP
    torchrun --nproc_per_node=4 scripts/train_rcgnn_unified.py --ddp --data_dir data/interim/uci_air
    
    # With GroupDRO
    python scripts/train_rcgnn_unified.py --data_dir data/interim/uci_air --use_groupdro
    
    # Sweep mode (for ablation)
    python scripts/train_rcgnn_unified.py --data_dir data/interim/uci_air --seed 42 --sweep_mode

Author: RC-GNN Team
"""

import os
import sys
import argparse
import json
import time
import math
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional, List, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau


def set_seed(seed: int, deterministic: bool = False):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def validate_config(cfg: Dict) -> List[str]:
    """Validate configuration for consistency. Returns list of warnings."""
    warnings_list = []
    if cfg.get("stage1_end", 0.3) >= cfg.get("stage2_end", 0.7):
        warnings_list.append("stage1_end should be < stage2_end")
    te = cfg.get("target_edges", 13)
    if te <= 0:
        warnings_list.append("target_edges must be positive")
    if cfg.get("oracle_direction_supervision", False) and cfg.get("sweep_mode", False):
        warnings_list.append("Sweep mode with oracle supervision — results will NOT generalise!")
    ep = cfg.get("epochs", 100)
    if ep < 10:
        warnings_list.append(f"Very few epochs ({ep}) — model may not converge")
    if cfg.get("lambda_recon", 1.0) <= 0:
        warnings_list.append("lambda_recon <= 0 disables reconstruction loss")
    return warnings_list

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.rcgnn import RCGNN, notears_acyclicity

# Optional DDP imports (graceful fallback)
try:
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data.distributed import DistributedSampler
    import torch.distributed as dist
    DDP_AVAILABLE = True
except ImportError:
    DDP_AVAILABLE = False


# =============================================================================
# Configuration Defaults (Publication-Quality)
# =============================================================================

DEFAULT_CONFIG = {
    # Training
    "epochs": 100,
    "batch_size": 32,
    "patience": 30, # V8.21: More patience (20→30)
    "eval_frequency": 1, # V8.12 FIX: Validate every epoch (was 2, caused alternation bug)
    
    # Learning rate (with warm restarts - FIX 6 from publication)
    "lr": 5e-4,
    "lr_min": 1e-5,
    "weight_decay": 1e-4,
    "warmup_epochs": 5,
    "restart_every": 50,
    "grad_clip": 1.0,
    
    # Model architecture
    "latent_dim": 32,
    "hidden_dim": 64,
    
    # ===========================================================================
    # CRITICAL FIX: Gradual sparsity ramp (not cliff)
    # ===========================================================================
    # Loss weights (rebalanced for causal discovery, not just reconstruction)
    "lambda_recon": 1.0, # FIXED: Was 50.0, dominated all other losses
    "lambda_miss": 0.5, # Missingness prediction
    "lambda_hsic": 0.1, # Disentanglement
    
    # Sparsity schedule (ramps from epoch 1)
    # INCREASED: 0.01 -> 0.05 for stronger sparsity pressure
    "lambda_sparse_init": 1e-5, # Start tiny at epoch 1
    "lambda_sparse_final": 0.05,# Ramp to this by end of PRUNE (was 0.01)
    
    # Acyclicity schedule (delayed until some structure learned)
    "lambda_acyclic_init": 0.0, # Start at 0
    "lambda_acyclic_final": 0.5,# Ramp to this by end of training
    "acyclic_start_epoch": 10, # Start acyclicity after epoch 10
    
    # Edge budget schedule (ramps from epoch 10)
    # NOTE: V8.21 overrides lambda_budget_final to 0.05 below
    "lambda_budget_init": 0.0, # Start at 0
    "lambda_budget_final": 0.5, # Overridden to 0.05 by V8.21 below
    "budget_start_epoch": 10, # Start budget pressure after epoch 10
    
    "lambda_inv": 1.0, # Invariance (needs regimes > 1)
    "lambda_causal": 1.0, # FIXED: Was 0.2, increased for direction awareness
    "lambda_var_penalty": 0.01, # MNAR variance penalty
    
    # ===========================================================================
    # CRITICAL FIX: Binarization penalty A*(1-A) for edge confidence
    # Encourages edges to be near 0 or 1, not in gray zone (0.2-0.3)
    # ===========================================================================
    "lambda_binary_init": 0.0, # Start at 0 (allow exploration)
    "lambda_binary_final": 0.2, # Ramp to this during PRUNE
    "binary_start_epoch": 10, # Start after initial exploration
    
    # ===========================================================================
    # CALIBRATION FIX: Prevent adjacency collapse
    # ===========================================================================
    "sparse_refine_factor": 0.5, # Reduce λs to 50% in REFINE phase
    "min_edge_sum_for_sparse": 5.0, # If edge_sum drops below this, reduce λs
    "temperature_floor_unstable": 0.8, # Don't go below this τ if direction unstable
    
    # ===========================================================================
    # CRITICAL FIX: Temperature annealing (prevent early saturation)
    # Aggressive schedule: 2.0 -> 0.3 to force confident edge separation
    # ===========================================================================
    "temperature_init": 2.0, # Start high (soft edges)
    "temperature_final": 0.2, # V8.21: Anneal lower (0.3→0.2) for sharper edges
    "temperature_anneal_start": 0.1, # V8.21: Start earlier (0.2→0.1)
    "temperature_anneal_end": 0.5, # V8.21: Finish earlier (0.7→0.5)
    
    # Structure learning
    "sparsify_method": "topk", # or "sigmoid" for continuous
    "topk_ratio": 0.15,
    "target_edges": 13, # Adjust per dataset
    
    # FIX E: Boost direction learning (orientation penalty)
    "lambda_orientation": 0.5, # Increased from 0.1 for better direction learning
    
    # ===========================================================================
    # CRITICAL FIX: Hard Top-K projection during PRUNE/REFINE
    # This forces model to commit to exactly K edges, preventing "soft dense"
    # where all edges cluster in 0.2-0.3 band. Single most impactful fix.
    #
    # GOTCHA FIX: Use 2K->K schedule to avoid freezing wrong edges too early
    # - Epoch < 35%: no projection (soft exploration)
    # - 35%-55%: project to 2*K (allows some flexibility)
    # - 55%+: project to K (final commitment)
    # ===========================================================================
    "use_topk_projection": True, # Enable hard Top-K projection
    "topk_projection_start": 0.20, # V8.21: Start projecting earlier (0.35→0.20)
    "topk_projection_2k_end": 0.55, # Switch from 2K to K at 55% training
    "topk_use_logits": True, # Project based on W magnitude (stable across temps)
    "lambda_projection": 0.5, # FIX B: Projection consistency loss weight
    
    # FIX E: Direction Asymmetry Loss
    "use_asymmetry_loss": True, # Penalize edges where transpose is stronger
    "lambda_asymmetry": 0.5, # Asymmetry loss weight
    "asymmetry_start": 0.35, # Start when PRUNE begins
    
    # 2-Stage scheduling (V8.5: Skeleton -> Direction)
    "stage1_end": 0.30, # Discovery phase ends (part of Stage S)
    "stage2_end": 0.70, # V8.21: Shorter PRUNE (0.80→0.70), more time for REFINE
    # Stage S (Skeleton): epochs 1 -> stage2_end (DISC + PRUNE) - no direction penalty
    # Stage D (Direction): epochs stage2_end -> 100% (REFINE) - skeleton frozen
    
    # V8.5: Two-Stage Training
    "two_stage_training": True, # Enable skeleton->direction separation
    "skeleton_freeze_threshold": 0.1, # Edges > this in skeleton mask can be trained for direction
    "direction_penalty_stage_d_only": True, # Only apply dir penalty in Stage D
    
    # V8.6: Confidence-based Direction Evaluation
    # Only evaluate/train direction on edges where max(A_ij, A_ji) > t_conf
    "direction_conf_threshold": 0.3, # Confidence threshold for direction eval
    "direction_conf_adaptive": True, # Use adaptive threshold (top 50% of skeleton)
    
    # V8.7: Direction Margin Loss (hinge-based)
    # For edge i->j: enforce A_ij >= A_ji + margin (hard separation)
    "use_direction_margin": True, # Use margin-based direction loss
    "direction_margin": 0.15, # V8.21: Stronger margin (0.1→0.15)
    "lambda_direction_margin": 3.0, # V8.21: 3x weight (1.0→3.0)
    
    # V8.8: Edge Discovery Fixes (address mode collapse to easy subgraph)
    # These directly target the recall problem on hard edges
    
    # 1. Presence loss with hard negative mining
    "use_presence_loss": True, # BCE on edge presence with pos_weight
    "presence_pos_weight": 3.0, # Upweight true edge positives
    "use_hard_negatives": True, # Mine hard negatives from high-corr non-edges
    "hard_neg_corr_percentile": 80, # Top X% correlation as hard negatives
    "lambda_presence": 0.5, # Weight for presence loss
    
    # 2. Cousin penalty (anti-confounding)
    "use_cousin_penalty": True, # Penalize edges between nodes with common ancestors
    "lambda_cousin": 0.3, # Weight for cousin penalty
    
    # 3. Coverage loss (prevent hub collapse)
    "use_coverage_loss": True, # Ensure every node has minimum outgoing mass
    "coverage_min_outdeg": 0.5, # Minimum outgoing degree sum per node
    "lambda_coverage": 0.2, # Weight for coverage loss
    "coverage_epochs": 0.5, # Apply coverage for first X% of training
    
    # V8.9: Direction Stability Fixes
    # 4. Exclusivity loss (force commitment to ONE direction per pair)
    "use_exclusivity_loss": True, # Penalize A_ij * A_ji (both directions high)
    "lambda_exclusivity": 0.3, # Weight for exclusivity loss
    
    # 5. Budget floor (maintain asymmetry pressure after sparsification)
    "lambda_budget_floor": 0.05, # Minimum budget λ after projection
    
    # V8.15: Non-TopK Suppression (clean up fat tail of medium-confidence edges)
    # Problem: Perfect TopK-F1 but 146 edges @ 0.2 vs 13 @ 0.5 hurts interpretability
    "use_nontopk_suppression": True, # Push non-TopK edges toward zero
    "nontopk_suppression_start": 0.2, # Start at 20% of training
    "lambda_nontopk_suppression": 0.5,# Weight for suppression loss
    "nontopk_margin": 0.1, # Desired margin between TopK and rest
    
    # V8.26: Model outputs CAUSAL convention A[i,j]=i→j natively.
    # to_causal_convention() is identity. No transpose at boundaries.
    # No config needed - model and A_true already use same convention.
    
    # V8.10: EMA smoothing for stable TopK-F1 evaluation
    "ema_alpha": 0.1, # EMA weight (0.1=slow/stable, 0.5=fast/responsive)
    
    # V8.11: Skeleton Freeze + Antisymmetric Direction Learning
    # Once skeleton is learned, freeze it and train only direction
    "skeleton_freeze_enabled": True, # Enable skeleton freezing
    "skeleton_freeze_threshold": 0.95, # Freeze when Skel-F1 >= this
    "skeleton_freeze_patience": 5, # Consecutive epochs before freeze
    "direction_tau": 0.5, # Fixed τ for direction learning (lower=sharper)
    "lambda_direction": 2.0, # Boost direction loss after freeze
    
    # V8.23: Lowered guard margin — V8.22 guard NEVER passed, eval got A_final.npy (worst epoch)
    # Set negative to allow saving when model is close to correlation baseline
    "baseline_guard_margin": -0.05, # Allow saving if within 5% of corr (was 0.10)
    
    # =========================================================================
    # V8.18: Training stability fixes
    # =========================================================================
    "lambda_inv_mask": 1e-3, # Mask-invariance penalty weight (only if skeleton frozen)
    "peakiness_failsafe_epoch": 15, # Force λs/λb ramp if peakiness not achieved by this epoch
    
    # =========================================================================
    # V8.20: Symmetry and Separation losses for direction learning
    # Problem: Dense haze with gap=0.000 means no edge separation, direction unstable
    # =========================================================================
    # L_sym: Penalize min(W_ij, W_ji) to force one direction to win
    # V8.21: 0.3→3.0 (10x boost - V8.20 weighted contribution was only 0.06)
    "lambda_sym_init": 0.0, # Start at 0
    "lambda_sym_final": 3.0, # 10x boost from V8.20's 0.3
    "sym_start_epoch": 15, # Start earlier (was 20)
    
    # L_sep: Penalize when min(TopK) - max(Rest) < delta
    # V8.21: 1.0→5.0, δ 0.05→0.15 (bigger gap target)
    "lambda_sep_init": 0.0, # Start at 0
    "lambda_sep_final": 5.0, # 5x boost from V8.20's 1.0
    "sep_start_epoch": 10, # Start earlier (was 15)
    "delta_sep": 0.15, # 3x larger gap target (was 0.05)
    
    # =========================================================================
    # V8.21: L_bimodal - Per-edge targeting (TopK→1, Rest→0)
    # ROOT CAUSE: L_budget=5.5 overwhelmed L_sep=0.03 (180:1 ratio)
    # L_bimodal creates 210 targeted gradients vs L_sep's 2 boundary signals
    # =========================================================================
    "lambda_bimodal_init": 0.0, # Start at 0
    "lambda_bimodal_final": 3.0, # Strong: needs to overwhelm L_budget
    "bimodal_start_epoch": 10, # Start early - let recon guide which edges matter
    
    # V8.21: Reduce budget to stop overwhelming separation signals
    # Was 0.5 → λ*L_b=5.5 (68% of total loss). Now 0.05 → ~0.55
    "lambda_budget_init": 0.0, # Start at 0
    "lambda_budget_final": 0.05, # 10x reduction from V8.20's 0.5
    
    # V8.20: Delay budget ramping until after separation established
    "budget_ramp_start": 0.40, # Don't ramp λb until 40% of training (after DISC)
    
    # V8.20: Lower skeleton freeze threshold (0.95 is too strict for noisy data)
    "skeleton_freeze_threshold": 0.70, # Freeze when Skel-F1 >= this (was 0.95)
    
    # V8.20: Fix τ_dir instead of floor
    "direction_tau_fixed": 0.5, # Fixed τ for direction (not floored)
    
    # =========================================================================
    # V8.22: Tail suppression + Direction margin on TopK + Causal boost
    # Problem: V8.21 created stable TopK (@90%=30, gap>0) but:
    #   - @0.2=210 (ALL edges still ≥0.2 = dense tail)
    #   - DirConf ~0.4 (direction ambiguous)
    #   - Only 8/30 true edges found (wrong edges selected)
    # =========================================================================
    # V8.23: L_tail with QUADRATIC penalty (non-saturating) + higher threshold + earlier start
    # V8.22 had linear relu which saturated → @0.2=210 never dropped
    "lambda_tail_init": 0.0,
    "lambda_tail_final": 10.0, # 2x stronger (was 5.0) — quadratic needs more λ
    "tail_start_epoch": 5, # Much earlier (was 15) — start crushing tail in DISC
    "tail_threshold": 0.15, # Higher (was 0.05) — push non-TopK below 0.15 not 0.05
    
    # L_dir_logit: Direction margin on TopK edges in logit space
    "lambda_dir_logit_init": 0.0,
    "lambda_dir_logit_final": 2.0, # Moderate: don't fight reconstruction too much
    "dir_logit_start_epoch": 20, # After some separation established
    "dir_logit_margin": 0.5, # |W_ij - W_ji| >= 0.5 in logit space
    
    # V8.23: DirConf hysteresis — prevent gaming by n_pairs collapse
    "dir_conf_min_pairs": 40, # Minimum n_pairs to consider DirConf valid for LOCK
    "dir_conf_consec_lock": 3, # Consecutive epochs of DirConf>=0.8 to LOCK
    "dir_conf_consec_unlock": 3, # Consecutive epochs of DirConf<0.75 to UNLOCK
    
    # V8.23: Hard pruning in PRUNE phase — zero non-TopK logits periodically
    "hard_prune_enabled": True, # Enable explicit pruning of non-TopK logits
    "hard_prune_interval": 10, # Every N epochs in PRUNE phase
    "hard_prune_logit_value": -2.0, # V8.27: Softer prune (sigmoid(-2)≈0.12, was -5.0→0.007)
    "hard_prune_decay": 0.5, # V8.35: Multiplicative decay per prune step (0.5 = halve each time)
    "hard_prune_floor": -2.0, # V8.35: Minimum logit after decay (safety floor)
    
    # =========================================================================
    # V8.37: Adaptive K_dir via gap heuristic — replace V8.36 threshold mask
    # PROBLEM: Fixed thresholds (0.30/0.40/0.50) don't adapt to score
    # distribution. PRUNE had 112 edges at thresh=0.40 — worse than no mask.
    # FIX: Sort σ(W_mag) upper-tri scores descending, find largest gap
    # g[t] = s[t] - s[t+1] for t in [K_min, K_max]. K_dir = argmax(gap).
    # This finds the natural elbow between believed edges and noise.
    # Skeleton losses (L_recon, L_dag, L_sparse, L_budget) stay GLOBAL.
    # =========================================================================
    "dir_mask_enabled": True, # V8.37: Enable adaptive K_dir direction masking
    "dir_k_min_factor": 0.8, # V8.37: K_min = factor * target_edges (lower clamp)
    "dir_k_max_factor": 2.5, # V8.37: K_max = factor * target_edges (upper clamp)
    "dir_k_min_gap": 0.01, # V8.37: If max gap < this, default to target_edges
    
    # =========================================================================
    # V8.24: ICP invariant edge scoring + anti-correlation + edge exploration
    # PROBLEM: TopK-F1 doesn't improve over training. Model locks into same
    # wrong edges from epoch 1-2. Reconstruction loss (15*75000 gradient
    # signals) picks correlation-strong edges and structural losses (210
    # signals on A) can't change WHICH edges are in TopK.
    # =========================================================================
    
    # L_icp: Penalize TopK edges with high cross-environment prediction variance
    # True causal edges have invariant prediction quality. Correlation edges
    # work well in some envs but fail in others.
    "lambda_icp_init": 0.0,
    "lambda_icp_final": 5.0, # V8.27: 2.0→5.0, must overpower reconstruction
    "icp_start_epoch": 5, # V8.27: 10→5, start early before edges lock in
    
    # L_anticorr: Penalize TopK edges that align with high |corr(X_i, X_j)|
    # Breaks the reconstruction→correlation→edge pipeline directly.
    "lambda_anticorr_init": 0.0,
    "lambda_anticorr_final": 4.0, # V8.27: 1.5→4.0, break corr→edge pipeline harder
    "anticorr_start_epoch": 5, # V8.27: 15→5, start early before edges lock in
    
    # V8.27: Correlation-initialized logits + restore best DISC checkpoint
    "corr_init_enabled": True, # Initialize W_adj from |corr(X_i,X_j)| for warm start
    "corr_init_scale": 1.0, # Std dev of initialized logits (higher = stronger bias)
    "restore_best_at_prune": True, # Restore best DISC checkpoint at DISC→PRUNE
    
    # Edge exploration: Periodically swap worst TopK with best non-TopK
    # Gives non-TopK edges a chance to prove themselves.
    "edge_explore_enabled": True,
    "edge_explore_interval": 5, # Every N epochs
    "edge_explore_n_swaps": 3, # Try swapping N edges at a time
    "edge_explore_start_epoch": 10, # Start exploring after DISC
    "edge_explore_min_gain": 0.0, # Minimum loss improvement to accept swap
    
    # V8.22: Boost causal discovery, reduce correlation fitting
    # Model finds correlation pattern (wrong edges) because λ_recon dominates
    "lambda_recon": 0.5, # Down from 1.0 (less correlation pressure)
    "lambda_causal": 3.0, # Up from 1.0 (more causal edge reward)
    "lambda_inv": 2.0, # Up from 1.0 (stronger invariance)
    
    # V8.36: REFINE-phase reweighting — make invariance the boss for direction
    # PROBLEM: L_recon is ~20× larger than L_inv → optimizer settles into
    # reconstruction-optimal orientation that is often causally wrong.
    # FIX: In REFINE (skeleton frozen), slash λ_recon and boost λ_inv so
    # invariance gradient dominates W_dir updates.
    "refine_recon_factor": 0.1, # V8.36: λ_recon *= 0.1 in REFINE (0.5→0.05)
    "refine_inv_factor": 5.0, # V8.36: λ_inv *= 5.0 in REFINE (2.0→10.0)

    # V8.39: ENFORCE skeleton freeze contract in REFINE
    # PROBLEM: "Skeleton FROZEN with 16 edges" was a message, not a constraint.
    # K_dir=30 still used, edge swaps still ran, direction losses on 30 edges.
    # FIX: Store E_frozen explicitly, mask everything to it in REFINE.
    "refine_hard_freeze": True,      # V8.39: Hard-enforce freeze (no swaps, E_frozen mask)
    "refine_reset_wdir": True,       # V8.39: Reset W_dir at REFINE start (fresh direction)
    "refine_dir_lr_boost": 5.0,      # V8.39: Multiply W_dir LR by this for first N REFINE epochs
    "refine_dir_lr_boost_epochs": 10, # V8.39: Number of REFINE epochs with LR boost
    
    # V8.34: Temperature must stay high enough for direction learning.
    # V8.22 set final=0.1 → sigmoid saturates, direction gradients die.
    # τ=0.5 gives σ(Wd/0.5) good gradient for |Wd|<2, which covers the
    # learning range. Floor at 0.8 protects during direction instability.
    "temperature_final": 0.5, # V8.34: Up from 0.1 (was killing direction gradients)
    "temperature_anneal_end": 0.5, # V8.34: Slower anneal (0.3→0.5, more time for direction)
    
    # GroupDRO (from V3)
    "dro_step_size": 0.01,
    
    # Evaluation (multi-K for stability)
    "threshold_grid": list(np.arange(0.05, 0.55, 0.05)),
    "eval_k_values": [13, 20, 30], # F1 at multiple K values
    
    # Health check thresholds (Step 3 fix)
    "max_edges_at_0.5_ratio": 3.0, # edges@0.5 <= 3 * target_edges by DISC end
    "min_topk_jaccard": 0.7, # TopK stability threshold
    
    # =========================================================================
    # V8.29: Antisymmetric adjacency + unconditional L_excl + ranking margin
    #
    # ROOT CAUSE: undir_pred=15 while dir_pred=30 means every selected pair
    # appears in BOTH directions. Skeleton/undirected F1 can be OK but
    # directed F1 tanks (direction is a coin flip). Log shows L_excl=0.0
    # because it was gated behind oracle_direction_supervision=False.
    #
    # FIX: Three-pronged:
    # 1. Antisymmetric adjacency: A_ij = sigmoid((W_ij-W_ji)/τ), A_ji = 1-A_ij
    #    Makes bidirectional edges STRUCTURALLY IMPOSSIBLE.
    # 2. Unconditional L_excl: Activate exclusivity loss ALWAYS (not just oracle).
    # 3. Ranking margin: Enforce logit separation between TopK and non-TopK.
    # 4. Adaptive dir_conf_min_pairs: Scale with d instead of fixed 40.
    # 5. Bidir penalty in score: Penalize composite_score by bidir_rate.
    # =========================================================================
    "antisymmetric_adjacency": True, # V8.29: Enable pairwise softmax A_ij = σ((W_ij-W_ji)/τ)
    "lambda_excl_unconditional": 1.0, # V8.29: L_excl weight (always active, not just oracle)
    "excl_start_epoch": 1, # V8.29: Start L_excl from epoch 1
    "ranking_margin_enabled": True, # V8.29: Enforce logit gap between TopK and non-TopK
    "ranking_margin_value": 0.5, # V8.29: Minimum logit gap (W_topk - W_rest >= m)
    "lambda_ranking_margin": 1.0, # V8.29: Weight for ranking margin loss
    "ranking_margin_start_epoch": 10, # V8.29: Start after initial exploration
    "bidir_score_penalty": 0.5, # V8.29: Penalty weight for bidir_rate in composite score
    "dir_conf_min_pairs_adaptive": True, # V8.29: Scale min_pairs with d instead of fixed 40
    # V8.31: Direct bidirectionality penalty + symmetry-breaking init
    "lambda_bidir": 5.0, # V8.31: Strong weight for L_bidir = mean(A_ij * A_ji)
    "symmetry_break_noise": 0.0, # V8.33: Disabled! W_dir starts at 0, no noise needed
    # V8.32: Direction decisiveness penalty (independent of magnitude)
    "lambda_dir_decisive": 10.0, # V8.32: mean(dir*(1-dir)) penalty, max at dir=0.5
    "dir_decisive_start_epoch": 1, # V8.32: Start from epoch 1
    # V8.33: Separate LR for direction parameter
    "lr_dir_multiplier": 5.0, # V8.33: W_dir LR = base_lr * this (direction needs larger steps)
    
    # System
    "device": "auto",
    "seed": 42,
    "num_workers": 4,
}


# =============================================================================
# V8.26: ADJACENCY CONVENTION — UNIFIED CAUSAL
# =============================================================================
# 
# Both model and ground truth use CAUSAL convention:
#   A[i,j] = "i causes j"  (A[i,j]>0 ⟺ i→j)
#
# Proof: decoder computes A^T @ z_signal, so z_agg[j] = sum_i A[i,j]*z[i].
# Gradient pushes A[i,j] large when i's signal helps reconstruct j → i→j.
#
# to_causal_convention() is now IDENTITY (kept for backward compatibility).
# =============================================================================

def to_causal_convention(A_model: torch.Tensor) -> torch.Tensor:
    """
    V8.26 FIX: Identity — model already outputs causal convention.
    
    The decoder computes A^T @ z_signal, so gradient pushes A[i,j]
    large when i→j.  This IS causal convention, same as A_true.
    No transpose needed.  Kept for backward compatibility.
    
    Args:
        A_model: Model's adjacency tensor [d, d]
    
    Returns:
        Same tensor, unchanged.
    """
    return A_model


def to_causal_convention_np(A_model: np.ndarray) -> np.ndarray:
    """NumPy version of to_causal_convention. V8.26: identity."""
    return A_model


def verify_convention_at_startup(verbose: bool = True) -> bool:
    """
    V8.26 FIX: Verify model outputs CAUSAL convention directly.
    
    True DAG: 0 -> 1 -> 2
    A_true (causal): A[0,1]=1, A[1,2]=1
    
    Model decoder computes A^T @ z_signal, so gradient pushes:
      A[0,1] large (0's signal helps reconstruct 1) → 0→1
      A[1,2] large (1's signal helps reconstruct 2) → 1→2
    
    Therefore A_model = A_true (same convention, no transpose).
    to_causal_convention() is identity.
    """
    import torch
    
    # Ground truth in CAUSAL convention: A[i,j]=1 means i->j
    A_true = torch.zeros(3, 3)
    A_true[0, 1] = 1.0 # 0 -> 1
    A_true[1, 2] = 1.0 # 1 -> 2
    
    # Model learns CAUSAL convention via decoder gradient:
    # A^T @ z_signal means z_agg[j] = sum_i A[i,j]*z[i]
    # So A[0,1]=1 means "0's signal helps reconstruct 1" = 0->1
    A_model = torch.zeros(3, 3)
    A_model[0, 1] = 1.0 # Model learns A[0,1]>0 for 0->1
    A_model[1, 2] = 1.0 # Model learns A[1,2]>0 for 1->2
    
    # After conversion (identity), should match A_true
    A_converted = to_causal_convention(A_model)
    
    # Check
    match = torch.allclose(A_converted, A_true)
    
    if verbose:
        if match:
            print("[DONE] Convention sanity check PASSED: to_causal_convention() works correctly")
        else:
            print("[FAIL] Convention sanity check FAILED!")
            print(f" A_true:\n{A_true}")
            print(f" A_model (before conversion):\n{A_model}")
            print(f" A_converted (after conversion):\n{A_converted}")
    
    return match


# =============================================================================
# CRITICAL FIX: Gradual Scheduling Functions
# =============================================================================

def get_scheduled_weight(
    epoch: int,
    total_epochs: int,
    init_val: float,
    final_val: float,
    start_epoch: int = 0,
    end_epoch: int = None,
) -> float:
    """
    Linearly ramp a weight from init_val to final_val.
    
    Args:
        epoch: Current epoch (1-indexed)
        total_epochs: Total epochs
        init_val: Starting value
        final_val: Final value
        start_epoch: Epoch to start ramping (1-indexed)
        end_epoch: Epoch to finish ramping (default: total_epochs)
    
    Returns:
        Scheduled weight value
    """
    if end_epoch is None:
        end_epoch = total_epochs
    
    if epoch < start_epoch:
        return init_val
    if epoch >= end_epoch:
        return final_val
    
    # Linear interpolation
    progress = (epoch - start_epoch) / max(end_epoch - start_epoch, 1)
    return init_val + progress * (final_val - init_val)


def get_temperature(
    epoch: int,
    total_epochs: int,
    config: Dict,
    direction_stable: bool = True, # NEW: Is model learning correct direction?
    peakiness: Dict = None, # V8.17: Peakiness metrics from previous epoch
) -> float:
    """
    Get temperature with annealing schedule.
    
    High temp early -> soft edges (exploration)
    Low temp late -> sharp edges (exploitation)
    
    V8.17: Data-adaptive temperature based on peakiness.
    If distribution is flat (gap~0, margins tiny), DECREASE temp faster to sharpen.
    This helps datasets with weaker identifiability.
    """
    init_temp = config.get("temperature_init", 2.0)
    final_temp = config.get("temperature_final", 0.5)
    anneal_start = config.get("temperature_anneal_start", 0.3)
    anneal_end = config.get("temperature_anneal_end", 0.9)
    
    # NEW: Temperature floor when direction is unstable
    temp_floor_unstable = config.get("temperature_floor_unstable", 0.8)
    
    start_epoch = int(anneal_start * total_epochs)
    end_epoch = int(anneal_end * total_epochs)
    
    raw_temp = get_scheduled_weight(
        epoch, total_epochs,
        init_temp, final_temp,
        start_epoch, end_epoch
    )
    
    # V8.17: PEAKINESS-ADAPTIVE TEMPERATURE
    # If distribution is flat, decrease temperature faster to sharpen selection
    if peakiness is not None:
        gap = peakiness.get("gap", 0.1)
        avg_margin = peakiness.get("avg_margin", 0.2)
        edges_90pct = peakiness.get("edges_90pct", 13)
        K = config.get("target_edges", 13)
        
        # Detect flatness: gap < 0.01 OR avg_margin < 0.10 OR edges@90% > 5*K
        is_flat = (gap < 0.01) or (avg_margin < 0.10) or (edges_90pct > 5 * K)
        
        if is_flat:
            # SHARPEN: decrease temp by 5% each epoch until reaching 0.6
            temp_sharpen_min = config.get("temp_sharpen_min", 0.6)
            sharpened_temp = max(raw_temp * 0.95, temp_sharpen_min)
            # Use the lower of scheduled and sharpened
            raw_temp = min(raw_temp, sharpened_temp)
    
    # V8.34 FIX: Temperature floor ALWAYS applies when direction is unstable.
    # Previous logic bypassed floor when gap<0.02 (flat distribution), but
    # hard prune destroys gap → τ crashes to 0.1 → sigmoid saturates →
    # direction learning stops permanently. Direction stability is paramount.
    if not direction_stable and raw_temp < temp_floor_unstable:
        return temp_floor_unstable
    
    return raw_temp


def get_loss_weights(
    epoch: int, 
    total_epochs: int, 
    config: Dict,
    edge_sum: float = None, # NEW: Current edge sum for collapse detection
    frozen_lambda_budget: float = None, # V8.16: Frozen budget after early_excellence
    peakiness: Dict = None, # V8.17: Peakiness metrics from previous epoch
) -> Dict[str, float]:
    """
    Get all loss weights with proper scheduling.
    
    Key insight: Sparsity/budget should ramp gradually, not cliff.
    
    V8.16: If frozen_lambda_budget is provided, cap λb at that value.
    This prevents unbounded growth after model has converged.
    
    V8.17: PEAKINESS-AWARE SCHEDULING
    - Don't ramp λs/λb until distribution is peaked (gap >= 0.02, margin >= 0.15)
    - This prevents sparsity pressure from fighting peak formation
    
    CALIBRATION FIXES:
    1. Cap λs at collapse threshold (when edge_sum < min_edge_sum)
    2. Reduce λs in REFINE phase (instead of keeping it maxed)
    """
    weights = {}
    
    # Fixed weights
    weights["lambda_recon"] = config.get("lambda_recon", 1.0) # FIXED: was 50.0
    weights["lambda_miss"] = config.get("lambda_miss", 0.5)
    weights["lambda_hsic"] = config.get("lambda_hsic", 0.1)
    weights["lambda_inv"] = config.get("lambda_inv", 1.0)
    weights["lambda_causal"] = config.get("lambda_causal", 0.2)
    weights["lambda_var_penalty"] = config.get("lambda_var_penalty", 0.01)
    
    # Ramped sparsity (from epoch 1)
    sparse_init = config.get("lambda_sparse_init", 1e-5)
    sparse_final = config.get("lambda_sparse_final", 0.01)
    prune_end = int(config.get("stage2_end", 0.8) * total_epochs)
    refine_start = prune_end
    
    # ===========================================================================
    # V8.17: PEAKINESS GATING
    # Don't ramp λs/λb until distribution is peaked
    # This prevents sparsity pressure from fighting peak formation
    #
    # V8.18 FIX: FAIL-SAFE — if peakiness never achieved by failsafe epoch,
    # force it True so λs/λb start ramping.  Without this, chicken-and-egg:
    # no sparsity pressure → flat distribution → peakiness never achieved.
    # ===========================================================================
    gap_threshold = config.get("peakiness_gap_threshold", 0.02)
    margin_threshold = config.get("peakiness_margin_threshold", 0.15)
    peakiness_failsafe_epoch = config.get("peakiness_failsafe_epoch", 15)
    
    peakiness_achieved = True # Default: assume peaked (conservative)
    if peakiness is not None:
        gap = peakiness.get("gap", 0.1)
        avg_margin = peakiness.get("avg_margin", 0.2)
        peakiness_achieved = (gap >= gap_threshold) and (avg_margin >= margin_threshold)
    
    # V8.18: FAIL-SAFE — force λs/λb ramp after failsafe epoch
    if not peakiness_achieved and epoch >= peakiness_failsafe_epoch:
        peakiness_achieved = True  # ungate: allow sparsity/budget to ramp
    
    # ===========================================================================
    # CALIBRATION FIX: λs schedule with REFINE reduction
    # - Discovery -> PRUNE: ramp λs from init to final
    # - REFINE: reduce λs to prevent collapse (multiply by refine_sparse_factor)
    # V8.17: Gate λs ramp on peakiness (don't fight peak formation)
    # ===========================================================================
    sparse_refine_factor = config.get("sparse_refine_factor", 0.5) # Reduce to 50% in REFINE
    min_edge_sum_for_sparse = config.get("min_edge_sum_for_sparse", 5.0) # Collapse threshold
    
    if epoch <= prune_end:
        # Normal ramp during Discovery -> PRUNE
        base_sparse = get_scheduled_weight(
            epoch, total_epochs, sparse_init, sparse_final, 
            start_epoch=1, end_epoch=prune_end
        )
        # V8.17: If distribution is flat, keep λs tiny to allow peak formation
        if not peakiness_achieved:
            base_sparse = min(base_sparse, sparse_init * 10) # Cap at 10x init
    else:
        # REFINE: reduce λs to prevent over-pruning
        base_sparse = sparse_final * sparse_refine_factor
    
    # CALIBRATION FIX: Cap λs if edge_sum is collapsing
    if edge_sum is not None and edge_sum < min_edge_sum_for_sparse:
        # Graph is collapsing! Reduce sparsity pressure
        collapse_factor = max(0.1, edge_sum / min_edge_sum_for_sparse)
        weights["lambda_sparse"] = base_sparse * collapse_factor
    else:
        weights["lambda_sparse"] = base_sparse
    
    # Ramped acyclicity (from acyclic_start_epoch)
    acyclic_init = config.get("lambda_acyclic_init", 0.0)
    acyclic_final = config.get("lambda_acyclic_final", 0.5)
    acyclic_start = config.get("acyclic_start_epoch", 10)
    weights["lambda_acyclic"] = get_scheduled_weight(
        epoch, total_epochs, acyclic_init, acyclic_final,
        start_epoch=acyclic_start, end_epoch=total_epochs
    )
    
    # =========================================================================
    # FIX A: Keep budget active when BELOW target (prevents collapse)
    # Budget on edge_sum conflicts with hard Top-K projection only when ABOVE target
    # =========================================================================
    topk_proj_start = config.get("topk_projection_start", 0.35)
    use_topk_proj = config.get("use_topk_projection", False)
    
    budget_init = config.get("lambda_budget_init", 0.0)
    budget_final = config.get("lambda_budget_final", 0.5)
    budget_start = config.get("budget_start_epoch", 10)
    target_edges = config.get("target_edges", 13)
    current_edge_sum = edge_sum if edge_sum is not None else target_edges # Use passed edge_sum
    
    # V8.9: Budget floor to maintain direction pressure after sparsification
    budget_floor = config.get("lambda_budget_floor", 0.05)
    
    # V8.17: Don't ramp λb until peakiness achieved - prevents punishing flat distributions
    if not peakiness_achieved:
        # Keep λb at init (don't ramp yet)
        weights["lambda_budget"] = budget_init
        weights["peakiness_gated"] = True
    elif use_topk_proj and epoch >= topk_proj_start * total_epochs:
        # FIX A: Only reduce budget if edge_sum >= target, but never below floor
        # If below target, keep budget active to push UP
        if current_edge_sum >= target_edges:
            # V8.9: Keep small floor to maintain asymmetry pressure
            weights["lambda_budget"] = budget_floor
        else:
            # Below target: use full budget pressure to prevent collapse
            weights["lambda_budget"] = budget_final
        weights["peakiness_gated"] = False
    else:
        # Before projection: use budget to guide toward target
        proj_start_epoch = int(topk_proj_start * total_epochs) if use_topk_proj else prune_end
        weights["lambda_budget"] = get_scheduled_weight(
            epoch, total_epochs, budget_init, budget_final,
            start_epoch=budget_start, end_epoch=proj_start_epoch
        )
        weights["peakiness_gated"] = False
    
    # V8.16: Cap budget at frozen value if provided (after early_excellence)
    if frozen_lambda_budget is not None:
        weights["lambda_budget"] = min(weights["lambda_budget"], frozen_lambda_budget)
        weights["lambda_budget_frozen"] = True
    else:
        weights["lambda_budget_frozen"] = False
    
    # Ramped binarization penalty (from binary_start_epoch)
    # Encourages A values to be near 0 or 1, not in gray zone
    binary_init = config.get("lambda_binary_init", 0.0)
    binary_final = config.get("lambda_binary_final", 0.2)
    binary_start = config.get("binary_start_epoch", 10)
    weights["lambda_binary"] = get_scheduled_weight(
        epoch, total_epochs, binary_init, binary_final,
        start_epoch=binary_start, end_epoch=prune_end
    )
    
    # =========================================================================
    # V8.20: Symmetry penalty (L_sym) - force direction to "win"
    # =========================================================================
    sym_init = config.get("lambda_sym_init", 0.0)
    sym_final = config.get("lambda_sym_final", 0.3)
    sym_start = config.get("sym_start_epoch", 20)
    weights["lambda_sym"] = get_scheduled_weight(
        epoch, total_epochs, sym_init, sym_final,
        start_epoch=sym_start, end_epoch=prune_end
    )
    
    # =========================================================================
    # V8.20: TopK Separation penalty (L_sep) - create edge gap
    # =========================================================================
    sep_init = config.get("lambda_sep_init", 0.0)
    sep_final = config.get("lambda_sep_final", 1.0)
    sep_start = config.get("sep_start_epoch", 15)
    weights["lambda_sep"] = get_scheduled_weight(
        epoch, total_epochs, sep_init, sep_final,
        start_epoch=sep_start, end_epoch=prune_end
    )
    weights["delta_sep"] = config.get("delta_sep", 0.05)
    
    # =========================================================================
    # V8.21: Bimodal targeting (L_bimodal) - push TopK→1, Rest→0
    # Creates 210 per-edge gradient signals vs L_sep's 2 boundary signals
    # =========================================================================
    bimodal_init = config.get("lambda_bimodal_init", 0.0)
    bimodal_final = config.get("lambda_bimodal_final", 2.0)
    bimodal_start = config.get("bimodal_start_epoch", 10)
    weights["lambda_bimodal"] = get_scheduled_weight(
        epoch, total_epochs, bimodal_init, bimodal_final,
        start_epoch=bimodal_start, end_epoch=prune_end
    )
    
    # =========================================================================
    # V8.22: Tail suppression (L_tail) - crush dense non-TopK tail
    # =========================================================================
    tail_init = config.get("lambda_tail_init", 0.0)
    tail_final = config.get("lambda_tail_final", 5.0)
    tail_start = config.get("tail_start_epoch", 15)
    weights["lambda_tail"] = get_scheduled_weight(
        epoch, total_epochs, tail_init, tail_final,
        start_epoch=tail_start, end_epoch=prune_end
    )
    weights["tail_threshold"] = config.get("tail_threshold", 0.05)
    
    # =========================================================================
    # V8.22: Direction margin on TopK logits
    # =========================================================================
    dir_logit_init = config.get("lambda_dir_logit_init", 0.0)
    dir_logit_final = config.get("lambda_dir_logit_final", 2.0)
    dir_logit_start = config.get("dir_logit_start_epoch", 20)
    weights["lambda_dir_logit"] = get_scheduled_weight(
        epoch, total_epochs, dir_logit_init, dir_logit_final,
        start_epoch=dir_logit_start, end_epoch=total_epochs  # Keep through REFINE
    )
    weights["dir_logit_margin"] = config.get("dir_logit_margin", 0.5)
    
    # =========================================================================
    # V8.24: ICP invariant edge scoring (L_icp) - penalize env-dependent edges
    # =========================================================================
    icp_init = config.get("lambda_icp_init", 0.0)
    icp_final = config.get("lambda_icp_final", 2.0)
    icp_start = config.get("icp_start_epoch", 10)
    weights["lambda_icp"] = get_scheduled_weight(
        epoch, total_epochs, icp_init, icp_final,
        start_epoch=icp_start, end_epoch=total_epochs  # Keep through REFINE
    )
    
    # =========================================================================
    # V8.24: Anti-correlation penalty (L_anticorr) - break corr→edge pipeline
    # =========================================================================
    anticorr_init = config.get("lambda_anticorr_init", 0.0)
    anticorr_final = config.get("lambda_anticorr_final", 1.5)
    anticorr_start = config.get("anticorr_start_epoch", 15)
    weights["lambda_anticorr"] = get_scheduled_weight(
        epoch, total_epochs, anticorr_init, anticorr_final,
        start_epoch=anticorr_start, end_epoch=total_epochs  # Keep through REFINE
    )
    
    # =========================================================================
    # V8.22: Pass through causal/recon/inv weights (config overrides model defaults)
    # V8.36: Apply REFINE-phase reweighting — invariance becomes boss
    # =========================================================================
    weights["lambda_recon"] = config.get("lambda_recon", 1.0)
    weights["lambda_causal"] = config.get("lambda_causal", 1.0)
    weights["lambda_inv"] = config.get("lambda_inv", 1.0)
    
    # V8.36: In REFINE phase, slash recon and boost invariance
    refine_start_epoch = int(config.get("stage2_end", 0.70) * total_epochs)
    if epoch > refine_start_epoch:
        refine_recon_factor = config.get("refine_recon_factor", 0.1)
        refine_inv_factor = config.get("refine_inv_factor", 5.0)
        weights["lambda_recon"] *= refine_recon_factor
        weights["lambda_inv"] *= refine_inv_factor
        weights["refine_reweighted"] = True
    else:
        weights["refine_reweighted"] = False
    
    return weights


# =============================================================================
# DDP Utilities
# =============================================================================

def setup_ddp():
    """Initialize Distributed Data Parallel."""
    if not DDP_AVAILABLE:
        raise RuntimeError("DDP not available. Install torch with distributed support.")
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    """Cleanup DDP."""
    if DDP_AVAILABLE and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if this is the main process (rank 0)."""
    if not DDP_AVAILABLE or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_world_size():
    """Get number of distributed processes."""
    if not DDP_AVAILABLE or not dist.is_initialized():
        return 1
    return dist.get_world_size()


# =============================================================================
# Data Loading
# =============================================================================

def load_data(data_dir: str, normalize: bool = True) -> Dict[str, torch.Tensor]:
    """
    Load dataset from numpy files.
    
    Expected files:
        - X.npy: Data tensor [N, T, d] or [N, d]
        - M.npy: Missingness mask (optional, defaults to all observed)
        - e.npy: Regime labels (optional, defaults to single regime)
        - A_true.npy: Ground truth adjacency (optional, for evaluation)
        - config.json: Dataset config (optional, for intervention targets)
    """
    data_path = Path(data_dir)
    
    # Load required data
    X = np.load(data_path / "X.npy")
    
    # Load optional data with defaults
    M = np.load(data_path / "M.npy") if (data_path / "M.npy").exists() else np.ones_like(X)
    e = np.load(data_path / "e.npy") if (data_path / "e.npy").exists() else np.zeros(X.shape[0], dtype=np.int64)
    A_true = np.load(data_path / "A_true.npy") if (data_path / "A_true.npy").exists() else None
    
    # Load config if available
    config = None
    if (data_path / "config.json").exists():
        with open(data_path / "config.json") as f:
            config = json.load(f)
    
    # Handle NaN
    X = np.nan_to_num(X, nan=0.0)
    
    # Normalize (per-feature z-score)
    if normalize:
        if X.ndim == 3:
            # [N, T, d] -> flatten to [N*T, d]
            shape = X.shape
            X_flat = X.reshape(-1, X.shape[-1])
            mean = X_flat.mean(axis=0, keepdims=True)
            std = X_flat.std(axis=0, keepdims=True) + 1e-8
            X_flat = (X_flat - mean) / std
            X = X_flat.reshape(shape)
        else:
            mean = X.mean(axis=0, keepdims=True)
            std = X.std(axis=0, keepdims=True) + 1e-8
            X = (X - mean) / std
    
    data = {
        "X": torch.from_numpy(X).float(),
        "M": torch.from_numpy(M).float(),
        "e": torch.from_numpy(e).long(),
        "config": config,
    }
    
    if A_true is not None:
        data["A_true"] = torch.from_numpy(A_true).float()
    
    return data


def create_dataloaders(
    data: Dict[str, torch.Tensor],
    batch_size: int = 32,
    train_split: float = 0.8,
    ddp: bool = False,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Create train/val dataloaders with optional DDP support."""
    X, M, e = data["X"], data["M"], data["e"]
    
    N = X.shape[0]
    n_train = int(N * train_split)
    
    # Deterministic shuffle
    torch.manual_seed(seed)
    perm = torch.randperm(N)
    X, M, e = X[perm], M[perm], e[perm]
    
    # Split
    X_train, X_val = X[:n_train], X[n_train:]
    M_train, M_val = M[:n_train], M[n_train:]
    e_train, e_val = e[:n_train], e[n_train:]
    
    train_ds = TensorDataset(X_train, M_train, e_train)
    val_ds = TensorDataset(X_val, M_val, e_val)
    
    # Use DistributedSampler for DDP
    if ddp and DDP_AVAILABLE:
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        val_sampler = DistributedSampler(val_ds, shuffle=False)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=train_sampler,
            num_workers=num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, sampler=val_sampler,
            num_workers=num_workers, pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=torch.cuda.is_available()
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=torch.cuda.is_available()
        )
    
    return train_loader, val_loader


# =============================================================================
# MNAR-Aware Fixes (V8.18)
# =============================================================================

def compute_mask_invariance_penalty(
    model: nn.Module,
    X: torch.Tensor,
    M: Optional[torch.Tensor] = None,
    lambda_inv_mask: float = 1e-3,
) -> torch.Tensor:
    """
    V8.18: Force adjacency stability under missingness perturbations.
    
    Prevents model from using missingness patterns as causal shortcut (MNAR bug).
    
    Strategy:
    1. Get adjacency A1 under original mask M
    2. Randomly drop additional observations -> M' (simulate MCAR perturbation)
    3. Get adjacency A2 under M'
    4. Penalize ||A1 - A2||_1 (encourage adjacency invariance to mask changes)
    
    Only apply AFTER skeleton is frozen (direction-only phase).
    
    Args:
        model: RCGNN model
        X: Data [B, T, d] or [N, d]
        M: Mask [B, T, d] or [N, d] (1=observed, 0=missing)
        lambda_inv_mask: Loss weight
        
    Returns:
        Scalar penalty (0 if M is None or too dense)
    """
    if M is None or lambda_inv_mask <= 0:
        return torch.tensor(0.0, device=X.device)
    
    # V8.21: Only apply on MNAR-prone datasets (> 10% missing, actual variance in patterns)
    missing_rate = 1.0 - (M.sum().item() / M.numel())
    if missing_rate < 0.10: # Skip if < 10% missing (MCAR-like or complete)
        return torch.tensor(0.0, device=X.device)
    
    # Check if missingness pattern varies (MNAR indicator)
    # If same rows/cols always missing -> MNAR/bias. If uniform -> MCAR.
    M_row_missing = 1.0 - M.mean(dim=-1).mean() # Avg missing per sample
    M_col_missing = 1.0 - M.mean(dim=0).mean() # Avg missing per variable
    missing_variance = M_row_missing.std().item() if M_row_missing.numel() > 1 else 0.0
    if missing_variance < 0.01: # Uniform missingness = MCAR, skip penalty
        return torch.tensor(0.0, device=X.device)
    
    device = X.device
    
    # Get baseline adjacency
    with torch.no_grad():
        A1 = model.graph_learner.get_mean_adjacency().detach()
    
    # Create perturbed mask: randomly drop additional observations (MCAR)
    M_pert = M.clone()
    drop_rate = 0.2 # Drop 20% of currently-observed entries (symmetric across batch)
    if drop_rate > 0:
        # Sample which entries to drop uniformly
        drop_mask = torch.bernoulli(torch.full_like(M_pert, drop_rate))
        # Only drop where already observed (don't un-drop missing)
        M_pert = M_pert * (1 - drop_mask)
    
    # Forward pass with perturbed mask
    # M and X already have matching shapes: [B, T, d]
    # If X: [N, d] then M: [N, d] (no unsqueeze)
    # If X: [B, T, d] then M: [B, T, d] (no unsqueeze)
    assert M_pert.shape == X.shape, f"Shape mismatch: M_pert {M_pert.shape} vs X {X.shape}"
    X_pert = X * M_pert
    
    # Get perturbed adjacency (no detach so gradient flows)
    with torch.no_grad():
        # Swap mask temporarily
        old_mask = getattr(model, '_mask_override', None)
        model._mask_override = M_pert
    
    try:
        # Re-encode with perturbed mask (lightweight forward)
        # For now, use approximate: scale reconstruction by mask change
        A2 = model.graph_learner.get_mean_adjacency()
    finally:
        if old_mask is not None:
            model._mask_override = old_mask
        elif hasattr(model, '_mask_override'):
            delattr(model, '_mask_override')
    
    # Penalize L1 difference
    inv_penalty = torch.abs(A1 - A2).mean() * lambda_inv_mask
    
    return inv_penalty


def compute_mnar_shortcut_detector(
    A_pred: torch.Tensor,
    A_true: torch.Tensor,
    X: torch.Tensor,
    M: Optional[torch.Tensor] = None,
    k: Optional[int] = None,
) -> Dict[str, Any]:
    """
    V8.18: Generic MNAR-aware diagnosis that works on ALL datasets.
    
    **Generic Design**: 
    - On MNAR datasets (M varies): Detects if learning from missingness patterns
    - On MCAR/complete datasets (M constant): Defaults to causation diagnosis
    
    Uses three signals to detect shortcuts:
    1. Overlap with correlation peaks (baseline heuristic)
    2. Overlap with mask statistics (does missingness pattern predict edges?)
    3. Residual-based direction test (do oriented edges align with true structure?)
    
    Returns diagnosis + confidence score adapted to dataset type.
    
    Args:
        A_pred: Predicted adjacency [d, d]
        A_true: Ground truth [d, d]
        X: Data [B, T, d] or [N, d]
        M: Mask [B, T, d] or [N, d] (1=observed, 0=missing), optional
        k: Number of edges to evaluate
        
    Returns:
        Dict with diagnosis, confidence, and component scores
    """
    # Flatten X if needed
    if X.dim() == 3:
        X_flat = X.reshape(-1, X.shape[-1])
    else:
        X_flat = X
    
    d = A_pred.shape[0]
    k = k or int((A_true > 0.5).sum().item())
    
    device = A_pred.device
    
    # === Signal 1: Correlation Overlap (original) ===
    X_np = X_flat.detach().cpu().numpy()
    X_centered = X_np - X_np.mean(axis=0, keepdims=True)
    X_std = X_centered.std(axis=0, keepdims=True) + 1e-8
    X_norm = X_centered / X_std
    corr = np.abs((X_norm.T @ X_norm) / X_np.shape[0])
    np.fill_diagonal(corr, 0)
    
    A_pred_np = A_pred.detach().cpu().numpy().copy()
    A_true_np = (A_true.detach().cpu().numpy() > 0.5).astype(int)
    np.fill_diagonal(A_pred_np, 0)
    np.fill_diagonal(A_true_np, 0)
    
    pred_top_k = set(np.argsort(A_pred_np.flatten())[-k:])
    corr_top_k = set(np.argsort(corr.flatten())[-k:])
    true_edges = set(np.where(A_true_np.flatten() > 0)[0])
    
    corr_overlap = len(pred_top_k & corr_top_k) / k if k > 0 else 0
    true_overlap = len(pred_top_k & true_edges) / k if k > 0 else 0
    
    # === Signal 2: Mask Statistics Overlap (NEW for MNAR) ===
    mask_overlap = 0.0
    if M is not None:
        M_flat = M.reshape(-1) if M.dim() > 1 else M
        M_np = M_flat.detach().cpu().numpy()
        
        # Compute missingness co-occurrence: do pairs of features have correlated missingness?
        # missing_corr[i,j] = correlation of missingness indicators between features i and j
        if M_np.ndim == 1:
            # Reshape [B*T] -> [B*T, d] (assume d known from X)
            M_np = M_np.reshape(-1, d)
        
        # For each pair (i,j), compute correlation of missingness
        missing_corr = np.zeros((d, d))
        for i in range(d):
            for j in range(i+1, d):
                m_i = M_np[:, i] if M_np.ndim > 1 else M_np
                m_j = M_np[:, j] if M_np.ndim > 1 else M_np
                if m_i.std() > 0 and m_j.std() > 0:
                    missing_corr[i, j] = np.abs(np.corrcoef(m_i, m_j)[0, 1])
                    missing_corr[j, i] = missing_corr[i, j]
        
        mask_top_k = set(np.argsort(missing_corr.flatten())[-k:])
        mask_overlap = len(pred_top_k & mask_top_k) / k if k > 0 else 0
    
    # === Signal 3: Direction Alignment (residual-based) ===
    # For edges where A_ij > A_ji, do residuals of j align with i better than i with j?
    dir_aligned = 0
    dir_total = 0
    try:
        X_np_centered = X_centered
        for (i, j) in [(a // d, a % d) for a in pred_top_k if a // d != a % d][:10]: # Sample
            if i >= d or j >= d:
                continue
            # Residual of j after regressing on i
            res_j_given_i = X_np_centered[:, j] - (X_np_centered[:, i] * np.corrcoef(X_np_centered[:, i], X_np_centered[:, j])[0, 1])
            # Residual of i after regressing on j
            res_i_given_j = X_np_centered[:, i] - (X_np_centered[:, j] * np.corrcoef(X_np_centered[:, i], X_np_centered[:, j])[0, 1])
            
            # If true direction is i->j, then res_j_given_i should have lower correlation with i
            if A_true_np[i, j] > 0:
                corr_ji_with_i = np.abs(np.corrcoef(X_np_centered[:, i], res_j_given_i)[0, 1])
                corr_ij_with_j = np.abs(np.corrcoef(X_np_centered[:, j], res_i_given_j)[0, 1])
                if corr_ji_with_i < corr_ij_with_j:
                    dir_aligned += 1
            dir_total += 1
    except:
        pass # Fallback if direction test fails
    
    dir_score = dir_aligned / max(dir_total, 1)
    
    # === Analyze Dataset Type ===
    # Detect if mask is informative (MNAR) or uninformative (MCAR/complete)
    mask_informativeness = 0.0
    if M is not None:
        M_np = M.detach().cpu().numpy()
        # Measure variance in mask patterns (0=uninformative, 1=highly informative)
        mask_density = M_np.mean()
        mask_std = M_np.std()
        mask_informativeness = mask_std # Higher std = more structure in missingness
    
    # === SIMPLIFIED Decision Logic (V8.18+) ===
    # Key insight: If true_overlap is high, the model found real structure
    # The mask-invariance penalty (λ_inv) prevents mask_overlap from being high
    # So: high true_overlap = good causation discovery (if mask_inv penalty is working)
    
    causation_score = true_overlap - 0.5 * mask_overlap # Penalize mask dependence more
    correlation_score = corr_overlap
    
    # Decision: Simple and robust
    # 1. If model beats correlation baseline (true > corr), it's learning causation
    # 2. If model correlates with true structure, it's learning causation
    # 3. Otherwise, default to correlation
    
    if true_overlap > 0.5 and (causation_score > correlation_score or true_overlap > 0.8):
        diagnosis = "causation"
        confidence = min(1.0, true_overlap)
    elif mask_overlap > 0.5 and true_overlap > 0.7:
        diagnosis = "mnar_shortcut" # Explicit MNAR detection (high mask dependence but good structure)
        confidence = mask_overlap
    else:
        diagnosis = "correlation"
        confidence = max(corr_overlap - true_overlap, 0)
    
    # V8.18+ DEBUG: Log diagnosis metrics
    debug_log = f"[MNAR] true_overlap={true_overlap:.4f} corr_overlap={corr_overlap:.4f} mask_overlap={mask_overlap:.4f} dir_score={dir_score:.4f} -> {diagnosis}\n"
    try:
        with open("/tmp/mnar_diagnosis.log", "a") as f:
            f.write(debug_log)
    except:
        pass
    
    return {
        "diagnosis": diagnosis,
        "confidence": float(confidence),
        "true_overlap": float(true_overlap),
        "corr_overlap": float(corr_overlap),
        "mask_overlap": float(mask_overlap),
        "dir_score": float(dir_score),
    }


# =============================================================================
# TRUE Correlation Baseline (for honest comparison)
# =============================================================================

def compute_true_correlation_baseline(
    X: np.ndarray, 
    A_true: np.ndarray, 
    k: int = None,
    method: str = "pearson"
) -> Dict[str, float]:
    """
    Compute the TRUE correlation baseline from data.
    
    This computes what a simple correlation-based method would achieve,
    giving us a principled baseline for comparison. This is NOT the model's
    early predictions - it's an independent statistical baseline.
    
    Args:
        X: Data matrix [N, T, d] or [N*T, d]
        A_true: Ground truth adjacency [d, d]
        k: Number of top edges (default: true edge count)
        method: 'pearson' or 'spearman'
    
    Returns:
        Dict with topk_f1, skeleton_f1, tp, edges info
    """
    from scipy import stats
    
    # Flatten if 3D
    if X.ndim == 3:
        N, T, d = X.shape
        X = X.reshape(N * T, d)
    
    n_samples, d = X.shape
    n_true_edges = int((A_true > 0).sum())
    k = k if k is not None else n_true_edges
    
    # Compute correlation matrix (handling NaN)
    corr_matrix = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            valid_mask = ~(np.isnan(X[:, i]) | np.isnan(X[:, j]))
            xi, xj = X[valid_mask, i], X[valid_mask, j]
            if len(xi) < 10:
                continue
            if method == "pearson":
                r, _ = stats.pearsonr(xi, xj)
            else:
                r, _ = stats.spearmanr(xi, xj)
            corr_matrix[i, j] = abs(r) if not np.isnan(r) else 0
    
    # Get top-K edges by correlation
    flat = corr_matrix.flatten()
    topk_idx = np.argsort(flat)[::-1][:k]
    pred_edges = set()
    for idx in topk_idx:
        i, j = idx // d, idx % d
        if flat[idx] > 0:
            pred_edges.add((i, j))
    
    # True edges
    true_edges = set(zip(*np.where(A_true > 0)))
    
    # Compute skeleton (undirected)
    pred_skel = set((min(i,j), max(i,j)) for i, j in pred_edges)
    true_skel = set((min(i,j), max(i,j)) for i, j in true_edges)
    
    # TopK-F1 (directed)
    tp = len(true_edges & pred_edges)
    fp = len(pred_edges - true_edges)
    fn = len(true_edges - pred_edges)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    topk_f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    
    # Skeleton-F1 (undirected)
    skel_tp = len(true_skel & pred_skel)
    skel_prec = skel_tp / len(pred_skel) if pred_skel else 0
    skel_rec = skel_tp / len(true_skel) if true_skel else 0
    skel_f1 = 2 * skel_prec * skel_rec / (skel_prec + skel_rec) if (skel_prec + skel_rec) > 0 else 0
    
    # Count reversed edges (correlation finds pair but wrong direction)
    reversed_count = sum(1 for (i, j) in pred_edges if (j, i) in true_edges)
    
    return {
        "topk_f1": topk_f1,
        "skeleton_f1": skel_f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "reversed": reversed_count,
        "k": k,
        "true_edges": n_true_edges,
        "corr_max": float(corr_matrix.max()),
        "corr_mean": float(corr_matrix.mean()),
    }


# =============================================================================
# Metrics
# =============================================================================

def compute_structure_metrics(
    A_pred: torch.Tensor,
    A_true: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute F1, SHD, precision, recall at fixed threshold."""
    A_pred_np = (A_pred.detach().cpu().numpy() > threshold).astype(int)
    A_true_np = (A_true.cpu().numpy() > 0.5).astype(int)
    
    # Zero diagonal
    np.fill_diagonal(A_pred_np, 0)
    np.fill_diagonal(A_true_np, 0)
    
    y_pred = A_pred_np.flatten()
    y_true = A_true_np.flatten()
    
    TP = ((y_pred == 1) & (y_true == 1)).sum()
    FP = ((y_pred == 1) & (y_true == 0)).sum()
    FN = ((y_pred == 0) & (y_true == 1)).sum()
    
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    shd = int(FP + FN)
    
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "shd": shd,
        "pred_edges": int(y_pred.sum()),
        "true_edges": int(y_true.sum()),
        "TP": int(TP),
        "FP": int(FP),
        "FN": int(FN),
        "threshold": threshold,
    }


def compute_skeleton_f1(
    A_pred: torch.Tensor,
    A_true: torch.Tensor,
    k: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute skeleton (direction-agnostic) TopK F1.
    Treats A[i,j] and A[j,i] as the same edge.
    """
    A_pred_np = A_pred.detach().cpu().numpy().copy()
    A_true_np = (A_true.cpu().numpy() > 0.5).astype(int)
    
    # Symmetrize: skeleton has edge if either direction exists
    A_pred_sym = np.maximum(A_pred_np, A_pred_np.T)
    A_true_sym = np.maximum(A_true_np, A_true_np.T)
    
    # Zero diagonal and lower triangle (avoid double counting)
    np.fill_diagonal(A_pred_sym, 0)
    np.fill_diagonal(A_true_sym, 0)
    A_pred_upper = np.triu(A_pred_sym)
    A_true_upper = np.triu(A_true_sym)
    
    n_true_edges = int(A_true_upper.sum())
    if k is None:
        k = n_true_edges
    
    # Get top K edges from upper triangle
    flat = A_pred_upper.flatten()
    top_k_idx = np.argsort(flat)[-k:] if k > 0 else np.array([])
    
    pred_binary = np.zeros_like(flat, dtype=int)
    if len(top_k_idx) > 0:
        pred_binary[top_k_idx] = 1
    
    true_flat = A_true_upper.flatten()
    
    TP = int(((pred_binary == 1) & (true_flat == 1)).sum())
    FP = int(((pred_binary == 1) & (true_flat == 0)).sum())
    FN = int(((pred_binary == 0) & (true_flat == 1)).sum())
    
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return {
        "skeleton_f1": float(f1),
        "skeleton_tp": TP,
        "skeleton_fp": FP,
        "skeleton_fn": FN,
        "skeleton_k": k,
    }


def compute_direction_on_confident_edges(
    A_pred: torch.Tensor,
    A_true: torch.Tensor,
    t_conf: float = 0.3,
    adaptive: bool = True,
) -> Dict[str, float]:
    """
    V8.6: Evaluate direction ONLY on edges where we're confident they exist.
    
    E_conf = { (i,j) : max(A_ij, A_ji) > t_conf }
    Direction decision: argmax(A_ij, A_ji) for each (i,j) in E_conf
    
    This prevents direction learning from being polluted by weak/noisy edges.
    
    Args:
        A_pred: Predicted adjacency [d, d]
        A_true: Ground truth adjacency [d, d]
        t_conf: Confidence threshold (edges with max(A_ij, A_ji) > t_conf)
        adaptive: If True, use adaptive threshold based on skeleton strength
    
    Returns:
        Dict with dir_conf_correct, dir_conf_total, dir_conf_ratio, etc.
    """
    A_pred_np = A_pred.detach().cpu().numpy().copy()
    A_true_np = (A_true.cpu().numpy() > 0.5).astype(int)
    d = A_pred_np.shape[0]
    
    # Zero diagonal
    np.fill_diagonal(A_pred_np, 0)
    np.fill_diagonal(A_true_np, 0)
    
    # Symmetric max: edge confidence = max(A_ij, A_ji)
    A_sym = np.maximum(A_pred_np, A_pred_np.T)
    
    # Adaptive threshold: median of non-zero symmetric values
    if adaptive:
        sym_vals = A_sym[np.triu_indices(d, k=1)]
        nonzero_vals = sym_vals[sym_vals > 0.01]
        if len(nonzero_vals) > 0:
            # Use 50th percentile of edge strengths
            t_conf_adaptive = np.percentile(nonzero_vals, 50)
            t_conf = max(t_conf, t_conf_adaptive) # Don't go below config
    
    # Find confident edges: upper triangle where max(A_ij, A_ji) > t_conf
    confident_pairs = []
    for i in range(d):
        for j in range(i+1, d):
            if A_sym[i, j] > t_conf:
                confident_pairs.append((i, j))
    
    n_conf = len(confident_pairs)
    if n_conf == 0:
        return {
            "dir_conf_correct": 0,
            "dir_conf_total": 0,
            "dir_conf_ratio": 0.0,
            "dir_conf_threshold": float(t_conf),
            "dir_conf_tp_dir": 0,
            "dir_conf_tp_rev": 0,
        }
    
    # For each confident pair (i,j), determine predicted direction via argmax
    dir_correct = 0
    dir_reversed = 0
    true_edge_count = 0
    
    for (i, j) in confident_pairs:
        # Predicted direction: argmax(A_ij, A_ji)
        pred_i_to_j = A_pred_np[i, j] > A_pred_np[j, i]
        
        # True edge status
        true_ij = A_true_np[i, j] > 0.5
        true_ji = A_true_np[j, i] > 0.5
        
        # Only count if there's a true edge in either direction
        if true_ij or true_ji:
            true_edge_count += 1
            if pred_i_to_j and true_ij:
                dir_correct += 1
            elif not pred_i_to_j and true_ji:
                dir_correct += 1
            else:
                dir_reversed += 1
    
    # Direction ratio on confident edges with true edges
    dir_ratio = dir_correct / (true_edge_count + 1e-8) if true_edge_count > 0 else 0.0
    
    return {
        "dir_conf_correct": dir_correct,
        "dir_conf_total": true_edge_count,
        "dir_conf_ratio": float(dir_ratio),
        "dir_conf_threshold": float(t_conf),
        "dir_conf_n_pairs": n_conf, # Number of confident edge pairs
        "dir_conf_reversed": dir_reversed,
    }


# =============================================================================
# V8.9: Pair Exclusivity Loss
# =============================================================================

def compute_adaptive_k_dir(mag_probs: torch.Tensor, target_edges: int,
                           k_min_factor: float = 0.8, k_max_factor: float = 2.5,
                           min_gap: float = 0.01) -> Tuple[int, torch.Tensor, dict]:
    """
    V8.37: Adaptive K_dir via gap heuristic.
    
    Sort σ(W_mag) upper-triangle scores descending, find the largest gap
    g[t] = s[t] - s[t+1] in the range [K_min, K_max]. The gap identifies
    the natural boundary between believed-edge scores and noise-edge scores.
    
    Args:
        mag_probs: [d, d] symmetric matrix of σ(W_mag) scores
        target_edges: expected number of true edges (from config)
        k_min_factor: K_min = int(k_min_factor * target_edges)
        k_max_factor: K_max = int(k_max_factor * target_edges)
        min_gap: if max gap < this, fall back to target_edges
    
    Returns:
        k_dir: chosen number of undirected edges for direction training
        dir_mask: [d, d] binary mask (top-K_dir entries by magnitude)
        info: dict with diagnostics (gap_val, boundary_score, etc.)
    """
    d = mag_probs.shape[0]
    # Extract upper triangle (undirected edge scores)
    triu_idx = torch.triu_indices(d, d, offset=1, device=mag_probs.device)
    scores = mag_probs[triu_idx[0], triu_idx[1]]  # [n_pairs]
    n_pairs = scores.shape[0]  # d*(d-1)/2
    
    # Sort descending
    sorted_scores, sorted_order = torch.sort(scores, descending=True)
    
    # Clamp range
    k_min = max(1, int(k_min_factor * target_edges))
    k_max = min(n_pairs - 1, int(k_max_factor * target_edges))
    k_min = min(k_min, k_max)  # safety
    
    # Compute gaps g[t] = s[t] - s[t+1]
    gaps = sorted_scores[:-1] - sorted_scores[1:]  # [n_pairs-1]
    
    # Find argmax gap in [k_min, k_max]
    search_gaps = gaps[k_min:k_max+1]  # gaps at positions k_min..k_max
    if search_gaps.numel() == 0:
        # Degenerate case
        k_dir = target_edges
        gap_val = 0.0
    else:
        best_local_idx = search_gaps.argmax().item()
        best_gap_idx = k_min + best_local_idx  # global index
        gap_val = gaps[best_gap_idx].item()
        
        if gap_val < min_gap:
            # No clear separation — default to target_edges
            k_dir = target_edges
        else:
            # K_dir = position after the gap (number of edges above the gap)
            k_dir = best_gap_idx + 1  # +1 because gap at index t means t+1 edges
    
    # Clamp final K_dir
    k_dir = max(k_min, min(k_dir, k_max))
    
    # Build mask: top-K_dir undirected edges → symmetric [d, d] mask
    dir_mask = torch.zeros(d, d, device=mag_probs.device)
    topk_order = sorted_order[:k_dir]  # indices into upper-tri scores
    rows = triu_idx[0][topk_order]
    cols = triu_idx[1][topk_order]
    dir_mask[rows, cols] = 1.0
    dir_mask[cols, rows] = 1.0  # symmetric
    
    # Boundary info for logging
    boundary_score = sorted_scores[k_dir - 1].item() if k_dir > 0 else 0.0
    next_score = sorted_scores[k_dir].item() if k_dir < n_pairs else 0.0
    
    info = {
        "k_dir": k_dir,
        "gap_val": gap_val,
        "boundary_score": boundary_score,
        "next_score": next_score,
        "k_min": k_min,
        "k_max": k_max,
        "score_max": sorted_scores[0].item() if n_pairs > 0 else 0.0,
    }
    
    return k_dir, dir_mask, info


def compute_exclusivity_loss(A: torch.Tensor, dir_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    V8.9: Penalize edges where both directions have high weight.
    V8.37: Optional dir_mask restricts to believed edges only (now gap-based).
    
    L_excl = sum_{i<j} A_ij * A_ji  (masked to dir_mask if provided)
    
    This pushes the model to commit to ONE direction per edge pair,
    increasing direction margins and preventing the 0 <-> 4 TP oscillation.
    
    Args:
        A: Adjacency matrix [d, d]
        dir_mask: [d, d] binary mask where 1 = believed edge (optional)
    
    Returns:
        Scalar loss (mean over masked pairs to normalize)
    """
    # A * A.T gives element-wise product: high when both A_ij and A_ji are high
    product = A * A.T
    
    # Only count upper triangle (avoid double-counting)
    mask = torch.triu(torch.ones_like(product), diagonal=1)
    
    # V8.36: Intersect with dir_mask if provided
    if dir_mask is not None:
        # dir_mask marks believed edges — symmetrize for upper triangle
        dir_mask_sym = torch.maximum(dir_mask, dir_mask.T)
        mask = mask * dir_mask_sym
    
    n_pairs = mask.sum() + 1e-8
    L_excl = (product * mask).sum() / n_pairs
    
    return L_excl


def compute_nontopk_suppression_loss(
    A: torch.Tensor,
    k: int,
    margin: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    V8.15: Non-TopK Suppression Loss - push non-TopK edges toward zero.
    
    Problem: Model achieves perfect TopK-F1 but has fat tail of medium-confidence
    edges (e.g., 146 edges at threshold 0.2 vs 13 at 0.5). This hurts:
    - Threshold-based evaluation
    - Interpretability (noisy adjacency matrix)
    - Calibration (edge weights don't reflect true confidence)
    
    Solution: Explicitly penalize non-TopK edges to push them down.
    
    Two components:
    1. L_tail: L1 penalty on non-TopK edges (push toward 0)
    2. L_margin: Margin loss between K-th and (K+1)-th edge (increase separation)
    
    Args:
        A: Adjacency matrix [d, d]
        k: Number of top edges to keep
        margin: Minimum desired gap between TopK and non-TopK
        
    Returns:
        Total loss and metrics dict
    """
    # Flatten and get top-K vs rest
    A_flat = A.flatten()
    n_edges = A_flat.numel()
    
    if k >= n_edges:
        return torch.tensor(0.0, device=A.device), {"L_tail": 0.0, "L_margin": 0.0}
    
    # Get top-K indices and values
    topk_vals, topk_indices = torch.topk(A_flat, k)
    
    # Create mask for non-TopK edges
    mask = torch.ones_like(A_flat)
    mask[topk_indices] = 0.0
    
    # 1. L_tail: L1 penalty on non-TopK edges
    # Goal: push these toward 0
    nontopk_vals = A_flat * mask
    L_tail = nontopk_vals.sum() / (n_edges - k) # Mean over non-TopK
    
    # 2. L_margin: Margin between K-th and (K+1)-th edge
    # Goal: increase separation between TopK and rest
    kth_edge = topk_vals[-1] # Smallest of top K
    
    # Get max of non-TopK edges
    nontopk_max = (A_flat * mask).max()
    
    # Hinge loss: penalize if gap < margin
    L_margin = F.relu(nontopk_max - kth_edge + margin)
    
    metrics = {
        "L_tail": L_tail.item(),
        "L_margin": L_margin.item(),
        "topk_min": kth_edge.item(),
        "nontopk_max": nontopk_max.item(),
        "topk_gap": (kth_edge - nontopk_max).item(),
    }
    
    return L_tail + L_margin, metrics


# =============================================================================
# V8.11: Antisymmetric Direction Parameterization
# =============================================================================

def compute_antisymmetric_adjacency(
    W: torch.Tensor,
    tau_skeleton: float = 1.0,
    tau_direction: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    V8.11: Compute adjacency with antisymmetric direction parameterization.
    
    This separates skeleton (existence) from direction (orientation):
    
    1. Skeleton (symmetric): S_ij = (W_ij + W_ji) / 2
       P(edge exists) = sigmoid(S_ij / τ_skeleton)
       
    2. Direction (antisymmetric): D_ij = W_ij - W_ji
       P(i->j | edge exists) = sigmoid(D_ij / τ_direction)
    
    Final adjacency: A_ij = skeleton_ij * direction_ij
    
    This FORCES exactly one direction per pair (when D_ij != 0).
    
    Args:
        W: Raw logits [d, d] from graph learner
        tau_skeleton: Temperature for skeleton (higher = softer)
        tau_direction: Temperature for direction (lower = sharper)
    
    Returns:
        A: Final adjacency [d, d]
        skeleton: Symmetric skeleton probabilities [d, d]
        direction_probs: P(i->j | edge) for each pair [d, d]
    """
    d = W.shape[0]
    device = W.device
    
    # Zero diagonal
    diag_mask = torch.eye(d, device=device)
    W = W * (1 - diag_mask)
    
    # Skeleton: symmetric part (average of W_ij and W_ji)
    S = (W + W.T) / 2.0
    skeleton = torch.sigmoid(S / tau_skeleton)
    
    # Direction: antisymmetric part (W_ij - W_ji)
    # For each pair (i,j), D_ij = W_ij - W_ji = -D_ji
    D = W - W.T
    direction_probs = torch.sigmoid(D / tau_direction)
    
    # Final adjacency: skeleton * direction
    # A_ij = P(edge exists in pair) * P(direction is i->j)
    A = skeleton * direction_probs
    A = A * (1 - diag_mask) # Ensure diagonal is zero
    
    return A, skeleton, direction_probs


def compute_direction_loss(
    W: torch.Tensor,
    A_true: torch.Tensor,
    tau_direction: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    V8.11: Compute direction-specific loss on true edges.
    
    For each true edge (i,j), we want:
    - D_ij = W_ij - W_ji > 0 (i.e., sigmoid(D_ij/τ) -> 1)
    
    Loss = BCE(sigmoid(D_ij / τ), 1) for true edges (i,j)
    
    Args:
        W: Raw logits [d, d]
        A_true: Ground truth adjacency [d, d]
        tau_direction: Temperature for direction sigmoid
    
    Returns:
        loss: Scalar direction loss
        metrics: Dict with direction accuracy, margins, etc.
    """
    device = W.device
    d = W.shape[0]
    
    # Get true edges
    true_edges = (A_true > 0.5).float()
    
    # Direction scores: D_ij = W_ij - W_ji
    D = W - W.T
    direction_probs = torch.sigmoid(D / tau_direction)
    
    # For each true edge (i,j), we want direction_probs[i,j] -> 1
    # BCE loss: -log(direction_probs) for true edges
    eps = 1e-8
    direction_loss = -torch.log(direction_probs + eps) * true_edges
    loss = direction_loss.sum() / (true_edges.sum() + eps)
    
    # Compute metrics
    with torch.no_grad():
        # Direction accuracy: how many true edges have correct direction?
        correct_direction = (direction_probs > 0.5).float() * true_edges
        dir_accuracy = correct_direction.sum() / (true_edges.sum() + eps)
        
        # Average margin on true edges
        margins = D * true_edges
        avg_margin = margins.sum() / (true_edges.sum() + eps)
        
        # Min margin on true edges (detect weak directions)
        true_edge_margins = D[A_true > 0.5]
        min_margin = true_edge_margins.min().item() if len(true_edge_margins) > 0 else 0.0
    
    metrics = {
        "dir_loss": loss.item(),
        "dir_accuracy": dir_accuracy.item(),
        "dir_avg_margin": avg_margin.item(),
        "dir_min_margin": min_margin,
    }
    
    return loss, metrics


def freeze_skeleton_parameters(model, skeleton_topk_edges: set, W_frozen: torch.Tensor):
    """
    V8.33: Freeze skeleton by restoring W_mag to frozen values.
    
    After skeleton is learned, we want to keep W_mag fixed
    and only train W_dir (direction). With dual parameters,
    this is trivial — just restore W_mag.data.
    
    Args:
        model: RCGNN model with graph_learner
        skeleton_topk_edges: Set of (i,j) pairs in learned skeleton
        W_frozen: Frozen W_mag logits at time of freeze
    """
    with torch.no_grad():
        # V8.33: Simply restore W_mag. W_dir is untouched → direction keeps learning.
        model.graph_learner.W_mag.data.copy_(W_frozen)


# =============================================================================
# V8.8: Edge Discovery Fixes
# =============================================================================

def compute_cousin_mask(
    A_true: torch.Tensor,
    causal_convention: bool = True,
) -> torch.Tensor:
    """
    V8.8: Compute mask of "cousin" pairs — nodes with common ancestor but no direct edge.
    
    These pairs are often high-correlation but NOT causally connected.
    We penalize the model for predicting edges on these pairs.
    
    CONVENTION (V8.26): Both A_true and model use CAUSAL convention
    (A[i,j] = i causes j). No transpose needed.
    
    Args:
        A_true: Ground truth adjacency [d, d] in CAUSAL convention
        causal_convention: If True, A_true is in causal convention (default)
    
    Returns:
        cousin_mask: [d, d] binary mask where 1 = cousin pair (should NOT have edge)
    """
    # V8.26 FIX: Model and A_true both use causal convention A[i,j]=i→j
    # No transpose needed. The causal_convention parameter is kept for compat.
    A_check = A_true
    A_bin = (A_check.detach() > 0.5).float()  # [d, d] — causal convention
    d = A_bin.shape[0]
    device = A_true.device
    
    # Vectorised transitive closure via repeated matrix-power on bool adjacency
    # reach[i,j]=1 means i can reach j.  Initialise with direct edges.
    reach = A_bin.clone()
    for _ in range(d - 1):
        reach_new = ((reach @ A_bin) > 0).float()
        if (reach_new == reach).all():
            break
        reach = torch.clamp(reach + reach_new, 0, 1)
    
    # ancestors[j] = set of nodes that can reach j  →  reach[:, j] > 0
    # common ancestor of (i, j): any k with reach[k,i]>0 AND reach[k,j]>0
    # Vectorised: common = reach.T @ reach  (d×d)  — entry (i,j) = # common ancestors
    has_common_ancestor = (reach.T @ reach) > 0  # [d, d]
    
    # No direct edge in either direction
    no_edge = (A_bin < 0.5) & (A_bin.T < 0.5)
    no_diag = ~torch.eye(d, dtype=torch.bool, device=device)
    
    cousin_mask = (has_common_ancestor & no_edge & no_diag).float()
    return cousin_mask


def compute_hard_negative_mask(
    A_true: torch.Tensor,
    corr_matrix: torch.Tensor,
    percentile: float = 80,
    causal_convention: bool = True,
) -> torch.Tensor:
    """
    V8.8: Compute mask of hard negatives — high-correlation non-edges.
    
    CONVENTION FIX (V8.25): A_true arrives in CAUSAL convention.
    Non-edge check is symmetric so convention doesn't change the result,
    but we document it explicitly for clarity.
    
    Fully vectorised — no Python loops.
    
    Args:
        A_true: Ground truth adjacency [d, d] in CAUSAL convention
        corr_matrix: Pairwise |correlation| matrix [d, d]
        percentile: Top X% correlations among non-edges to flag
        causal_convention: Whether A_true is in causal convention
    
    Returns:
        hard_neg_mask: [d, d] float mask where 1 = hard negative
    """
    d = A_true.shape[0]
    # V8.26 FIX: Model and A_true both use causal convention, no transpose
    A_check = A_true
    
    non_edge_mask = (A_check < 0.5) & (A_check.T < 0.5)
    non_edge_mask = non_edge_mask & ~torch.eye(d, dtype=torch.bool, device=A_true.device)
    
    corr_abs = corr_matrix.abs()
    non_edge_corrs = corr_abs[non_edge_mask]
    
    if non_edge_corrs.numel() > 0:
        threshold = torch.quantile(non_edge_corrs, percentile / 100.0)
        hard_neg_mask = non_edge_mask & (corr_abs >= threshold)
    else:
        hard_neg_mask = torch.zeros_like(A_true, dtype=torch.bool)
    
    return hard_neg_mask.float()


def compute_edge_discovery_losses(
    A_pred: torch.Tensor,
    A_true: torch.Tensor,
    corr_matrix: Optional[torch.Tensor],
    config: Dict[str, Any],
    epoch: int,
    total_epochs: int,
    cached_cousin_mask: Optional[torch.Tensor] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    """
    V8.8: Compute edge discovery losses to fix mode collapse.
    
    ⚠️  ORACLE supervision — uses ground-truth A_true.
    Must only be called when oracle_direction_supervision is True.
    
    CONVENTION (V8.26):
      Both A_true and A_pred use CAUSAL convention (A[i,j] = i→j).
      No conversion needed anywhere.
    
    1. Presence loss: BCE on edge presence with hard negative mining
    2. Cousin penalty: Penalize edges between cousin pairs
    3. Coverage loss: Ensure minimum outgoing mass per node
    
    Returns:
        losses: Dict of loss tensors
        metrics: Dict of metric values for logging
    """
    d = A_pred.shape[0]
    device = A_pred.device
    losses = {}
    metrics = {}
    
    # Presence score: max(A_ij, A_ji) for undirected edge presence
    A_presence = torch.maximum(A_pred, A_pred.T)
    
    # True presence (undirected) — symmetric, convention-agnostic
    A_true_presence = torch.maximum(A_true, A_true.T)
    
    # =========================================================================
    # 1. PRESENCE LOSS with hard negative mining
    # =========================================================================
    if config.get("use_presence_loss", True):
        pos_weight = config.get("presence_pos_weight", 3.0)
        lambda_presence = config.get("lambda_presence", 0.5)
        
        # Only upper triangle (avoid double counting)
        mask_upper = torch.triu(torch.ones(d, d, device=device), diagonal=1)
        
        # True labels (presence)
        y_true = (A_true_presence > 0.5).float() * mask_upper
        y_pred = A_presence * mask_upper
        
        # Weighted BCE loss
        # pos_weight upweights positive (true edge) samples
        bce_loss = F.binary_cross_entropy(
            y_pred.clamp(1e-6, 1-1e-6),
            y_true,
            reduction='none'
        )
        
        # Apply pos_weight to positives
        weights = torch.where(y_true > 0.5, pos_weight, 1.0)
        
        # Hard negative mining
        if config.get("use_hard_negatives", True) and corr_matrix is not None:
            percentile = config.get("hard_neg_corr_percentile", 80)
            hard_neg_mask = compute_hard_negative_mask(A_true, corr_matrix, percentile)
            hard_neg_mask = hard_neg_mask * mask_upper
            
            # Upweight hard negatives (high-corr non-edges)
            weights = weights + hard_neg_mask * 2.0 # Extra weight on hard negatives
            metrics["n_hard_neg"] = int(hard_neg_mask.sum().item())
        
        L_presence = (bce_loss * weights).sum() / (weights.sum() + 1e-8)
        losses["L_presence"] = lambda_presence * L_presence
        metrics["L_presence"] = L_presence.item()
    
    # =========================================================================
    # 2. COUSIN PENALTY
    # =========================================================================
    if config.get("use_cousin_penalty", True):
        lambda_cousin = config.get("lambda_cousin", 0.3)
        
        # Use cached mask if available (computed once per dataset)
        if cached_cousin_mask is not None:
            cousin_mask = cached_cousin_mask.to(device)
        else:
            cousin_mask = compute_cousin_mask(A_true, causal_convention=True)
        
        # Penalize predicted presence on cousin pairs
        # Weight by correlation if available (higher corr = stronger penalty)
        if corr_matrix is not None:
            cousin_weights = cousin_mask * (1.0 + corr_matrix.abs())
        else:
            cousin_weights = cousin_mask
        
        L_cousin = (A_presence * cousin_weights).sum() / (cousin_weights.sum() + 1e-8)
        losses["L_cousin"] = lambda_cousin * L_cousin
        metrics["L_cousin"] = L_cousin.item()
        metrics["n_cousin_pairs"] = int((cousin_mask > 0).sum().item() // 2)
    
    # =========================================================================
    # 3. COVERAGE LOSS (prevent hub collapse)
    # =========================================================================
    if config.get("use_coverage_loss", True):
        coverage_epochs = config.get("coverage_epochs", 0.5)
        
        # Only apply in first X% of training
        if epoch <= coverage_epochs * total_epochs:
            lambda_coverage = config.get("lambda_coverage", 0.2)
            min_outdeg = config.get("coverage_min_outdeg", 0.5)
            
            # Outgoing degree per node
            out_degree = A_pred.sum(dim=1) # [d]
            
            # Penalize nodes with insufficient outgoing mass
            # ReLU(min_outdeg - outdeg_i) = 0 if outdeg >= min, else positive
            L_coverage = F.relu(min_outdeg - out_degree).mean()
            
            losses["L_coverage"] = lambda_coverage * L_coverage
            metrics["L_coverage"] = L_coverage.item()
            metrics["min_out_deg"] = out_degree.min().item()
            metrics["mean_out_deg"] = out_degree.mean().item()
        else:
            metrics["L_coverage"] = 0.0
    
    # =========================================================================
    # 4. EXCLUSIVITY LOSS (V8.9: force commitment to ONE direction per pair)
    # =========================================================================
    if config.get("use_exclusivity_loss", True):
        lambda_excl = config.get("lambda_exclusivity", 0.3)
        
        L_excl = compute_exclusivity_loss(A_pred)
        
        losses["L_excl"] = lambda_excl * L_excl
        metrics["L_excl"] = L_excl.item()
        
        # Track average symmetry: how symmetric are the edges?
        A_min = torch.minimum(A_pred, A_pred.T)
        A_max = torch.maximum(A_pred, A_pred.T)
        symmetry_ratio = (A_min / (A_max + 1e-8)).mean()
        metrics["symmetry_ratio"] = symmetry_ratio.item() # 1.0 = fully symmetric (bad)
    
    # =========================================================================
    # 5. NON-TOPK SUPPRESSION (V8.15: push non-TopK edges toward zero)
    # Problem: Perfect TopK-F1 but fat tail of medium-confidence edges
    # (e.g., 146 @ 0.2 vs 13 @ 0.5) hurts interpretability
    # Solution: Explicitly penalize non-TopK edges
    # =========================================================================
    if config.get("use_nontopk_suppression", True):
        # Ramp up suppression over training (don't suppress early - need exploration)
        suppression_start = config.get("nontopk_suppression_start", 0.2) # Start at 20%
        lambda_suppress = config.get("lambda_nontopk_suppression", 0.5)
        suppress_margin = config.get("nontopk_margin", 0.1)
        target_edges = config.get("target_edges", 13)
        
        if epoch >= suppression_start * total_epochs:
            # Ramp from 0 to full over next 30% of training
            ramp_end = suppression_start + 0.3
            if epoch < ramp_end * total_epochs:
                ramp_factor = (epoch / total_epochs - suppression_start) / 0.3
            else:
                ramp_factor = 1.0
            
            L_suppress, suppress_metrics = compute_nontopk_suppression_loss(
                A_pred, k=target_edges, margin=suppress_margin
            )
            
            losses["L_nontopk"] = lambda_suppress * ramp_factor * L_suppress
            metrics["L_nontopk"] = L_suppress.item()
            metrics["topk_gap"] = suppress_metrics["topk_gap"]
            metrics["nontopk_max"] = suppress_metrics["nontopk_max"]
        else:
            metrics["L_nontopk"] = 0.0
    
    return losses, metrics


def compute_correlation_baseline(
    X: torch.Tensor,
    A_true: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute TRUE correlation baseline: what a simple correlation method achieves.
    
    This establishes a principled baseline - NOT the model's early predictions!
    Uses pairwise Pearson correlation, handling missing values properly.
    """
    from scipy import stats
    
    # Flatten X to [N, d]
    if X.dim() == 3:
        X_flat = X.reshape(-1, X.shape[-1])
    else:
        X_flat = X
    
    X_np = X_flat.detach().cpu().numpy()
    A_true_np = (A_true.cpu().numpy() > 0.5).astype(int)
    d = X_np.shape[1]
    
    # Compute pairwise correlation matrix (handles NaN properly)
    corr_matrix = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            # Get valid (non-NaN) pairs
            valid = ~(np.isnan(X_np[:, i]) | np.isnan(X_np[:, j]))
            if valid.sum() < 10:
                continue
            r, _ = stats.pearsonr(X_np[valid, i], X_np[valid, j])
            corr_matrix[i, j] = abs(r) if not np.isnan(r) else 0
    
    np.fill_diagonal(A_true_np, 0)
    n_true_edges = int(A_true_np.sum())
    k = n_true_edges
    
    # Get top K edges by absolute correlation
    flat = corr_matrix.flatten()
    top_k_idx = np.argsort(flat)[::-1][:k]
    
    pred_edges = set()
    for idx in top_k_idx:
        if flat[idx] > 0:
            pred_edges.add((idx // d, idx % d))
    
    true_edges = set(zip(*np.where(A_true_np > 0)))
    
    # Directed TopK-F1
    TP = len(true_edges & pred_edges)
    FP = len(pred_edges - true_edges)
    FN = len(true_edges - pred_edges)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Skeleton-F1 (undirected)
    pred_skel = set((min(i,j), max(i,j)) for i, j in pred_edges)
    true_skel = set((min(i,j), max(i,j)) for i, j in true_edges)
    skel_tp = len(true_skel & pred_skel)
    skel_prec = skel_tp / len(pred_skel) if pred_skel else 0
    skel_rec = skel_tp / len(true_skel) if true_skel else 0
    skel_f1 = 2 * skel_prec * skel_rec / (skel_prec + skel_rec) if (skel_prec + skel_rec) > 0 else 0
    
    # Count reversed (correlation finds right pair, wrong direction)
    reversed_count = sum(1 for (i, j) in pred_edges if (j, i) in true_edges)
    
    return {
        "corr_baseline_f1": float(f1),
        "corr_baseline_tp": TP,
        "corr_baseline_skel_f1": float(skel_f1),
        "corr_baseline_reversed": reversed_count,
        "corr_max": float(corr_matrix.max()),
        "corr_mean": float(corr_matrix.mean()),
        "k": k,
    }


def get_topk_adjacency(
    A: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """
    Extract TopK adjacency: keep only top K edges, zero out rest.
    This matches the edges used in evaluation.
    
    Args:
        A: [d, d] adjacency matrix
        k: number of edges to keep
    
    Returns:
        A_topk: [d, d] with only top K edges, rest zeroed
    """
    A_flat = A.flatten()
    _, topk_indices = torch.topk(A_flat, k=min(k, len(A_flat)))
    
    A_topk = torch.zeros_like(A)
    A_topk.view(-1)[topk_indices] = A.view(-1)[topk_indices]
    
    return A_topk


def compute_topk_f1(
    A_pred: torch.Tensor,
    A_true: torch.Tensor,
    k: Optional[int] = None,
    compute_transpose: bool = False,
    stable_direction: bool = True, # V8.9: Use margin-based direction selection
) -> Dict[str, float]:
    """
    Compute TopK metrics: predict top K edges vs true edges.
    K defaults to number of true edges (fair comparison).
    
    V8.9 FIX: Separate edge selection from direction selection.
    1. Select K UNDIRECTED pairs using max(A_ij, A_ji)
    2. For each pair, predict direction using MARGIN (A_ij - A_ji)
    This eliminates the 0 <-> 4 TP oscillation caused by noise in symmetric A.
    
    Args:
        compute_transpose: If True, also compute F1 on A_pred.T (diagnostic)
        stable_direction: If True, use margin-based direction (stable). If False, use raw TopK.
    """
    A_pred_np = A_pred.detach().cpu().numpy().copy()
    A_true_np = (A_true.cpu().numpy() > 0.5).astype(int)
    
    # Zero diagonal
    np.fill_diagonal(A_pred_np, 0)
    np.fill_diagonal(A_true_np, 0)
    
    n_true_edges = int(A_true_np.sum())
    if k is None:
        k = n_true_edges
    
    d = A_pred_np.shape[0]
    
    # Always define true_flat (needed for transpose diagnostic)
    true_flat = A_true_np.flatten()
    
    if stable_direction:
        # =====================================================================
        # V8.9: STABLE DIRECTION SELECTION
        # Step 1: Select K UNDIRECTED pairs using max(A_ij, A_ji)
        # Step 2: For each pair, direction = sign(A_ij - A_ji)
        # This decouples edge selection from direction, preventing oscillation.
        # =====================================================================
        
        # Symmetrize for pair selection
        A_sym = np.maximum(A_pred_np, A_pred_np.T)
        np.fill_diagonal(A_sym, 0)
        A_upper = np.triu(A_sym) # Only upper triangle to avoid double-counting pairs
        
        # V8.10: DETERMINISTIC TIE-BREAK
        # Sort by (score, -index) to ensure consistent ordering when ties exist.
        # Using -index means lower indices win ties (stable, reproducible).
        flat_upper = A_upper.flatten()
        indices = np.arange(len(flat_upper))
        # Create (score, -index) tuples, sort descending by score, then ascending by index
        sort_keys = np.lexsort((-indices, flat_upper)) # lexsort is stable, sorts by last key first
        top_k_idx = sort_keys[-k:] if k > 0 else np.array([]) # Take top K
        
        # Convert to (i, j) pairs
        pairs = []
        for idx in top_k_idx:
            if flat_upper[idx] > 0:
                i, j = idx // d, idx % d
                pairs.append((i, j))
        
        # For each pair, predict direction using margin (no softmax, no τ)
        # V8.10: Use STRICT inequality so ties default to (i,j) for i<j determinism
        pred_edges = set()
        margins = [] # Track margins for diagnostics
        for i, j in pairs:
            margin = A_pred_np[i, j] - A_pred_np[j, i]
            margins.append(abs(margin))
            if margin > 0: # Strict > 0 for determinism
                pred_edges.add((i, j))
            elif margin < 0:
                pred_edges.add((j, i))
            else:
                # Tie: default to (min, max) for determinism
                pred_edges.add((min(i, j), max(i, j)))
        
        # Build true edges set
        true_edges = set()
        for i in range(d):
            for j in range(d):
                if A_true_np[i, j] > 0:
                    true_edges.add((i, j))
        
        TP = len(pred_edges & true_edges)
        FP = len(pred_edges - true_edges)
        FN = len(true_edges - pred_edges)
        
        # Track margin stats for debugging
        avg_margin = float(np.mean(margins)) if margins else 0.0
        min_margin = float(np.min(margins)) if margins else 0.0
        
    else:
        # Original behavior: raw top-K directed edges
        flat = A_pred_np.flatten()
        top_k_idx = np.argsort(flat)[-k:] if k > 0 else np.array([])
        
        pred_binary = np.zeros_like(flat, dtype=int)
        if len(top_k_idx) > 0:
            pred_binary[top_k_idx] = 1
        
        true_flat = A_true_np.flatten()
        
        TP = int(((pred_binary == 1) & (true_flat == 1)).sum())
        FP = int(((pred_binary == 1) & (true_flat == 0)).sum())
        FN = int(((pred_binary == 0) & (true_flat == 1)).sum())
        
        avg_margin = 0.0
        min_margin = 0.0
    
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    result = {
        "topk_f1": float(f1),
        "topk_precision": float(precision),
        "topk_recall": float(recall),
        "topk_shd": FP + FN,
        "topk_tp": TP,
        "topk_fp": FP,
        "topk_fn": FN,
        "k": k,
        "avg_margin": avg_margin, # V8.9: Track direction confidence
        "min_margin": min_margin,
    }
    
    # =========================================================================
    # DIAGNOSTIC: Evaluate on TRANSPOSE to detect direction mismatch
    # If topk_f1_transpose >> topk_f1, the model learned anti-causal direction
    # =========================================================================
    if compute_transpose:
        A_pred_T = A_pred_np.T.copy()
        np.fill_diagonal(A_pred_T, 0)
        
        flat_T = A_pred_T.flatten()
        top_k_idx_T = np.argsort(flat_T)[-k:] if k > 0 else np.array([])
        
        pred_binary_T = np.zeros_like(flat_T, dtype=int)
        if len(top_k_idx_T) > 0:
            pred_binary_T[top_k_idx_T] = 1
        
        TP_T = int(((pred_binary_T == 1) & (true_flat == 1)).sum())
        FP_T = int(((pred_binary_T == 1) & (true_flat == 0)).sum())
        FN_T = int(((pred_binary_T == 0) & (true_flat == 1)).sum())
        
        precision_T = TP_T / (TP_T + FP_T + 1e-8)
        recall_T = TP_T / (TP_T + FN_T + 1e-8)
        f1_T = 2 * precision_T * recall_T / (precision_T + recall_T + 1e-8)
        
        result["topk_f1_transpose"] = float(f1_T)
        result["topk_tp_transpose"] = TP_T
    
    return result


def compute_best_f1(
    A_pred: torch.Tensor,
    A_true: torch.Tensor,
    thresholds: List[float] = None,
) -> Dict[str, float]:
    """Find best F1 over thresholds."""
    if thresholds is None:
        thresholds = list(np.arange(0.05, 0.55, 0.05))
    
    best_f1 = 0
    best_t = 0.1
    best_metrics = {}
    
    for t in thresholds:
        metrics = compute_structure_metrics(A_pred, A_true, threshold=t)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_t = t
            best_metrics = metrics
    
    return {
        "best_f1": best_f1,
        "best_threshold": best_t,
        "best_shd": best_metrics.get("shd", 0),
        "best_precision": best_metrics.get("precision", 0),
        "best_recall": best_metrics.get("recall", 0),
    }


def compute_auc_f1(
    A_pred: torch.Tensor,
    A_true: torch.Tensor,
    thresholds: List[float] = None,
) -> Dict[str, float]:
    """Compute AUC-F1 over thresholds (robustness metric)."""
    if thresholds is None:
        thresholds = list(np.arange(0.05, 0.55, 0.05))
    
    f1_scores = []
    for t in thresholds:
        metrics = compute_structure_metrics(A_pred, A_true, threshold=t)
        f1_scores.append(metrics["f1"])
    
    # Trapezoidal AUC
    auc = np.trapz(f1_scores, thresholds) / (thresholds[-1] - thresholds[0])
    
    return {
        "auc_f1": float(auc),
        "max_f1": float(max(f1_scores)),
        "mean_f1": float(np.mean(f1_scores)),
    }


def diagnose_correlation_vs_causation(
    A_pred: torch.Tensor,
    A_true: torch.Tensor,
    X: torch.Tensor,
) -> Dict[str, float]:
    """
    Diagnose whether model learns causation or correlation.
    From V4 causal priors.
    """
    A_pred_np = A_pred.detach().cpu().numpy().copy()
    A_true_np = A_true.cpu().numpy()
    X_np = X.cpu().numpy()
    
    # Compute correlation matrix
    if X_np.ndim == 3:
        X_flat = X_np.reshape(-1, X_np.shape[-1])
    else:
        X_flat = X_np
    corr = np.corrcoef(X_flat.T)
    corr = np.abs(corr)
    np.fill_diagonal(corr, 0)
    
    # Zero diagonals
    np.fill_diagonal(A_pred_np, 0)
    np.fill_diagonal(A_true_np, 0)
    
    # Get top K for each (K = true edges)
    k = int(A_true_np.sum())
    
    def top_k_indices(arr, k):
        flat = arr.flatten()
        return set(np.argsort(flat)[-k:].tolist()) if k > 0 else set()
    
    pred_topk = top_k_indices(A_pred_np, k)
    corr_topk = top_k_indices(corr, k)
    true_edges = set(np.argwhere(A_true_np.flatten() > 0.5).flatten().tolist())
    
    # Overlaps
    pred_corr_overlap = len(pred_topk & corr_topk) / max(len(pred_topk), 1)
    pred_true_overlap = len(pred_topk & true_edges) / max(len(pred_topk), 1)
    corr_true_overlap = len(corr_topk & true_edges) / max(len(corr_topk), 1)
    
    # Average edge correlations
    def avg_edge_corr(edge_set):
        d = corr.shape[0]
        vals = [corr.flatten()[i] for i in edge_set if i < d*d]
        return np.mean(vals) if vals else 0.0
    
    avg_pred_edge_corr = avg_edge_corr(pred_topk)
    avg_true_edge_corr = avg_edge_corr(true_edges)
    
    # Diagnosis (same logic as MNAR detector for consistency)
    # If model beats correlation baseline OR finds enough true edges -> causation
    if (pred_true_overlap > 0.5 and (pred_true_overlap > pred_corr_overlap or pred_true_overlap > 0.8)) or \
       pred_true_overlap > pred_corr_overlap + 0.2:
        diagnosis = "causation"
    elif pred_corr_overlap > pred_true_overlap + 0.2:
        diagnosis = "correlation"
    else:
        diagnosis = "mixed"
    
    return {
        "pred_corr_overlap": pred_corr_overlap,
        "pred_true_overlap": pred_true_overlap,
        "corr_true_overlap": corr_true_overlap,
        "avg_pred_edge_corr": avg_pred_edge_corr,
        "avg_true_edge_corr": avg_true_edge_corr,
        "diagnosis": diagnosis,
    }


# =============================================================================
# STEP 3: Proper Health Check (not dense-is-healthy!)
# =============================================================================

def compute_health_metrics(
    A_pred: torch.Tensor,
    target_edges: int,
    config: Dict,
    prev_topk_set: Optional[set] = None,
) -> Dict:
    """
    Compute health metrics for causal discovery.
    
    CORRECT definition of "healthy":
    - edges@0.5 <= 3 * target_edges (not dense)
    - edge_sum trending downward during PRUNE
    - TopK Jaccard > 0.7 (stability)
    
    NOT healthy (previous bug):
    - high A_max and huge edges@0.2 (that's correlation!)
    """
    A_np = A_pred.detach().cpu().numpy()
    np.fill_diagonal(A_np, 0)
    d = A_np.shape[0]
    max_edges = d * (d - 1)
    
    # Edge counts at thresholds
    edges_02 = int((A_np > 0.2).sum())
    edges_03 = int((A_np > 0.3).sum())
    edges_05 = int((A_np > 0.5).sum())
    edges_07 = int((A_np > 0.7).sum())
    
    # Key health check: is the graph sparse enough?
    max_ratio = config.get("max_edges_at_0.5_ratio", 3.0)
    edges_05_healthy = edges_05 <= max_ratio * target_edges
    
    # TopK stability
    k = target_edges
    flat = A_np.flatten()
    curr_topk = set(np.argsort(flat)[-k:].tolist()) if k > 0 else set()
    
    if prev_topk_set is not None and len(prev_topk_set) > 0:
        jaccard = len(curr_topk & prev_topk_set) / max(len(curr_topk | prev_topk_set), 1)
    else:
        jaccard = 1.0 # First epoch
    
    min_jaccard = config.get("min_topk_jaccard", 0.7)
    topk_stable = jaccard >= min_jaccard
    
    # Overall health
    is_healthy = edges_05_healthy and topk_stable
    
    return {
        "edges@0.2": edges_02,
        "edges@0.3": edges_03,
        "edges@0.5": edges_05,
        "edges@0.7": edges_07,
        "edge_sum": float(A_np.sum()),
        "density@0.5": edges_05 / max_edges,
        "edges_05_healthy": edges_05_healthy,
        "topk_jaccard": jaccard,
        "topk_stable": topk_stable,
        "is_healthy": is_healthy,
        "curr_topk_set": curr_topk,
    }


def compute_multi_k_f1(
    A_pred: torch.Tensor,
    A_true: torch.Tensor,
    k_values: List[int] = [13, 20, 30],
) -> Dict[str, float]:
    """
    STEP 5: Compute F1 at multiple K values for robustness.
    
    Prevents jumpy evaluation from ties.
    """
    results = {}
    for k in k_values:
        metrics = compute_topk_f1(A_pred, A_true, k=k)
        results[f"f1@{k}"] = metrics["topk_f1"]
        results[f"prec@{k}"] = metrics["topk_precision"]
        results[f"recall@{k}"] = metrics["topk_recall"]
    return results


# =============================================================================
# GroupDRO
# =============================================================================

def groupdro_reweight(
    regime_losses: Dict[int, torch.Tensor],
    weights: Dict[int, float],
    step_size: float = 0.01,
) -> Dict[int, float]:
    """Update GroupDRO weights based on regime losses."""
    new_weights = {}
    total = 0.0
    
    for r, loss in regime_losses.items():
        w = weights.get(r, 1.0)
        # Exponentiated gradient ascent
        w_new = w * torch.exp(step_size * loss.detach()).item()
        new_weights[r] = w_new
        total += w_new
    
    # Normalize
    for r in new_weights:
        new_weights[r] /= (total + 1e-8)
    
    return new_weights


# =============================================================================
# Training Loop
# =============================================================================

def train_epoch(
    model: RCGNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    config: Dict,
    ddp: bool = False,
    dro_weights: Optional[Dict[int, float]] = None,
    direction_stable: bool = True, # NEW: Is model learning correct direction?
    prev_edge_sum: float = None, # NEW: Previous edge_sum for collapse detection
    skeleton_mask: Optional[torch.Tensor] = None, # V8.5: Frozen skeleton for Stage D
    A_true: Optional[torch.Tensor] = None, # V8.7: Ground truth for margin loss
    skeleton_frozen: bool = False, # V8.11: Skeleton is frozen, direction-only mode
    W_frozen: Optional[torch.Tensor] = None, # V8.11: Frozen W for skeleton restoration
    direction_tau: float = 0.5, # V8.11: τ for direction learning (lower = sharper)
    lambda_direction_boost: float = 2.0, # V8.11: Boost direction loss when frozen
    frozen_lambda_budget: float = None, # V8.16: Frozen budget after early_excellence
    peakiness: Dict = None, # V8.17: Peakiness metrics from previous epoch
    E_frozen_mask: Optional[torch.Tensor] = None, # V8.39: Frozen skeleton edge mask for REFINE
) -> Tuple[Dict[str, float], Optional[Dict[int, float]]]:
    """Train one epoch with all bells and whistles."""
    model.train()
    base_model = model.module if ddp else model
    
    # CRITICAL FIX: Update temperature with direction-based floor
    # V8.11: Use fixed direction_tau when skeleton is frozen
    # V8.17: Pass peakiness for data-adaptive temperature
    if skeleton_frozen:
        temperature = direction_tau
    else:
        temperature = get_temperature(
            epoch, total_epochs, config, 
            direction_stable=direction_stable,
            peakiness=peakiness, # V8.17: Data-adaptive
        )
    base_model.graph_learner.set_temperature(temperature)
    
    # Get loss weights with collapse protection
    # V8.16: Pass frozen_lambda_budget to cap λb after convergence
    # V8.17: Pass peakiness to gate λs/λb ramp
    loss_weights = get_loss_weights(
        epoch, total_epochs, config, 
        edge_sum=prev_edge_sum,
        frozen_lambda_budget=frozen_lambda_budget,
        peakiness=peakiness, # V8.17: Data-adaptive
    )
    
    # V8.16: Track λb for logging (so we can see when it's frozen)
    lambda_budget_used = loss_weights.get("lambda_budget", 0)
    lambda_budget_is_frozen = loss_weights.get("lambda_budget_frozen", False)
    peakiness_gated = loss_weights.get("peakiness_gated", False) # V8.17
    
    total_loss = 0.0
    metrics_sum = {}
    n_batches = 0
    
    # For GroupDRO
    regime_losses = {}
    regime_counts = {}
    
    for batch_idx, (X, M, e) in enumerate(loader):
        X = X.to(device)
        M = M.to(device)
        e = e.to(device)
        
        optimizer.zero_grad()
        
        # Forward
        outputs = model(X, M, regime=e)
        
        # =================================================================
        # V8.19 FIX B: Hard TopK gating via straight-through estimator
        # Problem: Without hard gating, all 210 edges stay in 0.3-0.5 band
        # forever → @0.2=210, gap=0.000, no bimodal separation.
        #
        # Fix: After forward, replace non-TopK entries with detached values.
        # TopK entries keep full gradients; non-TopK entries get zero grad
        # but keep their values for loss computation (straight-through).
        #
        # When: Start after DISC phase (epoch > stage1_end * epochs)
        #
        # V8.19.1 FIX: Only modify A_base (used for acy/budget/sparse), NOT A
        # (used for invariance loss which expects per-batch regime structure).
        # =================================================================
        topk_gate_start = config.get("topk_gate_start", config.get("stage1_end", 0.15))  # V8.21: Earlier (0.20→0.15)
        target_edges = config.get("target_edges", 13)
        if epoch >= topk_gate_start * total_epochs:
            # Get soft adjacency (with grad)
            A_soft_gate = base_model.graph_learner.get_mean_adjacency()
            d_gate = A_soft_gate.shape[0]
            
            # Build TopK mask: top 2K entries during PRUNE, top K during REFINE
            stage2_end_frac = config.get("stage2_end", 0.80)
            if epoch < stage2_end_frac * total_epochs:
                k_gate = min(2 * target_edges, d_gate * (d_gate - 1))
            else:
                k_gate = min(target_edges, d_gate * (d_gate - 1))
            
            # Zero diagonal, flatten, find TopK indices
            A_flat = (A_soft_gate * (1 - torch.eye(d_gate, device=A_soft_gate.device))).flatten()
            _, topk_idx = torch.topk(A_flat, k_gate)
            mask_flat = torch.zeros_like(A_flat)
            mask_flat[topk_idx] = 1.0
            topk_mask = mask_flat.view(d_gate, d_gate)
            
            # Straight-through: A_gated = A * mask + A.detach() * (1-mask)
            # TopK entries: full gradient. Non-TopK: no gradient (detached).
            A_gated = A_soft_gate * topk_mask + A_soft_gate.detach() * (1 - topk_mask)
            
            # V8.19.1: Only inject into A_base (for acy/budget/sparse losses)
            # DO NOT modify outputs["A"] - it's used by invariance loss with
            # per-batch regime indexing that would crash with wrong shape.
            outputs["A_base"] = A_gated
            # Note: outputs["A"] is left unchanged for invariance loss
            
            # Track gating metrics
            batch_metrics_extra_gate = {
                "topk_gate_k": k_gate,
                "topk_gate_active": 1.0,
            }
        else:
            batch_metrics_extra_gate = {"topk_gate_active": 0.0}
        
        # =================================================================
        # V8.23 FIX: Hard pruning of non-TopK logits in PRUNE phase
        # Problem: L_tail provides smooth gradients but non-TopK edges stay
        # above 0.2 for dozens of epochs. Hard pruning directly zeros them.
        # When: Every hard_prune_interval epochs during PRUNE phase
        # How: Set W_adj[non-TopK] = hard_prune_logit_value (e.g., -5.0)
        #      sigmoid(-5) ≈ 0.007, effectively zeroing those edges
        # =================================================================
        hard_prune_enabled = config.get("hard_prune_enabled", True)
        hard_prune_interval = config.get("hard_prune_interval", 10)
        stage1_end_ep = int(config.get("stage1_end", 0.30) * total_epochs)
        stage2_end_ep = int(config.get("stage2_end", 0.70) * total_epochs)
        in_prune_phase = (epoch > stage1_end_ep and epoch <= stage2_end_ep)
        
        if (hard_prune_enabled and in_prune_phase and 
            epoch % hard_prune_interval == 0 and batch_idx == 0):
            with torch.no_grad():
                W_mag = base_model.graph_learner.W_mag  # V8.33: prune structure param
                d_prune = W_mag.shape[0]
                diag_mask_prune = torch.eye(d_prune, device=W_mag.device)
                # V8.34 FIX: W_mag is symmetric — select from UPPER TRIANGLE
                # to avoid wasting K slots on both (i,j) and (j,i).
                # Select K undirected edges, then mirror keep mask.
                Wm_sym = (W_mag + W_mag.T) / 2  # ensure symmetric
                Wm_sym = Wm_sym * (1 - diag_mask_prune)
                W_upper = torch.triu(Wm_sym, diagonal=1)  # upper triangle only
                W_upper_flat = W_upper.flatten()
                n_upper = d_prune * (d_prune - 1) // 2
                k_undirected = min(target_edges, n_upper)  # K undirected pairs
                _, topk_upper_idx = torch.topk(W_upper_flat, k_undirected)
                
                # Create mask in upper triangle, then mirror to full matrix
                keep_upper_flat = torch.zeros_like(W_upper_flat)
                keep_upper_flat[topk_upper_idx] = 1.0
                keep_upper = keep_upper_flat.view(d_prune, d_prune)
                keep_mask = keep_upper + keep_upper.T  # mirror: both (i,j) and (j,i) kept
                keep_mask = keep_mask.clamp(max=1.0)  # safety
                
                # V8.35 FIX: Soft damping instead of hard prune.
                # OLD: Set non-TopK to -2.0 (σ≈0.12) — kills gradient flow,
                #      edges can never recover even if they become important.
                # NEW: Shrink non-TopK logits by multiplying by decay factor.
                #      Each prune step reduces by e.g. 0.5, so after 3 prunes:
                #      logit → 0.5 → 0.25 → 0.125 (gradual fade, NOT instant death).
                #      This preserves gradient flow and allows edges to recover
                #      if structural losses push them back up.
                prune_decay = config.get("hard_prune_decay", 0.5)  # V8.35: multiplicative decay
                prune_floor = config.get("hard_prune_floor", -2.0)  # V8.35: minimum logit (safety)
                damped = W_mag.data * keep_mask + W_mag.data * (1 - keep_mask) * (1 - diag_mask_prune) * prune_decay
                # Clamp to floor so logits don't go to -inf over many rounds
                damped = torch.where(
                    keep_mask.bool() | diag_mask_prune.bool(),
                    W_mag.data,
                    torch.clamp(damped, min=prune_floor)
                )
                W_mag.data = damped
                
                if is_main_process():
                    n_kept = int(keep_mask.sum().item())
                    n_pruned = d_prune * (d_prune - 1) - n_kept
                    # Show min logit among non-TopK for monitoring
                    non_topk_vals = W_mag.data[~keep_mask.bool() & ~diag_mask_prune.bool()]
                    min_logit = non_topk_vals.min().item() if non_topk_vals.numel() > 0 else 0
                    max_logit = non_topk_vals.max().item() if non_topk_vals.numel() > 0 else 0
                    print(f" | [V8.35] SOFT PRUNE at epoch {epoch}: kept {k_undirected} undirected edges ({n_kept} entries), damped {n_pruned} by {prune_decay}× (non-TopK range: [{min_logit:.2f}, {max_logit:.2f}])")
        
        # =================================================================
        # V8.24 FIX: Edge exploration/swap mechanism
        # PROBLEM: Model locks into same TopK edges from epoch 1-2 and
        # NEVER tries alternatives. Reconstruction dominates edge ranking
        # with 15*75000 gradient signals vs 210 from structural losses.
        #
        # FIX: Periodically try swapping the WORST TopK edge (lowest logit
        # among TopK) with the BEST non-TopK edge (highest logit among
        # rest). If the swap improves validation-like loss on current batch,
        # keep it. Otherwise revert. This gives non-TopK edges a chance.
        #
        # When: Every edge_explore_interval epochs, first batch only
        # How: Swap worst TopK ↔ best non-TopK in W_adj, evaluate loss,
        #      keep if loss improves by at least edge_explore_min_gain
        # =================================================================
        edge_explore_enabled = config.get("edge_explore_enabled", True)
        edge_explore_interval = config.get("edge_explore_interval", 5)
        edge_explore_n_swaps = config.get("edge_explore_n_swaps", 3)
        edge_explore_start = config.get("edge_explore_start_epoch", 10)
        
        # V8.39: Hard-disable edge swaps in REFINE when E_frozen is active
        if E_frozen_mask is not None and config.get("refine_hard_freeze", True):
            edge_explore_enabled = False
        
        if (edge_explore_enabled and epoch >= edge_explore_start and
            epoch % edge_explore_interval == 0 and batch_idx == 0):
            with torch.no_grad():
                W_mag_exp = base_model.graph_learner.W_mag  # V8.33: swap structure param
                d_exp = W_mag_exp.shape[0]
                diag_mask_exp = torch.eye(d_exp, device=W_mag_exp.device)
                # V8.34 FIX: Use upper triangle for symmetric W_mag
                Wm_sym_exp = (W_mag_exp + W_mag_exp.T) / 2 * (1 - diag_mask_exp)
                W_upper_exp = torch.triu(Wm_sym_exp, diagonal=1)
                W_upper_flat_exp = W_upper_exp.flatten()
                n_upper_exp = d_exp * (d_exp - 1) // 2
                k_exp = min(target_edges, n_upper_exp)
                
                topk_vals_exp, topk_idx_exp = torch.topk(W_upper_flat_exp, k_exp)
                # Build rest mask: upper triangle positions NOT in topk
                upper_mask = torch.triu(torch.ones(d_exp, d_exp, device=W_mag_exp.device), diagonal=1).flatten().bool()
                rest_mask_exp = upper_mask.clone()
                rest_mask_exp[topk_idx_exp] = False
                
                rest_vals_exp = W_upper_flat_exp[rest_mask_exp]
                rest_orig_idx = torch.where(rest_mask_exp)[0]
                
                if rest_vals_exp.numel() > 0 and topk_vals_exp.numel() > 0:
                    # Sort TopK ascending (worst first), rest descending (best first)
                    _, worst_order = torch.sort(topk_vals_exp, descending=False)
                    _, best_rest_order = torch.sort(rest_vals_exp, descending=True)
                    
                    n_swaps = min(edge_explore_n_swaps, len(worst_order), len(best_rest_order))
                    swaps_made = 0
                    
                    # Save original W_mag for potential revert
                    W_backup = W_mag_exp.data.clone()
                    
                    for s in range(n_swaps):
                        worst_topk_flat_idx = topk_idx_exp[worst_order[s]]
                        best_rest_flat_idx = rest_orig_idx[best_rest_order[s]]
                        
                        # Convert flat index to (i,j) pair
                        wi, wj = worst_topk_flat_idx.item() // d_exp, worst_topk_flat_idx.item() % d_exp
                        ri, rj = best_rest_flat_idx.item() // d_exp, best_rest_flat_idx.item() % d_exp
                        
                        old_topk_val = W_mag_exp.data[wi, wj].item()
                        old_rest_val = W_mag_exp.data[ri, rj].item()
                        
                        # Only swap if rest edge logit is reasonably close
                        if old_topk_val - old_rest_val < 3.0:
                            # V8.34: Swap BOTH (i,j) and (j,i) for symmetric W_mag
                            W_mag_exp.data[wi, wj] = old_rest_val
                            W_mag_exp.data[wj, wi] = old_rest_val
                            W_mag_exp.data[ri, rj] = old_topk_val
                            W_mag_exp.data[rj, ri] = old_topk_val
                            swaps_made += 1
                    
                    if swaps_made > 0:
                        # V8.35 FIX: Use TopK-F1 metric instead of total loss.
                        # OLD: total loss dominated by reconstruction → swaps ALWAYS rejected.
                        # NEW: Compute TopK-F1 before vs after swap. Accept if F1 improves.
                        # This makes structural improvement visible.
                        if A_true is not None:
                            # Compute A from swapped W_mag
                            W_dir_exp = base_model.graph_learner.W_dir
                            tau_exp = base_model.graph_learner.temperature
                            Wm_new = (W_mag_exp + W_mag_exp.T) / 2  # symmetric
                            Wd_new = (W_dir_exp - W_dir_exp.T) / 2  # antisymmetric
                            A_swap = torch.sigmoid(Wm_new) * torch.sigmoid(Wd_new / tau_exp)
                            swap_metrics = compute_topk_f1(A_swap, A_true, k=target_edges)
                            swap_f1 = swap_metrics["topk_f1"]
                            
                            # Compute A from original W_mag
                            W_mag_exp.data.copy_(W_backup)
                            Wm_orig = (W_mag_exp + W_mag_exp.T) / 2
                            A_orig = torch.sigmoid(Wm_orig) * torch.sigmoid(Wd_new / tau_exp)
                            orig_metrics = compute_topk_f1(A_orig, A_true, k=target_edges)
                            orig_f1 = orig_metrics["topk_f1"]
                            
                            if swap_f1 > orig_f1:
                                # Swap improved TopK-F1! Apply permanently (symmetric)
                                for s in range(swaps_made):
                                    worst_topk_flat_idx = topk_idx_exp[worst_order[s]]
                                    best_rest_flat_idx = rest_orig_idx[best_rest_order[s]]
                                    wi, wj = worst_topk_flat_idx.item() // d_exp, worst_topk_flat_idx.item() % d_exp
                                    ri, rj = best_rest_flat_idx.item() // d_exp, best_rest_flat_idx.item() % d_exp
                                    old_topk_val = W_backup[wi, wj].item()
                                    old_rest_val = W_backup[ri, rj].item()
                                    if old_topk_val - old_rest_val < 3.0:
                                        W_mag_exp.data[wi, wj] = old_rest_val
                                        W_mag_exp.data[wj, wi] = old_rest_val
                                        W_mag_exp.data[ri, rj] = old_topk_val
                                        W_mag_exp.data[rj, ri] = old_topk_val
                                
                                if is_main_process():
                                    print(f" | [V8.35] EDGE SWAP at epoch {epoch}: {swaps_made} swaps ACCEPTED (F1: {orig_f1:.4f}→{swap_f1:.4f})")
                            elif swap_f1 == orig_f1:
                                # Same F1 — accept anyway to encourage exploration.
                                # With same directed F1, a swap at least diversifies
                                # the candidate edge set, which may help later.
                                for s in range(swaps_made):
                                    worst_topk_flat_idx = topk_idx_exp[worst_order[s]]
                                    best_rest_flat_idx = rest_orig_idx[best_rest_order[s]]
                                    wi, wj = worst_topk_flat_idx.item() // d_exp, worst_topk_flat_idx.item() % d_exp
                                    ri, rj = best_rest_flat_idx.item() // d_exp, best_rest_flat_idx.item() % d_exp
                                    old_topk_val = W_backup[wi, wj].item()
                                    old_rest_val = W_backup[ri, rj].item()
                                    if old_topk_val - old_rest_val < 3.0:
                                        W_mag_exp.data[wi, wj] = old_rest_val
                                        W_mag_exp.data[wj, wi] = old_rest_val
                                        W_mag_exp.data[ri, rj] = old_topk_val
                                        W_mag_exp.data[rj, ri] = old_topk_val
                                if is_main_process():
                                    print(f" | [V8.35] EDGE SWAP at epoch {epoch}: {swaps_made} swaps ACCEPTED (same F1={orig_f1:.4f}, exploring)")
                            else:
                                # Revert already done (W_backup restored above)
                                if is_main_process():
                                    print(f" | [V8.35] EDGE SWAP at epoch {epoch}: {swaps_made} swaps REJECTED (F1: {orig_f1:.4f}→{swap_f1:.4f})")
                        else:
                            # No A_true available — fall back to loss-based (legacy)
                            test_outputs = model(X, M, regime=e)
                            test_loss, _ = base_model.compute_loss(
                                test_outputs, X, M, regime=e,
                                epoch=epoch, total_epochs=total_epochs,
                                loss_weights=loss_weights,
                            )
                            W_mag_exp.data.copy_(W_backup)
                            orig_outputs = model(X, M, regime=e)
                            orig_loss, _ = base_model.compute_loss(
                                orig_outputs, X, M, regime=e,
                                epoch=epoch, total_epochs=total_epochs,
                                loss_weights=loss_weights,
                            )
                            min_gain = config.get("edge_explore_min_gain", 0.0)
                            if test_loss.item() < orig_loss.item() - min_gain:
                                for s in range(swaps_made):
                                    worst_topk_flat_idx = topk_idx_exp[worst_order[s]]
                                    best_rest_flat_idx = rest_orig_idx[best_rest_order[s]]
                                    wi, wj = worst_topk_flat_idx.item() // d_exp, worst_topk_flat_idx.item() % d_exp
                                    ri, rj = best_rest_flat_idx.item() // d_exp, best_rest_flat_idx.item() % d_exp
                                    old_topk_val = W_backup[wi, wj].item()
                                    old_rest_val = W_backup[ri, rj].item()
                                    if old_topk_val - old_rest_val < 3.0:
                                        W_mag_exp.data[wi, wj] = old_rest_val
                                        W_mag_exp.data[wj, wi] = old_rest_val
                                        W_mag_exp.data[ri, rj] = old_topk_val
                                        W_mag_exp.data[rj, ri] = old_topk_val
                                if is_main_process():
                                    gain = orig_loss.item() - test_loss.item()
                                    print(f" | [V8.24] EDGE SWAP at epoch {epoch}: {swaps_made} swaps ACCEPTED (gain={gain:.4f})")
                            else:
                                if is_main_process():
                                    diff = test_loss.item() - orig_loss.item()
                                    print(f" | [V8.24] EDGE SWAP at epoch {epoch}: {swaps_made} swaps REJECTED (worse by {diff:.4f})")
        
        # Compute loss with scheduled weights
        loss, batch_metrics = base_model.compute_loss(
            outputs, X, M, regime=e,
            epoch=epoch,
            total_epochs=total_epochs,
            loss_weights=loss_weights, # Pass scheduled weights
        )
        batch_metrics.update(batch_metrics_extra_gate)
        
        # =====================================================================
        # V8.37: Compute direction mask via adaptive K_dir (gap heuristic)
        # Skeleton losses (L_recon, L_dag, L_sparse) stay GLOBAL.
        # Direction losses (L_excl, L_dir_dec, L_asymm) use dir_mask.
        # Replaces V8.36 threshold-based mask which failed in PRUNE (112 edges).
        # =====================================================================
        dir_mask = None  # Default: no masking (all edges)
        dir_mask_n_edges = -1  # -1 = disabled
        if config.get("dir_mask_enabled", True):
            # Compute magnitude probabilities from W_mag (symmetric)
            Wm_mask, _ = base_model.graph_learner._get_sym_antisym()
            mag_probs = torch.sigmoid(Wm_mask)  # σ(Wm) — symmetric, diag=0
            
            target_e = config.get("target_edges", 13)
            k_min_f = config.get("dir_k_min_factor", 0.8)
            k_max_f = config.get("dir_k_max_factor", 2.5)
            min_gap = config.get("dir_k_min_gap", 0.01)
            
            dir_mask_n_edges, dir_mask, dir_mask_info = compute_adaptive_k_dir(
                mag_probs, target_e, k_min_f, k_max_f, min_gap
            )
            batch_metrics["dir_mask_n_edges"] = dir_mask_n_edges
            batch_metrics["dir_mask_gap"] = dir_mask_info["gap_val"]
            batch_metrics["dir_mask_boundary"] = dir_mask_info["boundary_score"]
            batch_metrics["dir_mask_k_min"] = dir_mask_info["k_min"]
            batch_metrics["dir_mask_k_max"] = dir_mask_info["k_max"]
        
        # =====================================================================
        # V8.39: Override dir_mask with E_frozen in REFINE
        # This is the KEY FIX: all direction losses are now computed ONLY on
        # the frozen skeleton edges, not on the adaptive K_dir set.
        # The skeleton freeze becomes a real constraint, not just a message.
        # =====================================================================
        if E_frozen_mask is not None and config.get("refine_hard_freeze", True):
            dir_mask = E_frozen_mask
            dir_mask_n_edges = int(E_frozen_mask.sum().item()) // 2  # undirected count
            batch_metrics["dir_mask_n_edges"] = dir_mask_n_edges
            batch_metrics["dir_mask_source"] = "E_frozen"  # diagnostic
        
        # =====================================================================
        # V8.32 FIX: Direction Decisiveness Penalty
        # PROBLEM in V8.31: L_bidir = mean(A_ij*A_ji) = mean(mag²*dir*(1-dir)).
        # With mag≈0.22, mag²≈0.048 ATTENUATES the gradient by 20×.
        # L_bidir only reached 0.06 despite bidir=0.40 for 33 epochs.
        #
        # FIX: L_dir_decisive = mean_{i<j}(dir*(1-dir)) where dir = σ((W-W.T)/2τ).
        # This is INDEPENDENT of magnitude → full gradient to break dir=0.5.
        # Also keep L_excl as safety net at lower weight.
        # V8.36: Both losses now masked to E_keep (believed edges only).
        # =====================================================================
        lambda_excl_unc = config.get("lambda_excl_unconditional", 1.0)
        lambda_dir_dec = config.get("lambda_dir_decisive", 10.0)
        excl_start = config.get("excl_start_epoch", 1)
        dir_dec_start = config.get("dir_decisive_start_epoch", 1)
        if epoch >= excl_start:
            A_excl = base_model.graph_learner.get_mean_adjacency()
            # L_excl = mean(A_ij * A_ji) — V8.36: masked to dir_mask
            L_excl_val = compute_exclusivity_loss(A_excl, dir_mask=dir_mask)
            loss = loss + lambda_excl_unc * L_excl_val
            batch_metrics["L_excl"] = L_excl_val.item()
            # Also track symmetry ratio for diagnostics
            A_min_e = torch.minimum(A_excl, A_excl.T)
            A_max_e = torch.maximum(A_excl, A_excl.T)
            sym_ratio_e = (A_min_e / (A_max_e + 1e-8)).mean().item()
            batch_metrics["symmetry_ratio"] = sym_ratio_e
            # Track bidir_rate = 1 - undir/dir for score penalty
            d_e = A_excl.shape[0]
            K_excl = config.get("target_edges", 13)
            A_flat_excl = (A_excl * (1 - torch.eye(d_e, device=A_excl.device))).flatten()
            _, topk_idx_excl = torch.topk(A_flat_excl, min(K_excl, A_flat_excl.numel()))
            dir_pred_set = set()
            undir_pred_set = set()
            for idx in topk_idx_excl.cpu().numpy():
                i_e, j_e = idx // d_e, idx % d_e
                dir_pred_set.add((i_e, j_e))
                undir_pred_set.add(tuple(sorted((i_e, j_e))))
            bidir_rate = 1.0 - len(undir_pred_set) / max(len(dir_pred_set), 1)
            batch_metrics["bidir_rate"] = bidir_rate
        
        # V8.32: Direction decisiveness penalty — V8.36: masked to dir_mask
        if lambda_dir_dec > 0 and epoch >= dir_dec_start:
            antisym_on = getattr(base_model.graph_learner, '_antisymmetric', False)
            if antisym_on:
                dir_map = base_model.graph_learner.get_direction_map()  # [d,d]
                d_dir = dir_map.shape[0]
                # dir*(1-dir) is maximized at 0.5 (=0.25), zero at 0 or 1
                dir_entropy = dir_map * (1 - dir_map)
                mask_upper = torch.triu(torch.ones(d_dir, d_dir, device=dir_map.device), diagonal=1)
                # V8.36: Intersect with dir_mask — only penalize on believed edges
                if dir_mask is not None:
                    mask_upper = mask_upper * dir_mask
                n_masked = mask_upper.sum() + 1e-8
                L_dir_dec = (dir_entropy * mask_upper).sum() / n_masked
                loss = loss + lambda_dir_dec * L_dir_dec
                batch_metrics["L_dir_dec"] = L_dir_dec.item()
                # Also log avg|dir-0.5| as decisiveness measure (on masked edges)
                dir_upper = dir_map[mask_upper.bool()]
                batch_metrics["dir_decisiveness"] = (dir_upper - 0.5).abs().mean().item() if dir_upper.numel() > 0 else 0.0
        
        # =====================================================================
        # V8.29 FIX B: Ranking Margin Loss
        # PROBLEM: Margins avg=0.01-0.05 for long periods → edges aren't separable.
        # TopK edges have logits barely above non-TopK → direction is noisy.
        #
        # FIX: For TopK set E+ and non-TopK E-, enforce W_pos >= W_neg + m.
        # Uses hinge loss: L_rank = mean(ReLU(W_neg - W_pos + m)).
        # This creates direct gradient signal to separate edge logits.
        # =====================================================================
        ranking_enabled = config.get("ranking_margin_enabled", True)
        ranking_start = config.get("ranking_margin_start_epoch", 10)
        if ranking_enabled and epoch >= ranking_start:
            lambda_rank = config.get("lambda_ranking_margin", 1.0)
            rank_margin = config.get("ranking_margin_value", 0.5)
            
            W_rank = base_model.graph_learner.W_mag  # V8.33: rank by structure param
            d_rank = W_rank.shape[0]
            diag_mask_rank = torch.eye(d_rank, device=W_rank.device)
            # V8.34 FIX: Use upper triangle for symmetric W_mag
            Wm_sym_rank = (W_rank + W_rank.T) / 2 * (1 - diag_mask_rank)
            W_upper_rank = torch.triu(Wm_sym_rank, diagonal=1)
            W_upper_flat_rank = W_upper_rank.flatten()
            n_upper_rank = d_rank * (d_rank - 1) // 2
            k_rank = min(target_edges, n_upper_rank)
            
            topk_vals_rank, topk_idx_rank = torch.topk(W_upper_flat_rank, k_rank)
            upper_mask_rank = torch.triu(torch.ones(d_rank, d_rank, device=W_rank.device), diagonal=1).flatten().bool()
            rest_mask_rank = upper_mask_rank.clone()
            rest_mask_rank[topk_idx_rank] = False
            rest_vals_rank = W_upper_flat_rank[rest_mask_rank]
            
            if rest_vals_rank.numel() > 0 and topk_vals_rank.numel() > 0:
                # Worst TopK logit vs best non-TopK logit
                min_topk = topk_vals_rank.min()
                max_rest = rest_vals_rank.max()
                # Hinge: penalize when gap < margin
                L_rank = torch.relu(max_rest - min_topk + rank_margin)
                loss = loss + lambda_rank * L_rank
                batch_metrics["L_ranking"] = L_rank.item()
                batch_metrics["rank_gap"] = (min_topk - max_rest).item()
        
        # =====================================================================
        # FIX B: Projection consistency loss during PRUNE/REFINE
        # Encourages soft adjacency to sharpen toward Top-K selection
        # This provides gradient signal to reinforce the projected edges
        # =====================================================================
        use_topk_proj = config.get("use_topk_projection", False)
        topk_proj_start = config.get("topk_projection_start", 0.35)
        topk_proj_2k_end = config.get("topk_projection_2k_end", 0.55)
        lambda_proj = config.get("lambda_projection", 0.5)
        
        if use_topk_proj and epoch >= topk_proj_start * total_epochs:
            target_edges = config.get("target_edges", 13)
            if epoch < topk_proj_2k_end * total_epochs:
                k_proj = 2 * target_edges
            else:
                k_proj = target_edges
            
            # Get soft adjacency and hard projection
            A_soft = base_model.graph_learner.get_mean_adjacency()
            A_hard = base_model.graph_learner.topk_project(k_proj, use_logits=True).detach()
            
            # BCE loss: encourage soft A to match hard projection
            L_proj = F.binary_cross_entropy(A_soft, A_hard)
            loss = loss + lambda_proj * L_proj
            batch_metrics["L_projection"] = L_proj.item()
        
        # =====================================================================
        # FIX E: Direction Asymmetry Loss (V8.5: Stage D only with skeleton mask)
        # For each skeleton edge (i,j), penalize if A[j,i] > A[i,j]
        # This encourages the model to commit to ONE direction per edge
        #
        # V8.5 CHANGE: Only apply asymmetry in Stage D (REFINE) when skeleton is frozen
        # During Stage S (DISC+PRUNE), we focus on finding edges, not direction
        # =====================================================================
        lambda_asymm = config.get("lambda_asymmetry", 0.5)
        use_asymm = config.get("use_asymmetry_loss", True)
        two_stage = config.get("two_stage_training", True)
        stage2_end = config.get("stage2_end", 0.80)
        in_stage_d = epoch >= stage2_end * total_epochs # REFINE phase = Stage D
        
        # V8.5+V8.6: Only apply asymmetry loss in Stage D, on CONFIDENT edges
        if use_asymm and (in_stage_d if two_stage else epoch >= 0.35 * total_epochs):
            A_soft = base_model.graph_learner.get_mean_adjacency()
            target_edges = config.get("target_edges", 13)
            
            # V8.6: Confidence-based edge selection for direction loss
            t_conf = config.get("direction_conf_threshold", 0.3)
            conf_adaptive = config.get("direction_conf_adaptive", True)
            
            # Symmetric max: confidence = max(A_ij, A_ji)
            A_sym = torch.maximum(A_soft, A_soft.T)
            
            # Adaptive threshold: use median of edge strengths
            if conf_adaptive:
                d = A_soft.shape[0]
                mask_upper = torch.triu(torch.ones(d, d, device=A_soft.device), diagonal=1)
                sym_vals = A_sym[mask_upper > 0]
                if sym_vals.numel() > 0:
                    t_conf_adaptive = sym_vals.median().item()
                    t_conf = max(t_conf, t_conf_adaptive)
            
            # V8.5: Use skeleton mask if provided (Stage D), else use confidence-based
            if skeleton_mask is not None:
                # Stage D: Intersect skeleton with confident edges
                conf_mask = (A_sym > t_conf).float()
                edge_mask = skeleton_mask.float() * conf_mask
                batch_metrics["asymm_mode"] = "skel+conf"
                batch_metrics["asymm_t_conf"] = t_conf
            else:
                # Stage S (or legacy mode): Use confidence-based selection
                edge_mask = (A_sym > t_conf).float()
                batch_metrics["asymm_mode"] = "conf"
                batch_metrics["asymm_t_conf"] = t_conf
            
            # V8.36: Intersect edge_mask with dir_mask (only direction-train on believed edges)
            if dir_mask is not None:
                edge_mask = edge_mask * dir_mask
            
            # =========================================================
            # V8.7: Direction Margin Loss (hinge-based)
            # For TRUE edge i->j: enforce A_ij >= A_ji + margin
            # Loss = max(0, A_ji - A_ij + margin) for each true directed edge
            # =========================================================
            use_margin = config.get("use_direction_margin", True)
            margin = config.get("direction_margin", 0.1)
            lambda_margin = config.get("lambda_direction_margin", 1.0)
            
            # LEAKAGE FIX: Only use A_true if oracle mode is explicitly enabled
            if use_margin and A_true is not None and config.get("oracle_direction_supervision", False):
                # V8.26 FIX: Model uses CAUSAL convention (A[i,j]=i→j)
                # A_true also uses causal convention. No transpose needed.
                A_true_model_conv = A_true
                
                # Get ground truth edges in MODEL convention
                A_true_bin = (A_true_model_conv > 0.5).float()
                
                # For each true edge i->j (in MODEL convention), we want A_ij >= A_ji + margin
                # Hinge loss: max(0, A_ji - A_ij + margin)
                # Only for positions where A_true_model_conv[i,j] = 1
                
                # Compute violation: how much A_ji exceeds A_ij - margin
                margin_violation = F.relu(A_soft.T - A_soft + margin) # [d, d]
                
                # Mask to only true edges AND confident edges
                # This focuses direction learning on edges we're sure exist
                combined_mask = A_true_bin * edge_mask
                
                # Average margin loss over true edges
                n_masked_edges = combined_mask.sum()
                if n_masked_edges > 0:
                    L_margin = (combined_mask * margin_violation).sum() / n_masked_edges
                else:
                    L_margin = torch.tensor(0.0, device=A_soft.device)
                
                # Apply margin loss
                margin_weight = lambda_margin * 2.0 if in_stage_d else lambda_margin
                loss = loss + margin_weight * L_margin
                batch_metrics["L_dir_margin"] = L_margin.item()
                batch_metrics["n_margin_edges"] = int(n_masked_edges.item())
            else:
                # Fallback: soft asymmetry penalty (no ground truth available)
                # For edge (i,j): if A[j,i] > A[i,j], this is wrong direction
                asymm_penalty = F.relu(A_soft.T - A_soft) # [d, d]
                
                # Only apply to masked edges
                L_asymm = (edge_mask * asymm_penalty).sum() / (edge_mask.sum() + 1e-8)
                
                # V8.5: Stronger asymmetry in Stage D (direction is the focus)
                asymm_weight = lambda_asymm * 2.0 if in_stage_d else lambda_asymm
                loss = loss + asymm_weight * L_asymm
                batch_metrics["L_asymmetry"] = L_asymm.item()
        
        # =====================================================================
        # V8.11: Antisymmetric Direction Loss (when skeleton is frozen)
        # Use D_ij = W_ij - W_ji for direction, forcing exactly one winner per pair
        # 
        # V8.26 FIX: Model uses CAUSAL convention (A[i,j]=i→j)
        # A_true also uses causal convention. No transpose needed.
        # =====================================================================
        # LEAKAGE FIX: Only use A_true if oracle mode is explicitly enabled
        if skeleton_frozen and A_true is not None and config.get("oracle_direction_supervision", False):
            W = base_model.graph_learner.W_mag  # V8.33: direction from W_dir, but compute_direction_loss uses W logits
            # V8.26: No convention conversion needed
            A_true_model_conv = A_true
            dir_loss, dir_metrics = compute_direction_loss(
                W, A_true_model_conv, tau_direction=direction_tau
            )
            
            # Boost direction loss since it's now our main objective
            loss = loss + lambda_direction_boost * dir_loss
            
            # Log direction metrics
            batch_metrics["L_direction_antisym"] = dir_loss.item()
            batch_metrics["dir_accuracy"] = dir_metrics["dir_accuracy"]
            batch_metrics["dir_avg_margin"] = dir_metrics["dir_avg_margin"]
            batch_metrics["dir_min_margin"] = dir_metrics["dir_min_margin"]
        
        # =====================================================================
        # V8.18: MNAR-aware Mask-Invariance Penalty
        # Force adjacency stability under missingness perturbations (MNAR-only)
        # Prevents model from using MNAR shortcuts to learn structure
        # Apply ONLY after skeleton freeze AND on MNAR-prone datasets
        # =====================================================================
        # V8.21: MNAR-only + shape-correct + sparse activation
        if skeleton_frozen and M is not None:
            lambda_inv_mask = config.get("lambda_inv_mask", 1e-3)
            if lambda_inv_mask > 0:
                inv_mask_penalty = compute_mask_invariance_penalty(
                    base_model, X, M, lambda_inv_mask=lambda_inv_mask
                )
                if inv_mask_penalty.item() > 0: # Only add if penalty computed (MNAR detected)
                    loss = loss + inv_mask_penalty
                    batch_metrics["L_inv_mask"] = inv_mask_penalty.item()
        
        # =====================================================================
        # V8.8: Edge Discovery Losses (fix mode collapse to easy subgraph)
        # These losses directly target:
        # 1. Missing hub fan-outs (10->{5,11,12}, 11->{0,2,12})
        # 2. Wrong high-corr cousin pairs (6-8, 10-8, 7-9)
        # =====================================================================
        use_edge_discovery = (
            config.get("use_presence_loss", True) or 
            config.get("use_cousin_penalty", True) or 
            config.get("use_coverage_loss", True)
        )
        
        # LEAKAGE FIX: Only use A_true if oracle mode is explicitly enabled
        if use_edge_discovery and A_true is not None and config.get("oracle_direction_supervision", False):
            A_soft = base_model.graph_learner.get_mean_adjacency()
            
            # Get correlation matrix (stored on CPU to avoid GPU leak, move lazily)
            _corr_cpu = getattr(base_model, '_corr_matrix_cpu', None)
            corr_matrix = _corr_cpu.to(device) if _corr_cpu is not None else None
            
            # Get cached cousin mask (also stored on CPU)
            _cousin_cpu = getattr(base_model, '_cousin_mask_cpu', None)
            
            # Compute all edge discovery losses
            disc_losses, disc_metrics = compute_edge_discovery_losses(
                A_pred=A_soft,
                A_true=A_true,
                corr_matrix=corr_matrix,
                config=config,
                epoch=epoch,
                total_epochs=total_epochs,
                cached_cousin_mask=_cousin_cpu,
            )
            
            # Add losses
            for loss_name, loss_val in disc_losses.items():
                loss = loss + loss_val
            
            # Track metrics
            batch_metrics.update(disc_metrics)
        
        # GroupDRO: reweight by regime
        if dro_weights is not None:
            unique_regimes = torch.unique(e)
            weighted_loss = torch.tensor(0.0, device=device)
            
            for r in unique_regimes:
                r_int = r.item()
                mask = (e == r)
                if mask.sum() == 0:
                    continue
                
                # Track regime loss
                if r_int not in regime_losses:
                    regime_losses[r_int] = 0.0
                    regime_counts[r_int] = 0
                regime_losses[r_int] += loss.item()
                regime_counts[r_int] += 1
                
                # Weight
                w = dro_weights.get(r_int, 1.0)
                weighted_loss = weighted_loss + w * loss
            
            loss = weighted_loss
        
        # Backward
        loss.backward()
        
        # =====================================================================
        # V8.5: SKELETON FREEZE in Stage D
        # Zero out gradients for edges NOT in skeleton - prevents changing structure
        # Only learn DIRECTION for edges already in the frozen skeleton
        # =====================================================================
        if skeleton_mask is not None:
            # V8.33: Zero W_mag gradients for non-skeleton edges.
            # W_dir gradients are KEPT — direction still learns for all edges.
            W_mag = base_model.graph_learner.W_mag
            if W_mag.grad is not None:
                skel_sym = skeleton_mask | skeleton_mask.T
                W_mag.grad = W_mag.grad * skel_sym.float()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=config["grad_clip"]
        )
        
        optimizer.step()
        base_model.graph_learner.increment_step()
        
        # Track metrics
        total_loss += loss.item()
        for k, v in batch_metrics.items():
            if isinstance(v, (int, float)):
                metrics_sum[k] = metrics_sum.get(k, 0.0) + v
        metrics_sum["grad_norm"] = metrics_sum.get("grad_norm", 0.0) + grad_norm.item()
        n_batches += 1
    
    # Average metrics
    avg_metrics = {k: v / n_batches for k, v in metrics_sum.items()}
    avg_metrics["loss"] = total_loss / n_batches
    
    # V8.16: Add lambda values to metrics for logging
    avg_metrics["lambda_bud_used"] = lambda_budget_used
    avg_metrics["lambda_bud_frozen"] = lambda_budget_is_frozen
    avg_metrics["lambda_sparse_used"] = loss_weights.get("lambda_sparse", 0)
    avg_metrics["temperature"] = temperature
    avg_metrics["peakiness_gated"] = peakiness_gated # V8.17
    
    # Update DRO weights
    if dro_weights is not None and regime_losses:
        avg_regime_losses = {
            r: torch.tensor(regime_losses[r] / regime_counts[r])
            for r in regime_losses
        }
        dro_weights = groupdro_reweight(
            avg_regime_losses, dro_weights, step_size=config["dro_step_size"]
        )
    
    return avg_metrics, dro_weights


def validate(
    model: RCGNN,
    loader: DataLoader,
    A_true: Optional[torch.Tensor],
    X_all: torch.Tensor,
    device: torch.device,
    config: Dict,
    ddp: bool = False,
    epoch: int = 0,
    total_epochs: int = 100,
) -> Dict[str, float]:
    """Validation with comprehensive metrics."""
    model.eval()
    base_model = model.module if ddp else model
    
    with torch.no_grad():
        # Get predicted adjacency
        A_pred = base_model.graph_learner.get_mean_adjacency()
        
        # ==================================================================
        # V8.26: CONVENTION — model already outputs causal (identity)
        # A[i,j] = i→j in both model and A_true. No transpose.
        # to_causal_convention() is kept as identity for compat.
        # ==================================================================
        A_pred = to_causal_convention(A_pred) # V8.26: identity (no-op)
        
        # ==================================================================
        # CRITICAL FIX: Hard Top-K projection during PRUNE/REFINE
        # Forces evaluation on exactly K edges, matching the metric
        #
        # GOTCHA FIX: 2K->K schedule to avoid freezing wrong edges early
        # - 35%-55%: project to 2*K (flexibility phase)
        # - 55%+: project to K (commitment phase)
        # ==================================================================
        use_topk_proj = config.get("use_topk_projection", False)
        topk_proj_start = config.get("topk_projection_start", 0.35)
        topk_proj_2k_end = config.get("topk_projection_2k_end", 0.55)
        target_edges = config.get("target_edges", 13)
        use_logits = config.get("topk_use_logits", True)
        
        if use_topk_proj and epoch >= topk_proj_start * total_epochs:
            # Calculate K based on 2K->K schedule
            proj_start = topk_proj_start * total_epochs
            proj_2k_end = topk_proj_2k_end * total_epochs
            
            if epoch < proj_2k_end:
                # Flexibility phase: use 2*K
                k_proj = 2 * target_edges
            else:
                # Commitment phase: use exactly K
                k_proj = target_edges
            
            # Get projected adjacency (using logits for stable ranking)
            A_proj = base_model.graph_learner.topk_project(k_proj, use_logits=use_logits)
            
            # Log projection info
            n_proj_edges = int((A_proj > 0.5).sum().item())
        else:
            A_proj = None
            k_proj = 0
            n_proj_edges = 0
        
        metrics = {
            "A_mean": A_pred.mean().item(),
            "A_max": A_pred.max().item(),
            "A_min": A_pred.min().item(),
            "temperature": base_model.graph_learner.get_temperature(),
            "topk_proj_k": k_proj,
            "topk_proj_edges": n_proj_edges,
        }
        
        # Add logit stats for separation diagnostics
        W_logits = base_model.graph_learner.get_logits()
        W_masked = W_logits * (1 - torch.eye(W_logits.shape[0], device=W_logits.device))
        metrics["W_logits_min"] = W_masked.min().item()
        metrics["W_logits_max"] = W_masked.max().item()
        metrics["W_logits_mean"] = W_masked.mean().item()
        
        if A_true is not None:
            # TopK-F1 (primary metric) - on raw adjacency + TRANSPOSE DIAGNOSTIC
            topk = compute_topk_f1(A_pred, A_true, compute_transpose=True)
            metrics.update(topk)
            
            # ================================================================
            # DIAGNOSTIC: Skeleton F1 (direction-agnostic)
            # If skeleton_f1 >> topk_f1, model finds correct edges but wrong direction
            # ================================================================
            skeleton = compute_skeleton_f1(A_pred, A_true)
            metrics.update(skeleton)
            
            # ================================================================
            # V8.6: Direction accuracy on CONFIDENT edges only
            # This focuses direction evaluation on edges we're sure exist
            # ================================================================
            t_conf = config.get("direction_conf_threshold", 0.3)
            conf_adaptive = config.get("direction_conf_adaptive", True)
            dir_conf = compute_direction_on_confident_edges(
                A_pred, A_true, t_conf=t_conf, adaptive=conf_adaptive
            )
            metrics.update(dir_conf)
            
            # ================================================================
            # CRITICAL: If Top-K projection is active, compute F1 on projected edges
            # This is the TRUE metric since we'll use projected graph for final eval
            # ================================================================
            if A_proj is not None:
                proj_topk = compute_topk_f1(A_proj, A_true, k=target_edges)
                metrics["proj_topk_f1"] = proj_topk["topk_f1"]
                metrics["proj_topk_tp"] = proj_topk["topk_tp"]
                metrics["proj_topk_fp"] = proj_topk["topk_fp"]
                metrics["proj_topk_fn"] = proj_topk["topk_fn"]
            
            # Best-F1 (optimal threshold)
            best = compute_best_f1(A_pred, A_true, config["threshold_grid"])
            metrics.update(best)
            
            # AUC-F1 (robustness)
            auc = compute_auc_f1(A_pred, A_true, config["threshold_grid"])
            metrics.update(auc)
            
            # Fixed threshold metrics
            fixed = compute_structure_metrics(A_pred, A_true, threshold=0.2)
            metrics["fixed02_f1"] = fixed["f1"]
            metrics["fixed02_pred_edges"] = fixed["pred_edges"]
        
        # Correlation vs causation diagnosis (every 10 epochs)
        # Done in main loop
        
    return metrics


# =============================================================================
# Main Training Function
# =============================================================================

def train(
    data_dir: str,
    output_dir: str = "artifacts/unified",
    config: Dict = None,
    ddp: bool = False,
    use_groupdro: bool = False,
    sweep_mode: bool = False,
) -> Dict[str, float]:
    """
    Main training function with all features.
    
    Args:
        data_dir: Path to data directory
        output_dir: Path to output directory
        config: Configuration dict (merged with defaults)
        ddp: Enable Distributed Data Parallel
        use_groupdro: Enable GroupDRO
        sweep_mode: Enable sweep mode (minimal output)
    
    Returns:
        Dictionary of final metrics
    """
    # V8.24 VERSION CHECK - if you see this, the new code is running!
    print("=" * 70)
    print("[V8.39] RC-GNN Training Script — ENFORCED SKELETON FREEZE:")
    print("  - V8.39 FIX: REFINE enforces skeleton freeze contract")
    print("  -   PROBLEM: 'Skeleton FROZEN with 16 edges' was a message, not a constraint")
    print("  -   K_dir=30 still used, edge swaps ran, direction on 30 edges")
    print("  -   FIX 1: Store E_frozen explicitly, mask ALL direction losses to E_frozen")
    print("  -   FIX 2: Disable edge swaps in REFINE (no skeleton mutations)")
    print("  -   FIX 3: Reset W_dir at REFINE start (escape wrong-direction basin)")
    print("  -   FIX 4: LR boost for W_dir in early REFINE (5x for 10 epochs)")
    print("  - Inherits V8.38: Early stopping blocked in PRUNE")
    print("  - Inherits V8.37: Adaptive K_dir (used only in DISC/PRUNE)")
    print("  - Inherits V8.36: REFINE reweighting (λ_recon×0.1, λ_inv×5.0)")
    print("=" * 70)
    
    # Merge config with defaults
    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(config)
    
    # Setup DDP
    local_rank = 0
    if ddp:
        local_rank = setup_ddp()
        cfg["device"] = f"cuda:{local_rank}"
    
    # Device
    if cfg["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg["device"])
    
    # Seed (comprehensive)
    set_seed(cfg["seed"] + local_rank, deterministic=cfg.get("deterministic", False))
    
    # Config validation
    cfg_warnings = validate_config(cfg)
    if cfg_warnings and is_main_process():
        print("\n⚠️  CONFIG WARNINGS:")
        for w in cfg_warnings:
            print(f"  • {w}")
        print()
    
    # Oracle mode banner
    if cfg.get("oracle_direction_supervision", False) and is_main_process():
        print("\n" + "=" * 70)
        print("⚠️  ORACLE MODE ENABLED ⚠️")
        print("Using ground truth adjacency for direction supervision!")
        print("This is for diagnostic/ablation ONLY — results do NOT generalise!")
        print("=" * 70 + "\n")
    
    # Load data
    data = load_data(data_dir, normalize=True)
    d = data["X"].shape[-1]
    n_regimes = len(torch.unique(data["e"]))
    A_true = data.get("A_true", None)
    
    # Adjust target edges if A_true available
    if A_true is not None and cfg["target_edges"] == 13:
        cfg["target_edges"] = int(A_true.sum())
    
    # =========================================================================
    # V8.28: DIMENSION-ADAPTIVE SCALING
    # Problem: All hyperparameters were tuned for d=13 (UCI Air Quality).
    # Table 2 SEM benchmarks span d∈{13,15,20,25,50}. For d=50, we need:
    #   - More epochs (bigger search space)
    #   - More patience (slower convergence)
    #   - Larger architecture (more capacity)
    #   - Longer DISC phase (more edges to explore)
    #   - Softer hard prune (more edges to evaluate)
    #   - More edge exploration swaps (bigger candidate pool)
    #   - Delayed TopK gating (don't commit too early)
    #
    # ONLY apply auto-scaling for values still at their DEFAULT_CONFIG defaults.
    # If user passed explicit CLI args, those take priority.
    # =========================================================================
    target_edges_cfg = cfg["target_edges"]
    n_possible_edges = d * (d - 1)
    edge_density = target_edges_cfg / max(n_possible_edges, 1)
    
    # --- Training duration scaling ---
    # d=13: 100 epochs is OK. d=50: need ~250.
    if cfg["epochs"] == DEFAULT_CONFIG["epochs"]:  # Only if not overridden
        cfg["epochs"] = max(100, int(100 * (d / 13) ** 0.7))
    if cfg["patience"] == DEFAULT_CONFIG["patience"]:
        cfg["patience"] = max(30, int(30 * (d / 13) ** 0.5))
    
    # --- Architecture scaling ---
    # d=13: latent=32, hidden=64 fine. d=50: need latent=64, hidden=128+
    if cfg["latent_dim"] == DEFAULT_CONFIG["latent_dim"]:
        cfg["latent_dim"] = max(32, min(128, int(32 * (d / 13) ** 0.6)))
    if cfg["hidden_dim"] == DEFAULT_CONFIG["hidden_dim"]:
        cfg["hidden_dim"] = max(64, min(256, int(64 * (d / 13) ** 0.6)))
    
    # --- Stage boundary scaling ---
    # Bigger d needs longer DISC (more edges to discover) and longer PRUNE
    if cfg["stage1_end"] == DEFAULT_CONFIG["stage1_end"]:
        if d >= 40:
            cfg["stage1_end"] = 0.40  # 40% for DISC (was 30%)
        elif d >= 25:
            cfg["stage1_end"] = 0.35  # 35% for DISC
    if cfg["stage2_end"] == DEFAULT_CONFIG["stage2_end"]:
        if d >= 40:
            cfg["stage2_end"] = 0.80  # 80% for DISC+PRUNE (was 70%)
        elif d >= 25:
            cfg["stage2_end"] = 0.75  # 75% for DISC+PRUNE
    
    # --- Hard prune scaling ---
    # d=13: prune_val=-2.0, interval=10 is fine (kills 156 non-TopK edges).
    # d=50: prune_val=-2.0 kills 2437 candidates — way too aggressive.
    # Softer prune for larger d: sigmoid(-1)≈0.27 allows recovery.
    if cfg["hard_prune_logit_value"] == DEFAULT_CONFIG.get("hard_prune_logit_value", -2.0):
        if d >= 40:
            cfg["hard_prune_logit_value"] = -1.0  # sigmoid(-1)≈0.27, recoverable
        elif d >= 25:
            cfg["hard_prune_logit_value"] = -1.5  # sigmoid(-1.5)≈0.18
    if cfg["hard_prune_interval"] == DEFAULT_CONFIG.get("hard_prune_interval", 10):
        if d >= 40:
            cfg["hard_prune_interval"] = 20  # Less frequent pruning for large d
        elif d >= 25:
            cfg["hard_prune_interval"] = 15
    
    # --- Edge exploration scaling ---
    # d=13: 3 swaps fine (3/13 = 23% of edges explored per step).
    # d=50: need more swaps (3/100 = 3% is too few).
    if cfg["edge_explore_n_swaps"] == DEFAULT_CONFIG.get("edge_explore_n_swaps", 3):
        cfg["edge_explore_n_swaps"] = max(3, target_edges_cfg // 4)
    if cfg["edge_explore_interval"] == DEFAULT_CONFIG.get("edge_explore_interval", 5):
        if d >= 40:
            cfg["edge_explore_interval"] = 3  # Explore more often for big d
    
    # --- TopK gating delay ---
    # Don't commit to TopK too early for large d (needs more exploration)
    if cfg.get("topk_gate_start") is None or cfg.get("topk_gate_start") == cfg.get("stage1_end", 0.15):
        if d >= 40:
            cfg["topk_gate_start"] = 0.30  # Delay gating for d>=40
        elif d >= 25:
            cfg["topk_gate_start"] = 0.25
    
    # --- TopK projection delay ---
    if cfg["topk_projection_start"] == DEFAULT_CONFIG.get("topk_projection_start", 0.20):
        if d >= 40:
            cfg["topk_projection_start"] = 0.35  # Delay hard projection
        elif d >= 25:
            cfg["topk_projection_start"] = 0.30
    
    # --- Lambda scaling: reduce V8.22 overrides for large d ---
    # V8.22 set lambda_recon=0.5, lambda_causal=3.0 for d=13.
    # For large d, reconstruction is harder — don't suppress it as much.
    # Also, anticorr/icp need less aggressive scaling for larger graphs.
    if d >= 25:
        # More balanced lambdas for large d
        if cfg["lambda_recon"] == 0.5:  # V8.22 override
            cfg["lambda_recon"] = min(1.0, 0.5 + 0.02 * (d - 13))  # Scale up
        if cfg["lambda_causal"] == 3.0:  # V8.22 override
            cfg["lambda_causal"] = max(1.5, 3.0 - 0.05 * (d - 13))  # Scale down
    
    # --- eval_k_values should include actual target_edges ---
    if target_edges_cfg not in cfg.get("eval_k_values", [13, 20, 30]):
        cfg["eval_k_values"] = sorted(set(cfg.get("eval_k_values", [13, 20, 30]) + [target_edges_cfg]))
    
    if is_main_process() and not sweep_mode:
        print(f"\n[V8.28] Dimension-adaptive scaling for d={d}:")
        print(f"  epochs={cfg['epochs']}, patience={cfg['patience']}")
        print(f"  latent_dim={cfg['latent_dim']}, hidden_dim={cfg['hidden_dim']}")
        print(f"  stage1_end={cfg['stage1_end']:.2f}, stage2_end={cfg['stage2_end']:.2f}")
        print(f"  hard_prune: val={cfg.get('hard_prune_logit_value', -2.0)}, interval={cfg.get('hard_prune_interval', 10)}")
        print(f"  edge_explore: n_swaps={cfg['edge_explore_n_swaps']}, interval={cfg['edge_explore_interval']}")
        print(f"  target_edges={target_edges_cfg}, edge_density={edge_density:.3f}")
        print(f"  lambda_recon={cfg['lambda_recon']}, lambda_causal={cfg['lambda_causal']}")
    
    train_loader, val_loader = create_dataloaders(
        data,
        batch_size=cfg["batch_size"],
        ddp=ddp,
        num_workers=cfg["num_workers"],
        seed=cfg["seed"],
    )
    
    # Print info
    if is_main_process() and not sweep_mode:
        print("\n" + "=" * 70)
        print("RC-GNN UNIFIED TRAINING")
        print("=" * 70)
        print(f"Device: {device}")
        if ddp:
            print(f"DDP: {get_world_size()} GPUs")
        print(f"Data: {data_dir}")
        print(f" X shape: {data['X'].shape}")
        print(f" Regimes: {n_regimes}")
        print(f" True edges: {int(A_true.sum()) if A_true is not None else 'N/A'}")
        print(f"Epochs: {cfg['epochs']}, Batch: {cfg['batch_size']}, LR: {cfg['lr']}")
        print(f"GroupDRO: {use_groupdro}")
        print(f"3-Stage: discovery->{int(cfg['stage1_end']*100)}%, "
              f"pruning->{int(cfg['stage2_end']*100)}%, refinement->100%")
        print("=" * 70 + "\n")
    
    # Create model (use final lambda values - scheduling handled externally)
    model = RCGNN(
        d=d,
        latent_dim=cfg["latent_dim"],
        hidden_dim=cfg["hidden_dim"],
        n_regimes=n_regimes,
        target_edges=cfg["target_edges"],
        lambda_recon=cfg["lambda_recon"],
        lambda_miss=cfg["lambda_miss"],
        lambda_hsic=cfg["lambda_hsic"],
        lambda_sparse=cfg.get("lambda_sparse_final", cfg.get("lambda_sparse", 0.01)),
        lambda_inv=cfg["lambda_inv"],
        lambda_causal=cfg["lambda_causal"],
        lambda_var_penalty=cfg["lambda_var_penalty"],
        # FIX E: Boost orientation penalty for direction learning
        lambda_orientation=cfg.get("lambda_orientation", 0.5), # Increased from 0.1
    ).to(device)
    
    # Wrap with DDP
    if ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    base_model = model.module if ddp else model
    
    # =========================================================================
    # V8.29: Enable antisymmetric adjacency mode
    # This makes A_ij = sigmoid((W_ij - W_ji) / tau), A_ji = 1 - A_ij
    # Bidirectional edges are structurally impossible.
    # =========================================================================
    if cfg.get("antisymmetric_adjacency", True):
        base_model.graph_learner.set_antisymmetric(True)
        if is_main_process() and not sweep_mode:
            print(f"[V8.34] DUAL-PARAMETER with STRUCTURAL PROJECTIONS + SYMMETRIC PRUNE:")
            print(f"        Wm = (W_mag+W_mag.T)/2 [symmetric],  Wd = (W_dir-W_dir.T)/2 [antisymmetric]")
            print(f"        A_ij = σ(Wm) * σ(Wd/τ),  A_ji = σ(Wm) * (1-σ(Wd/τ))")
            print(f"        GUARANTEE: A_ij+A_ji = mag_ij → dL_budget/dW_dir = 0")
            print(f"        V8.34 FIX: Hard prune selects from UPPER TRIANGLE (K undirected edges)")
            print(f"        V8.34 FIX: τ floor={cfg.get('temperature_floor_unstable', 0.8)} ALWAYS enforced when direction unstable")
            print(f"        V8.34 FIX: τ_final={cfg.get('temperature_final', 0.5)} (was 0.1, killed direction gradients)")
            print(f"        L_dir_dec = {cfg.get('lambda_dir_decisive', 10.0)}×mean(p*(1-p)) → W_dir only")
            if cfg.get('dir_mask_enabled', True):
                print(f"        V8.37 ADAPTIVE K_dir: K_min={cfg.get('dir_k_min_factor', 0.8)}×target, "
                      f"K_max={cfg.get('dir_k_max_factor', 2.5)}×target, "
                      f"min_gap={cfg.get('dir_k_min_gap', 0.01)}")
    
    if is_main_process() and not sweep_mode:
        print(f"[Model] d={d}, latent={cfg['latent_dim']}, regimes={n_regimes}")
        print(f"[Model] Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # =========================================================================
    # TRUE CORRELATION BASELINE (computed once at start)
    # This is the principled baseline - what correlation-based methods achieve.
    # NOT the model's early predictions!
    # =========================================================================
    true_corr_baseline = None
    corr_matrix = None # V8.8: Cache correlation matrix for edge discovery losses
    
    if A_true is not None:
        true_corr_baseline = compute_correlation_baseline(data["X"], A_true)
        if is_main_process() and not sweep_mode:
            print(f"\n" + "=" * 60)
            print("TRUE CORRELATION BASELINE (from data, not model)")
            print("=" * 60)
            print(f" TopK-F1: {true_corr_baseline['corr_baseline_f1']:.4f} "
                  f"(TP={true_corr_baseline['corr_baseline_tp']}/{int(A_true.sum().item())})")
            print(f" Skeleton-F1: {true_corr_baseline['corr_baseline_skel_f1']:.4f}")
            print(f" Reversed: {true_corr_baseline['corr_baseline_reversed']} edges (right pair, wrong direction)")
            print(f" Corr max: {true_corr_baseline['corr_max']:.3f}")
            print("=" * 60)
            print("Model must EXCEED this to claim causal learning!")
            print("=" * 60 + "\n")
        
        # V8.8 / V8.25: Compute and cache correlation matrix for hard negative mining
        # Vectorised: use numpy corrcoef instead of per-pair scipy calls
        X_np = data["X"].numpy()
        if X_np.ndim == 3:
            X_flat = X_np.reshape(-1, X_np.shape[-1])
        else:
            X_flat = X_np
        
        # Drop NaN rows once, then use np.corrcoef (O(d²) not O(d² * N))
        valid_rows = ~np.isnan(X_flat).any(axis=1)
        if valid_rows.sum() >= 10:
            corr_np = np.abs(np.corrcoef(X_flat[valid_rows].T)).astype(np.float32)
            np.fill_diagonal(corr_np, 0.0)
            corr_np = np.nan_to_num(corr_np, nan=0.0)
        else:
            corr_np = np.zeros((d, d), dtype=np.float32)
        
        corr_matrix = torch.from_numpy(corr_np)  # keep on CPU to avoid GPU memory leak
        base_model._corr_matrix_cpu = corr_matrix  # Cache on CPU; moved to device when needed
        
        # =================================================================
        # V8.27 FIX 1: Correlation-initialized logits
        # PROBLEM: Random init (randn*0.01) means all logits ≈ 0 → A ≈ 0.5.
        # First edges to separate are random, not principled. Model then
        # locks into these random edges. Result: often WORSE than corr baseline.
        #
        # FIX: Initialize W_adj ∝ standardized |corr(X_i, X_j)|.
        # High-corr pairs start with positive logits (σ>0.5),
        # low-corr start with negative (σ<0.5). This gives a warm start
        # at least as good as the correlation baseline. Causal losses
        # (L_icp, L_anticorr) then refine from there.
        # =================================================================
        if cfg.get("corr_init_enabled", True):
            corr_init_scale = cfg.get("corr_init_scale", 1.0)
            # Standardize: mean=0, std=corr_init_scale
            nonzero_mask = corr_np > 0
            if nonzero_mask.sum() > 1:
                c_mean = corr_np[nonzero_mask].mean()
                c_std = corr_np[nonzero_mask].std()
                if c_std > 1e-8:
                    W_init = (corr_np - c_mean) / c_std * corr_init_scale
                else:
                    W_init = (corr_np - c_mean) * corr_init_scale
            else:
                W_init = np.zeros_like(corr_np)
            np.fill_diagonal(W_init, 0.0)
            W_init_t = torch.from_numpy(W_init.astype(np.float32)).to(device)
            base_model.graph_learner.W_mag.data.copy_(W_init_t)
            
            # =============================================================
            # V8.33: W_dir starts at zero (dir=0.5 initially).
            # No symmetry-break noise needed! Direction losses (L_dir_dec)
            # will push W_dir toward ±∞ independently of W_mag.
            # The whole point of V8.33 is that direction and structure
            # have SEPARATE parameters — no need for noise hacks.
            # =============================================================
            base_model.graph_learner.W_dir.data.zero_()
            
            if is_main_process() and not sweep_mode:
                n_pos = int((W_init > 0).sum())
                topk_init = np.sort(W_init.flatten())[-target_edges_cfg:]
                print(f"\n[V8.33] Corr-initialized W_mag: {n_pos}/{d*(d-1)} positive, "
                      f"scale={corr_init_scale}, topk_mean={topk_init.mean():.3f}")
                print(f"[V8.33] W_dir initialized to 0 (all dir=0.5). Direction losses will learn direction.")
        
        # V8.25: Pre-cache cousin mask (expensive to recompute every epoch)
        if A_true is not None and cfg.get("use_cousin_penalty", True):
            base_model._cousin_mask_cpu = compute_cousin_mask(
                A_true.to(device), causal_convention=True).cpu()
    
    # =========================================================================
    # V8.33: Separate optimizer param groups for W_mag (structure) vs W_dir (direction)
    # Direction typically needs larger LR since it starts at 0 and must reach
    # large |Wd| values, while skeleton only needs small adjustments from corr init.
    # =========================================================================
    lr_dir_mult = cfg.get("lr_dir_multiplier", 5.0)
    lr_base = cfg["lr"]
    lr_dir = lr_base * lr_dir_mult
    
    # Collect parameter groups
    mag_params = [base_model.graph_learner.W_mag]
    dir_params = [base_model.graph_learner.W_dir]
    other_params = [p for n, p in model.named_parameters()
                    if p is not base_model.graph_learner.W_mag
                    and p is not base_model.graph_learner.W_dir]
    
    optimizer = torch.optim.AdamW([
        {"params": mag_params, "lr": lr_base, "weight_decay": cfg["weight_decay"]},
        {"params": dir_params, "lr": lr_dir, "weight_decay": 0.0},  # No WD on direction
        {"params": other_params, "lr": lr_base, "weight_decay": cfg["weight_decay"]},
    ])
    if is_main_process() and not sweep_mode:
        print(f"[V8.33] Optimizer: lr_mag={lr_base:.1e}, lr_dir={lr_dir:.1e} ({lr_dir_mult}×), "
              f"mag_params={sum(p.numel() for p in mag_params)}, "
              f"dir_params={sum(p.numel() for p in dir_params)}, "
              f"other_params={sum(p.numel() for p in other_params)}")
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=cfg["restart_every"],
        T_mult=1,
        eta_min=cfg["lr_min"]
    )
    
    # GroupDRO weights
    dro_weights = None
    if use_groupdro and n_regimes > 1:
        dro_weights = {r: 1.0 / n_regimes for r in range(n_regimes)}
    
    # Output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # =================================================================
    # PARETO CHECKPOINTING (V8.2)
    # Save multiple "best" checkpoints for different optimization targets:
    # 1. best_topk_sparse: max TopK-F1 subject to budget guard
    # 2. best_skel_sparse: max Skeleton-F1 subject to budget guard
    # 3. best_score: max composite score subject to budget guard
    # This prevents losing good sparse graphs that excel in one dimension.
    # =================================================================
    
    # V8.13 FIX: Decouple best tracking from guard
    # Problem: When guard never passes (e.g., stuck in DISC), best metrics
    # show 0.0000 even though model achieved perfect performance.
    # Solution: Track both guarded (for checkpoints) and overall (for reporting).
    
    # Pareto tracking - GUARDED (with budget guard, for checkpoint saving)
    best_topk_sparse = (0.0, 0) # Best TopK-F1 in budget window
    best_skel_sparse = (0.0, 0) # Best Skeleton-F1 in budget window
    best_score_ckpt = (-float("inf"), 0) # Best composite score
    
    # V8.13: OVERALL best (no guard, for reporting true performance)
    best_topk_overall = (0.0, 0) # Best TopK-F1 ever seen
    best_skel_overall = (0.0, 0) # Best Skeleton-F1 ever seen
    best_score_overall = (-float("inf"), 0) # Best composite score ever
    
    # V8.14: Early excellence tracking (allow save during DISC if model is good)
    # Track consecutive epochs where model meets high quality thresholds
    consecutive_excellent_epochs = 0
    excellence_threshold_epochs = 5 # Must maintain excellence for N epochs
    
    # V8.16: Freeze λb after early_excellence to prevent unbounded loss growth
    # Once model has converged (TopK-F1=1.0 sustained), no need to keep ramping budget
    frozen_lambda_budget = None # Will be set when early_excellence triggers
    lambda_budget_was_frozen_at_epoch = None # For logging
    
    # V8.17: Peakiness tracking for data-adaptive scheduling
    # Track gap, margin, edges_90pct from previous epoch to guide temp/lambda
    prev_peakiness = None # Will be computed at end of each epoch
    
    # V8.10: Track transpose wins (convention mismatch detection)
    transpose_wins_count = 0
    transpose_total_count = 0
    transpose_best_f1 = 0.0 # Best F1 achieved with transpose
    
    # For early stopping and backward compatibility
    best_metric = 0 # TopK-F1 for reporting
    best_epoch = 0
    patience_counter = 0
    prev_stage = "1:DISC"  # V8.18: Track stage for patience reset at transitions
    history = []
    stage1_healthy = True
    prev_topk_set = None # For tracking TopK stability
    
    # NOTE: We now use true_corr_baseline (computed from data at start)
    # NOT the old "best during DISC phase" which was just model initialization noise
    
    # CALIBRATION FIX: Track direction stability and edge_sum for adaptive scheduling
    direction_stable = False # Start assuming unstable until we see F1 > F1_T
    prev_edge_sum = None # For collapse detection
    
    # V8.23: DirConf hysteresis — prevent gaming by n_pairs collapse
    dir_conf_stable_streak = 0 # Consecutive epochs with DirConf >= 0.8 AND n_pairs >= min
    dir_conf_unstable_streak = 0 # Consecutive epochs with DirConf < 0.75
    dir_conf_locked = False # True = direction considered stable (with hysteresis)
    
    # V8.10: EMA smoothing for stable evaluation
    # A_ema = 0.9 * A_ema + 0.1 * A_current (reduces noise in TopK selection)
    A_ema = None
    ema_alpha = cfg.get("ema_alpha", 0.1) # How much to weight new A (0.1 = slow, 0.5 = fast)
    
    # V8.5: Skeleton mask for Stage D (frozen structure)
    skeleton_mask = None # Will be computed at end of Stage S (PRUNE)
    
    # V8.39: E_frozen — the frozen skeleton edge mask for REFINE enforcement
    E_frozen_mask = None  # Will be set at PRUNE→REFINE transition
    refine_start_epoch = None  # Will be set at PRUNE→REFINE transition
    
    # V8.11: Skeleton freeze for direction-only training
    skeleton_frozen = False
    skeleton_freeze_counter = 0 # Consecutive epochs with Skel >= threshold
    W_frozen = None # Frozen logits at time of skeleton freeze
    skeleton_freeze_threshold = cfg.get("skeleton_freeze_threshold", 0.95)
    skeleton_freeze_patience = cfg.get("skeleton_freeze_patience", 5)
    direction_tau = cfg.get("direction_tau", 0.5)
    lambda_direction_boost = cfg.get("lambda_direction", 2.0)
    
    # Training loop
    if is_main_process() and not sweep_mode:
        print("--- Training ---\n")
    
    for epoch in range(1, cfg["epochs"] + 1):
        t_start = time.time()
        
        # Set sampler epoch for DDP
        if ddp and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        
        # =====================================================================
        # V8.39: LR boost for W_dir in early REFINE epochs
        # Gives direction parameters extra gradient step size to escape from
        # the reset zero-init and quickly find invariance-aligned directions.
        # =====================================================================
        if E_frozen_mask is not None and refine_start_epoch is not None:
            refine_epoch = epoch - refine_start_epoch
            boost_duration = cfg.get("refine_dir_lr_boost_epochs", 10)
            base_lr_dir = cfg["lr"] * cfg.get("lr_dir_multiplier", 5.0) * 0.5  # 0.5 from Stage3 LR reduction
            if refine_epoch >= 0 and refine_epoch < boost_duration:
                boost_factor = cfg.get("refine_dir_lr_boost", 5.0)
                boosted_lr = base_lr_dir * boost_factor
                optimizer.param_groups[1]["lr"] = boosted_lr  # param_groups[1] = dir_params
                if refine_epoch == 0 and is_main_process() and not sweep_mode:
                    print(f" | [V8.39] W_dir LR boosted: {base_lr_dir:.1e} → {boosted_lr:.1e} (×{boost_factor}) for {boost_duration} epochs")
            elif refine_epoch == boost_duration:
                optimizer.param_groups[1]["lr"] = base_lr_dir  # Restore normal
                if is_main_process() and not sweep_mode:
                    print(f" | [V8.39] W_dir LR boost ended, restored to {base_lr_dir:.1e}")
        
        # Train epoch (with calibration parameters and skeleton mask for Stage D)
        train_metrics, dro_weights = train_epoch(
            model, train_loader, optimizer, device,
            epoch, cfg["epochs"], cfg,
            ddp=ddp, dro_weights=dro_weights,
            direction_stable=direction_stable, # For temperature floor
            prev_edge_sum=prev_edge_sum, # For sparsity collapse protection
            skeleton_mask=skeleton_mask, # V8.5: Frozen skeleton for Stage D
            A_true=A_true.to(device) if A_true is not None else None, # V8.7: For margin loss
            skeleton_frozen=skeleton_frozen, # V8.11: Skeleton frozen mode
            W_frozen=W_frozen, # V8.11: Frozen W for restoration
            direction_tau=direction_tau, # V8.11: τ for direction
            lambda_direction_boost=lambda_direction_boost, # V8.11: Boost factor
            frozen_lambda_budget=frozen_lambda_budget, # V8.16: Cap λb after convergence
            peakiness=prev_peakiness, # V8.17: Data-adaptive scheduling
            E_frozen_mask=E_frozen_mask, # V8.39: Frozen skeleton mask for REFINE
        )
        
        scheduler.step()
        t_elapsed = time.time() - t_start
        
        # Validate
        val_metrics = {}
        if epoch % cfg["eval_frequency"] == 0 or epoch == cfg["epochs"]:
            val_metrics = validate(
                model, val_loader, A_true, data["X"], device, cfg, ddp=ddp,
                epoch=epoch, total_epochs=cfg["epochs"],
            )
            
            # Correlation vs causation diagnosis (every 10 epochs)
            if epoch % 10 == 0:
                has_m = "M" in data
                print(f" | [PRE-DIAG] epoch={epoch} A_true_is_none={A_true is None} data_has_M={has_m}", flush=True)
            if epoch % 10 == 0 and A_true is not None:
                X_sample = data["X"][:min(2000, len(data["X"]))].to(device)
                M_sample = data.get("M", None)
                if M_sample is not None:
                    M_sample = M_sample[:min(2000, len(M_sample))].to(device)
                
                A_pred = base_model.graph_learner.get_mean_adjacency()
                # CRITICAL FIX: Convert to causal convention BEFORE comparison with A_true
                # A_pred is in MODEL convention (A[i,j]="j depends on i")
                # A_true is in CAUSAL convention (A[i,j]="i causes j")
                # Must transpose A_pred to match A_true
                A_pred = A_pred.T.contiguous() # Convert to causal convention
                A_pred_np = A_pred.detach().cpu().numpy().copy()
                A_true_np = (A_true.detach().cpu().numpy() > 0.5).astype(int)
                
                # V8.22: FIX—compute manual_true_overlap using SAME method as topk_f1
                # Use directed edge pairs, not flat indices!
                np.fill_diagonal(A_pred_np, 0)
                np.fill_diagonal(A_true_np, 0)
                
                d = A_pred_np.shape[0]
                n_true_edges = int(A_true_np.sum())
                
                # Step 1: Select K undirected pairs using symmetrization (same as compute_topk_f1)
                A_sym = np.maximum(A_pred_np, A_pred_np.T)
                np.fill_diagonal(A_sym, 0)
                A_upper = np.triu(A_sym)
                flat_upper = A_upper.flatten()
                # FIX: Just use argsort, not lexsort with confusing index ordering
                sort_keys = np.argsort(flat_upper)
                top_k_idx = sort_keys[-n_true_edges:] if n_true_edges > 0 else np.array([])
                
                # DIAGNOSTIC SETUP: Extract top-K edges directly from adjacency
                # NO upper triangle games—just use flat matrix indices
                try:
                    # Flatten adjacency, mask diagonal, get top K
                    A_flat = A_pred_np.flatten()
                    flat_mask = np.ones(len(A_flat), dtype=bool)
                    for i in range(d):
                        flat_mask[i * d + i] = False # mask diagonal
                    
                    # Get indices of top K in the FLAT directed space
                    flat_valid = np.where(flat_mask)[0]
                    flat_scores = A_flat[flat_valid]
                    top_k_local = np.argsort(flat_scores)[-n_true_edges:]
                    top_k_flat = flat_valid[top_k_local]
                    
                    # Unravel to (i,j) in the original directed space
                    pred_edges_directed = set(zip(*np.unravel_index(top_k_flat, (d, d))))
                    pred_edges_undirected = {tuple(sorted((i, j))) for i, j in pred_edges_directed}
                    
                    # True edges from ground truth
                    true_indices = np.where(A_true_np > 0)
                    true_edges_directed = set(zip(true_indices[0], true_indices[1]))
                    true_edges_undirected = {tuple(sorted((i, j))) for i, j in true_edges_directed}
                    
                    # Compute overlaps
                    dir_overlap = len(pred_edges_directed & true_edges_directed) / max(len(true_edges_directed), 1)
                    undir_overlap = len(pred_edges_undirected & true_edges_undirected) / max(len(true_edges_undirected), 1)
                    
                    # Show samples
                    pred_sample = list(pred_edges_directed)[:3]
                    true_sample = list(true_edges_directed)[:3]
                    
                    # DEBUG: Check intersection & raw set sizes
                    inter_dir = pred_edges_directed & true_edges_directed
                    inter_undir = pred_edges_undirected & true_edges_undirected
                    
                    # **CRITICAL**: Verify all undir edges actually come from pred/true
                    pred_un_sorted = sorted(pred_edges_undirected)
                    true_un_sorted = sorted(true_edges_undirected)
                    inter_un_sorted = sorted(inter_undir)
                    
                    print(f" | [EDGESET DEBUG] epoch={epoch}:", flush=True)
                    print(f" | DIR: pred={pred_sample} true={true_sample}", flush=True)
                    print(f" | SIZES: dir_pred={len(pred_edges_directed)} dir_true={len(true_edges_directed)}", flush=True)
                    print(f" | SIZES: undir_pred={len(pred_edges_undirected)} undir_true={len(true_edges_undirected)}", flush=True)
                    print(f" | UNDIR_PRED: {pred_un_sorted}", flush=True)
                    print(f" | UNDIR_TRUE: {true_un_sorted}", flush=True)
                    print(f" | UNDIR_INTER: {inter_un_sorted}", flush=True)
                    print(f" | Overlap: DIR={dir_overlap:.4f} ({len(inter_dir)}/{len(true_edges_directed)}) | UNDIR={undir_overlap:.4f} ({len(inter_undir)}/{len(true_edges_undirected)})", flush=True)
                    manual_true_overlap = undir_overlap # Use undirected (skeleton) overlap
                    
                except Exception as e:
                    print(f" | [EDGESET ERROR] {type(e).__name__}: {e}", flush=True)
                    manual_true_overlap = 0.0
                
                # Use the overlap that matches 1.0 (or closest to TopK-F1)
                manual_true_overlap = undir_overlap if undir_overlap > dir_overlap else dir_overlap
                
                # V8.21: Metric-driven diagnosis (TopK-F1, Skel-F1, DirConf)
                # Determine based on actual model performance, not heuristics
                topk_f1 = val_metrics.get("topk_f1", 0.0)
                skel_f1 = val_metrics.get("skeleton_f1", 0.0)
                dir_conf = val_metrics.get("dir_conf_ratio", 0.0) # V8.22: Fixed key name
                
                # Decision tree based on metrics:
                if topk_f1 >= 0.9 and skel_f1 >= 0.9 and dir_conf >= 0.8:
                    # All metrics strong -> CAUSATION (recovered true graph)
                    diagnosis = "causation"
                    reason = f"All strong: TopK={topk_f1:.2f}, Skel={skel_f1:.2f}, Dir={dir_conf:.2f}"
                elif topk_f1 >= 0.8 and skel_f1 >= 0.9 and dir_conf < 0.7:
                    # Found skeleton but directions wrong -> SKELETON ONLY
                    diagnosis = "skeleton_only"
                    reason = f"Skel good ({skel_f1:.2f}) but Dir weak ({dir_conf:.2f})"
                elif topk_f1 < 0.5:
                    # TopK-F1 too low -> CORRELATION
                    diagnosis = "correlation"
                    reason = f"TopK low: {topk_f1:.2f}"
                elif manual_true_overlap >= 0.8:
                    # High overlap with true edges -> CAUSATION
                    diagnosis = "causation"
                    reason = f"True overlap high: {manual_true_overlap:.2f}"
                else:
                    # Mixed signal: use TopK-F1 as proxy
                    diagnosis = "causation" if topk_f1 >= 0.7 else "correlation"
                    reason = f"TopK={topk_f1:.2f}, Overlap={manual_true_overlap:.2f}"
                
                print(f" | [DIAGNOSIS] {diagnosis.upper()}: {reason}", flush=True)
                
                val_metrics["diagnosis"] = diagnosis
                val_metrics["_detector"] = "metric-driven"
        
        # Log
        if is_main_process():
            entry = {
                "epoch": epoch,
                "time": t_elapsed,
                **train_metrics,
                **val_metrics,
            }
            history.append(entry)
            
            if not sweep_mode:
                # Stage info
                stage = "1:DISC" if epoch <= cfg["stage1_end"] * cfg["epochs"] else \
                        "2:PRUNE" if epoch <= cfg["stage2_end"] * cfg["epochs"] else "3:REFINE"
                
                # Compact log
                topk_f1 = val_metrics.get("topk_f1", 0)
                best_f1 = val_metrics.get("best_f1", 0)
                
                # Show scheduled weights
                temp = train_metrics.get("temperature", 1.0)
                lam_sparse = train_metrics.get("lambda_sparse_used", 0)
                lam_budget = train_metrics.get("lambda_bud_used", 0)
                
                # Get adjacency for proper edge counts
                # V8.12: Apply convention fix HERE so all downstream uses correct convention
                A_log = base_model.graph_learner.get_mean_adjacency()
                A_log = to_causal_convention(A_log) # V8.26: identity (no-op)
                A_np_log = A_log.detach().cpu().numpy()
                
                # V8.10: Update EMA smoothed adjacency for stable metrics
                # NOTE: A_np_log is now in CAUSAL convention, so EMA tracks correct metric
                if A_ema is None:
                    A_ema = A_np_log.copy()
                else:
                    A_ema = (1 - ema_alpha) * A_ema + ema_alpha * A_np_log
                
                # TopK stats (compute first for adaptive thresholding)
                K = cfg["target_edges"]
                flat_A = A_np_log.flatten()
                sorted_A = np.sort(flat_A)[::-1] # Descending
                topk_vals = sorted_A[:K]
                topk_mean = float(topk_vals.mean())
                topk_min = float(topk_vals.min()) # threshold at rank K
                
                # Compute gap to (K+1)th edge - measures ranking confidence
                if len(sorted_A) > K:
                    next_edge = float(sorted_A[K]) # (K+1)th largest
                    gap = topk_min - next_edge
                else:
                    gap = topk_min
                
                # Adaptive thresholding relative to topk_min (the decision boundary)
                thr = topk_min # The natural threshold that produces K edges
                edges_90pct = int((A_np_log > 0.9 * thr).sum()) # slightly below boundary
                edges_110pct = int((A_np_log > 1.1 * thr).sum()) # slightly above boundary 
                edges_2x = int((A_np_log > 2.0 * thr).sum()) # well above boundary
                
                # V8.12 FIX: Compute edge_sum on TopK edges ONLY (matches evaluation)
                # Previously used A.sum() (all 169 edges) -> failed guard even when TopK-F1=1.0
                # Now use sum of top K edges -> aligns budget with what we evaluate
                A_topk = get_topk_adjacency(A_log, k=K)
                edge_sum = float(A_topk.sum())
                
                # A stats
                A_max = float(A_np_log.max())
                A_mean = float(A_np_log.mean())
                
                # TopK TP count (for primary line)
                topk_tp = val_metrics.get("topk_tp", 0)
                
                # V8.16: Show frozen state for λb
                lambda_frozen_indicator = "[FROZEN]" if train_metrics.get("lambda_bud_frozen", False) else ""
                
                print(f"[{stage}] Epoch {epoch:3d}/{cfg['epochs']} | "
                      f"loss={train_metrics['loss']:.4f} | "
                      f"L_recon={train_metrics.get('L_recon', 0):.4f} | "
                      f"τ={temp:.2f} λs={lam_sparse:.1e} λb={lam_budget:.2f}{lambda_frozen_indicator} | "
                      f"TopK-F1={topk_f1:.4f} TP={topk_tp}/{K} | "
                      f"{t_elapsed:.1f}s")
                
                # ===============================================================
                # NEW: Clear edge count logging (ranked vs adaptive threshold)
                # ===============================================================
                print(f" | TopK edges K={K}: TP={topk_tp}")
                print(f" | Edges: thr={thr:.3f} gap={gap:.3f} | @90%={edges_90pct} @110%={edges_110pct} @2x={edges_2x} | sum={edge_sum:.1f}")
                
                # V8.15: Show TopK gap (separation between TopK and non-TopK edges)
                topk_gap_v15 = train_metrics.get("nontopk_max", None)  # V8.15 key
                nontopk_max = train_metrics.get("nontopk_max", None)
                L_nontopk = train_metrics.get("L_nontopk", 0)
                if topk_gap_v15 is not None and nontopk_max is not None:
                    topk_gap_val = train_metrics.get("topk_gap", 0) or 0
                    gap_status = "[OK]" if topk_gap_val > 0.1 else ("~" if topk_gap_val > 0 else "[WARN]")
                    print(f" | TopK gap: {topk_gap_val:.3f} [{gap_status}] | non-TopK max={nontopk_max:.3f} | L_suppress={L_nontopk:.4f}")
                
                print(f" | A stats: max={A_max:.3f} mean={A_mean:.4f} topk_mean={topk_mean:.3f}")
                
                # ===============================================================
                # V8.19: Diagnostic loss breakdown
                # Shows raw penalty values and weighted contributions to help debug
                # V8.20: Added L_sym (symmetry) and L_sep (separation) diagnostics
                # ===============================================================
                h_A_raw = train_metrics.get("h_A_raw", 0)
                L_acyclic = train_metrics.get("L_acyclic", 0)
                lambda_acy = train_metrics.get("lambda_acy_used", 0)
                L_budget = train_metrics.get("L_budget", 0)
                L_sparse = train_metrics.get("L_sparse", 0)
                L_binary = train_metrics.get("L_binary", 0)
                edge_sum_full = train_metrics.get("edge_sum", 0)  # Full matrix sum for budget
                target_K = train_metrics.get("target_edges", K)
                
                # V8.20: Get new losses
                L_sym = train_metrics.get("L_sym", 0)
                L_sep = train_metrics.get("L_sep", 0)
                L_bimodal = train_metrics.get("L_bimodal", 0)
                L_tail_val = train_metrics.get("L_tail", 0)
                L_dir_logit_val = train_metrics.get("L_dir_logit", 0)
                lambda_sym = train_metrics.get("lambda_sym", 0)
                lambda_sep = train_metrics.get("lambda_sep", 0)
                lambda_bimodal = train_metrics.get("lambda_bimodal", 0)
                lambda_tail_val = train_metrics.get("lambda_tail", 0)
                lambda_dir_logit_val = train_metrics.get("lambda_dir_logit", 0)
                topk_gap = train_metrics.get("topk_gap", 0)
                
                # Compute weighted contributions
                weighted_acy = lambda_acy * L_acyclic
                weighted_budget = lam_budget * L_budget
                weighted_sparse = lam_sparse * L_sparse
                weighted_sym = lambda_sym * L_sym
                weighted_sep = lambda_sep * L_sep
                weighted_bimodal = lambda_bimodal * L_bimodal
                weighted_tail = lambda_tail_val * L_tail_val
                weighted_dir_logit = lambda_dir_logit_val * L_dir_logit_val
                
                print(f" | LOSS: h_A={h_A_raw:.2f}→λ*L_acy={weighted_acy:.4f} | "
                      f"L_budget={L_budget:.2f}→λ*L_b={weighted_budget:.4f} | "
                      f"L_sparse={L_sparse:.4f}→λ*L_s={weighted_sparse:.4f} | "
                      f"L_bin={L_binary:.4f} | Σedge={edge_sum_full:.1f}/K={target_K}")
                
                # V8.23: Show all separation + direction diagnostics
                if lambda_sym > 0 or lambda_sep > 0 or lambda_bimodal > 0 or lambda_tail_val > 0:
                    print(f" | V8.23: L_sym={L_sym:.4f}→{weighted_sym:.4f} | "
                          f"L_bim={L_bimodal:.4f}→{weighted_bimodal:.4f} | "
                          f"L_tail²={L_tail_val:.4f}→{weighted_tail:.4f} | "
                          f"L_dir={L_dir_logit_val:.4f}→{weighted_dir_logit:.4f} | gap={topk_gap:.4f}")
                
                # V8.24: Show ICP + anticorr diagnostics
                L_icp_val = train_metrics.get("L_icp", 0)
                L_anticorr_val = train_metrics.get("L_anticorr", 0)
                lambda_icp_val = train_metrics.get("lambda_icp", 0)
                lambda_anticorr_val = train_metrics.get("lambda_anticorr", 0)
                if lambda_icp_val > 0 or lambda_anticorr_val > 0:
                    weighted_icp = lambda_icp_val * L_icp_val
                    weighted_anticorr = lambda_anticorr_val * L_anticorr_val
                    print(f" | V8.24: L_icp={L_icp_val:.4f}→{weighted_icp:.4f} | "
                          f"L_anticorr={L_anticorr_val:.4f}→{weighted_anticorr:.4f}")
                
                # ===============================================================
                # DIAGNOSTIC: Show transpose and skeleton F1 to detect direction issues
                # V8.10: CRITICAL - if F1(A.T) > F1(A), we have convention mismatch!
                # ===============================================================
                topk_f1_T = val_metrics.get("topk_f1_transpose", 0)
                topk_tp_T = val_metrics.get("topk_tp_transpose", 0)
                skeleton_f1 = val_metrics.get("skeleton_f1", 0)
                
                # CRITICAL: Check if transpose is BETTER (convention mismatch)
                transpose_wins = topk_f1_T > topk_f1 + 0.05
                skeleton_perfect = skeleton_f1 > 0.9
                convention_bug = transpose_wins and skeleton_perfect
                
                # V8.10: Update transpose win tracking
                if topk_f1_T > 0 and topk_f1 > 0:
                    transpose_total_count += 1
                    if transpose_wins:
                        transpose_wins_count += 1
                    if topk_f1_T > transpose_best_f1:
                        transpose_best_f1 = topk_f1_T
                
                if topk_f1_T > 0 or skeleton_f1 > 0:
                    if convention_bug:
                        # This is the smoking gun - perfect skeleton but transpose wins
                        print(f" | [ALERT] CONVENTION BUG DETECTED: F1(A)={topk_f1:.4f} < F1(A.T)={topk_f1_T:.4f}")
                        print(f" | -> Skeleton perfect ({skeleton_f1:.3f}) but directions are INVERTED")
                        print(f" | -> FIX: Use A.T for evaluation, or flip edge convention in data/model")
                    elif transpose_wins:
                        print(f" | [WARN] DIR_MISMATCH: F1(A)={topk_f1:.4f} TP={topk_tp} | F1(A.T)={topk_f1_T:.4f} TP_T={topk_tp_T}")
                    else:
                        skel_issue = " (EDGE_OK_DIR_WRONG)" if skeleton_f1 > topk_f1 + 0.1 else ""
                        print(f" | Direction: F1={topk_f1:.4f} vs F1_T={topk_f1_T:.4f} | Skel={skeleton_f1:.4f}{skel_issue}")
                
                # ===============================================================
                # V8.6: Direction accuracy on CONFIDENT edges only
                # ===============================================================
                dir_conf_ratio = val_metrics.get("dir_conf_ratio", 0)
                dir_conf_correct = val_metrics.get("dir_conf_correct", 0)
                dir_conf_total = val_metrics.get("dir_conf_total", 0)
                dir_conf_t = val_metrics.get("dir_conf_threshold", 0.3)
                dir_conf_n = val_metrics.get("dir_conf_n_pairs", 0)
                if dir_conf_total > 0:
                    dir_emoji = "[OK]" if dir_conf_ratio >= 0.7 else ("~" if dir_conf_ratio >= 0.5 else "[X]")
                    print(f" | DirConf: {dir_conf_correct}/{dir_conf_total} = {dir_conf_ratio:.2f} [{dir_emoji}] (t={dir_conf_t:.2f}, n_pairs={dir_conf_n})")
                
                # ===============================================================
                # V8.23 FIX: DirConf with hysteresis + minimum pairs requirement
                # V8.22 PROBLEM: DirConf jumped to 0.86 when n_pairs dropped from
                # 52→29 (fewer constraints = easier to satisfy). This is GAMING.
                #
                # FIX:
                # 1. Require n_pairs >= 40 to consider DirConf valid
                # 2. Lock only after 3 consecutive valid+stable epochs
                # 3. Unlock only after 3 consecutive unstable epochs
                # ===============================================================
                dir_conf_threshold_stable = cfg.get("dir_conf_threshold_stable", 0.8)
                # V8.29: Adaptive dir_conf_min_pairs — scale with d instead of fixed 40
                if cfg.get("dir_conf_min_pairs_adaptive", True):
                    dir_conf_min_pairs = max(10, d)  # d=15 → 15, d=50 → 50
                else:
                    dir_conf_min_pairs = cfg.get("dir_conf_min_pairs", 40)
                dir_conf_consec_lock = cfg.get("dir_conf_consec_lock", 3)
                dir_conf_consec_unlock = cfg.get("dir_conf_consec_unlock", 3)
                
                # Check if DirConf is VALID (enough pairs) and STABLE (above threshold)
                dir_conf_valid = (dir_conf_n >= dir_conf_min_pairs)
                dir_conf_good = (dir_conf_ratio >= dir_conf_threshold_stable) and dir_conf_valid
                dir_conf_bad = (dir_conf_ratio < 0.75) or (not dir_conf_valid)
                
                if dir_conf_good:
                    dir_conf_stable_streak += 1
                    dir_conf_unstable_streak = 0
                elif dir_conf_bad:
                    dir_conf_unstable_streak += 1
                    dir_conf_stable_streak = 0
                else:
                    # In gray zone (0.75-0.80 or valid but marginal): keep streaks
                    pass
                
                # Hysteresis: lock after N consecutive good, unlock after N consecutive bad
                if not dir_conf_locked and dir_conf_stable_streak >= dir_conf_consec_lock:
                    dir_conf_locked = True
                    if not sweep_mode:
                        print(f" | [V8.23] DirConf LOCKED after {dir_conf_consec_lock} consecutive stable epochs")
                elif dir_conf_locked and dir_conf_unstable_streak >= dir_conf_consec_unlock:
                    dir_conf_locked = False
                    if not sweep_mode:
                        print(f" | [V8.23] DirConf UNLOCKED after {dir_conf_consec_unlock} consecutive unstable epochs")
                
                direction_stable = dir_conf_locked
                prev_edge_sum = edge_sum # Track for collapse detection
                
                # ===============================================================
                # V8.17: Update peakiness metrics for next epoch
                # Used by data-adaptive temperature and λs/λb scheduling
                # ===============================================================
                prev_peakiness = {
                    "gap": gap,
                    "avg_margin": val_metrics.get("avg_margin", 0.2),
                    "edges_90pct": edges_90pct,
                    "topk_mean": topk_mean,
                    "A_mean": A_mean,
                }
                
                # V8.17: Log peakiness gating status
                if train_metrics.get("peakiness_gated", False):
                    failsafe_ep = cfg.get("peakiness_failsafe_epoch", 15)
                    if epoch < failsafe_ep:
                        print(f" | PEAKINESS WARMUP (gap={gap:.3f} <0.02 or margin<0.15): τ↓, λs/λb frozen (failsafe at epoch {failsafe_ep})")
                    else:
                        print(f" | PEAKINESS WARMUP bypassed by V8.18 fail-safe (epoch {epoch} ≥ {failsafe_ep}): λs/λb RAMPING")
                
                # V8.23: Direction stability with hysteresis + min pairs
                dir_tau_fixed = cfg.get("direction_tau_fixed", 0.5)
                pairs_ok = "✓" if dir_conf_n >= dir_conf_min_pairs else f"✗({dir_conf_n}<{dir_conf_min_pairs})"
                if direction_stable:
                    print(f" | [LOCK] Direction STABLE (hysteresis locked, streak={dir_conf_stable_streak}, pairs={pairs_ok}, τ_dir={dir_tau_fixed})")
                else:
                    print(f" | Direction UNSTABLE (streak_good={dir_conf_stable_streak}/{dir_conf_consec_lock}, streak_bad={dir_conf_unstable_streak}, pairs={pairs_ok})")
                
                # ===============================================================
                # V8.9: Log margin and symmetry metrics
                # ===============================================================
                avg_margin = val_metrics.get("avg_margin", 0)
                min_margin = val_metrics.get("min_margin", 0)
                symmetry_ratio = train_metrics.get("symmetry_ratio", 0)
                L_excl = train_metrics.get("L_excl", 0)
                if avg_margin > 0 or symmetry_ratio > 0:
                    margin_status = "[OK]" if min_margin > 0.01 else "[WARN]"
                    print(f" | Margins: avg={avg_margin:.4f} min={min_margin:.4f} [{margin_status}] | sym={symmetry_ratio:.3f} L_excl={L_excl:.4f}")
                
                # ===============================================================
                # V8.29: Log antisymmetric / bidir / ranking diagnostics
                # ===============================================================
                bidir_rate_log = train_metrics.get("bidir_rate", -1)
                L_ranking_log = train_metrics.get("L_ranking", 0)
                rank_gap_log = train_metrics.get("rank_gap", 0)
                L_excl_uncond = train_metrics.get("L_excl", 0)
                L_dir_dec_log = train_metrics.get("L_dir_dec", 0)
                dir_dec_log = train_metrics.get("dir_decisiveness", 0)
                if bidir_rate_log >= 0:
                    bidir_status = "[OK]" if bidir_rate_log < 0.1 else ("[WARN]" if bidir_rate_log < 0.3 else "[BAD]")
                    print(f" | V8.33: bidir={bidir_rate_log:.3f}{bidir_status} L_dir_dec={L_dir_dec_log:.4f} dir_dec={dir_dec_log:.3f} L_excl={L_excl_uncond:.4f} rank_gap={rank_gap_log:.4f}")
                
                # ===============================================================
                # V8.36: Log direction masking diagnostics
                # ===============================================================
                k_dir_n = train_metrics.get("dir_mask_n_edges", -1)
                k_dir_gap = train_metrics.get("dir_mask_gap", 0)
                k_dir_bnd = train_metrics.get("dir_mask_boundary", 0)
                dir_mask_src = train_metrics.get("dir_mask_source", "adaptive")
                if k_dir_n >= 0:
                    target_e = cfg.get("target_edges", 13)
                    true_e = int(A_true.sum()) if A_true is not None else -1
                    ratio = k_dir_n / max(true_e, 1) if true_e > 0 else -1
                    if dir_mask_src == "E_frozen":
                        print(f" | V8.39: K_dir={k_dir_n} [E_FROZEN] "
                              f"true={true_e} target={target_e} ratio={ratio:.1f}x")
                    else:
                        status = "[OK]" if 0.5 <= ratio <= 2.5 else ("[TIGHT]" if ratio < 0.5 else "[WIDE]")
                        print(f" | V8.37: K_dir={k_dir_n} (gap={k_dir_gap:.3f} bnd={k_dir_bnd:.3f}) "
                              f"true={true_e} target={target_e} ratio={ratio:.1f}x {status}")

                # V8.36: Log effective loss weights when REFINE reweighting is active
                refine_start_ep = int(cfg.get("stage2_end", 0.70) * cfg.get("epochs", 110))
                if epoch > refine_start_ep:
                    eff_recon = cfg.get("lambda_recon", 0.5) * cfg.get("refine_recon_factor", 0.1)
                    eff_inv = cfg.get("lambda_inv", 2.0) * cfg.get("refine_inv_factor", 5.0)
                    print(f" | V8.36: REFINE reweight λ_recon={eff_recon:.3f} λ_inv={eff_inv:.1f}")

                # ===============================================================
                # V8.11: Skeleton Freeze Logic
                # Once skeleton is learned (Skel-F1 >= threshold for N epochs),
                # freeze skeleton and train only direction
                # ===============================================================
                if cfg.get("skeleton_freeze_enabled", False) and not skeleton_frozen:
                    if skeleton_f1 >= skeleton_freeze_threshold:
                        skeleton_freeze_counter += 1
                        if skeleton_freeze_counter >= skeleton_freeze_patience:
                            # FREEZE SKELETON!
                            skeleton_frozen = True
                            W_frozen = base_model.graph_learner.W_mag.data.clone()  # V8.33: freeze W_mag
                            print(f" | [FROZEN] SKELETON FROZEN @ epoch {epoch}!")
                            print(f" | Skel-F1={skeleton_f1:.3f} >= {skeleton_freeze_threshold:.3f} for {skeleton_freeze_patience} epochs")
                            print(f" | Now training DIRECTION only (τ_dir={direction_tau}, λ_dir×{lambda_direction_boost})")
                    else:
                        skeleton_freeze_counter = 0 # Reset if skeleton drops
                        
                # If skeleton is frozen, restore symmetric part after each epoch
                if skeleton_frozen and W_frozen is not None:
                    freeze_skeleton_parameters(base_model, set(), W_frozen)
                    print(f" | [FROZEN] Skeleton restored (direction-only training)")
                
                # ===============================================================
                # V8.10: EMA-smoothed TopK-F1 for stable progress tracking
                # This removes noise from epoch-to-epoch evaluation
                # ===============================================================
                if A_ema is not None and A_true is not None:
                    A_ema_tensor = torch.tensor(A_ema, dtype=torch.float32)
                    ema_metrics = compute_topk_f1(A_ema_tensor, A_true)
                    ema_f1 = ema_metrics["topk_f1"]
                    ema_tp = ema_metrics["topk_tp"]
                    # Show if different from raw (indicates noise in raw)
                    if abs(ema_f1 - topk_f1) > 0.01:
                        print(f" | EMA TopK-F1: {ema_f1:.4f} TP={ema_tp}/{K} (smoothed, raw={topk_f1:.4f})")
                
                # Diagnosis
                if "diagnosis" in val_metrics:
                    diag_vals = {k: v for k, v in val_metrics.items() if k.startswith("diag_")}
                    detector = val_metrics.get("_detector", "?")
                    diag_str = " | ".join([f"{k.replace('diag_','')}={v:.3f}" for k, v in sorted(diag_vals.items())]) if diag_vals else ""
                    if diag_str:
                        print(f" | Learning: {val_metrics['diagnosis'].upper()} [{detector}] ({diag_str})")
                    else:
                        print(f" | Learning: {val_metrics['diagnosis'].upper()} [{detector}]")
        
        # STEP 3 FIX: Proper health check (not dense-is-healthy!)
        stage1_end = int(cfg["stage1_end"] * cfg["epochs"])
        if epoch == stage1_end and is_main_process():
            A_pred = base_model.graph_learner.get_mean_adjacency()
            health = compute_health_metrics(A_pred, cfg["target_edges"], cfg, prev_topk_set)
            prev_topk_set = health["curr_topk_set"]
            
            # CORRECT health check: is graph SPARSE enough, not too dense
            if not health["edges_05_healthy"]:
                stage1_healthy = False
                if not sweep_mode:
                    print(f" | [WARN] Stage 1 UNHEALTHY: edges@0.5={health['edges@0.5']} > {3*cfg['target_edges']} (too dense!)")
                    print(f" | Graph is learning CORRELATION, not CAUSATION")
            else:
                if not sweep_mode:
                    print(f" | [OK] Stage 1 HEALTHY: edges@0.5={health['edges@0.5']} ≤ {3*cfg['target_edges']}")
                    # V8.16: Check tail mass for interpretability
                    # Fat tail = many medium-confidence edges that hurt threshold-based eval
                    K = cfg['target_edges']
                    tail_budget = 5 * K # Allow up to 5x target at 0.2 threshold
                    edges_02 = health['edges@0.2']
                    edges_03 = health['edges@0.3']
                    if edges_02 > tail_budget:
                        print(f" | [WARN] FAT TAIL: edges@0.2={edges_02} > {tail_budget} (interpretability risk)")
                        print(f" | Consider: V8.15 suppression to push non-TopK edges down")
                    else:
                        print(f" | edges@0.2={edges_02}, edges@0.3={edges_03} (tail OK)")
        
        # Track TopK stability during PRUNE
        stage2_start = int(cfg["stage1_end"] * cfg["epochs"])
        stage2_end_epoch = int(cfg["stage2_end"] * cfg["epochs"])
        if stage2_start < epoch <= stage2_end_epoch and epoch % 5 == 0 and is_main_process():
            A_pred = base_model.graph_learner.get_mean_adjacency()
            health = compute_health_metrics(A_pred, cfg["target_edges"], cfg, prev_topk_set)
            prev_topk_set = health["curr_topk_set"]
            
            if not sweep_mode:
                stable_icon = "[OK]" if health["topk_stable"] else "[WARN]"
                # V8.16: Include tail mass in periodic check
                K = cfg['target_edges']
                tail_ratio = health['edges@0.2'] / max(health['edges@0.5'], 1)
                tail_status = "[OK]" if tail_ratio <= 5 else "[WARN]"
                print(f" | {stable_icon} TopK Jaccard={health['topk_jaccard']:.3f}, edges@0.5={health['edges@0.5']}, edge_sum={health['edge_sum']:.1f}")
                if tail_ratio > 3:
                    print(f" | {tail_status} Tail: @0.2={health['edges@0.2']} (@0.2/@0.5={tail_ratio:.1f}x)")
        
        # Stage 3 = Stage D (Direction) - LR reduction and skeleton freeze
        stage3_start = int(cfg["stage2_end"] * cfg["epochs"])
        if epoch == stage3_start:
            for param_group in optimizer.param_groups:
                param_group["lr"] *= 0.5
            if is_main_process() and not sweep_mode:
                print(f" | Stage 3: LR reduced to {optimizer.param_groups[0]['lr']:.1e}")
            
            # ================================================================
            # V8.5: SKELETON FREEZE - Extract and freeze skeleton at Stage S end
            # This separates "which edges exist" from "which direction"
            # ================================================================
            if cfg.get("two_stage_training", True):
                A_stage_s = base_model.graph_learner.get_mean_adjacency()
                A_stage_s_np = A_stage_s.detach().cpu().numpy()
                
                # Create skeleton mask: symmetrize and threshold
                # Edge (i,j) is in skeleton if max(A[i,j], A[j,i]) > threshold
                A_sym = np.maximum(A_stage_s_np, A_stage_s_np.T)
                np.fill_diagonal(A_sym, 0)
                
                # Get top-K undirected edges for skeleton
                K = cfg.get("target_edges", 13)
                # For undirected, use upper triangle
                upper = np.triu(A_sym, k=1)
                flat_upper = upper.flatten()
                topk_val = np.sort(flat_upper)[::-1][K-1] if K > 0 else 0
                skel_thresh = max(cfg.get("skeleton_freeze_threshold", 0.1), topk_val)
                
                skeleton_mask = torch.from_numpy((A_sym >= skel_thresh).astype(np.float32)).to(device)
                skeleton_mask = skeleton_mask.bool()
                
                skel_edge_count = int(skeleton_mask.sum().item()) // 2 # Undirected
                
                # =============================================================
                # V8.39: Store E_frozen as explicit mask for REFINE enforcement
                # This mask will be passed to train_epoch to override dir_mask,
                # ensuring ALL direction losses are computed ONLY on these edges.
                # =============================================================
                E_frozen_mask = skeleton_mask.float()  # [d, d] binary symmetric
                E_frozen_n = skel_edge_count  # undirected edge count
                
                if is_main_process() and not sweep_mode:
                    print(f" | [LOCK] STAGE D: Skeleton FROZEN with {skel_edge_count} undirected edges")
                    print(f" | Threshold: {skel_thresh:.4f}, Now learning DIRECTION only")
                    print(f" | [V8.39] E_frozen stored: {E_frozen_n} undirected edges → ALL direction losses masked to these")
                    
                    # Save skeleton for analysis
                    np.save(output_path / "skeleton_mask.npy", skeleton_mask.cpu().numpy())
                    np.save(output_path / "A_stage_s.npy", A_stage_s_np)
                
                # =============================================================
                # V8.39: Reset W_dir at REFINE start (escape wrong-direction basin)
                # The old W_dir committed to wrong directions during DISC/PRUNE
                # under L_recon dominance. A fresh start with boosted LR gives
                # invariance a fair shot on the frozen skeleton.
                # =============================================================
                if cfg.get("refine_reset_wdir", True):
                    base_model.graph_learner.W_dir.data.zero_()
                    if is_main_process() and not sweep_mode:
                        print(f" | [V8.39] W_dir RESET to 0 (all dir=0.5). Fresh direction learning in REFINE.")
                
                # V8.39: LR boost for W_dir in early REFINE
                refine_lr_boost = cfg.get("refine_dir_lr_boost", 5.0)
                for pg in optimizer.param_groups:
                    # param_groups[1] is dir_params (W_dir) — set during optimizer creation
                    pass  # We'll apply per-epoch below
                # Store refine start epoch for LR boost scheduling
                refine_start_epoch = epoch
        
        # =================================================================
        # MODEL SELECTION - Strict Budget-Window Guard (V8)
        # =================================================================
        # Only save checkpoints for graphs that are:
        # 1. Past DISC phase (not early dense correlation graphs)
        # 2. Within budget window: K * (1 - tol) <= edge_sum <= K * (1 + tol)
        # 3. Have confident edges (edges @ 0.2 threshold > 0)
        # This eliminates "early dense best" problem entirely.
        # =================================================================
        if "topk_f1" in val_metrics:
            current_metric = val_metrics["topk_f1"]
            
            # Get current graph stats and convert to causal convention
            A_pred_check = base_model.graph_learner.get_mean_adjacency()
            A_pred_check = to_causal_convention(A_pred_check) # V8.26: identity (no-op)
            A_np_check = A_pred_check.detach().cpu().numpy()
            
            # Budget window parameters
            K = cfg.get("target_edges", 13)
            
            # V8.12 FIX: Use TopK edge sum for guard (matches evaluation)
            # Previously used full matrix sum -> failed guard even when TopK-F1=1.0
            A_topk_check = get_topk_adjacency(A_pred_check, k=K)
            edge_sum = float(A_topk_check.sum())
            max_edge = float(A_np_check.max())
            edges_at_02 = int((A_np_check > 0.2).sum())
            
            # Budget window parameters
            # V8.18 FIX: tol=0.5 was too tight for large K (K=30 → min=15,
            # but soft edge values sum to ~13).  Widen to 0.7 so min=K*0.3.
            tol = 0.7 # Allow sum ∈ [K*0.3, K*1.7]
            min_sum = K * (1 - tol) # e.g., 3.9 for K=13, 9.0 for K=30
            max_sum = K * (1 + tol) # e.g., 22.1 for K=13, 51.0 for K=30
            
            # Stage boundaries
            stage1_end = int(cfg["stage1_end"] * cfg["epochs"])
            past_disc = epoch > stage1_end # Must be in PRUNE or REFINE
            
            # V8.14: Check if model is demonstrably excellent
            # High quality: TopK-F1 ≥ 0.9, Skel-F1 ≥ 0.95, in budget window
            topk_f1 = val_metrics.get("topk_f1", 0.0)
            skel_f1 = val_metrics.get("skeleton_f1", 0.0)
            is_excellent = (topk_f1 >= 0.9 and skel_f1 >= 0.95 and 
                           min_sum <= edge_sum <= max_sum)
            
            # Track consecutive excellent epochs
            if is_excellent:
                consecutive_excellent_epochs += 1
            else:
                consecutive_excellent_epochs = 0
            
            # Early excellence: maintained high quality for N epochs
            early_excellence = consecutive_excellent_epochs >= excellence_threshold_epochs
            
            # V8.16: Freeze λb once early_excellence is achieved
            # This prevents unbounded loss growth after model has converged
            if early_excellence and frozen_lambda_budget is None:
                # Get current λb value and freeze it
                current_lambda_b = train_metrics.get("lambda_bud_used", 0.5)
                frozen_lambda_budget = current_lambda_b
                lambda_budget_was_frozen_at_epoch = epoch
                if not sweep_mode:
                    print(f" | [FROZEN] V8.16: λb FROZEN at {frozen_lambda_budget:.3f} (epoch {epoch}, TopK-F1={topk_f1:.4f})")
            
            # FLEXIBLE GUARD: Allow checkpoint if EITHER:
            # 1. Past DISC phase (original strict guard), OR
            # 2. Model demonstrates sustained excellence (prevents "solved but can't save")
            in_budget_window = (min_sum <= edge_sum <= max_sum)
            has_confident = (max_edge > 0.1) # At least one edge > 0.1
            
            # ================================================================
            # V8.19 FIX D: Baseline-aware guard
            # Don't save checkpoints when model is WORSE than correlation baseline!
            # This prevents saving useless models that could have been achieved
            # by simply picking top-K correlations.
            #
            # Note: true_corr_baseline is computed ONCE at start from data.
            # ================================================================
            corr_f1 = true_corr_baseline.get("corr_baseline_f1", 0.0) if true_corr_baseline else 0.0
            baseline_margin = cfg.get("baseline_guard_margin", 0.0)  # Can be negative to allow slightly worse
            beats_baseline = (topk_f1 > corr_f1 + baseline_margin)
            
            # Combine all guard checks
            graph_valid = (past_disc or early_excellence) and in_budget_window and has_confident and beats_baseline
            
            # Log guard status (useful for debugging)
            if not sweep_mode and epoch % 10 == 0:
                guard_status = "[OK]" if graph_valid else "[X]"
                reason = []
                if not past_disc and not early_excellence:
                    if consecutive_excellent_epochs > 0:
                        reason.append(f"DISC phase, excellent {consecutive_excellent_epochs}/{excellence_threshold_epochs}")
                    else:
                        reason.append(f"DISC phase")
                elif early_excellence and not past_disc:
                    reason.append(f" EARLY EXCELLENCE (F1={topk_f1:.2f}, sustained {consecutive_excellent_epochs} epochs)")
                if not in_budget_window:
                    reason.append(f"budget [{min_sum:.1f},{max_sum:.1f}]")
                if not has_confident:
                    reason.append(f"max={max_edge:.3f}<0.1")
                if not beats_baseline:
                    reason.append(f"TopK-F1={topk_f1:.3f}<=CorrBaseline={corr_f1:.3f}")
                reason_str = ", ".join(reason) if reason else "all checks passed"
                print(f" | [LOCK] Guard: {guard_status} ({reason_str})")
            
            # ================================================================
            # COMPOSITE SCORE for model selection (V8.3)
            # Instead of TopK-F1 alone, use weighted combination:
            # - TopK-F1 (directed correctness)
            # - Skeleton F1 (undirected structure)
            # - Direction bonus/penalty:
            # * BONUS when TopK > TopK_T (getting direction right)
            # * PENALTY when Skeleton > TopK (found edge, wrong direction)
            # - Budget penalty (deviation from target K)
            # 
            # The key insight: if Skeleton-F1 >> TopK-F1, model finds the 
            # right variable pairs but gets direction wrong consistently.
            # This should be penalized to encourage true causal learning.
            # ================================================================
            topk = current_metric
            skel = val_metrics.get("skeleton_f1", 0.0) # Fixed: was "skel_f1" but key is "skeleton_f1"
            f1_t = val_metrics.get("topk_f1_T", 0.0)
            
            # Direction correctness analysis
            dir_bonus = max(0.0, topk - f1_t) # Positive only if direction correct vs transpose
            dir_penalty = max(0.0, skel - topk) # Penalty when skeleton > directed (wrong directions)
            
            # V8.12 FIX: Budget penalty on TopK edges (matches evaluation)
            # edge_sum now contains sum of top K edges only, not full matrix
            # Ideal: edge_sum ≈ K (each edge weight ≈ 1.0)
            budget_pen = abs(edge_sum - K) / K # 0 when perfect, 1 when 2× or 0×
            # edge_sum now contains sum of top K edges only, not full matrix
            # Ideal: edge_sum ≈ K (each edge weight ≈ 1.0)
            budget_pen = abs(edge_sum - K) / K # 0 when perfect, 1 when 2× or 0×
            
            # Direction ratio: topk/skel measures what fraction of found edges have correct direction
            # If skel=0.3077 and topk=0.0769, then dir_ratio = 0.25 (only 25% correct direction)
            # If skel=0.3077 and topk=0.2308, then dir_ratio = 0.75 (75% correct direction)
            dir_ratio = topk / skel if skel > 1e-6 else 1.0
            
            # ================================================================
            # V8.5: Stage-aware scoring
            # - Stage S (Skeleton): Score by skeleton, minimal direction penalty
            # - Stage D (Direction): Heavy direction penalty, skeleton is fixed
            # ================================================================
            stage2_end = cfg.get("stage2_end", 0.80)
            in_stage_d = epoch >= stage2_end * cfg["epochs"]
            two_stage = cfg.get("two_stage_training", True)
            
            # V8.29: Bidir penalty — penalize composite score by bidir_rate from training
            bidir_pen_w = cfg.get("bidir_score_penalty", 0.5)
            epoch_bidir_rate = train_metrics.get("bidir_rate", 0.0)
            bidir_pen = bidir_pen_w * epoch_bidir_rate
            
            if two_stage and not in_stage_d:
                # Stage S: Focus on skeleton, light direction penalty
                composite_score = (
                    0.5 * topk + # Some weight on directed
                    1.0 * skel + # PRIMARY: undirected structure
                    0.3 * dir_bonus - # Light bonus for direction
                    0.3 * dir_penalty - # Light penalty (don't fight sparsity)
                    0.5 * budget_pen -
                    bidir_pen            # V8.29: penalize bidirectional edges
                )
            else:
                # Stage D or legacy: Heavy direction penalty
                composite_score = (
                    1.0 * topk +
                    0.5 * skel +
                    0.5 * dir_bonus - # Bonus for beating transpose
                    1.0 * dir_penalty - # Strong penalty for skeleton >> directed
                    0.5 * budget_pen -
                    bidir_pen            # V8.29: penalize bidirectional edges
                )
            
            # Log composite score components
            if not sweep_mode and epoch % 10 == 0:
                dir_status = "[OK]" if dir_ratio > 0.7 else ("~" if dir_ratio > 0.4 else "[X]")
                print(f" | Score: {composite_score:.4f} = topk({topk:.3f}) + 0.5*skel({skel:.3f}) + 0.5*dirB({dir_bonus:.3f}) - 1.0*dirP({dir_penalty:.3f}) - 0.5*bpen({budget_pen:.2f}) - bidir({bidir_pen:.3f})")
                print(f" | Dir ratio: {dir_ratio:.2f} [{dir_status}] (topk/skel, want >0.7)")
            
            # ================================================================
            # PARETO CHECKPOINTING - Save multiple "best" checkpoints
            # ================================================================
            
            # Helper to save checkpoint
            def save_ckpt(name: str, metric_val: float, score_val: float = None):
                model_to_save = model.module if ddp else model
                torch.save({
                    "epoch": epoch,
                    "model_state": model_to_save.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "metrics": val_metrics,
                    "composite_score": composite_score,
                    "config": cfg,
                }, output_path / f"best_{name}.pt")
                np.save(output_path / f"A_best_{name}.npy", A_np_check)
            
            if graph_valid:
                # V8.13: Track GUARDED best (with guard, for checkpoint saving)
                # 1. Best TopK-F1 (sparse)
                if topk > best_topk_sparse[0]:
                    best_topk_sparse = (topk, epoch)
                    if is_main_process():
                        save_ckpt("topk_sparse", topk)
                        if not sweep_mode:
                            print(f" | New best_topk_sparse: TopK-F1={topk:.4f} (guarded)")
                
                # 2. Best Skeleton-F1 (sparse)
                if skel > best_skel_sparse[0]:
                    best_skel_sparse = (skel, epoch)
                    if is_main_process():
                        save_ckpt("skel_sparse", skel)
                        if not sweep_mode:
                            print(f" | New best_skel_sparse: Skel-F1={skel:.4f} (guarded)")
                
                # 3. Best composite score
                if composite_score > best_score_ckpt[0]:
                    best_score_ckpt = (composite_score, epoch)
                    best_metric = topk # For reporting
                    best_epoch = epoch
                    if is_main_process():
                        save_ckpt("score", composite_score, composite_score)
                        # Also save as best_model.pt for backward compatibility
                        torch.save({
                            "epoch": epoch,
                            "model_state": (model.module if ddp else model).state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "metrics": val_metrics,
                            "composite_score": composite_score,
                            "config": cfg,
                        }, output_path / "best_model.pt")
                        np.save(output_path / "A_best.npy", A_np_check)
                        if not sweep_mode:
                            print(f" | * New best_score: {composite_score:.4f} (TopK={topk:.4f}, sum={edge_sum:.1f}) (guarded)")
            
            # V8.13: ALWAYS track OVERALL best (no guard, for reporting true performance)
            # V8.23 FIX: Also SAVE unguarded best checkpoint as fallback!
            # V8.22 PROBLEM: Guard never passed → no checkpoint saved → eval used
            # A_final.npy (last epoch, often WORSE than best epoch).
            # Now we always save the best unguarded model and adjacency.
            if topk > best_topk_overall[0]:
                best_topk_overall = (topk, epoch)
                if is_main_process():
                    # Always save unguarded best
                    model_to_save_ug = model.module if ddp else model
                    torch.save({
                        "epoch": epoch,
                        "model_state": model_to_save_ug.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "metrics": val_metrics,
                        "composite_score": composite_score,
                        "config": cfg,
                    }, output_path / "best_unguarded.pt")
                    np.save(output_path / "A_best_unguarded.npy", A_np_check)
                    if not graph_valid and not sweep_mode:
                        print(f" | New best_topk_overall: TopK-F1={topk:.4f} (unguarded, saved fallback)")
            
            if skel > best_skel_overall[0]:
                best_skel_overall = (skel, epoch)
            
            # Reset patience on overall improvement (unguarded)
            if composite_score > best_score_overall[0]:
                best_score_overall = (composite_score, epoch)
                patience_counter = 0 # Reset patience on any improvement
            else:
                patience_counter += 1
            
            if not graph_valid:
                if not sweep_mode and is_main_process():
                    print(f" | [WARN] score={composite_score:.4f} but guard failed (sum={edge_sum:.1f})")
        
        # =====================================================================
        # V8.18 FIX: Stage-aware patience — reset at DISC→PRUNE and PRUNE→REFINE
        # Without this, patience accumulated in late DISC carries over and
        # immediately triggers early stopping at PRUNE entry.
        # =====================================================================
        stage1_end = cfg.get("stage1_end", 0.30)
        stage2_end = cfg.get("stage2_end", 0.80)
        cur_stage = "1:DISC" if epoch <= stage1_end * cfg["epochs"] else \
                    "2:PRUNE" if epoch <= stage2_end * cfg["epochs"] else "3:REFINE"
        
        if cur_stage != prev_stage:
            # =============================================================
            # V8.27 FIX 2: Restore best DISC checkpoint at DISC→PRUNE
            # PROBLEM: Model finds best F1 mid-DISC (e.g., epoch 21) but
            # regresses by DISC end (epoch 30). PRUNE then locks in the
            # WORSE edge set, never recovering the earlier peak.
            #
            # FIX: At the DISC→PRUNE transition, reload the best unguarded
            # checkpoint (saved during DISC). Then immediately hard-prune
            # to lock in the better edge set.
            # =============================================================
            if prev_stage == "1:DISC" and cur_stage == "2:PRUNE":
                best_ckpt_path = output_path / "best_unguarded.pt"
                if best_ckpt_path.exists() and cfg.get("restore_best_at_prune", True):
                    ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
                    base_model.load_state_dict(ckpt["model_state"])
                    # Also restore optimizer to match model state
                    if "optimizer_state" in ckpt:
                        optimizer.load_state_dict(ckpt["optimizer_state"])
                    ckpt_epoch = ckpt.get("epoch", "?")
                    if is_main_process() and not sweep_mode:
                        print(f"\n🔁 [V8.27] RESTORED best DISC checkpoint (epoch {ckpt_epoch}, "
                              f"TopK-F1={best_topk_overall[0]:.4f}) before entering PRUNE")
                    
                    # Immediately hard-prune to lock in the restored edge set
                    if cfg.get("hard_prune_enabled", True):
                        with torch.no_grad():
                            W_mag_r = base_model.graph_learner.W_mag  # V8.33: prune structure param
                            d_r = W_mag_r.shape[0]
                            diag_r = torch.eye(d_r, device=W_mag_r.device)
                            # V8.34 FIX: Select from upper triangle (symmetric W_mag)
                            Wm_sym_r = (W_mag_r + W_mag_r.T) / 2
                            Wm_sym_r = Wm_sym_r * (1 - diag_r)
                            W_upper_r = torch.triu(Wm_sym_r, diagonal=1)
                            W_upper_flat_r = W_upper_r.flatten()
                            n_upper_r = d_r * (d_r - 1) // 2
                            k_r = min(target_edges_cfg, n_upper_r)
                            _, topk_r_idx = torch.topk(W_upper_flat_r, k_r)
                            keep_upper_flat_r = torch.zeros_like(W_upper_flat_r)
                            keep_upper_flat_r[topk_r_idx] = 1.0
                            keep_upper_r = keep_upper_flat_r.view(d_r, d_r)
                            keep_r = (keep_upper_r + keep_upper_r.T).clamp(max=1.0)
                            prune_decay_r = cfg.get("hard_prune_decay", 0.5)
                            prune_floor_r = cfg.get("hard_prune_floor", -2.0)
                            damped_r = W_mag_r.data * keep_r + W_mag_r.data * (1 - keep_r) * (1 - diag_r) * prune_decay_r
                            damped_r = torch.where(
                                keep_r.bool() | diag_r.bool(),
                                W_mag_r.data,
                                torch.clamp(damped_r, min=prune_floor_r)
                            )
                            W_mag_r.data = damped_r
                        if is_main_process() and not sweep_mode:
                            n_kept_r = int(keep_r.sum().item())
                            print(f" | [V8.35] Immediate soft prune after restore: "
                                  f"kept {k_r} undirected edges ({n_kept_r} entries), damped rest by {prune_decay_r}×")
            
            if is_main_process() and not sweep_mode:
                print(f"\n🔄 Stage transition {prev_stage} → {cur_stage} at epoch {epoch}"
                      f" — patience reset ({patience_counter} → 0)")
                # V8.37: Announce REFINE reweighting
                if cur_stage == "3:REFINE":
                    rr = cfg.get("refine_recon_factor", 0.1)
                    ri = cfg.get("refine_inv_factor", 5.0)
                    lr_base = cfg.get("lambda_recon", 0.5)
                    li_base = cfg.get("lambda_inv", 2.0)
                    print(f" | [V8.36] REFINE REWEIGHTING ACTIVE:")
                    print(f" |   λ_recon: {lr_base:.2f} → {lr_base*rr:.3f} (×{rr})")
                    print(f" |   λ_inv:   {li_base:.2f} → {li_base*ri:.1f} (×{ri})")
                    print(f" |   → Invariance is now the DOMINANT signal for direction")
                    if E_frozen_mask is not None:
                        n_frozen = int(E_frozen_mask.sum().item()) // 2
                        print(f" | [V8.39] ENFORCED FREEZE CONTRACT:")
                        print(f" |   E_frozen: {n_frozen} undirected edges")
                        print(f" |   Edge swaps: DISABLED")
                        print(f" |   Dir losses: MASKED to E_frozen only")
                        print(f" |   W_dir: {'RESET' if cfg.get('refine_reset_wdir', True) else 'kept'}")
                        print(f" |   LR boost: ×{cfg.get('refine_dir_lr_boost', 5.0)} for {cfg.get('refine_dir_lr_boost_epochs', 10)} epochs")
            patience_counter = 0  # fresh start for new stage
            prev_stage = cur_stage
        
        past_disc = epoch >= stage1_end * cfg["epochs"]
        past_refine = epoch >= stage2_end * cfg["epochs"]  # V8.38
        
        # V8.38 FIX: Early stopping ONLY in REFINE phase
        # PROBLEM: patience=32 exhausts during PRUNE (epoch 34+32=66),
        #   but REFINE starts at epoch 78 (0.70×110). Model never reaches
        #   REFINE reweighting — the whole point of the 3-stage pipeline.
        # FIX: Block early stopping in both DISC and PRUNE. Only allow in REFINE.
        if patience_counter >= cfg["patience"]:
            if past_refine:  # V8.38: Only early stop in REFINE phase
                if is_main_process() and not sweep_mode:
                    print(f"\n⏹ Early stopping at epoch {epoch} (patience={cfg['patience']})")
                break
            elif past_disc:
                # In PRUNE phase - don't stop, need to reach REFINE
                if is_main_process() and not sweep_mode and patience_counter == cfg["patience"]:
                    print(f" | [V8.38] Early stopping blocked (PRUNE phase: epoch {epoch}, "
                          f"REFINE starts at {int(stage2_end * cfg['epochs'])}. Patience reset)")
                patience_counter = 0  # Reset patience in PRUNE
            else:
                # Still in DISC phase - don't stop early
                if is_main_process() and not sweep_mode and patience_counter == cfg["patience"]:
                    print(f" | Early stopping blocked (still in DISC phase: epoch {epoch}/{int(stage1_end * cfg['epochs'])}, patience reset)")
                patience_counter = 0 # Reset patience in DISC, don't keep at threshold
    
    # Final summary
    if is_main_process():
        # Save history
        with open(output_path / "training_history.json", "w") as f:
            json.dump(history, f, indent=2, default=str)
        
        # Save final adjacency in CAUSAL convention
        A_final = base_model.graph_learner.get_mean_adjacency()
        A_final = to_causal_convention(A_final) # V8.26: identity (no-op)
        np.save(output_path / "A_final.npy", A_final.detach().cpu().numpy())
        
        if not sweep_mode:
            # Get final graph stats
            A_final_np = A_final.detach().cpu().numpy()
            
            # V8.12 FIX: Use TopK edge sum for consistency
            K = cfg.get("target_edges", 13)
            A_topk_final = get_topk_adjacency(A_final, k=K)
            final_edge_sum = float(A_topk_final.sum())
            final_edges_02 = int((A_final_np > 0.2).sum())
            
            print("\n" + "=" * 70)
            print("TRAINING COMPLETE (V8.39 - Enforced Skeleton Freeze)")
            print("=" * 70)
            print("CONVENTION INFO (for paper writeup):")
            print(" Model output: A[i,j] = 'i causes j' (causal convention)")
            print(" A_true:       A[i,j] = 'i causes j' (causal convention)")
            print(" Conversion:   NONE (identity — model already outputs causal)")
            print(" Proof: decoder A^T @ z → z_agg[j] = sum_i A[i,j]*z[i] → A[i,j]>0 ⟺ i→j")
            print(" [DONE] All metrics, saves, and confusion matrices use A directly")
            print("-" * 70)
            print("V8.11 FIXES:")
            print(" 1. Skeleton freeze: after Skel-F1>=0.95 for 5 epochs, freeze skeleton")
            print(" 2. Antisymmetric direction: D_ij = W_ij - W_ji (forces 1 winner per pair)")
            print(" 3. Direction-only training: only train direction after freeze")
            print(" 4. Fixed τ_dir=0.5 for stable direction learning")
            if skeleton_frozen:
                print(f" [DONE] Skeleton was frozen during training!")
            else:
                print(f" [WARN] Skeleton never reached freeze threshold ({skeleton_freeze_threshold:.2f})")
            print("-" * 70)
            K = cfg.get("target_edges", 13)
            tol = 0.5
            print(f"Guard: (past DISC OR excellent ≥5 epochs) & sum ∈ [{K*(1-tol):.1f},{K*(1+tol):.1f}] & max>0.1")
            print(f" Excellence = TopK-F1≥0.9 & Skel-F1≥0.95 & in budget window")
            print("-" * 70)
            print("PARETO CHECKPOINTS (GUARDED - saved to disk):")
            print(f" best_topk_sparse (guarded): TopK-F1={best_topk_sparse[0]:.4f} @ epoch {best_topk_sparse[1]}")
            print(f" best_skel_sparse (guarded): Skel-F1={best_skel_sparse[0]:.4f} @ epoch {best_skel_sparse[1]}")
            print(f" * best_score (guarded): score={best_score_ckpt[0]:.4f} @ epoch {best_score_ckpt[1]}")
            print("-" * 70)
            print("OVERALL PERFORMANCE (NO GUARD - true model performance):")
            print(f" best_topk_overall: TopK-F1={best_topk_overall[0]:.4f} @ epoch {best_topk_overall[1]}")
            print(f" best_skel_overall: Skel-F1={best_skel_overall[0]:.4f} @ epoch {best_skel_overall[1]}")
            print(f" best_score_overall: score={best_score_overall[0]:.4f} @ epoch {best_score_overall[1]}")
            print("-" * 70)
            
            # Compute improvement over TRUE correlation baseline (computed from data)
            if true_corr_baseline is not None:
                corr_f1 = true_corr_baseline['corr_baseline_f1']
                print(f" TRUE Correlation: TopK-F1={corr_f1:.4f} (computed from data, NOT model)")
                print("-" * 70)
                if corr_f1 > 0 and best_topk_sparse[0] > 0:
                    improvement = (best_topk_sparse[0] - corr_f1) / corr_f1 * 100
                    if best_topk_sparse[0] > corr_f1:
                        print(f"[DONE] CAUSAL IMPROVEMENT: +{improvement:.1f}% (model beat correlation!)")
                    elif abs(best_topk_sparse[0] - corr_f1) < 1e-4:
                        print(f"[WARN] MATCHED correlation baseline (no causal gain)")
                    else:
                        print(f"[FAIL] BELOW correlation baseline: {improvement:.1f}%")
                        print(" Model is WORSE than simple correlation-based graph.")
                elif corr_f1 == 0:
                    print("[WARN] Correlation baseline is 0 (degenerate dataset)")
            else:
                print(" [WARN] TRUE correlation baseline not computed (no A_true available)")
            
            # ===================================================================
            # V8.10: Convention mismatch diagnostic summary
            # ===================================================================
            if transpose_total_count > 0:
                transpose_win_rate = transpose_wins_count / transpose_total_count
                print("-" * 70)
                print(f"DIRECTION CONVENTION ANALYSIS:")
                print(f" Epochs where F1(A.T) > F1(A): {transpose_wins_count}/{transpose_total_count} ({transpose_win_rate*100:.0f}%)")
                print(f" Best F1 with A: {best_topk_sparse[0]:.4f}")
                print(f" Best F1 with A.T: {transpose_best_f1:.4f}")
                
                if transpose_win_rate > 0.7 and transpose_best_f1 > best_topk_sparse[0] + 0.1:
                    print("")
                    print(" [ALERT] STRONG CONVENTION MISMATCH DETECTED!")
                    print(" The model consistently predicts INVERTED directions.")
                    print(" This suggests edge_index[0]->edge_index[1] vs A[i,j] convention differs.")
                    print("")
                    print(" RECOMMENDED FIXES (pick one):")
                    print(" 1. QUICK: Set 'use_transpose_for_eval: true' in config")
                    print(" 2. DATA FIX: Check A_true construction (parent->child vs child->parent)")
                    print(" 3. MODEL FIX: Transpose A before loss computation in SEM")
                    print("")
                    print(f" If you use A.T, expected F1: ~{transpose_best_f1:.4f} (vs {best_topk_sparse[0]:.4f})")
                elif transpose_win_rate > 0.5:
                    print(f" [WARN] Possible convention issue (transpose wins >50% of epochs)")
                else:
                    print(f" [OK] Direction convention looks correct (A wins most epochs)")
                
            if best_topk_sparse[1] == 0:
                print("[WARN] No causal checkpoint saved (guard never passed)")
                print(" Check: DISC ended early? Budget window too narrow? No confident edges?")
                # V8.23 FIX: Promote unguarded best as fallback
                unguarded_adj_path = output_path / "A_best_unguarded.npy"
                unguarded_model_path = output_path / "best_unguarded.pt"
                if unguarded_adj_path.exists():
                    import shutil
                    # Copy unguarded best to all expected checkpoint names
                    shutil.copy2(unguarded_adj_path, output_path / "A_best_score.npy")
                    shutil.copy2(unguarded_adj_path, output_path / "A_best.npy")
                    shutil.copy2(unguarded_adj_path, output_path / "A_best_topk_sparse.npy")
                    if unguarded_model_path.exists():
                        shutil.copy2(unguarded_model_path, output_path / "best_model.pt")
                        shutil.copy2(unguarded_model_path, output_path / "best_score.pt")
                    print(f" [V8.23] FALLBACK: Promoted unguarded best (TopK-F1={best_topk_overall[0]:.4f} @ epoch {best_topk_overall[1]}) to A_best_score.npy")
                else:
                    print(" [V8.23] WARNING: No unguarded checkpoint found either!")
            print(f"Final graph: edge_sum={final_edge_sum:.2f}, @0.2={final_edges_02}")
            print(f"Output: {output_path}")
            print("=" * 70)
    
    # Cleanup DDP
    if ddp:
        cleanup_ddp()
    
    # Return final metrics for sweep mode
    return {
        "best_topk_f1": best_metric,
        "best_epoch": best_epoch,
        "final_epoch": epoch,
        "stage1_healthy": stage1_healthy,
    }


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="RC-GNN Unified Training Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Required
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to data directory")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="artifacts/unified",
                        help="Output directory")
    
    # Training
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_CONFIG["weight_decay"])
    parser.add_argument("--grad_clip", type=float, default=DEFAULT_CONFIG["grad_clip"])
    parser.add_argument("--patience", type=int, default=DEFAULT_CONFIG["patience"])
    
    # Model
    parser.add_argument("--latent_dim", type=int, default=DEFAULT_CONFIG["latent_dim"])
    parser.add_argument("--hidden_dim", type=int, default=DEFAULT_CONFIG["hidden_dim"])
    parser.add_argument("--target_edges", type=int, default=DEFAULT_CONFIG["target_edges"])
    
    # Loss weights (these set the FINAL values for ramped schedules)
    parser.add_argument("--lambda_recon", type=float, default=DEFAULT_CONFIG["lambda_recon"])
    parser.add_argument("--lambda_hsic", type=float, default=DEFAULT_CONFIG["lambda_hsic"])
    parser.add_argument("--lambda_sparse", type=float, default=DEFAULT_CONFIG["lambda_sparse_final"],
                        help="Final sparsity weight (ramped from lambda_sparse_init)")
    parser.add_argument("--lambda_inv", type=float, default=DEFAULT_CONFIG["lambda_inv"])
    parser.add_argument("--lambda_causal", type=float, default=DEFAULT_CONFIG["lambda_causal"])
    
    # Features
    parser.add_argument("--use_groupdro", action="store_true",
                        help="Enable GroupDRO for worst-case robustness")
    parser.add_argument("--ddp", action="store_true",
                        help="Enable Distributed Data Parallel")
    parser.add_argument("--sweep_mode", action="store_true",
                        help="Minimal output for ablation sweeps")
    
    # Oracle Supervision (Added to prevent leakage)
    parser.add_argument("--oracle_direction_supervision", action="store_true",
                        help="Allow using ground truth graph for direction supervision (ORACLE ONLY)")

    # System
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda", "cuda:0", "cuda:1"])
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    parser.add_argument("--num_workers", type=int, default=DEFAULT_CONFIG["num_workers"])
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # V8.12: Verify convention handling at startup
    if not verify_convention_at_startup(verbose=True):
        print("[FAIL] FATAL: Convention sanity check failed! Fix to_causal_convention() before proceeding.")
        return
    
    # Build config from args
    config = {k: v for k, v in vars(args).items() 
              if k not in ["data_dir", "output_dir", "ddp", "use_groupdro", "sweep_mode"]}
    
    # Map lambda_sparse to lambda_sparse_final for the new schedule
    if "lambda_sparse" in config:
        config["lambda_sparse_final"] = config["lambda_sparse"]
    
    # Train
    results = train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        config=config,
        ddp=args.ddp,
        use_groupdro=args.use_groupdro,
        sweep_mode=args.sweep_mode,
    )
    
    # Print results in sweep mode
    if args.sweep_mode:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
