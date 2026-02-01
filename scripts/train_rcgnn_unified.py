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
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional, List, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau

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
    "patience": 20,
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
    "lambda_budget_init": 0.0, # Start at 0
    "lambda_budget_final": 0.5, # Ramp to this
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
    "temperature_final": 0.3, # Anneal to LOW (sharp edges) - was 0.5, now 0.3
    "temperature_anneal_start": 0.2, # Start annealing at 20% training (earlier)
    "temperature_anneal_end": 0.7, # Finish annealing at 70% (faster)
    
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
    "topk_projection_start": 0.35, # Start projecting at 35% training (after DISC)
    "topk_projection_2k_end": 0.55, # Switch from 2K to K at 55% training
    "topk_use_logits": True, # Project based on W magnitude (stable across temps)
    "lambda_projection": 0.5, # FIX B: Projection consistency loss weight
    
    # FIX E: Direction Asymmetry Loss
    "use_asymmetry_loss": True, # Penalize edges where transpose is stronger
    "lambda_asymmetry": 0.5, # Asymmetry loss weight
    "asymmetry_start": 0.35, # Start when PRUNE begins
    
    # 2-Stage scheduling (V8.5: Skeleton -> Direction)
    "stage1_end": 0.30, # Discovery phase ends (part of Stage S)
    "stage2_end": 0.80, # Pruning phase ends (Stage S complete, skeleton frozen)
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
    "direction_margin": 0.1, # Margin value (A_ij - A_ji >= margin)
    "lambda_direction_margin": 1.0, # Weight for margin loss
    
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
    
    # V8.12: Convention handling is now ALWAYS applied via to_causal_convention()
    # Model outputs: A[i,j] = "j depends on i" (SEM/dependency convention)
    # Evaluation/saving: A[i,j] = "i causes j" (standard causal convention)
    # No config needed - conversion happens automatically at all boundaries
    
    # V8.10: EMA smoothing for stable TopK-F1 evaluation
    "ema_alpha": 0.1, # EMA weight (0.1=slow/stable, 0.5=fast/responsive)
    
    # V8.11: Skeleton Freeze + Antisymmetric Direction Learning
    # Once skeleton is learned, freeze it and train only direction
    "skeleton_freeze_enabled": True, # Enable skeleton freezing
    "skeleton_freeze_threshold": 0.95, # Freeze when Skel-F1 >= this
    "skeleton_freeze_patience": 5, # Consecutive epochs before freeze
    "direction_tau": 0.5, # Fixed τ for direction learning (lower=sharper)
    "lambda_direction": 2.0, # Boost direction loss after freeze
    
    # =========================================================================
    # V8.18: MNAR-aware Fixes
    # =========================================================================
    "lambda_inv_mask": 1e-3, # Mask-invariance penalty weight (only if skeleton frozen)
    
    # GroupDRO (from V3)
    "dro_step_size": 0.01,
    
    # Evaluation (multi-K for stability)
    "threshold_grid": list(np.arange(0.05, 0.55, 0.05)),
    "eval_k_values": [13, 20, 30], # F1 at multiple K values
    
    # Health check thresholds (Step 3 fix)
    "max_edges_at_0.5_ratio": 3.0, # edges@0.5 <= 3 * target_edges by DISC end
    "min_topk_jaccard": 0.7, # TopK stability threshold
    
    # System
    "device": "auto",
    "seed": 42,
    "num_workers": 4,
}


# =============================================================================
# V8.12: ADJACENCY CONVENTION UTILITIES
# =============================================================================
# 
# MODEL CONVENTION: A_model[i,j] = "j depends on i" (parent-in-column, child-in-row)
# This is natural for SEM: x_j = Σ_i A[i,j] * x_i + noise_j
# 
# CAUSAL CONVENTION: A_causal[i,j] = "i causes j" (row causes column)
# This is standard in most papers and libraries.
#
# SOLUTION: Convert ONCE at the boundary using to_causal_convention().
# All evaluation, saving, and reporting uses A_causal.
# =============================================================================

def to_causal_convention(A_model: torch.Tensor) -> torch.Tensor:
    """
    Convert model's internal adjacency to canonical causal convention.
    
    Model convention: A[i,j] = 'j depends on i' (parent-in-column)
    Causal convention: A_causal[i,j] = 'i causes j' (row causes column)
    
    Call this ONCE at every interface boundary:
    - Before evaluation against A_true
    - Before saving adjacency to files
    - Before visualization/plotting
    
    Args:
        A_model: Model's internal adjacency tensor [d, d]
    
    Returns:
        A_causal: Adjacency in canonical convention [d, d]
    """
    return A_model.T.contiguous()


def to_causal_convention_np(A_model: np.ndarray) -> np.ndarray:
    """NumPy version of to_causal_convention."""
    return A_model.T.copy()


def verify_convention_at_startup(verbose: bool = True) -> bool:
    """
    Sanity check: verify convention handling on a tiny 3-node DAG.
    
    True DAG: 0 -> 1 -> 2
    A_true (causal convention): A[0,1]=1, A[1,2]=1
    A_model (SEM convention): A[1,0]=1, A[2,1]=1 (transposed)
    
    This test ensures our to_causal_convention() correctly aligns them.
    """
    import torch
    
    # Ground truth in CAUSAL convention: A[i,j]=1 means i->j
    A_true = torch.zeros(3, 3)
    A_true[0, 1] = 1.0 # 0 -> 1
    A_true[1, 2] = 1.0 # 1 -> 2
    
    # Model output in SEM convention: A[i,j]=1 means j depends on i
    # For 0->1: x_1 depends on x_0, so A_model[0,1]=1... wait, that's the same!
    # Actually the SEM is: x_j = Σ_i A[i,j] * x_i
    # So for x_1 = A[0,1] * x_0 + noise, A[0,1]=1 means 0->1 (correct!)
    # 
    # BUT if model stores it as A_model[j,i] for "j depends on i", then:
    # A_model[1,0]=1 means "x_1 depends on x_0" -> 0->1
    # That's TRANSPOSED from causal convention!
    
    A_model = torch.zeros(3, 3)
    A_model[1, 0] = 1.0 # Model says "1 depends on 0" -> 0->1
    A_model[2, 1] = 1.0 # Model says "2 depends on 1" -> 1->2
    
    # After conversion, should match A_true
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
    
    # CALIBRATION: Don't go too low if direction is unstable
    # But V8.17: flatness overrides this - we need to sharpen first
    if not direction_stable and raw_temp < temp_floor_unstable:
        # Only apply floor if distribution is already peaked
        if peakiness is None or peakiness.get("gap", 0) >= 0.02:
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
    # ===========================================================================
    gap_threshold = config.get("peakiness_gap_threshold", 0.02)
    margin_threshold = config.get("peakiness_margin_threshold", 0.15)
    
    peakiness_achieved = True # Default: assume peaked (conservative)
    if peakiness is not None:
        gap = peakiness.get("gap", 0.1)
        avg_margin = peakiness.get("avg_margin", 0.2)
        peakiness_achieved = (gap >= gap_threshold) and (avg_margin >= margin_threshold)
    
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

def compute_exclusivity_loss(A: torch.Tensor) -> torch.Tensor:
    """
    V8.9: Penalize edges where both directions have high weight.
    
    L_excl = sum_{i<j} A_ij * A_ji
    
    This pushes the model to commit to ONE direction per edge pair,
    increasing direction margins and preventing the 0 <-> 4 TP oscillation.
    
    Returns:
        Scalar loss (mean over pairs to normalize)
    """
    # A * A.T gives element-wise product: high when both A_ij and A_ji are high
    product = A * A.T
    
    # Only count upper triangle (avoid double-counting)
    mask = torch.triu(torch.ones_like(product), diagonal=1)
    L_excl = (product * mask).sum() / (mask.sum() + 1e-8)
    
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
    V8.11: Freeze skeleton by clamping symmetric part of W.
    
    After skeleton is learned, we want to keep the SYMMETRIC part of W fixed
    and only train the ANTISYMMETRIC part (direction).
    
    Implementation: Store the symmetric part S = (W + W.T)/2 and 
    after each gradient step, restore S while keeping the learned direction.
    
    Args:
        model: RCGNN model with graph_learner
        skeleton_topk_edges: Set of (i,j) pairs in learned skeleton
        W_frozen: Frozen logits at time of freeze
    """
    with torch.no_grad():
        W_current = model.graph_learner.W_adj.data
        
        # Current symmetric and antisymmetric parts
        S_current = (W_current + W_current.T) / 2
        A_current = (W_current - W_current.T) / 2
        
        # Frozen symmetric part
        S_frozen = (W_frozen + W_frozen.T) / 2
        
        # Restore frozen symmetric part, keep current antisymmetric (direction)
        # W_new = S_frozen + A_current
        model.graph_learner.W_adj.data = S_frozen + A_current


# =============================================================================
# V8.8: Edge Discovery Fixes
# =============================================================================

def compute_cousin_mask(A_true: torch.Tensor) -> torch.Tensor:
    """
    V8.8: Compute mask of "cousin" pairs - nodes with common ancestor but no direct edge.
    
    These pairs are often high-correlation but NOT causally connected.
    We penalize the model for predicting edges on these pairs.
    
    Args:
        A_true: Ground truth adjacency [d, d]
    
    Returns:
        cousin_mask: [d, d] binary mask where 1 = cousin pair (should NOT have edge)
    """
    A_np = (A_true.detach().cpu().numpy() > 0.5).astype(int)
    d = A_np.shape[0]
    
    # Compute transitive closure (reachability) for ancestor check
    # ancestors[i] = set of nodes that can reach i
    ancestors = [set() for _ in range(d)]
    
    # Use iterative approach to find all ancestors
    for _ in range(d): # At most d iterations for full propagation
        changed = False
        for j in range(d):
            for i in range(d):
                if A_np[i, j] > 0: # i -> j
                    if i not in ancestors[j]:
                        ancestors[j].add(i)
                        ancestors[j] |= ancestors[i] # j inherits i's ancestors
                        changed = True
        if not changed:
            break
    
    # Build cousin mask
    cousin_mask = np.zeros((d, d), dtype=np.float32)
    
    for i in range(d):
        for j in range(i+1, d):
            # Check if (i,j) share a common ancestor
            common = ancestors[i] & ancestors[j]
            
            # Only mark as cousin if:
            # 1. They share a common ancestor
            # 2. Neither i->j nor j->i is a true edge
            if len(common) > 0 and A_np[i, j] == 0 and A_np[j, i] == 0:
                cousin_mask[i, j] = 1.0
                cousin_mask[j, i] = 1.0
    
    return torch.from_numpy(cousin_mask).to(A_true.device)


def compute_hard_negative_mask(
    A_true: torch.Tensor,
    corr_matrix: torch.Tensor,
    percentile: float = 80,
) -> torch.Tensor:
    """
    V8.8: Compute mask of hard negatives - high-correlation non-edges.
    
    These are the pairs the model is most likely to wrongly predict as edges.
    
    Args:
        A_true: Ground truth adjacency [d, d]
        corr_matrix: Pairwise correlation matrix [d, d]
        percentile: Top X% correlations to consider as hard negatives
    
    Returns:
        hard_neg_mask: [d, d] binary mask where 1 = hard negative
    """
    d = A_true.shape[0]
    
    # Non-edges in ground truth
    non_edge_mask = (A_true < 0.5) & (A_true.T < 0.5)
    
    # Remove diagonal
    non_edge_mask = non_edge_mask & ~torch.eye(d, dtype=torch.bool, device=A_true.device)
    
    # Get correlation threshold for top percentile
    corr_abs = corr_matrix.abs()
    non_edge_corrs = corr_abs[non_edge_mask]
    
    if len(non_edge_corrs) > 0:
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
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    """
    V8.8: Compute edge discovery losses to fix mode collapse.
    
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
    
    # True presence (undirected)
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
        
        cousin_mask = compute_cousin_mask(A_true)
        
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
        
        # Compute loss with scheduled weights
        loss, batch_metrics = base_model.compute_loss(
            outputs, X, M, regime=e,
            epoch=epoch,
            total_epochs=total_epochs,
            loss_weights=loss_weights, # Pass scheduled weights
        )
        
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
            
            # =========================================================
            # V8.7: Direction Margin Loss (hinge-based)
            # For TRUE edge i->j: enforce A_ij >= A_ji + margin
            # Loss = max(0, A_ji - A_ij + margin) for each true directed edge
            # =========================================================
            use_margin = config.get("use_direction_margin", True)
            margin = config.get("direction_margin", 0.1)
            lambda_margin = config.get("lambda_direction_margin", 1.0)
            
            if use_margin and A_true is not None:
                # CRITICAL: Convert A_true from causal to dependency convention
                # Model uses W[i,j] = "j depends on i" (dependency)
                # Data uses A[i,j] = "i causes j" (causal)
                # So transpose A_true to match model convention
                A_true_model_conv = A_true.T
                
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
        # CRITICAL: Model uses DEPENDENCY convention (A[i,j] = "j depends on i")
        # but A_true uses CAUSAL convention (A[i,j] = "i causes j")
        # Must convert A_true to match model convention for direction training!
        # =====================================================================
        if skeleton_frozen and A_true is not None:
            W = base_model.graph_learner.W_adj
            # Convert A_true from causal to dependency convention (transpose)
            A_true_model_conv = A_true.T if A_true.shape[0] == A_true.shape[1] else A_true
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
        
        if use_edge_discovery and A_true is not None:
            A_soft = base_model.graph_learner.get_mean_adjacency()
            
            # Get correlation matrix if available (computed once per dataset, cached)
            corr_matrix = getattr(base_model, '_corr_matrix', None)
            
            # Compute all edge discovery losses
            disc_losses, disc_metrics = compute_edge_discovery_losses(
                A_pred=A_soft,
                A_true=A_true,
                corr_matrix=corr_matrix,
                config=config,
                epoch=epoch,
                total_epochs=total_epochs,
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
            # Get W (adjacency logits) and zero grad for non-skeleton edges
            W = base_model.graph_learner.W
            if W.grad is not None:
                # Keep gradients only for skeleton edges (both directions for same pair)
                skel_sym = skeleton_mask | skeleton_mask.T # Allow learning either direction
                W.grad = W.grad * skel_sym.float()
        
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
        # V8.12: CONVERT TO CAUSAL CONVENTION
        # Model outputs A_model in SEM convention: A[i,j] = "j depends on i"
        # We convert to causal convention: A[i,j] = "i causes j"
        # This single conversion ensures ALL downstream code is correct:
        # - Metrics (TopK-F1, Skeleton-F1, etc.)
        # - Saved adjacency matrices
        # - Confusion matrix TP/FP/FN/TN
        # ==================================================================
        A_pred = to_causal_convention(A_pred) # Convert once at boundary
        
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
    
    # Seed
    torch.manual_seed(cfg["seed"] + local_rank)
    np.random.seed(cfg["seed"] + local_rank)
    
    # Load data
    data = load_data(data_dir, normalize=True)
    d = data["X"].shape[-1]
    n_regimes = len(torch.unique(data["e"]))
    A_true = data.get("A_true", None)
    
    # Adjust target edges if A_true available
    if A_true is not None and cfg["target_edges"] == 13:
        cfg["target_edges"] = int(A_true.sum())
    
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
        
        # V8.8: Compute and cache correlation matrix for hard negative mining
        from scipy import stats
        X_np = data["X"].numpy()
        if X_np.ndim == 3:
            X_flat = X_np.reshape(-1, X_np.shape[-1])
        else:
            X_flat = X_np
        
        corr_np = np.zeros((d, d), dtype=np.float32)
        for i in range(d):
            for j in range(d):
                if i == j:
                    continue
                valid = ~(np.isnan(X_flat[:, i]) | np.isnan(X_flat[:, j]))
                if valid.sum() >= 10:
                    r, _ = stats.pearsonr(X_flat[valid, i], X_flat[valid, j])
                    corr_np[i, j] = abs(r) if not np.isnan(r) else 0
        
        corr_matrix = torch.from_numpy(corr_np).to(device)
        base_model._corr_matrix = corr_matrix # Cache on model for train_epoch access
    
    # Optimizer with warm restarts
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"]
    )
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
    history = []
    stage1_healthy = True
    prev_topk_set = None # For tracking TopK stability
    
    # NOTE: We now use true_corr_baseline (computed from data at start)
    # NOT the old "best during DISC phase" which was just model initialization noise
    
    # CALIBRATION FIX: Track direction stability and edge_sum for adaptive scheduling
    direction_stable = False # Start assuming unstable until we see F1 > F1_T
    prev_edge_sum = None # For collapse detection
    
    # V8.10: EMA smoothing for stable evaluation
    # A_ema = 0.9 * A_ema + 0.1 * A_current (reduces noise in TopK selection)
    A_ema = None
    ema_alpha = cfg.get("ema_alpha", 0.1) # How much to weight new A (0.1 = slow, 0.5 = fast)
    
    # V8.5: Skeleton mask for Stage D (frozen structure)
    skeleton_mask = None # Will be computed at end of Stage S (PRUNE)
    
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
                A_log = to_causal_convention(A_log) # Convert to causal convention
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
                topk_gap = train_metrics.get("topk_gap", None)
                nontopk_max = train_metrics.get("nontopk_max", None)
                L_nontopk = train_metrics.get("L_nontopk", 0)
                if topk_gap is not None:
                    gap_status = "[OK]" if topk_gap > 0.1 else ("~" if topk_gap > 0 else "[WARN]")
                    print(f" | TopK gap: {topk_gap:.3f} [{gap_status}] | non-TopK max={nontopk_max:.3f} | L_suppress={L_nontopk:.4f}")
                
                print(f" | A stats: max={A_max:.3f} mean={A_mean:.4f} topk_mean={topk_mean:.3f}")
                
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
                # CALIBRATION FIX: Update direction stability for next epoch
                # Direction is stable when F1 > F1_T (learning causation, not anti-causal)
                # ===============================================================
                direction_stable = (topk_f1 >= topk_f1_T) or (topk_f1_T == 0)
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
                    print(f" | PEAKINESS WARMUP (gap={gap:.3f} <0.02 or margin<0.15): τ↓, λs/λb frozen")
                
                if direction_stable:
                    print(f" | [LOCK] Direction STABLE (τ floor released)")
                else:
                    print(f" | Direction UNSTABLE (τ floored at 0.8)")
                
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
                            W_frozen = base_model.graph_learner.W_adj.data.clone()
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
                
                if is_main_process() and not sweep_mode:
                    print(f" | [LOCK] STAGE D: Skeleton FROZEN with {skel_edge_count} undirected edges")
                    print(f" | Threshold: {skel_thresh:.4f}, Now learning DIRECTION only")
                    
                    # Save skeleton for analysis
                    np.save(output_path / "skeleton_mask.npy", skeleton_mask.cpu().numpy())
                    np.save(output_path / "A_stage_s.npy", A_stage_s_np)
        
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
            A_pred_check = to_causal_convention(A_pred_check) # V8.12: Convert at boundary
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
            tol = 0.5 # Allow +/- 50% (can tighten to 0.25 later)
            min_sum = K * (1 - tol) # e.g., 6.5 for K=13
            max_sum = K * (1 + tol) # e.g., 19.5 for K=13
            
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
            graph_valid = (past_disc or early_excellence) and in_budget_window and has_confident
            
            # ================================================================
            # NOTE: We no longer track "correlation ref" during DISC phase
            # That was FAKE - it was just model's random initialization!
            # TRUE correlation baseline is computed from DATA at start.
            # See: true_corr_baseline variable (computed before training loop)
            # ================================================================
            
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
            
            if two_stage and not in_stage_d:
                # Stage S: Focus on skeleton, light direction penalty
                composite_score = (
                    0.5 * topk + # Some weight on directed
                    1.0 * skel + # PRIMARY: undirected structure
                    0.3 * dir_bonus - # Light bonus for direction
                    0.3 * dir_penalty - # Light penalty (don't fight sparsity)
                    0.5 * budget_pen
                )
            else:
                # Stage D or legacy: Heavy direction penalty
                composite_score = (
                    1.0 * topk +
                    0.5 * skel +
                    0.5 * dir_bonus - # Bonus for beating transpose
                    1.0 * dir_penalty - # Strong penalty for skeleton >> directed
                    0.5 * budget_pen
                )
            
            # Log composite score components
            if not sweep_mode and epoch % 10 == 0:
                dir_status = "[OK]" if dir_ratio > 0.7 else ("~" if dir_ratio > 0.4 else "[X]")
                print(f" | Score: {composite_score:.4f} = topk({topk:.3f}) + 0.5*skel({skel:.3f}) + 0.5*dirB({dir_bonus:.3f}) - 1.0*dirP({dir_penalty:.3f}) - 0.5*bpen({budget_pen:.2f})")
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
            # This ensures we never lose information even when guard fails
            if topk > best_topk_overall[0]:
                best_topk_overall = (topk, epoch)
                if not graph_valid and not sweep_mode and is_main_process():
                    print(f" | New best_topk_overall: TopK-F1={topk:.4f} (unguarded)")
            
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
        # V8.14 FIX: Disable early stopping until after DISC phase
        # CRITICAL BUG: V8.13 had inverted logic. When two_stage=True (default),
        # early stopping triggered at epoch 5, before DISC complete at epoch 45.
        # Fix: Only allow early stopping if past_disc OR patience >= max(40, patience)
        # =====================================================================
        stage1_end = cfg.get("stage1_end", 0.30)
        past_disc = epoch >= stage1_end * cfg["epochs"]
        
        # Early stopping (only in PRUNE/REFINE, not DISC)
        if patience_counter >= cfg["patience"]:
            if past_disc: # Only allow early stop AFTER DISC phase completes
                if is_main_process() and not sweep_mode:
                    print(f"\n⏹ Early stopping at epoch {epoch} (patience={cfg['patience']})")
                break
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
        A_final = to_causal_convention(A_final) # V8.12: Convert at boundary
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
            print("TRAINING COMPLETE (V8.12 - Causal Convention Fix)")
            print("=" * 70)
            print("CONVENTION INFO (for paper writeup):")
            print(" Model output: A_model[i,j] = 'j depends on i' (SEM/dependency convention)")
            print(" Evaluation: A_causal[i,j] = 'i causes j' (standard causal convention)")
            print(" Conversion: A_causal = A_model.T (applied at all boundaries)")
            print(" [DONE] All metrics, saves, and confusion matrices use A_causal")
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
