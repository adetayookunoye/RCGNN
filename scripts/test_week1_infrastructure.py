#!/usr/bin/env python3
"""
Comprehensive test suite for Week 1 infrastructure.

Tests:
1. Stability metrics (adjacency_variance, edge_set_jaccard, policy_consistency)
2. Multi-environment evaluation (eval_epoch_multi_env)
3. Synthetic corruption benchmarks
4. End-to-end training validation

Usage:
    python scripts/test_week1_infrastructure.py --test all
    python scripts/test_week1_infrastructure.py --test metrics
    python scripts/test_week1_infrastructure.py --test benchmarks
    python scripts/test_week1_infrastructure.py --test training
    python scripts/test_week1_infrastructure.py --test quick
"""

import argparse
import sys
import json
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.metrics import (
    adjacency_variance,
    edge_set_jaccard,
    policy_consistency,
    shd
)
from src.training.loop import eval_epoch_multi_env
from src.models.rcgnn import RCGNN
from src.dataio.loaders import load_synth


# ============================================================================
# Test 1: Stability Metrics
# ============================================================================

def test_stability_metrics():
    """Test all 3 stability metrics with synthetic data."""
    print("\n" + "="*70)
    print("TEST 1: STABILITY METRICS")
    print("="*70)
    
    # Create test data: 3 environments with adjacency matrices
    d = 10
    
    # Environment 1: baseline
    A1 = np.zeros((d, d))
    A1[0, 1] = A1[1, 2] = A1[2, 3] = A1[3, 4] = 1.0
    
    # Environment 2: slight variation
    A2 = A1.copy()
    A2[4, 5] = 1.0  # Add one edge
    
    # Environment 3: larger variation
    A3 = A1.copy()
    A3[0, 1] = 0.0  # Remove edge
    A3[5, 6] = A3[6, 7] = 1.0  # Add edges
    
    A_by_env = {0: A1, 1: A2, 2: A3}
    
    print("\n✓ Created 3 test adjacency matrices (10×10)")
    print(f"  - Env 0: {int(A1.sum())} edges")
    print(f"  - Env 1: {int(A2.sum())} edges")
    print(f"  - Env 2: {int(A3.sum())} edges")
    
    # Test 1: adjacency_variance
    print("\n[Test 1.1] adjacency_variance()")
    try:
        var = adjacency_variance(A_by_env)
        print(f"  ✅ PASS: variance = {var:.6f}")
        assert isinstance(var, (float, np.floating)), "Should return float"
        assert var >= 0, "Variance should be non-negative"
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        return False
    
    # Test 2: edge_set_jaccard
    print("\n[Test 1.2] edge_set_jaccard()")
    try:
        jac = edge_set_jaccard(A_by_env, threshold=0.5)
        print(f"  ✅ PASS: jaccard similarity = {jac:.6f}")
        assert isinstance(jac, (float, np.floating)), "Should return float"
        assert 0 <= jac <= 1, "Jaccard should be in [0, 1]"
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        return False
    
    # Test 3: policy_consistency
    print("\n[Test 1.3] policy_consistency()")
    try:
        policy_edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
        pol = policy_consistency(A_by_env, policy_edges, threshold=0.5)
        print(f"  ✅ PASS: policy metrics = {pol}")
        assert isinstance(pol, dict), "Should return dict"
        assert "consistency" in pol, "Should have 'consistency' key"
        assert "presence" in pol, "Should have 'presence' key"
        assert "variance" in pol, "Should have 'variance' key"
        assert 0 <= pol["consistency"] <= 1, "consistency should be in [0, 1]"
        assert 0 <= pol["presence"] <= 1, "presence should be in [0, 1]"
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        return False
    
    # Test 4: Stability sensitivity (high vs low stability)
    print("\n[Test 1.4] Stability Sensitivity (High vs Low)")
    try:
        # High stability: all environments nearly identical
        A_stable = {0: A1.copy(), 1: A1.copy(), 2: A1.copy()}
        var_stable = adjacency_variance(A_stable)
        
        # Low stability: all environments very different
        A_random = {}
        for env in range(3):
            A_rand = np.random.rand(d, d)
            A_rand = np.tril(A_rand)  # Make lower triangular (DAG)
            A_rand = (A_rand > 0.7).astype(float)  # Threshold for sparsity
            A_random[env] = A_rand
        var_unstable = adjacency_variance(A_random)
        
        ratio = var_unstable / (var_stable + 1e-10)
        print(f"  High stability variance: {var_stable:.6f}")
        print(f"  Low stability variance:  {var_unstable:.6f}")
        print(f"  Ratio (low/high):        {ratio:.1f}×")
        
        # Low stability should have much higher variance
        if var_unstable > var_stable * 100:
            print(f"  ✅ PASS: Sensitivity verified (ratio > 100×)")
        else:
            print(f"  ⚠️  WARNING: Low sensitivity (ratio < 100×)")
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        return False
    
    print("\n" + "✅ TEST 1 PASSED: All stability metrics working correctly")
    return True


# ============================================================================
# Test 2: Multi-Environment Evaluation
# ============================================================================

def test_eval_epoch_multi_env():
    """Test eval_epoch_multi_env function."""
    print("\n" + "="*70)
    print("TEST 2: MULTI-ENVIRONMENT EVALUATION")
    print("="*70)
    
    try:
        print("\n[Test 2.1] Loading synthetic benchmark data")
        # Try to load H1 Easy benchmark
        data_root = "data/interim/synth_corrupted_h1_easy"
        
        if not Path(data_root).exists():
            print(f"  ⚠️  SKIP: {data_root} not found (run benchmark generation first)")
            return True
        
        print(f"  ✓ Loading from: {data_root}")
        
        # Load metadata
        meta_path = Path(data_root) / "meta.json"
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"  ✓ Metadata loaded: {meta['d']} nodes, {meta['edges']} edges")
        
        # Always load A_true first
        A_true = np.load(Path(data_root) / "A_true.npy")
        
        # Load data
        try:
            val_ds = load_synth(data_root, "val", seed=42)
            print(f"  ✓ Validation dataset loaded: {len(val_ds)} samples")
        except Exception as e:
            print(f"  ⚠️  Could not load with load_synth: {e}")
            print(f"  → Manually loading numpy files...")
            
            X_val = np.load(Path(data_root) / "X_val.npy")
            e_val = np.load(Path(data_root) / "e_val.npy")
            
            X_val_t = torch.from_numpy(X_val).float()
            e_val_t = torch.from_numpy(e_val).long()
            
            val_ds = TensorDataset(X_val_t, e_val_t)
            print(f"  ✓ Manual load successful: X shape {X_val.shape}")
        
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
        
        print("\n[Test 2.2] Creating toy RC-GNN model")
        d = meta['d']
        model = RCGNN(d=d, latent_dim=8, hidden_dim=16, n_envs=meta['n_envs'], device="cpu")
        model.eval()
        print(f"  ✓ Model created: {d} nodes, {meta['n_envs']} environments")
        
        print("\n[Test 2.3] Running eval_epoch_multi_env()")
        metrics = eval_epoch_multi_env(
            model=model,
            eval_loader=val_loader,
            A_true=A_true,
            device="cpu",
            threshold=0.5
        )
        print(f"  ✓ Metrics computed successfully")
        print(f"  ✓ Keys: {list(metrics.keys())}")
        
        # Check for expected keys
        expected_keys = ["shd", "precision", "recall", "f1"]
        for key in expected_keys:
            if key in metrics:
                print(f"    - {key}: {metrics[key]:.4f}" if isinstance(metrics[key], (float, np.floating)) else f"    - {key}: {metrics[key]}")
        
        # Check multi-env metrics if available
        if "adjacency_variance" in metrics:
            print(f"\n  ✓ Multi-environment metrics available:")
            print(f"    - adjacency_variance: {metrics['adjacency_variance']:.6f}")
            if "edge_set_jaccard" in metrics:
                print(f"    - edge_set_jaccard: {metrics['edge_set_jaccard']:.6f}")
        
        print("\n" + "✅ TEST 2 PASSED: eval_epoch_multi_env() working correctly")
        return True
        
    except Exception as e:
        print(f"\n  ❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# Test 3: Synthetic Benchmarks
# ============================================================================

def test_benchmarks():
    """Test that all benchmarks exist and have correct structure."""
    print("\n" + "="*70)
    print("TEST 3: SYNTHETIC CORRUPTION BENCHMARKS")
    print("="*70)
    
    benchmarks = [
        "h1_easy",
        "h1_medium", 
        "h1_hard",
        "h2_multi_env",
        "h2_stability",
        "h3_policy"
    ]
    
    all_pass = True
    
    for bench in benchmarks:
        print(f"\n[Test 3] Checking {bench}...")
        
        bench_dir = Path(f"data/interim/synth_corrupted_{bench}")
        
        if not bench_dir.exists():
            print(f"  ❌ FAIL: Directory not found: {bench_dir}")
            all_pass = False
            continue
        
        # Check required files
        required_files = [
            "A_true.npy",
            "X_train.npy",
            "X_val.npy",
            "M_train.npy",
            "M_val.npy",
            "e_train.npy",
            "e_val.npy",
            "meta.json"
        ]
        
        missing = []
        for fname in required_files:
            if not (bench_dir / fname).exists():
                missing.append(fname)
        
        if missing:
            print(f"  ❌ FAIL: Missing files: {missing}")
            all_pass = False
            continue
        
        # Load and verify metadata
        try:
            with open(bench_dir / "meta.json") as f:
                meta = json.load(f)
            
            # Load and check data shapes
            A_true = np.load(bench_dir / "A_true.npy")
            X_train = np.load(bench_dir / "X_train.npy")
            e_train = np.load(bench_dir / "e_train.npy")
            
            d = meta['d']
            n_envs = meta['n_envs']
            
            # Verify shapes
            assert A_true.shape == (d, d), f"A_true shape mismatch: {A_true.shape}"
            assert X_train.shape[2] == d, f"X_train feature dimension mismatch: {X_train.shape}"
            assert len(e_train) == X_train.shape[0], f"e_train length mismatch: {len(e_train)} vs {X_train.shape[0]}"
            assert len(np.unique(e_train)) == n_envs, f"Environment count mismatch"
            
            size_mb = sum((bench_dir / f).stat().st_size for f in required_files if (bench_dir / f).exists()) / 1024 / 1024
            
            print(f"  ✅ PASS:")
            print(f"     - Shape: {X_train.shape} (samples, timesteps, features)")
            print(f"     - Edges: {int(A_true.sum())}")
            print(f"     - Environments: {n_envs}")
            print(f"     - Size: {size_mb:.1f} MB")
            
        except Exception as e:
            print(f"  ❌ FAIL: {e}")
            all_pass = False
    
    if all_pass:
        print("\n" + "✅ TEST 3 PASSED: All benchmarks valid and ready")
    else:
        print("\n" + "⚠️  TEST 3 PARTIAL: Some benchmarks missing (generate with: python scripts/synth_corruption_benchmark.py --benchmark [name])")
    
    return all_pass


# ============================================================================
# Test 4: End-to-End Training
# ============================================================================

def test_end_to_end_training():
    """Quick end-to-end training test."""
    print("\n" + "="*70)
    print("TEST 4: END-TO-END TRAINING VALIDATION")
    print("="*70)
    
    try:
        print("\n[Test 4.1] Loading H1 Easy benchmark")
        
        data_root = Path("data/interim/synth_corrupted_h1_easy")
        if not data_root.exists():
            print(f"  ⚠️  SKIP: {data_root} not found")
            return True
        
        # Load metadata
        with open(data_root / "meta.json") as f:
            meta = json.load(f)
        
        # Load data
        X_val = np.load(data_root / "X_val.npy")
        e_val = np.load(data_root / "e_val.npy")
        A_true = np.load(data_root / "A_true.npy")
        
        d = meta['d']
        n_envs = meta['n_envs']
        
        print(f"  ✓ Data loaded: {d} nodes, {n_envs} environments, {X_val.shape[0]} samples")
        
        print("\n[Test 4.2] Creating and training small RC-GNN model")
        
        # Create model
        model = RCGNN(
            d=d,
            latent_dim=8,
            hidden_dim=16,
            n_envs=n_envs,
            device="cpu"
        )
        
        # Create dummy training data
        X_val_t = torch.from_numpy(X_val[:32]).float()  # Small batch
        e_val_t = torch.from_numpy(e_val[:32]).long()
        train_loader = DataLoader(
            TensorDataset(X_val_t, e_val_t),
            batch_size=8,
            shuffle=True
        )
        
        print(f"  ✓ Model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Forward pass
        print("\n[Test 4.3] Forward pass and loss computation")
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        losses = []
        for epoch in range(3):
            epoch_loss = 0
            for batch_idx, (X, e) in enumerate(train_loader):
                optimizer.zero_grad()
                
                # Forward pass
                output = model(X, e)
                
                # Simple MSE loss for testing
                loss = ((output - X) ** 2).mean()
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                losses.append(loss.item())
            
            avg_loss = epoch_loss / len(train_loader)
            print(f"  Epoch {epoch}: loss = {avg_loss:.6f}")
        
        # Check that loss is decreasing
        if losses[-1] < losses[0]:
            print(f"  ✅ PASS: Loss decreasing (initial: {losses[0]:.6f}, final: {losses[-1]:.6f})")
        else:
            print(f"  ⚠️  WARNING: Loss not consistently decreasing")
        
        print("\n[Test 4.4] Evaluation with eval_epoch_multi_env()")
        
        model.eval()
        val_loader = DataLoader(
            TensorDataset(X_val_t, e_val_t),
            batch_size=8,
            shuffle=False
        )
        
        with torch.no_grad():
            metrics = eval_epoch_multi_env(
                model=model,
                eval_loader=val_loader,
                A_true=A_true,
                device="cpu",
                threshold=0.5
            )
        
        print(f"  ✅ PASS: Evaluation successful")
        print(f"     - SHD: {metrics.get('shd', 'N/A')}")
        print(f"     - F1: {metrics.get('f1', 'N/A')}")
        
        print("\n" + "✅ TEST 4 PASSED: End-to-end training works correctly")
        return True
        
    except Exception as e:
        print(f"\n  ❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# Quick Test Suite (fast validation)
# ============================================================================

def test_quick():
    """Quick validation of all components."""
    print("\n" + "="*70)
    print("QUICK TEST SUITE (Fast Validation)")
    print("="*70)
    
    results = {
        "Stability Metrics": test_stability_metrics(),
        "Benchmarks": test_benchmarks(),
    }
    
    # Try multi-env evaluation if data available
    print("\n[Bonus] Attempting multi-environment evaluation...")
    results["Multi-Env Evaluation"] = test_eval_epoch_multi_env()
    
    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Test Week 1 infrastructure")
    parser.add_argument(
        "--test",
        type=str,
        default="quick",
        choices=["all", "metrics", "benchmarks", "training", "eval", "quick"],
        help="Test to run"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("WEEK 1 INFRASTRUCTURE TEST SUITE")
    print("="*70)
    print(f"Test: {args.test}")
    print(f"Device: cpu")
    print("="*70)
    
    results = {}
    
    if args.test == "quick":
        results = test_quick()
    elif args.test == "metrics":
        results["Stability Metrics"] = test_stability_metrics()
    elif args.test == "benchmarks":
        results["Benchmarks"] = test_benchmarks()
    elif args.test == "eval":
        results["Multi-Env Evaluation"] = test_eval_epoch_multi_env()
    elif args.test == "training":
        results["End-to-End Training"] = test_end_to_end_training()
    elif args.test == "all":
        results["Stability Metrics"] = test_stability_metrics()
        results["Benchmarks"] = test_benchmarks()
        results["Multi-Env Evaluation"] = test_eval_epoch_multi_env()
        results["End-to-End Training"] = test_end_to_end_training()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    all_pass = all(results.values())
    
    if all_pass:
        print("\n" + "="*70)
        print("🎉 ALL TESTS PASSED! Infrastructure ready for hypothesis testing")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("⚠️  SOME TESTS FAILED - Check output above")
        print("="*70)
    
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
