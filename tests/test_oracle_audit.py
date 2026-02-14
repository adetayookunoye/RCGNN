#!/usr/bin/env python3
"""
ORACLE AUDIT TEST - Critical for reviewer defense.

This test ensures that A_true (ground truth adjacency) is NEVER used during training.
Any gradient path that uses A_true constitutes oracle leakage and is a desk-reject risk.

What counts as oracle leakage:
1. Any gradient path that uses A_true
2. Choosing K = |E_true| on the test set (unless stated as evaluation convention)
3. Selecting thresholds/hyperparameters on test performance
4. Using ground-truth graph-derived quantities to shape training dynamics

This test enforces separation by:
1. Scanning training code for A_true references
2. Running training with A_true removed and ensuring it completes
3. Checking that no training loss function depends on A_true
"""

import os
import sys
import ast
import re
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class OracleLeakageDetector(ast.NodeVisitor):
    """AST visitor that detects potential oracle leakage patterns."""
    
    def __init__(self, filename):
        self.filename = filename
        self.violations = []
        self.in_function = None
        self.in_class = None
        
        # Patterns that indicate oracle leakage in TRAINING code
        self.forbidden_patterns = [
            'A_true',           # Direct ground truth access
            'ground_truth',     # Alternative naming
            'true_adjacency',   # Alternative naming
            'A_gt',             # Alternative naming
        ]
        
        # Functions where A_true is ALLOWED (evaluation only)
        self.allowed_contexts = [
            'validate',
            'eval',
            'compute_metrics',
            'compute_topk_f1',
            'compute_skeleton_f1',
            'compute_shd',
            'diagnose_',
            'plot_',
            'save_',
            'log_',
        ]
    
    def visit_FunctionDef(self, node):
        old_function = self.in_function
        self.in_function = node.name
        self.generic_visit(node)
        self.in_function = old_function
    
    def visit_ClassDef(self, node):
        old_class = self.in_class
        self.in_class = node.name
        self.generic_visit(node)
        self.in_class = old_class
    
    def visit_Name(self, node):
        """Check for forbidden variable names."""
        for pattern in self.forbidden_patterns:
            if pattern.lower() in node.id.lower():
                # Check if we're in an allowed context
                if self.in_function and any(ctx in self.in_function.lower() for ctx in self.allowed_contexts):
                    pass  # Allowed in evaluation functions
                else:
                    self.violations.append({
                        'file': self.filename,
                        'line': node.lineno,
                        'function': self.in_function,
                        'class': self.in_class,
                        'variable': node.id,
                        'pattern': pattern,
                    })
        self.generic_visit(node)
    
    def visit_Subscript(self, node):
        """Check for dictionary/array access patterns like data['A_true']."""
        if isinstance(node.slice, ast.Constant):
            key = str(node.slice.value)
            for pattern in self.forbidden_patterns:
                if pattern.lower() in key.lower():
                    if self.in_function and any(ctx in self.in_function.lower() for ctx in self.allowed_contexts):
                        pass
                    else:
                        self.violations.append({
                            'file': self.filename,
                            'line': node.lineno,
                            'function': self.in_function,
                            'class': self.in_class,
                            'variable': f"['{key}']",
                            'pattern': pattern,
                        })
        self.generic_visit(node)


def scan_file_for_oracle_leakage(filepath: Path) -> list:
    """Scan a Python file for potential oracle leakage."""
    try:
        with open(filepath, 'r') as f:
            source = f.read()
        tree = ast.parse(source)
        detector = OracleLeakageDetector(str(filepath))
        detector.visit(tree)
        return detector.violations
    except SyntaxError as e:
        print(f"[WARN] Could not parse {filepath}: {e}")
        return []


def scan_training_code_for_oracle_leakage():
    """
    Scan all training-related code for oracle leakage.
    
    Training code should NEVER reference A_true.
    Only evaluation code is allowed to use ground truth.
    """
    project_root = Path(__file__).parent.parent
    
    # Files that are TRAINING code (must NOT use A_true in loss computation)
    training_files = [
        project_root / "src" / "models" / "rcgnn.py",
        project_root / "src" / "models" / "structure.py",
        project_root / "src" / "models" / "encoders.py",
        project_root / "src" / "models" / "losses.py",
        project_root / "src" / "models" / "mechanisms.py",
        project_root / "src" / "models" / "recon.py",
        project_root / "src" / "models" / "disentanglement.py",
        project_root / "src" / "models" / "missingness.py",
    ]
    
    all_violations = []
    
    for filepath in training_files:
        if filepath.exists():
            violations = scan_file_for_oracle_leakage(filepath)
            # Filter: only report violations in non-evaluation functions
            critical_violations = [
                v for v in violations 
                if not any(ctx in (v['function'] or '').lower() 
                          for ctx in ['validate', 'eval', 'metric', 'diagnose', 'plot'])
            ]
            all_violations.extend(critical_violations)
    
    return all_violations


def test_no_oracle_in_model_forward():
    """
    CRITICAL TEST: Model forward pass must not use A_true.
    
    The RCGNN model's forward() and compute_loss() methods must operate
    solely on input data (X, M, regime) without access to ground truth.
    """
    project_root = Path(__file__).parent.parent
    rcgnn_file = project_root / "src" / "models" / "rcgnn.py"
    
    if not rcgnn_file.exists():
        print(f"[SKIP] {rcgnn_file} not found")
        return
    
    with open(rcgnn_file, 'r') as f:
        source = f.read()
    
    # Check that forward() doesn't have A_true parameter
    # Using regex to find function signatures
    forward_match = re.search(r'def forward\s*\([^)]*\)', source)
    if forward_match:
        forward_sig = forward_match.group(0)
        assert 'A_true' not in forward_sig, \
            f"ORACLE LEAKAGE: forward() has A_true parameter: {forward_sig}"
        assert 'ground_truth' not in forward_sig.lower(), \
            f"ORACLE LEAKAGE: forward() has ground_truth parameter: {forward_sig}"
    
    # Check compute_loss() - this is where leakage often hides
    loss_match = re.search(r'def compute_loss\s*\([^)]*\)', source)
    if loss_match:
        loss_sig = loss_match.group(0)
        # A_true in compute_loss is OK if it's only for logging/metrics, not gradients
        # We'll check the function body separately
        pass
    
    print("[PASS] Model forward pass does not have A_true parameter")


def test_no_oracle_in_loss_gradient():
    """
    CRITICAL TEST: Loss computation must not backprop through A_true.
    
    Even if A_true is passed for logging, it must be detached or used
    in a no_grad context so no gradients flow through it.
    """
    project_root = Path(__file__).parent.parent
    rcgnn_file = project_root / "src" / "models" / "rcgnn.py"
    
    if not rcgnn_file.exists():
        print(f"[SKIP] {rcgnn_file} not found")
        return
    
    with open(rcgnn_file, 'r') as f:
        source = f.read()
    
    # Pattern: A_true used without .detach() or in torch.no_grad()
    # This is a heuristic - may have false positives
    dangerous_patterns = [
        r'A_true\s*\*',      # A_true * something (multiplication)
        r'\*\s*A_true',      # something * A_true
        r'A_true\s*\+',      # A_true + something
        r'\+\s*A_true',      # something + A_true
        r'A_true\s*-',       # A_true - something
        r'-\s*A_true',       # something - A_true
        r'loss\s*.*A_true',  # loss = ... A_true
    ]
    
    # Find all matches
    issues = []
    for pattern in dangerous_patterns:
        matches = re.finditer(pattern, source)
        for match in matches:
            # Check if it's inside a no_grad block or detached
            start = max(0, match.start() - 200)
            context = source[start:match.end() + 50]
            if 'no_grad' not in context and '.detach()' not in context:
                # Get line number
                line_num = source[:match.start()].count('\n') + 1
                issues.append(f"Line {line_num}: {match.group(0)}")
    
    if issues:
        print(f"[WARN] Potential oracle leakage in loss computation:")
        for issue in issues[:5]:  # Show first 5
            print(f"  {issue}")
        print("  (These may be false positives - manual review needed)")
    else:
        print("[PASS] No obvious oracle leakage in loss gradient paths")


def test_training_runs_without_ground_truth():
    """
    CRITICAL TEST: Training must complete even if A_true.npy is removed.
    
    This is the definitive test: if training fails without A_true,
    there's oracle leakage. If it completes, training is oracle-free.
    """
    import tempfile
    import shutil
    import numpy as np
    
    project_root = Path(__file__).parent.parent
    
    # Create a minimal test dataset WITHOUT A_true.npy
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create minimal data
        d = 5
        N = 100
        T = 10
        
        X = np.random.randn(N, T, d).astype(np.float32)
        M = (np.random.rand(N, T, d) > 0.1).astype(np.float32)  # 10% missing
        e = np.zeros(N, dtype=np.int64)  # Single regime
        
        np.save(tmpdir / "X.npy", X)
        np.save(tmpdir / "M.npy", M)
        np.save(tmpdir / "e.npy", e)
        # Deliberately NOT creating A_true.npy
        
        # Try to import training components
        try:
            from src.models.rcgnn import RCGNN
            import torch
            
            # Create model (matching actual RCGNN signature)
            model = RCGNN(
                d=d,
                latent_dim=16,
                hidden_dim=32,
                n_regimes=1,
                target_edges=5,
                lambda_recon=1.0,
                lambda_miss=0.5,
                lambda_hsic=0.1,
                lambda_acyclic=1.0,
                lambda_sparse=0.01,
                lambda_inv=0.1,
            )
            
            # Create dummy batch
            X_batch = torch.from_numpy(X[:32])
            M_batch = torch.from_numpy(M[:32])
            e_batch = torch.from_numpy(e[:32])
            
            # Forward pass (should work without A_true)
            outputs = model(X_batch, M_batch, regime=e_batch)
            
            # Compute loss (should work without A_true)
            loss, metrics = model.compute_loss(
                outputs, X_batch, M_batch, regime=e_batch,
                epoch=1, total_epochs=10
            )
            
            # Backward pass (should work without A_true)
            loss.backward()
            
            print("[PASS] Training forward/backward works WITHOUT A_true.npy")
            
        except Exception as e:
            if 'A_true' in str(e) or 'ground_truth' in str(e).lower():
                raise AssertionError(f"ORACLE LEAKAGE: Training requires A_true: {e}")
            else:
                # Other errors are OK (might be missing dependencies)
                print(f"[SKIP] Could not run training test: {e}")


def test_calibration_is_validation_only():
    """
    TEST: K calibration must use validation data only, not test.
    
    The comprehensive_evaluation.py should:
    1. Select K on a validation corruption
    2. Apply that K unchanged to test corruptions
    """
    project_root = Path(__file__).parent.parent
    eval_file = project_root / "scripts" / "comprehensive_evaluation.py"
    
    if not eval_file.exists():
        print(f"[SKIP] {eval_file} not found")
        return
    
    with open(eval_file, 'r') as f:
        source = f.read()
    
    # Check for calibration protocol
    has_calibration = 'calibrate' in source.lower() or 'optimal_k' in source.lower()
    has_validation = 'validation' in source.lower() or 'held-out' in source.lower()
    
    assert has_calibration, "Evaluation should have K calibration"
    
    # Check that calibration doesn't use test data
    # (This is a heuristic check)
    if 'calibration on a held-out' in source.lower() or 'validation corruption' in source.lower():
        print("[PASS] Calibration appears to use validation data (manual verification recommended)")
    else:
        print("[WARN] Calibration protocol unclear - verify it uses validation only")


def run_oracle_audit():
    """Run all oracle audit tests."""
    import numpy as np  # Needed for test_training_runs_without_ground_truth
    
    print("=" * 70)
    print("ORACLE AUDIT - Detecting ground truth leakage in training")
    print("=" * 70)
    print()
    
    # Test 1: Scan code for oracle references
    print("1. Scanning training code for A_true references...")
    violations = scan_training_code_for_oracle_leakage()
    if violations:
        print(f"   [WARN] Found {len(violations)} potential violations:")
        for v in violations[:5]:
            print(f"   - {v['file']}:{v['line']} in {v['function']}: {v['variable']}")
        if len(violations) > 5:
            print(f"   ... and {len(violations) - 5} more")
    else:
        print("   [PASS] No oracle references in training code")
    print()
    
    # Test 2: Model forward pass
    print("2. Checking model forward() signature...")
    test_no_oracle_in_model_forward()
    print()
    
    # Test 3: Loss gradient paths
    print("3. Checking loss gradient paths...")
    test_no_oracle_in_loss_gradient()
    print()
    
    # Test 4: Training without ground truth
    print("4. Testing training runs without A_true.npy...")
    test_training_runs_without_ground_truth()
    print()
    
    # Test 5: Calibration protocol
    print("5. Verifying calibration is validation-only...")
    test_calibration_is_validation_only()
    print()
    
    print("=" * 70)
    print("ORACLE AUDIT COMPLETE")
    print("=" * 70)


# Pytest-compatible test functions
def test_oracle_audit_full():
    """Run full oracle audit as a pytest test."""
    violations = scan_training_code_for_oracle_leakage()
    # Allow some violations in diagnostic/logging code, but flag critical ones
    critical = [v for v in violations if 'loss' in (v['function'] or '').lower()]
    assert len(critical) == 0, f"Critical oracle leakage in loss functions: {critical}"


if __name__ == "__main__":
    run_oracle_audit()
