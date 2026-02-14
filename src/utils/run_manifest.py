#!/usr/bin/env python3
"""
RUN MANIFEST - Provenance logging for reproducibility.

Generates a complete manifest of the experimental setup including:
- Git commit hash
- Environment (pip freeze)
- CUDA/cuDNN versions
- Dataset checksum
- Full resolved config
- Calibration parameters

This ensures every experiment is fully reproducible and auditable.
"""

import json
import hashlib
import subprocess
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np


def get_git_info() -> Dict[str, str]:
    """Get git repository information."""
    info = {
        'commit': 'unknown',
        'branch': 'unknown',
        'dirty': 'unknown',
        'remote': 'unknown',
    }
    
    try:
        # Git commit hash
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            info['commit'] = result.stdout.strip()
        
        # Git branch
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            info['branch'] = result.stdout.strip()
        
        # Check if dirty (uncommitted changes)
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            info['dirty'] = len(result.stdout.strip()) > 0
        
        # Git remote URL
        result = subprocess.run(
            ['git', 'remote', 'get-url', 'origin'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            info['remote'] = result.stdout.strip()
            
    except Exception as e:
        info['error'] = str(e)
    
    return info


def get_python_env() -> Dict[str, Any]:
    """Get Python environment information."""
    info = {
        'python_version': platform.python_version(),
        'python_implementation': platform.python_implementation(),
        'platform': platform.platform(),
        'machine': platform.machine(),
    }
    
    # Get installed packages
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'freeze'],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            packages = result.stdout.strip().split('\n')
            info['packages'] = {
                p.split('==')[0]: p.split('==')[1] if '==' in p else 'unknown'
                for p in packages if p and '==' in p
            }
            # Highlight key packages
            key_packages = ['torch', 'numpy', 'scipy', 'pandas', 'scikit-learn', 'networkx']
            info['key_packages'] = {
                k: info['packages'].get(k, 'not installed')
                for k in key_packages
            }
    except Exception as e:
        info['packages_error'] = str(e)
    
    return info


def get_cuda_info() -> Dict[str, Any]:
    """Get CUDA/cuDNN information."""
    info = {
        'cuda_available': False,
        'cuda_version': 'N/A',
        'cudnn_version': 'N/A',
        'gpu_count': 0,
        'gpu_names': [],
    }
    
    try:
        import torch
        info['cuda_available'] = torch.cuda.is_available()
        
        if torch.cuda.is_available():
            info['cuda_version'] = torch.version.cuda
            info['cudnn_version'] = str(torch.backends.cudnn.version())
            info['gpu_count'] = torch.cuda.device_count()
            info['gpu_names'] = [
                torch.cuda.get_device_name(i) 
                for i in range(torch.cuda.device_count())
            ]
    except ImportError:
        info['torch_import_error'] = True
    except Exception as e:
        info['error'] = str(e)
    
    return info


def compute_file_checksum(filepath: Path) -> str:
    """Compute MD5 checksum of a file."""
    if not filepath.exists():
        return 'file_not_found'
    
    hash_md5 = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def compute_dataset_checksums(data_dir: Path) -> Dict[str, str]:
    """Compute checksums for all dataset files."""
    checksums = {}
    
    data_files = ['X.npy', 'M.npy', 'e.npy', 'A_true.npy', 'config.json']
    
    for filename in data_files:
        filepath = data_dir / filename
        if filepath.exists():
            checksums[filename] = compute_file_checksum(filepath)
    
    # Also compute checksum for entire directory (detects any extra files)
    all_files = sorted(data_dir.glob('*'))
    combined = ''.join([compute_file_checksum(f) for f in all_files if f.is_file()])
    checksums['_combined'] = hashlib.md5(combined.encode()).hexdigest()
    
    return checksums


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate configuration for invalid combinations.
    
    Returns dict with 'valid' bool and 'issues' list.
    """
    issues = []
    
    # Check for conflicting settings
    if config.get('use_topk_projection', False):
        if config.get('topk_projection_start', 0) >= config.get('stage2_end', 1):
            issues.append("topk_projection_start >= stage2_end: projection starts after pruning ends")
    
    # Check temperature schedule
    if config.get('temperature_init', 2.0) < config.get('temperature_final', 0.5):
        issues.append("temperature_init < temperature_final: temperature should decrease")
    
    # Check lambda schedules
    if config.get('lambda_sparse_init', 0) > config.get('lambda_sparse_final', 0.01):
        issues.append("lambda_sparse_init > lambda_sparse_final: sparsity should increase")
    
    # Check for missing required fields
    required_fields = ['epochs', 'batch_size', 'lr', 'seed']
    for field in required_fields:
        if field not in config:
            issues.append(f"Missing required field: {field}")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues
    }


def create_run_manifest(
    config: Dict[str, Any],
    data_dir: Path,
    output_dir: Path,
    calibration_info: Optional[Dict[str, Any]] = None,
    extra_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create a complete run manifest for reproducibility.
    
    Args:
        config: Resolved configuration dictionary
        data_dir: Path to dataset directory
        output_dir: Path to output directory
        calibration_info: Optional calibration parameters (K, validation corruption, etc.)
        extra_info: Optional additional information to include
    
    Returns:
        Complete manifest dictionary
    """
    manifest = {
        'timestamp': datetime.now().isoformat(),
        'manifest_version': '1.0',
        
        # Git information
        'git': get_git_info(),
        
        # Python environment
        'python_env': get_python_env(),
        
        # CUDA information
        'cuda': get_cuda_info(),
        
        # Dataset checksums (for data integrity)
        'dataset': {
            'path': str(data_dir),
            'checksums': compute_dataset_checksums(data_dir) if data_dir.exists() else {},
        },
        
        # Full resolved configuration
        'config': config,
        
        # Config validation
        'config_validation': validate_config(config),
        
        # Output paths
        'output': {
            'directory': str(output_dir),
        },
    }
    
    # Add calibration information if provided
    if calibration_info:
        manifest['calibration'] = calibration_info
    
    # Add extra information if provided
    if extra_info:
        manifest['extra'] = extra_info
    
    return manifest


def save_run_manifest(manifest: Dict[str, Any], output_path: Path):
    """Save manifest to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Custom JSON encoder for numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    
    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2, cls=NumpyEncoder)
    
    print(f"[MANIFEST] Saved to {output_path}")


def load_run_manifest(manifest_path: Path) -> Dict[str, Any]:
    """Load manifest from JSON file."""
    with open(manifest_path, 'r') as f:
        return json.load(f)


def compare_manifests(manifest1: Dict, manifest2: Dict) -> Dict[str, Any]:
    """
    Compare two manifests to identify differences.
    
    Useful for debugging reproducibility issues.
    """
    differences = {
        'git_differs': manifest1.get('git', {}).get('commit') != manifest2.get('git', {}).get('commit'),
        'config_differs': manifest1.get('config') != manifest2.get('config'),
        'dataset_differs': manifest1.get('dataset', {}).get('checksums') != manifest2.get('dataset', {}).get('checksums'),
        'cuda_differs': manifest1.get('cuda', {}).get('cuda_version') != manifest2.get('cuda', {}).get('cuda_version'),
    }
    
    # Detailed config differences
    if differences['config_differs']:
        config1 = manifest1.get('config', {})
        config2 = manifest2.get('config', {})
        all_keys = set(config1.keys()) | set(config2.keys())
        differences['config_changes'] = {
            k: {'old': config1.get(k), 'new': config2.get(k)}
            for k in all_keys
            if config1.get(k) != config2.get(k)
        }
    
    return differences


# =============================================================================
# CORE CONFIG SURFACE (≤12 knobs for reviewer-friendly ablation)
# =============================================================================

CORE_CONFIG_KNOBS = {
    # Training basics
    'seed': 'Random seed for reproducibility',
    'epochs': 'Number of training epochs',
    'batch_size': 'Batch size',
    'lr': 'Learning rate',
    
    # Model architecture
    'latent_dim': 'Latent dimension for encoders',
    'hidden_dim': 'Hidden dimension for networks',
    
    # Core loss weights
    'lambda_sparse': 'Sparsity regularization weight',
    'lambda_acyclic': 'DAG acyclicity penalty weight',
    
    # Temperature schedule
    'temperature_init': 'Initial softmax temperature',
    'temperature_final': 'Final softmax temperature',
    
    # Calibration
    'k_calibration_mode': 'How to select K: "validation" or "fixed"',
    
    # Data handling
    'missingness_strategy': 'How to handle missing data: "impute" or "mask"',
}

ADVANCED_CONFIG_KNOBS = {
    # Everything else is "advanced/experimental"
    # Users can set these but they're not part of official ablation
}


def extract_core_config(full_config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract only the core config knobs for reporting."""
    return {k: full_config.get(k) for k in CORE_CONFIG_KNOBS.keys() if k in full_config}


def print_core_config(config: Dict[str, Any]):
    """Print core config in a reviewer-friendly format."""
    print("\n" + "=" * 60)
    print("CORE CONFIGURATION (official ablation surface)")
    print("=" * 60)
    
    core = extract_core_config(config)
    for key, value in core.items():
        desc = CORE_CONFIG_KNOBS.get(key, '')
        print(f"  {key:25s} = {value:>10}  # {desc}")
    
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Demo: create a manifest for current environment
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate run manifest")
    parser.add_argument('--data-dir', type=Path, default=Path('data/interim/uci_air'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/run_manifest.json'))
    args = parser.parse_args()
    
    # Example config
    example_config = {
        'seed': 42,
        'epochs': 100,
        'batch_size': 32,
        'lr': 5e-4,
        'latent_dim': 32,
        'hidden_dim': 64,
        'lambda_sparse': 0.05,
        'lambda_acyclic': 0.5,
        'temperature_init': 2.0,
        'temperature_final': 0.3,
    }
    
    manifest = create_run_manifest(
        config=example_config,
        data_dir=args.data_dir,
        output_dir=args.output.parent,
        calibration_info={
            'K_used': 13,
            'calibration_corruption': 'compound_full',
            'method': 'validation F1 maximization'
        }
    )
    
    save_run_manifest(manifest, args.output)
    print_core_config(example_config)
    
    # Validate
    validation = manifest['config_validation']
    if validation['valid']:
        print("[PASS] Configuration is valid")
    else:
        print("[WARN] Configuration issues:")
        for issue in validation['issues']:
            print(f"  - {issue}")
