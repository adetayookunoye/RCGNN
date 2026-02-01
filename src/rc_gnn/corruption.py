"""
Compound Corruption Generator for RC-GNN v2

Generates realistic compound corruptions:
1. Missingness: MCAR, MAR, MNAR patterns
2. Noise: Heteroscedastic, outliers, drift
3. Bias: Constant offset, scaling, non-linear

Key design: All corruptions are differentiable where possible
to allow end-to-end training and gradient analysis.
"""

import math
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List, Union


# ============================================================================
# MISSINGNESS GENERATORS
# ============================================================================

def generate_mcar_mask(
    shape: Tuple[int, ...],
    missing_rate: float = 0.2,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Missing Completely At Random (MCAR).
    
    Each entry is independently missing with probability `missing_rate`.
    
    Args:
        shape: Shape of the mask
        missing_rate: Probability of missingness per entry
        device: Device for tensor
        
    Returns:
        mask: Binary mask (1 = observed, 0 = missing)
    """
    mask = torch.rand(shape, device=device) > missing_rate
    return mask.float()


def generate_mar_mask(
    X: torch.Tensor,
    missing_rate: float = 0.2,
    n_conditioning: int = 2,
) -> torch.Tensor:
    """
    Missing At Random (MAR).
    
    Missingness in variable j depends on observed values in other variables.
    P(M_j = 1 | X) = sigmoid(W @ X[:, others] + b)
    
    Args:
        X: Input data [B, d] or [B, T, d]
        missing_rate: Target average missing rate
        n_conditioning: Number of conditioning variables
        
    Returns:
        mask: Binary mask (1 = observed, 0 = missing)
    """
    device = X.device
    original_shape = X.shape
    
    # Flatten to [N, d]
    if X.dim() == 3:
        B, T, d = X.shape
        X_flat = X.view(-1, d)
    else:
        X_flat = X
        d = X.shape[-1]
    
    N = X_flat.shape[0]
    mask = torch.ones(N, d, device=device)
    
    for j in range(d):
        # Select conditioning variables (not j)
        others = [i for i in range(d) if i != j]
        if len(others) > n_conditioning:
            idx = torch.randperm(len(others))[:n_conditioning]
            others = [others[i] for i in idx]
        
        # Conditioning values
        X_cond = X_flat[:, others] # [N, n_cond]
        
        # Random linear combination
        weights = torch.randn(len(others), device=device)
        logits = X_cond @ weights
        
        # Calibrate threshold to achieve target missing rate
        threshold = torch.quantile(logits, 1 - missing_rate)
        mask[:, j] = (logits < threshold).float()
    
    # Reshape
    if len(original_shape) == 3:
        mask = mask.view(B, T, d)
    
    return mask


def generate_mnar_mask(
    X: torch.Tensor,
    missing_rate: float = 0.2,
    mechanism: str = "threshold", # "threshold", "self", "extreme"
) -> torch.Tensor:
    """
    Missing Not At Random (MNAR).
    
    Missingness depends on the value itself (potentially unobserved).
    
    Mechanisms:
    - "threshold": Missing if value exceeds threshold
    - "self": P(missing | x) depends on x
    - "extreme": Missing for extreme values
    
    Args:
        X: Input data [B, d] or [B, T, d]
        missing_rate: Target missing rate
        mechanism: Type of MNAR mechanism
        
    Returns:
        mask: Binary mask (1 = observed, 0 = missing)
    """
    device = X.device
    original_shape = X.shape
    
    if X.dim() == 3:
        X_flat = X.view(-1, X.shape[-1])
    else:
        X_flat = X
    
    N, d = X_flat.shape
    
    if mechanism == "threshold":
        # Missing if value > threshold (calibrated per variable)
        mask = torch.ones_like(X_flat)
        for j in range(d):
            threshold = torch.quantile(X_flat[:, j], 1 - missing_rate)
            mask[:, j] = (X_flat[:, j] < threshold).float()
            
    elif mechanism == "self":
        # Probability of missing proportional to value
        # P(missing | x) = sigmoid(alpha * (x - median))
        X_centered = X_flat - X_flat.median(dim=0, keepdim=True).values
        alpha = 2.0 # Controls steepness
        
        prob_missing = torch.sigmoid(alpha * X_centered)
        # Calibrate to achieve target rate
        scale = missing_rate / prob_missing.mean()
        prob_missing = (prob_missing * scale).clamp(0, 1)
        
        mask = (torch.rand_like(X_flat) > prob_missing).float()
        
    elif mechanism == "extreme":
        # Missing for extreme values (both tails)
        mask = torch.ones_like(X_flat)
        for j in range(d):
            lower = torch.quantile(X_flat[:, j], missing_rate / 2)
            upper = torch.quantile(X_flat[:, j], 1 - missing_rate / 2)
            mask[:, j] = ((X_flat[:, j] >= lower) & (X_flat[:, j] <= upper)).float()
    else:
        raise ValueError(f"Unknown mechanism: {mechanism}")
    
    # Reshape
    if len(original_shape) == 3:
        mask = mask.view(original_shape)
    
    return mask


def generate_structural_mnar_mask(
    X: torch.Tensor,
    A_true: torch.Tensor,
    missing_rate: float = 0.2,
) -> torch.Tensor:
    """
    Structural MNAR: Missingness depends on PARENT node values.
    
    This is the most realistic corruption for causal discovery:
    P(M_j = 1 | X) depends on X_pa(j) where pa(j) are parents in the DAG.
    
    Args:
        X: Input data [B, T, d] or [B, d]
        A_true: Ground truth adjacency [d, d] where A[i,j]=1 means i->j
        missing_rate: Target missing rate
        
    Returns:
        mask: Binary mask (1 = observed, 0 = missing)
    """
    device = X.device
    original_shape = X.shape
    d = X.shape[-1]
    
    if X.dim() == 3:
        X_flat = X.view(-1, d)
    else:
        X_flat = X
    
    N = X_flat.shape[0]
    mask = torch.ones(N, d, device=device)
    
    A_true = A_true.to(device)
    
    for j in range(d):
        # Find parents of node j
        parents = torch.where(A_true[:, j] > 0)[0]
        
        if len(parents) == 0:
            # No parents: fall back to self-dependent MNAR
            X_cond = X_flat[:, j:j+1]
        else:
            # Parents exist: missingness depends on parent values
            X_cond = X_flat[:, parents] # [N, n_parents]
        
        # Linear combination of parent values
        weights = torch.randn(X_cond.shape[1], device=device)
        logits = X_cond @ weights
        
        # Calibrate threshold for target missing rate
        threshold = torch.quantile(logits, 1 - missing_rate)
        mask[:, j] = (logits < threshold).float()
    
    if len(original_shape) == 3:
        mask = mask.view(original_shape)
    
    return mask


# ============================================================================
# REGIME/ENVIRONMENT SHIFT GENERATORS
# ============================================================================

def generate_regime_shifts(
    X: torch.Tensor,
    n_regimes: int = 3,
    shift_type: str = "intervention", # "intervention", "distribution", "mechanism"
    intervention_strength: float = 1.0,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate regime/environment shifts for multi-environment causal discovery.
    
    This creates heterogeneous environments where:
    - Regime 0: Original data (observational)
    - Regime 1+: Shifted data (interventional or distributional shifts)
    
    Args:
        X: Input data [B, T, d]
        n_regimes: Number of distinct regimes
        shift_type: Type of shift
            - "intervention": Hard intervention on random nodes
            - "distribution": Mean/variance shift across environments
            - "mechanism": Change in causal mechanisms
        intervention_strength: Magnitude of intervention
        seed: Random seed
        
    Returns:
        X_shifted: Data with regime-specific shifts [B, T, d]
        e: Regime indicator [B]
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    device = X.device
    B, T, d = X.shape
    
    # Assign samples to regimes uniformly
    e = torch.randint(0, n_regimes, (B,), device=device)
    
    X_shifted = X.clone()
    
    if shift_type == "intervention":
        # Hard intervention: replace values in intervened nodes
        for regime in range(1, n_regimes):
            regime_mask = (e == regime)
            n_samples = regime_mask.sum().item()
            
            if n_samples == 0:
                continue
            
            # Intervene on random subset of nodes (1-3 nodes per regime)
            n_intervene = min(d, np.random.randint(1, 4))
            intervened_nodes = np.random.choice(d, n_intervene, replace=False)
            
            for node in intervened_nodes:
                # Replace with fixed value + noise
                intervention_value = intervention_strength * (2 * np.random.rand() - 1)
                noise = torch.randn(n_samples, T, device=device) * 0.1
                X_shifted[regime_mask, :, node] = intervention_value + noise
                
    elif shift_type == "distribution":
        # Distribution shift: change mean and variance per regime
        for regime in range(1, n_regimes):
            regime_mask = (e == regime)
            
            # Random mean shift per variable
            mean_shift = (torch.randn(d, device=device) * intervention_strength).view(1, 1, d)
            # Random variance scaling
            var_scale = (0.5 + torch.rand(d, device=device)).view(1, 1, d)
            
            X_regime = X[regime_mask]
            X_regime = (X_regime - X_regime.mean(dim=(0,1), keepdim=True)) * var_scale
            X_regime = X_regime + mean_shift
            X_shifted[regime_mask] = X_regime
            
    elif shift_type == "mechanism":
        # Mechanism change: modify relationships between variables
        for regime in range(1, n_regimes):
            regime_mask = (e == regime)
            
            # Add cross-variable interactions
            perm = torch.randperm(d)
            interaction_weight = intervention_strength * 0.3
            
            X_regime = X[regime_mask].clone()
            for j in range(d):
                # Add contribution from a random other variable
                other = perm[j].item()
                if other != j:
                    X_regime[:, :, j] += interaction_weight * X_regime[:, :, other]
            
            X_shifted[regime_mask] = X_regime
    else:
        raise ValueError(f"Unknown shift_type: {shift_type}")
    
    return X_shifted, e


# ============================================================================
# NOISE GENERATORS
# ============================================================================

def add_gaussian_noise(
    X: torch.Tensor,
    scale: float = 0.1,
    heteroscedastic: bool = False,
) -> torch.Tensor:
    """
    Add Gaussian noise.
    
    Args:
        X: Input data
        scale: Noise scale (std dev)
        heteroscedastic: If True, scale varies by position
        
    Returns:
        X_noisy: Noisy data
    """
    if heteroscedastic:
        # Scale proportional to |x|
        local_scale = scale * (1 + X.abs())
        noise = torch.randn_like(X) * local_scale
    else:
        noise = torch.randn_like(X) * scale
    
    return X + noise


def add_outliers(
    X: torch.Tensor,
    outlier_rate: float = 0.05,
    outlier_scale: float = 5.0,
) -> torch.Tensor:
    """
    Add outliers (extreme values).
    
    Args:
        X: Input data
        outlier_rate: Fraction of entries to corrupt
        outlier_scale: Magnitude of outliers (in std devs)
        
    Returns:
        X_with_outliers: Data with outliers
    """
    device = X.device
    
    # Outlier mask
    outlier_mask = torch.rand_like(X) < outlier_rate
    
    # Random sign and magnitude
    signs = 2 * (torch.rand_like(X) > 0.5).float() - 1
    magnitudes = outlier_scale * X.std() * torch.rand_like(X)
    
    outliers = signs * magnitudes
    
    # Apply
    X_out = X.clone()
    X_out[outlier_mask] = X[outlier_mask] + outliers[outlier_mask]
    
    return X_out


def add_drift(
    X: torch.Tensor,
    drift_rate: float = 0.01,
    temporal_dim: int = 1,
) -> torch.Tensor:
    """
    Add temporal drift (AR(1) process).
    
    Args:
        X: Input data [B, T, d]
        drift_rate: Magnitude of drift per timestep
        temporal_dim: Which dimension is temporal
        
    Returns:
        X_drifted: Data with temporal drift
    """
    if X.dim() != 3:
        return X # No temporal dimension
    
    B, T, d = X.shape
    device = X.device
    
    # Generate AR(1) drift per variable
    drift = torch.zeros(B, T, d, device=device)
    drift[:, 0, :] = torch.randn(B, d, device=device) * drift_rate
    
    for t in range(1, T):
        innovation = torch.randn(B, d, device=device) * drift_rate
        drift[:, t, :] = 0.95 * drift[:, t-1, :] + innovation
    
    return X + drift


# ============================================================================
# BIAS GENERATORS
# ============================================================================

def add_constant_bias(
    X: torch.Tensor,
    bias_magnitude: float = 0.5,
    per_variable: bool = True,
) -> torch.Tensor:
    """
    Add constant bias to each variable.
    
    Args:
        X: Input data
        bias_magnitude: Maximum bias magnitude
        per_variable: If True, different bias per variable
        
    Returns:
        X_biased: Biased data
    """
    device = X.device
    d = X.shape[-1]
    
    if per_variable:
        bias = (2 * torch.rand(d, device=device) - 1) * bias_magnitude
        # Reshape for broadcasting
        if X.dim() == 3:
            bias = bias.view(1, 1, d)
        else:
            bias = bias.view(1, d)
    else:
        bias = (2 * torch.rand(1, device=device) - 1) * bias_magnitude
    
    return X + bias


def add_scaling_bias(
    X: torch.Tensor,
    scale_range: Tuple[float, float] = (0.8, 1.2),
) -> torch.Tensor:
    """
    Add multiplicative scaling bias.
    
    Args:
        X: Input data
        scale_range: Range of scaling factors
        
    Returns:
        X_scaled: Scaled data
    """
    device = X.device
    d = X.shape[-1]
    
    low, high = scale_range
    scales = low + (high - low) * torch.rand(d, device=device)
    
    if X.dim() == 3:
        scales = scales.view(1, 1, d)
    else:
        scales = scales.view(1, d)
    
    return X * scales


def add_nonlinear_bias(
    X: torch.Tensor,
    polynomial_degree: int = 2,
    magnitude: float = 0.1,
) -> torch.Tensor:
    """
    Add nonlinear (polynomial) bias.
    
    Args:
        X: Input data
        polynomial_degree: Degree of polynomial
        magnitude: Magnitude of nonlinear component
        
    Returns:
        X_biased: Data with nonlinear bias
    """
    device = X.device
    d = X.shape[-1]
    
    # Random polynomial coefficients per variable
    bias = torch.zeros_like(X)
    
    for deg in range(2, polynomial_degree + 1):
        coef = torch.randn(d, device=device) * magnitude / deg
        if X.dim() == 3:
            coef = coef.view(1, 1, d)
        else:
            coef = coef.view(1, d)
        bias = bias + coef * (X ** deg)
    
    return X + bias


# ============================================================================
# COMPOUND CORRUPTION CLASS
# ============================================================================

class CompoundCorruptionGenerator:
    """
    Generate compound corruptions combining missingness, noise, and bias.
    
    Usage:
        generator = CompoundCorruptionGenerator(
            missing_rate=0.2,
            missing_mechanism="mcar",
            noise_scale=0.1,
            noise_type="heteroscedastic",
            bias_type="constant",
            bias_magnitude=0.5,
        )
        X_corrupt, mask = generator(X)
    """
    
    def __init__(
        self,
        # Missingness
        missing_rate: float = 0.2,
        missing_mechanism: str = "mcar", # "mcar", "mar", "mnar"
        mnar_type: str = "threshold",
        # Noise
        noise_scale: float = 0.1,
        noise_type: str = "gaussian", # "gaussian", "outliers", "drift", "combined"
        heteroscedastic: bool = False,
        outlier_rate: float = 0.05,
        outlier_scale: float = 5.0,
        drift_rate: float = 0.01,
        # Bias
        bias_type: str = "none", # "none", "constant", "scaling", "nonlinear"
        bias_magnitude: float = 0.5,
        scale_range: Tuple[float, float] = (0.8, 1.2),
        polynomial_degree: int = 2,
        # General
        seed: Optional[int] = None,
    ):
        self.missing_rate = missing_rate
        self.missing_mechanism = missing_mechanism
        self.mnar_type = mnar_type
        
        self.noise_scale = noise_scale
        self.noise_type = noise_type
        self.heteroscedastic = heteroscedastic
        self.outlier_rate = outlier_rate
        self.outlier_scale = outlier_scale
        self.drift_rate = drift_rate
        
        self.bias_type = bias_type
        self.bias_magnitude = bias_magnitude
        self.scale_range = scale_range
        self.polynomial_degree = polynomial_degree
        
        self.seed = seed
    
    def __call__(
        self,
        X: torch.Tensor,
        return_components: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Apply compound corruptions.
        
        Args:
            X: Clean input data [B, d] or [B, T, d]
            return_components: If True, return dict with all components
            
        Returns:
            If return_components=False: (X_corrupt, mask)
            If return_components=True: dict with X_corrupt, mask, X_noisy, X_biased, etc.
        """
        if self.seed is not None:
            torch.manual_seed(self.seed)
        
        X_clean = X.clone()
        
        # 1. Apply bias first (before noise)
        if self.bias_type == "constant":
            X_biased = add_constant_bias(X_clean, self.bias_magnitude)
        elif self.bias_type == "scaling":
            X_biased = add_scaling_bias(X_clean, self.scale_range)
        elif self.bias_type == "nonlinear":
            X_biased = add_nonlinear_bias(X_clean, self.polynomial_degree, self.bias_magnitude)
        else:
            X_biased = X_clean
        
        # 2. Apply noise
        if self.noise_type == "gaussian":
            X_noisy = add_gaussian_noise(X_biased, self.noise_scale, self.heteroscedastic)
        elif self.noise_type == "outliers":
            X_noisy = add_outliers(X_biased, self.outlier_rate, self.outlier_scale)
        elif self.noise_type == "drift":
            X_noisy = add_drift(X_biased, self.drift_rate)
        elif self.noise_type == "combined":
            X_noisy = add_gaussian_noise(X_biased, self.noise_scale, self.heteroscedastic)
            X_noisy = add_outliers(X_noisy, self.outlier_rate, self.outlier_scale)
            if X.dim() == 3:
                X_noisy = add_drift(X_noisy, self.drift_rate)
        else:
            X_noisy = X_biased
        
        # 3. Generate missingness mask
        if self.missing_mechanism == "mcar":
            mask = generate_mcar_mask(X.shape, self.missing_rate, X.device)
        elif self.missing_mechanism == "mar":
            mask = generate_mar_mask(X_noisy, self.missing_rate)
        elif self.missing_mechanism == "mnar":
            mask = generate_mnar_mask(X_noisy, self.missing_rate, self.mnar_type)
        else:
            mask = torch.ones_like(X)
        
        # 4. Apply mask (set missing values to NaN or 0)
        X_corrupt = X_noisy * mask # 0 for missing values
        
        if return_components:
            return {
                "X_clean": X_clean,
                "X_biased": X_biased,
                "X_noisy": X_noisy,
                "X_corrupt": X_corrupt,
                "mask": mask,
                "missing_rate_actual": (1 - mask.float().mean()).item(),
            }
        else:
            return X_corrupt, mask
    
    def get_config(self) -> Dict:
        """Return configuration dict."""
        return {
            "missing_rate": self.missing_rate,
            "missing_mechanism": self.missing_mechanism,
            "mnar_type": self.mnar_type,
            "noise_scale": self.noise_scale,
            "noise_type": self.noise_type,
            "heteroscedastic": self.heteroscedastic,
            "outlier_rate": self.outlier_rate,
            "outlier_scale": self.outlier_scale,
            "drift_rate": self.drift_rate,
            "bias_type": self.bias_type,
            "bias_magnitude": self.bias_magnitude,
            "scale_range": self.scale_range,
            "polynomial_degree": self.polynomial_degree,
            "seed": self.seed,
        }


# ============================================================================
# CORRUPTION PRESETS
# ============================================================================

def get_corruption_preset(name: str) -> CompoundCorruptionGenerator:
    """
    Get a preset corruption configuration.
    
    Available presets:
    - "mild": Low corruption (10% MCAR, low noise)
    - "moderate": Moderate corruption (20% MAR, medium noise)
    - "severe": High corruption (30% MNAR, high noise, drift)
    - "extreme": Very high corruption (50% MNAR, outliers, nonlinear bias)
    - "sensor_failure": Realistic sensor failure pattern
    """
    presets = {
        "mild": CompoundCorruptionGenerator(
            missing_rate=0.1,
            missing_mechanism="mcar",
            noise_scale=0.05,
            noise_type="gaussian",
            bias_type="none",
        ),
        "moderate": CompoundCorruptionGenerator(
            missing_rate=0.2,
            missing_mechanism="mar",
            noise_scale=0.1,
            noise_type="gaussian",
            heteroscedastic=True,
            bias_type="constant",
            bias_magnitude=0.2,
        ),
        "severe": CompoundCorruptionGenerator(
            missing_rate=0.3,
            missing_mechanism="mnar",
            mnar_type="self",
            noise_scale=0.15,
            noise_type="combined",
            outlier_rate=0.05,
            drift_rate=0.02,
            bias_type="scaling",
            scale_range=(0.7, 1.3),
        ),
        "extreme": CompoundCorruptionGenerator(
            missing_rate=0.5,
            missing_mechanism="mnar",
            mnar_type="extreme",
            noise_scale=0.2,
            noise_type="combined",
            outlier_rate=0.1,
            outlier_scale=10.0,
            drift_rate=0.05,
            bias_type="nonlinear",
            bias_magnitude=0.3,
            polynomial_degree=3,
        ),
        "sensor_failure": CompoundCorruptionGenerator(
            missing_rate=0.25,
            missing_mechanism="mnar",
            mnar_type="threshold", # Sensor fails when reading high
            noise_scale=0.1,
            noise_type="gaussian",
            heteroscedastic=True, # More noise at extreme values
            bias_type="scaling", # Calibration drift
            scale_range=(0.9, 1.1),
        ),
    }
    
    if name not in presets:
        raise ValueError(f"Unknown preset: {name}. Available: {list(presets.keys())}")
    
    return presets[name]
