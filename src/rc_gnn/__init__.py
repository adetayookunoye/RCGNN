"""
RC-GNN: Robust Causal Graph Neural Network

A novel, SOTA-surpassing implementation for causal discovery under compound corruptions.

Key components:
- DisentangledEncoder: Separates signal from corruption with HSIC independence
- CausalGraphLearner: Learns adjacency A with per-environment deltas
- InvarianceRegularizer: Structure-level stability across environments
- CausalPriorLoss: Intervention, orientation, necessity, mechanism invariance
- CompoundCorruptionGenerator: MCAR/MAR/MNAR + noise + bias

Theory:
- HSIC (Hilbert-Schmidt Independence Criterion) ensures statistical independence
  between signal and corruption latent spaces
- NOTEARS constraint tr(exp(A∘A)) - d = 0 is necessary and sufficient for DAG
- Variance-based invariance across environments identifies stable causal edges
"""

from .encoder import DisentangledEncoder, DisentanglementLoss, hsic_penalty, rbf_kernel
from .graph_learner import (
    CausalGraphLearner,
    GraphLearnerLoss,
    notears_h,
    notears_h_poly,
)
from .invariance import (
    InvarianceRegularizer,
    BatchInvariance,
    IRM_Invariance,
)
# Import RCGNN from the main models package
from src.models.rcgnn import RCGNN, notears_acyclicity
from .corruption import (
    CompoundCorruptionGenerator,
    generate_mcar_mask,
    generate_mar_mask,
    generate_mnar_mask,
    add_gaussian_noise,
    add_outliers,
    add_drift,
    get_corruption_preset,
)
from .metrics import (
    compute_shd,
    compute_edge_metrics,
    compute_auroc_auprc,
    compute_fdr_tpr,
    compute_sid,
    compute_all_metrics,
    find_optimal_threshold,
    MetricsTracker,
)

__all__ = [
    # Encoder
    "DisentangledEncoder",
    "DisentanglementLoss",
    "hsic_penalty",
    "rbf_kernel",
    # Graph Learner
    "CausalGraphLearner",
    "GraphLearnerLoss",
    "notears_h",
    "notears_h_poly",
    # Invariance
    "InvarianceRegularizer",
    "BatchInvariance",
    "IRM_Invariance",
    # Model
    "RCGNN",
    "notears_acyclicity",
    # Corruption
    "CompoundCorruptionGenerator",
    "generate_mcar_mask",
    "generate_mar_mask",
    "generate_mnar_mask",
    "add_gaussian_noise",
    "add_outliers",
    "add_drift",
    "get_corruption_preset",
    # Metrics
    "compute_shd",
    "compute_edge_metrics",
    "compute_auroc_auprc",
    "compute_fdr_tpr",
    "compute_sid",
    "compute_all_metrics",
    "find_optimal_threshold",
    "MetricsTracker",
]

__version__ = "2.0.0"
