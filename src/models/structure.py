import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import notears_acyclicity, topk_per_column
from .sparsification import sparsemax, entmax15, GumbelTopK


class StructureLearner(nn.Module):
    """Temporal structure learner with multiple sparsification strategies."""

    def __init__(
        self,
        d: int,
        n_lags: int = 1,
        z_dim: int | None = None,
        hidden: int = 64,
        sparsify_method: str = "topk",
        sparsify_k: int | None = 3,
        temp_start: float = 5.0,
        temp_end: float = 0.5,
        steps: int | None = None,
        temporal_prior: str | None = None,
        rank: int | None = None,  # unused but kept for backward compatibility
    ) -> None:
        super().__init__()
        self.d = d
        self.n_lags = n_lags
        self.sparsify_method = sparsify_method
        self.sparsify_k = sparsify_k if sparsify_k is not None else d
        self.temp_start = temp_start
        self.temp_end = temp_end
        self.temporal_prior = temporal_prior
        self.steps = steps

        if self.sparsify_method in {"topk", "gumbel_topk"}:
            if self.sparsify_k <= 0:
                raise ValueError("sparsify_k must be positive for top-k based sparsification.")
            self.sparsify_k = min(self.sparsify_k, d)

        in_dim = d if z_dim is None else z_dim
        self.d_model = hidden

        self.node_proj = nn.Linear(in_dim, self.d_model)
        self.bilinear = nn.Parameter(torch.randn(n_lags, self.d_model, self.d_model) * 0.02)
        nn.init.kaiming_uniform_(self.node_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.node_proj.bias)
        nn.init.xavier_uniform_(self.bilinear)

        self.register_buffer("step", torch.tensor(0.0))
        self.register_buffer("n_envs", torch.tensor(0))

        if self.sparsify_method == "gumbel_topk":
            self.gumbel = GumbelTopK(self.sparsify_k)
        else:
            self.gumbel = None

        if temporal_prior == "exp":
            decay = torch.exp(-torch.arange(n_lags, dtype=torch.float32))
            self.register_buffer("lag_decay", decay)
        elif temporal_prior == "gamma":
            self.lag_alpha = nn.Parameter(torch.ones(1))
            self.lag_beta = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer("lag_decay", torch.ones(n_lags, dtype=torch.float32))

        self.register_parameter(
            "env_deltas", nn.Parameter(torch.zeros(0, self.d, self.d, self.n_lags))
        )

    def temperature(self) -> float:
        """Annealed temperature schedule."""
        if self.steps is None or self.steps <= 0:
            s = min(1.0, float(self.step.item()) / 100.0)
        else:
            s = min(1.0, float(self.step.item()) / float(self.steps))
        return self.temp_start * (1 - s) + self.temp_end * s

    def init_env_deltas(self, n_envs: int) -> None:
        """Initialise environment-specific adjustment tensors."""
        device = self.bilinear.device
        env = torch.zeros(n_envs, self.d, self.d, self.n_lags, device=device)
        self._parameters["env_deltas"] = nn.Parameter(env)
        self.n_envs.fill_(n_envs)

    def forward(self, ZS: torch.Tensor):
        """Compute adjacency matrices for each lag."""
        env_idx = getattr(ZS, "_env_idx", None)
        unbatched = False
        if ZS.dim() == 2:
            ZS = ZS.unsqueeze(0)
            unbatched = True

        proj = self.node_proj(ZS)  # [B, d, d_model]
        proj = torch.tanh(proj)

        temperature = self.temperature()
        adjacencies: list[torch.Tensor] = []
        logits_all: list[torch.Tensor] = []

        if isinstance(env_idx, torch.Tensor):
            env_idx = int(env_idx.item())
        elif env_idx is not None:
            env_idx = int(env_idx)

        for lag in range(self.n_lags):
            kernel = self.bilinear[lag]  # [d_model, d_model]
            proj_l = torch.matmul(proj, kernel)  # [B, d, d_model]
            logits = torch.matmul(proj_l, proj.transpose(-1, -2))  # [B, d, d]
            logits = logits / math.sqrt(self.d_model)
            logits = logits / temperature
            logits = logits - torch.diag_embed(torch.diagonal(logits, dim1=-2, dim2=-1))
            prior_weight = 1.0

            if self.temporal_prior == "exp":
                logits = logits * self.lag_decay[lag]
                prior_weight = float(self.lag_decay[lag].item())
            elif self.temporal_prior == "gamma":
                alpha = F.softplus(self.lag_alpha) + 1.0
                beta = F.softplus(self.lag_beta) + 1.0
                weight = alpha * torch.exp(-float(lag) / beta)
                logits = logits * weight
                prior_weight = float(weight.item())

            if self.env_deltas.numel() > 0 and env_idx is not None:
                delta = self.env_deltas[env_idx, :, :, lag]
                logits = logits + delta.unsqueeze(0)

            logits_all.append(logits)
            adj = self._sparsify(logits)
            adj = adj * prior_weight
            adjacencies.append(adj)

        if unbatched:
            adjacencies = [A.squeeze(0) for A in adjacencies]
            logits_all = [L.squeeze(0) for L in logits_all]

        # Enforce monotonic non-increasing influence with lag: ensure for each
        # edge (i,j) the adjacency at lag t+1 is no greater than at lag t.
        # This avoids cases where sparsification/thresholding produces larger
        # activations at higher lags due to discrete selection effects.
        if len(adjacencies) > 1:
            for lag in range(1, len(adjacencies)):
                # adjacencies[lag] and adjacencies[lag-1] can be batched or
                # unbatched tensors; torch.min will broadcast correctly.
                adjacencies[lag] = torch.min(adjacencies[lag], adjacencies[lag - 1])

        return adjacencies, logits_all

    def _sparsify(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply sparsification/activation to raw logits."""
        if logits.dim() == 2:
            logits = logits.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        if self.sparsify_method == "topk":
            weights = torch.relu(logits)
            mats = [topk_per_column(mat, self.sparsify_k) for mat in weights]
            result = torch.stack(mats, dim=0)
        elif self.sparsify_method == "gumbel_topk":
            if self.gumbel is not None:
                self.gumbel.step.data.copy_(self.step.data)
            weights = torch.relu(logits)
            masks = []
            for mat in weights:
                mask = self.gumbel(mat.transpose(0, 1), dim=-1).transpose(0, 1)
                masks.append(mask * mat)
            result = torch.stack(masks, dim=0)
        elif self.sparsify_method == "sparsemax":
            mat = logits.transpose(-1, -2)
            flat = mat.reshape(-1, mat.size(-1))
            flat = sparsemax(flat, dim=-1)
            attn = flat.view_as(mat).transpose(-1, -2)
            result = torch.relu(attn)
            max_active = max(1, math.ceil(result.size(-1) * 0.4))
            topk_vals, topk_idx = torch.topk(result, max_active, dim=-1)
            sparse = torch.zeros_like(result)
            sparse.scatter_(-1, topk_idx, topk_vals)
            result = sparse
        elif self.sparsify_method == "entmax":
            mat = logits.transpose(-1, -2)
            flat = mat.reshape(-1, mat.size(-1))
            flat = entmax15(flat, dim=-1)
            attn = flat.view_as(mat).transpose(-1, -2)
            result = torch.relu(attn)
            max_active = max(1, math.ceil(result.size(-1) * 0.4))
            topk_vals, topk_idx = torch.topk(result, max_active, dim=-1)
            sparse = torch.zeros_like(result)
            sparse.scatter_(-1, topk_idx, topk_vals)
            result = sparse
        else:
            result = torch.sigmoid(logits)

        if squeeze:
            result = result.squeeze(0)
        return result

    def acyclicity(self, A_all: list[torch.Tensor]):
        """Compute NOTEARS acyclicity penalty per lag."""
        penalties = []
        total = torch.tensor(0.0, device=A_all[0].device)

        for A in A_all:
            if A.dim() == 3:
                A_use = A.mean(0)
            else:
                A_use = A
            pen = notears_acyclicity(A_use)
            penalties.append(pen)
            total = total + pen

        return total, penalties
