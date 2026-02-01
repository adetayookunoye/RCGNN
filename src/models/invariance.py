import torch
import torch.nn as nn
import torch.nn.functional as F

class IRMStructureInvariance(nn.Module):
    """IRM-style invariance regularization for causal structure learning.
    
    Combines:
    1. IRM gradient penalty on per-environment structural risks
    2. Structure variance penalty across environments
    3. Shared mean logits with environment-specific deltas
    """
    def __init__(self, n_features, n_envs, gamma=0.1):
        super().__init__()
        self.n_features = n_features
        self.n_envs = n_envs
        self.gamma = gamma
        
        # Register dummy parameters for gradient tracking
        self.register_buffer('env_risks', torch.zeros(n_envs))
        self.register_buffer('grad_norms', torch.zeros(n_envs))
        
    def compute_env_risk(self, A, logits, X, M, e):
        """Compute structural risk per environment.
        
        Args:
            A: Adjacency matrix [B,d,d] or [d,d]
            logits: Raw adjacency logits [B,d,d] or [d,d]
            X: Input data [B,T,d] or [T,d]
            M: Mask tensor [B,T,d] or [T,d]
            e: Environment indices [B] or scalar
            
        Returns:
            risk: Per-environment risk
            grad_penalty: IRM gradient penalty
        """
        # CRITICAL FIX: During evaluation (torch.no_grad()), we cannot compute
        # gradients for the IRM penalty. Return zeros in that case.
        if not torch.is_grad_enabled():
            zeros = torch.zeros(self.n_envs, device=A.device, dtype=A.dtype)
            return zeros, torch.tensor(0.0, device=A.device, dtype=A.dtype)
        
        if A.dim() == 2:
            A = A.unsqueeze(0)
            logits = logits.unsqueeze(0) if logits is not None else None
        if logits is not None and logits.dim() == 2:
            logits = logits.unsqueeze(0)
        if X.dim() == 2:
            X = X.unsqueeze(0)
            M = M.unsqueeze(0)
        if not isinstance(e, torch.Tensor):
            e = torch.tensor([e], device=A.device, dtype=torch.long)
        else:
            e = e.to(A.device)

        if logits is None:
            logits = A

        B, d, _ = A.shape

        risks = torch.zeros(self.n_envs, device=A.device, dtype=A.dtype)
        grads = []

        for env_idx in range(self.n_envs):
            env_mask = (e == env_idx)
            if not env_mask.any():
                continue

            A_e = A[env_mask]
            logits_e = logits[env_mask]
            X_e = X[env_mask]
            M_e = M[env_mask]

            scale = torch.ones(1, device=A.device, requires_grad=True)
            support = torch.sigmoid(logits_e)
            effective_adj = A_e * support

            # CRITICAL FIX: Remove transpose to match causal convention
            # A[i,j]=1 means i->j (i causes j)
            # For prediction: X_next[j] = sum_i A[i,j] * X_prev[i]
            # This is X_prev @ A (not X_prev @ A.T)
            preds = torch.matmul(X_e[:, :-1], effective_adj)
            targets = X_e[:, 1:]

            mask = M_e[:, 1:].to(preds.dtype)
            err = ((preds - targets) ** 2) * mask
            risk = err.mean()

            # Preferred: compute gradient of risk w.r.t. logits directly so the
            # resulting grad_penalty depends on logits and backpropagates into
            # them (logits.grad will be populated after total_loss.backward()).
            try:
                grad_logits = torch.autograd.grad(risk, logits_e, create_graph=True)[0]
                # Keep edge-level gradients for proper variance computation
                # Average over batch dimension, keep [d, d] structure
                grad_edges = grad_logits.mean(dim=0) # [d, d]
                grads.append(grad_edges.flatten()) # [d*d]
            except RuntimeError:
                # Fallback to the original scale-based finite-diff style when
                # logits are not differentiable for some reason.
                scale = torch.ones(1, device=A.device, requires_grad=True)
                # CRITICAL FIX: Remove transpose to match causal convention
                preds_s = torch.matmul(X_e[:, :-1], effective_adj * scale)
                err_s = ((preds_s - targets) ** 2) * mask
                risk_s = err_s.mean()
                grad = torch.autograd.grad(risk_s, [scale], create_graph=True)[0]
                grads.append(grad.expand(self.n_features * self.n_features))

            risks[env_idx] = risk

        if grads:
            grads_tensor = torch.stack(grads) # [n_envs, d*d]
            if grads_tensor.shape[0] > 1:
                # Variance across environments for each edge
                edge_var = torch.var(grads_tensor, dim=0, unbiased=False) # [d*d]
                # Mean variance across all edges, scaled by gamma
                grad_penalty = edge_var.mean() * self.gamma
            else:
                grad_penalty = torch.zeros((), device=A.device, dtype=A.dtype)
        else:
            grad_penalty = torch.zeros((), device=A.device, dtype=A.dtype)

        return risks, grad_penalty
        
    def structure_variance(self, A, logits, e):
        """Compute variance of adjacency across environments.
        
        Args:
            A: Adjacency matrix [B,d,d] or [d,d]
            logits: Raw adjacency logits [B,d,d] or [d,d]
            e: Environment indices [B] or scalar
            
        Returns:
            var_penalty: Structure variance penalty
        """
        if A.dim() == 2:
            A = A.unsqueeze(0)
        if logits is not None and logits.dim() == 2:
            logits = logits.unsqueeze(0)
        if logits is None:
            logits = A
        if not isinstance(e, torch.Tensor):
            e = torch.tensor([e], device=A.device, dtype=torch.long)
        else:
            e = e.to(A.device)

        env_As = []
        env_logits = []
        for env_idx in range(self.n_envs):
            env_mask = (e == env_idx)
            if env_mask.any():
                A_e = A[env_mask].mean(0)
                env_As.append(A_e)
                if logits is not None:
                    logits_e = torch.sigmoid(logits[env_mask]).mean(0)
                    env_logits.append(logits_e)

        if not env_As:
            return torch.zeros((), device=A.device, dtype=A.dtype)

        env_As = torch.stack(env_As)
        if env_As.shape[0] < 2:
            return torch.zeros((), device=A.device, dtype=A.dtype)

        var_penalty = torch.var(env_As, dim=0, unbiased=False).mean()
        if env_As.shape[0] > 1:
            pairwise = []
            for i in range(env_As.shape[0]):
                for j in range(i + 1, env_As.shape[0]):
                    pairwise.append(torch.mean(torch.abs(env_As[i] - env_As[j])))
            if pairwise:
                var_penalty = var_penalty + torch.stack(pairwise).mean()
        if env_logits:
            env_logits = torch.stack(env_logits)
            if env_logits.shape[0] > 1:
                var_penalty = var_penalty + torch.var(env_logits, dim=0, unbiased=False).mean()

        return var_penalty
        
    def forward(self, A, logits, X, M, e):
        """Compute total invariance loss.
        
        Args:
            A: Adjacency matrix [B,d,d] or [d,d]
            logits: Raw adjacency logits [B,d,d] or [d,d]
            X: Input data [B,T,d] or [T,d]
            M: Mask tensor [B,T,d] or [T,d]
            e: Environment indices [B] or scalar
            
        Returns:
            total_loss: Combined invariance penalty
            metrics: Dict of individual terms
        """
        # IRM gradient penalty
        risks, grad_penalty = self.compute_env_risk(A, logits, X, M, e)
        
        # Structure variance penalty
        var_penalty = self.structure_variance(A, logits, e)
        
        # Total loss
        total_loss = grad_penalty + var_penalty
        
        # Save metrics
        if risks.numel() == self.env_risks.numel():
            self.env_risks.copy_(risks.detach())
        
        metrics = {
            'env_risks': risks.detach(),
            'grad_penalty': grad_penalty.detach(),
            'var_penalty': var_penalty.detach(),
            'total_invariance': total_loss.detach()
        }
        
        return total_loss, metrics
