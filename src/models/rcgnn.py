import torch, torch.nn as nn
from .encoders import SimpleImputer, EncoderS, EncoderN, EncoderB
from .structure import StructureLearner
from .mechanisms import Mechanisms
from .recon import Recon
from .losses import masked_l1, hsic_xy, adj_variance
from .disentanglement import reset_disentangle_temperature

class RCGNN(nn.Module):
    def __init__(self, d, cfg):
        super().__init__()
        hS = cfg["encoders"]["z_s"]["d_hidden"]
        hN = cfg["encoders"]["z_n"]["d_hidden"]
        hB = cfg["encoders"]["z_b"]["d_hidden"]
        self.imputer = SimpleImputer(d_in=d, d_model=cfg["imputer"]["d_model"], n_layers=cfg["imputer"]["n_layers"])
        self.encS = EncoderS(d, hS)
        self.encN = EncoderN(d, hN)
        self.encB = EncoderB(4*d, hB)  # stats len 4*d - [mu, sigma, skew, kurt] per feature
        self.struct = StructureLearner(d, z_dim=hS, hidden=cfg["structure"]["gnn_hidden"], sparsify_method=cfg["structure"]["sparsify"]["method"],
                                       sparsify_k=cfg["structure"]["sparsify"]["k"],
                                       temp_start=cfg["structure"]["temp_start"], temp_end=cfg["structure"]["temp_end"])
        # optionally initialize per-environment deltas if provided in config
        if "n_envs" in cfg.get("structure", {}):
            try:
                n_envs = int(cfg["structure"]["n_envs"])
                self.struct.init_env_deltas(n_envs)
            except Exception:
                pass
        self.mech = Mechanisms(d, hidden=cfg["mechanisms"]["node_mlp"][0])
        self.recon = Recon(d, hidden_dim=hB)
        self.lw = cfg["loss"]
        reset_disentangle_temperature()

    def forward(self, batch):
        return self.forward_batch(batch)

    def forward_batch(self, batch):
        X, M, e = batch["X"], batch["M"], batch["e"]
        _ = batch.get("S", None)  # unused but kept for API compatibility
        # impute
        X_imp, imp_sigma, _ = self.imputer(X, M)
        if X_imp.dim() == 2:
            X_imp_batched = X_imp.unsqueeze(0)
        else:
            X_imp_batched = X_imp
        if imp_sigma.dim() == 2:
            imp_sigma_use = imp_sigma
        else:
            imp_sigma_use = imp_sigma.squeeze(0) if imp_sigma.size(0) == 1 else imp_sigma.mean(0)
        X_imp_for_mech = X_imp_batched.squeeze(0) if X_imp_batched.size(0) == 1 else X_imp_batched

        B, T, d = X_imp_batched.shape

        # Ensure all tensors have batch dimension
        if X_imp_for_mech.dim() == 2:  # [T, d] -> [1, T, d]
            X_imp_for_mech = X_imp_for_mech.unsqueeze(0)

        # Ensure imp_sigma has shape [B, T, d]
        if imp_sigma_use.dim() == 2:  # [T, d] -> [B, T, d]
            current_sigma = imp_sigma_use.unsqueeze(0).expand(B, -1, -1)
        elif imp_sigma_use.dim() == 3:  # Already [B, T, d]
            current_sigma = imp_sigma_use
        else:  # Single value or 1D -> expand to [B, T, d]
            current_sigma = imp_sigma_use.expand(B, T, d)
            
        # Compute statistics consistently over time dimension
        mu = X_imp_for_mech.mean(dim=1)  # [B, d]
        sigma_mean = current_sigma.mean(dim=1)  # [B, d]
        
        X_centered = X_imp_for_mech - mu.unsqueeze(1)  # [B, T, d]
        X_standardized = X_centered / (current_sigma + 1e-6)  # [B, T, d]
        
        skew = X_standardized.pow(3).mean(dim=1)  # [B, d]
        kurt = X_standardized.pow(4).mean(dim=1) - 3  # [B, d]
        
        # Combine all statistics along feature dimension
        stats = torch.cat([mu, sigma_mean, skew, kurt], dim=1)  # [B, 4d]
        
        # Add batch dimension if needed
        if stats.dim() == 1:  # [4d] -> [1, 4d]
            stats = stats.unsqueeze(0)
        
        # Tri-latent encodings
        ZS = self.encS(X_imp_for_mech) # [B,T,h]
        ZN = self.encN(X_imp_for_mech) # [B,T,h]
        ZB = self.encB(stats)         # [B,h_b]
        
        # Aggregate ZS for structure learning
        ZS_nodes = ZS.mean(1) # [B, h] -> but structure learner expects [B, d, h] or [d,h]
        
        # Reshape ZS for structure learning
        ZS_nodes = ZS.mean(1)  # [B, d, hS]
        
        ZN = self.encN(X_imp_batched).mean(1) # [B, hN]

        if B == 1:
            ZS_struct = ZS_nodes.squeeze(0)
        else:
            ZS_struct = ZS_nodes
        
        # Expand ZB for temporal mechanisms
        ZB_temporal = ZB.unsqueeze(1).expand(-1, T, -1) # [B,T,h_b]

        try:
            env_idx = int(e.item()) if isinstance(e, torch.Tensor) else int(e)
            ZS_struct._env_idx = env_idx
        except Exception:
            pass

        A_all, logits_all = self.struct(ZS_struct)
        if isinstance(A_all, (list, tuple)):
            A_stack = torch.stack(A_all, dim=0)
            A_mean = A_stack.mean(0)
        else:
            A_mean = A_all
            A_stack = A_mean.unsqueeze(0)
        if A_mean.dim() == 3:
            A_mean = A_mean.squeeze(0) if A_mean.size(0) == 1 else A_mean.mean(0)

        X_series = X_imp_for_mech if X_imp_for_mech.dim() == 2 else X_imp_for_mech.mean(0)
        S_hat = self.mech(X_series, A_mean)
        # reconstruct (mean + uncertainty)
        ZB_recon = ZB.squeeze(0) if ZB.dim() == 2 and ZB.size(0) == 1 else ZB
        X_hat, unc, _ = self.recon(S_hat, ZB_recon)

        # losses
        L_rec = masked_l1(X_hat, X, M, unc=unc)
        # compute HSIC across nodes: treat each node's ZS as a sample and match ZN per-node
        ZS_hsic = ZS_struct if ZS_struct.dim() == 2 else ZS_struct.reshape(-1, ZS_struct.size(-1))
        ZN_base = ZN if ZN.dim() == 2 else ZN.unsqueeze(0)
        try:
            if ZN_base.size(0) == 1:
                ZN_rep = ZN_base.repeat(ZS_hsic.size(0), 1)
            else:
                if ZN_base.size(0) != ZS_hsic.size(0):
                    ZN_rep = ZN_base.mean(dim=0, keepdim=True).repeat(ZS_hsic.size(0), 1)
                else:
                    ZN_rep = ZN_base
            L_hsic = hsic_xy(ZS_hsic, ZN_rep)
        except Exception:
            ZN_scalar = ZN_base.mean(dim=0)
            L_hsic = hsic_xy(ZS_hsic.mean(dim=0), ZN_scalar)

        L_acy, _ = self.struct.acyclicity(list(A_stack))
        L_sparse = torch.mean(torch.abs(A_mean))
        total = (L_rec +
                 self.lw["lambda_dis"]*L_hsic +
                 self.lw["lambda_acy"]*L_acy +
                 self.lw["lambda_sparse"]*L_sparse)

        out = {"loss": total, "L_rec": L_rec.detach(), "L_hsic": L_hsic.detach(),
               "L_acy": L_acy.detach(), "L_sparse": L_sparse.detach(), "A": A_mean.detach(),
               "z_s": ZS_nodes, "z_n": ZN, "z_b": ZB}
        return out

    def invariance_penalty(self, Abatch_by_regime):
        if len(Abatch_by_regime)<=1: return torch.tensor(0.0)
        return adj_variance([A for A in Abatch_by_regime])
