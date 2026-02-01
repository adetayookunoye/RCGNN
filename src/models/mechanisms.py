import torch, torch.nn as nn

class Mechanisms(nn.Module):
    def __init__(self, d, hidden=64):
        super().__init__()
        self.edge_mlp = nn.Sequential(nn.Linear(1, 8), nn.ReLU(), nn.Linear(8,1))
        # node_mlp maps a scalar per-node aggregated feature to a scalar output
        self.node_mlp = nn.Sequential(nn.Linear(1, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, ZS_time, A):
        # ZS_time: [T,d], features per node over time (here just X_imp)
        # For simplicity, predict S_hat(t) = node_mlp( A^T * ZS_time(t) )
        T, d = ZS_time.shape
        out = []
        for t in range(T):
            agg = torch.matmul(ZS_time[t], A) # [d]
            # apply node_mlp per node
            agg_in = agg.unsqueeze(-1) # [d,1]
            node_out = self.node_mlp(agg_in).squeeze(-1) # [d]
            out.append(node_out)
        return torch.stack(out, dim=0) # [T,d]
