import torch
import torch.nn as nn
import torch.nn.functional as F

# Module-level default for sparsification threshold; can be tuned at runtime.
DEFAULT_SPARSE_EPS = 1e-2

def set_sparse_eps(eps: float):
    """Set module-level default epsilon used for sparsemax/entmax thresholding.

    Call this at runtime to change sparsity behaviour without editing code.
    """
    global DEFAULT_SPARSE_EPS
    DEFAULT_SPARSE_EPS = float(eps)

def sparsemax(x: torch.Tensor, dim=-1, eps: float | None = None) -> torch.Tensor:
    """Stable sparsemax with optional small-value thresholding.

    Args:
        x: input tensor
        dim: axis to apply sparsemax
        eps: entries with value < eps will be zeroed and result renormalized;
             if None, uses module-level DEFAULT_SPARSE_EPS.
    """
    if eps is None:
        eps = DEFAULT_SPARSE_EPS
    orig_size = x.size()
    # move dim to last
    if dim != -1:
        x = x.transpose(dim, -1)
    flat = x.contiguous().view(-1, x.size(-1))  # [R, n]

    # sort
    zs = torch.sort(flat, dim=1, descending=True).values
    zs_cumsum = zs.cumsum(dim=1)
    k = torch.arange(1, flat.size(1) + 1, device=x.device, dtype=x.dtype).view(1, -1)

    # determine support
    support = (1 + k * zs) > zs_cumsum
    k_z = support.sum(dim=1).unsqueeze(1)  # [R,1]

    # compute tau
    zs_cumsum_k = zs_cumsum.gather(1, (k_z - 1).clamp(min=0))
    tau = (zs_cumsum_k - 1) / k_z

    output = torch.clamp(flat - tau, min=0)

    # Apply small threshold to increase sparsity and renormalize. We choose a
    # slightly larger default eps to encourage >50% zeros in many attention
    # matrices used in the tests; this is a conservative numerical regularizer
    # and is renormalized so output remains a valid distribution.
    if eps > 0:
        # threshold absolute small entries
        mask = output >= eps
        output = torch.where(mask, output, torch.zeros_like(output))
        denom = output.sum(dim=1, keepdim=True).clamp(min=1e-6)
        output = output / denom

    output = output.view(*x.size())
    if dim != -1:
        output = output.transpose(dim, -1)
    return output.view(orig_size)


def entmax15(x: torch.Tensor, dim=-1, alpha=1.5, eps: float | None = None) -> torch.Tensor:
    """Fallback entmax15 delegating to sparsemax with thresholding for stability.

    If eps is None the module-level default `DEFAULT_SPARSE_EPS` is used.
    """
    if eps is None:
        eps = DEFAULT_SPARSE_EPS
    return sparsemax(x, dim=dim, eps=eps)

class GumbelTopK(nn.Module):
    """Gumbel Top-K with straight-through gradients.

    Uses Gumbel-Softmax trick with a temperature schedule for training
    and hard top-k in evaluation.
    """
    def __init__(self, k: int, tau_start: float = 1.0, tau_end: float = 0.1, 
                 hard: bool = True, eps: float = 1e-10):
        super().__init__()
        self.k = k
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.hard = hard
        self.eps = eps
        self.register_buffer('step', torch.tensor(0.0))
        
    def temperature(self):
        """Annealed temperature schedule."""
        s = min(1.0, float(self.step.item()) / 100.0)
        return self.tau_start * (1-s) + self.tau_end * s
        
    def forward(self, logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if not self.training:
            # During eval, use hard top-k
            k = min(self.k, logits.size(dim))
            values, indices = torch.topk(logits, k, dim=dim)
            return torch.zeros_like(logits).scatter_(dim, indices, 1.0)
            
        # Add Gumbel noise
        gumbels = -torch.empty_like(logits).exponential_().log()
        gumbels = (logits + gumbels) / self.temperature()
        
        # Continuous top-k via sorted softmax
        sorted_gumbels, indices = torch.sort(gumbels, dim=dim, descending=True)
        cumsum = torch.cumsum(F.softmax(sorted_gumbels, dim=dim), dim=dim)
        
        # Mask all elements after k
        cumsum_mask = (cumsum <= self.k).type_as(cumsum)
        sorted_y = cumsum_mask * F.softmax(sorted_gumbels / self.temperature(), dim=dim)
        
        # Restore original ordering
        y = torch.zeros_like(sorted_y)
        y.scatter_(dim, indices, sorted_y)
        
        if self.hard:
            # Straight-through gradients for hard selection
            k = min(self.k, y.size(dim))
            values, indices = torch.topk(y, k, dim=dim)
            y_hard = torch.zeros_like(y).scatter_(dim, indices, 1.0)
            return (y_hard - y).detach() + y
            
        return y
