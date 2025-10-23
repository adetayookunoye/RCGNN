import torch

def make_optimizer(params, cfg):
    # defensive: coerce yaml values to numeric types
    lr = float(cfg["optimizer"].get("lr", 1e-3))
    betas = tuple(float(b) for b in cfg["optimizer"].get("betas", [0.9,0.999]))
    weight_decay = float(cfg["optimizer"].get("weight_decay", 0.0))
    opt = torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)
    return opt
