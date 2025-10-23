import torch, numpy as np
from ..models.utils import threshold_adj
from ..models import disentanglement

def train_epoch(model, loader, optimizer, inv_weight=0.0, device="cpu"):
    model.train()
    agg = {"loss":0.0, "L_rec":0.0, "L_acy":0.0}
    # process minibatches; model.forward_batch expects a single example dict (T,d)
    for batch in loader:
        # move tensors to device
        for k in batch:
            batch[k] = batch[k].to(device)
        B = batch["X"].shape[0] if batch["X"].dim()>=3 else 1
        per_sample_As = {}
        total_loss = None  # Start as None instead of 0.0
        total_L_rec = 0.0
        total_L_acy = 0.0
        # iterate samples to preserve current single-sample forward implementation
        for i in range(B):
            sample = {k: (batch[k][i] if batch[k].dim()>0 else batch[k]) for k in batch}
            out = model.forward_batch(sample)
            e = int(sample["e"].item())
            per_sample_As.setdefault(e, []).append(out["A"])
            # Accumulate loss as tensor to preserve computation graph
            if total_loss is None:
                total_loss = out["loss"]
            else:
                total_loss = total_loss + out["loss"]
            total_L_rec += float(out.get("L_rec", 0.0))
            total_L_acy += float(out.get("L_acy", 0.0))
        # invariance penalty per minibatch (variance across regime means in this minibatch)
        inv_pen = 0.0
        if inv_weight>0 and len(per_sample_As)>1:
            means = [torch.stack(v,0).mean(0) for v in per_sample_As.values()]
            inv_pen = model.invariance_penalty(means)
            total_loss = total_loss + inv_weight * inv_pen

        # backward once per minibatch
        optimizer.zero_grad()
        # Ensure we have a valid loss tensor before backward
        if total_loss is None or not isinstance(total_loss, torch.Tensor):
            # This should not happen, but handle gracefully
            print("[WARNING] Invalid loss tensor encountered, skipping batch")
            continue
        total_loss.backward()
        # --- gradient-norm logging: record per-parameter grad L2 norms ---
        try:
            grad_norms = []
            for name, p in model.named_parameters():
                if p.grad is None:
                    continue
                gnorm = p.grad.detach().norm().item()
                grad_norms.append((name, gnorm))
            if len(grad_norms) > 0:
                # summarize: max, mean and top-3
                norms = [g for _, g in grad_norms]
                max_n = max(norms)
                mean_n = sum(norms) / len(norms)
                top3 = sorted(grad_norms, key=lambda x: x[1], reverse=True)[:3]
                top3s = ", ".join([f"{n}:{g:.3g}" for n,g in top3])
                print(f"[GRAD] max={max_n:.3g} mean={mean_n:.3g} top3=[{top3s}]")
        except Exception:
            # be conservative: don't crash training if logging fails
            pass
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.struct.step += 1
        agg["loss"] += float(total_loss)
        agg["L_rec"] += total_L_rec
        agg["L_acy"] += total_L_acy

    disentanglement.decay_disentangle_temperature()
    n = len(loader)
    for k in agg: agg[k] /= n
    return agg

@torch.no_grad()
def eval_epoch(model, loader, A_true=None, thr=0.5, device="cpu"):
    model.eval()
    As = []
    for batch in loader:
        for k in batch: batch[k] = batch[k].to(device)
        # support batched loader by iterating samples
        B = batch["X"].shape[0] if batch["X"].dim()>=3 else 1
        for i in range(B):
            sample = {k: (batch[k][i] if batch[k].dim()>0 else batch[k]) for k in batch}
            out = model.forward_batch(sample)
            As.append(out["A"].cpu().numpy())
    if len(As)==0:
        # no data in loader: return zero adjacency of correct shape
        d = getattr(model.struct, "d", None)
        if d is None:
            A_mean = np.zeros((1,1))
        else:
            A_mean = np.zeros((d,d))
    else:
        A_mean = np.stack(As,0).mean(0)
    A_bin = (A_mean>=thr).astype(int)
    metrics = {"A_mean":A_mean, "A_bin":A_bin}
    if A_true is not None:
        from .metrics import shd
        metrics["shd"] = shd(A_bin, A_true)
    return metrics
