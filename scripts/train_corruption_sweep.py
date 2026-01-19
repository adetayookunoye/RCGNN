#!/usr/bin/env python3
"""
Train RC-GNN across corruption levels with multiple seeds.
This trains FROM SCRATCH at each corruption level.

DDP mode:
    torchrun --nproc_per_node=4 scripts/train_corruption_sweep.py --ddp --corruption 0.4 --seed 1
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import json
import argparse
from tqdm import tqdm

from src.dataio.loaders import load_synth, SynthDataset
from src.models.rcgnn import RCGNN
from src.models.utils import evaluate_adjacency, best_threshold_f1


def setup_ddp():
    """Initialize DDP."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    """Cleanup DDP."""
    dist.destroy_process_group()


def is_main_process():
    """Check if this is the main process."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def train_one_run(data_dir, seed, epochs=40, device='cpu', ddp=False):
    """Train one RC-GNN run with given seed."""
    local_rank = 0
    if ddp:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = f"cuda:{local_rank}"
    
    torch.manual_seed(seed + local_rank)
    np.random.seed(seed)
    
    # Load data
    dataset = SynthDataset(root=data_dir, split='train')
    
    if ddp:
        sampler = DistributedSampler(dataset)
        loader = DataLoader(dataset, batch_size=64, sampler=sampler, num_workers=2)
    else:
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Get dimensions
    sample = dataset[0]
    N, T, d = sample['X'].shape
    
    # Create model
    model = RCGNN(
        d_nodes=d,
        d_signal=32,
        d_noise=16,
        d_bias=8,
        lambda_sparse=0.0001,
        lambda_disentangle=0.1,
        gamma_acyclic=0.01,
        device=device
    ).to(device)
    
    if ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    model_for_eval = model.module if ddp else model
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    best_f1 = 0
    best_auprc = 0
    best_A = None
    patience = 15
    no_improve = 0
    
    for epoch in range(epochs):
        if ddp:
            loader.sampler.set_epoch(epoch)
        
        model.train()
        epoch_loss = 0
        
        for batch in loader:
            X = batch['X'].to(device)
            M = batch['M'].to(device)
            e = batch['e'].to(device)
            
            optimizer.zero_grad()
            loss_dict = model_for_eval.compute_loss(X, M, e) if not ddp else model.module.compute_loss(X, M, e)
            loss = loss_dict['total']
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # Evaluation (main process only)
        if is_main_process():
            model_for_eval.eval()
            with torch.no_grad():
                A_pred = model_for_eval.get_adjacency().cpu().numpy()
                A_true = np.load(Path(data_dir) / "A_true.npy")
            
            metrics = evaluate_adjacency(A_pred, A_true)
            best_f1_info = best_threshold_f1(A_pred, A_true)
            
            # Update best
            if best_f1_info['f1'] > best_f1:
                best_f1 = best_f1_info['f1']
                best_A = A_pred.copy()
                no_improve = 0
            else:
                if epoch >= 20:  # Only count after warmup
                    no_improve += 1
            
            if metrics['auprc'] > best_auprc:
                best_auprc = metrics['auprc']
        
        # Early stopping
        if no_improve >= patience and epoch >= 20:
            break
    
    # Final evaluation
    A_true = np.load(Path(data_dir) / "A_true.npy")
    final_metrics = evaluate_adjacency(best_A, A_true)
    best_f1_info = best_threshold_f1(best_A, A_true)
    
    return {
        'f1': best_f1_info['f1'],
        'precision': best_f1_info['precision'],
        'recall': best_f1_info['recall'],
        'auprc': final_metrics['auprc'],
        'auroc': final_metrics['auroc'],
        'shd': final_metrics['shd'],
        'nnz': final_metrics['nnz'],
        'epochs_trained': epoch + 1
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corruption', type=float, required=True, help='Corruption rate (0.0-0.4)')
    parser.add_argument('--seed', type=int, required=True, help='Random seed')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--ddp', action='store_true', help='Enable DDP training')
    args = parser.parse_args()
    
    # Setup DDP if enabled
    if args.ddp:
        setup_ddp()
    
    # Construct data directory
    data_dir = f"data/interim/uci_air_{int(args.corruption*100):02d}"
    
    if is_main_process():
        print("=" * 60)
        print(f"TRAINING RC-GNN: {args.corruption:.0%} missing, seed={args.seed}")
        print("=" * 60)
        print(f"Data: {data_dir}")
        print(f"Epochs: {args.epochs}")
        print(f"Device: {args.device}")
        if args.ddp:
            print(f"DDP: {dist.get_world_size()} GPUs")
        print()
    
    # Train
    results = train_one_run(data_dir, args.seed, args.epochs, args.device, args.ddp)
    
    # Save results (main process only)
    if is_main_process():
        output_dir = Path("artifacts") / "corruption_sweep" / f"corruption_{int(args.corruption*100):02d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"rcgnn_seed{args.seed}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print()
        print("=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"F1:       {results['f1']:.3f}")
        print(f"AUPRC:    {results['auprc']:.3f}")
        print(f"SHD:      {results['shd']}")
        print(f"Epochs:   {results['epochs_trained']}")
        print()
        print(f"Saved to: {output_file}")
    
    # Cleanup DDP
    if args.ddp:
        cleanup_ddp()

if __name__ == "__main__":
    main()
