from __future__ import annotations

import torch.cuda.nvtx as nvtx

import timeit
import torch
import torch.nn as nn
from typing import Optional, Tuple
import yaml
from pathlib import Path
import sys
sys.path.append("/home/aiscuser/repos/assignment2-systems/cs336-basics")
from cs336_basics.model import BasicsTransformerLM # type: ignore
from cs336_basics.optimizer import AdamW, get_cosine_lr
from benchmark import create_random_batch, load_config
import torch.cuda.nvtx as nvtx
import os
from contextlib import nullcontext


print (os.getcwd())

def benchmark_model_with_nvtx(
    model: nn.Module,
    batch: torch.Tensor,
    warmup_steps: int,
    benchmark_steps: int,
    forward_only: bool,
    device: str,
) -> Tuple[float, float]:
    """
    Benchmark the model's forward and backward passes.
    
    Args:
        model: The model to benchmark
        batch: Input batch tensor
        warmup_steps: Number of warmup steps
        benchmark_steps: Number of steps to benchmark
        forward_only: Whether to only benchmark forward pass
        device: Device to run on
        
    Returns:
        Tuple of (forward_time, backward_time) in seconds
    """
    optimizer = AdamW(model.parameters(), lr=1e-3)
    max_lr = 1e-3
    min_lr = 1e-5
    warmup_iters = warmup_steps
    total_iters = warmup_steps + benchmark_steps
    global_step = 0


    print ("start warmup:")
    # Warmup
    with nvtx.range("Warmup"):
        for _ in range(warmup_steps):
            lr = get_cosine_lr(global_step, max_lr, min_lr, warmup_iters, total_iters)
            for group in optimizer.param_groups:
                group['lr'] = lr
            
            outputs = model(batch)
            if not forward_only:
                loss = outputs.mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            if device == "cuda":
                torch.cuda.synchronize()
            
        global_step += 1
    
    torch.cuda.memory._record_memory_history(max_entries=10000)
    ctx = nullcontext()
    print ("start training:")
    with ctx:
        for step in range(benchmark_steps):
            with nvtx.range(f"step_{step}"):
                with nvtx.range("get_consine_lr"):
                    lr = get_cosine_lr(global_step, max_lr, min_lr, warmup_iters, total_iters)
                for group in optimizer.param_groups:
                    group['lr'] = lr

            with nvtx.range("Forward"):
                # with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(batch)
                if device == "cuda":
                    torch.cuda.synchronize()
            
            if not forward_only:
                loss = outputs.mean()
                with nvtx.range("backward"):
                    loss.backward()
                with nvtx.range("optimizer step"):
                    optimizer.step()
                    optimizer.zero_grad()
                if device == "cuda":
                    torch.cuda.synchronize()
            global_step += 1
    
    torch.cuda.memory._dump_snapshot("./memory_profile/memory_snapshot_fw_2.7b_no_auto.pickle")
    torch.cuda.memory._record_memory_history(enabled=None)




def main():
    # Load configuration
    config_path = Path("configures/2.7B.yaml")
    config = load_config(config_path)

    if torch.cuda.is_available():
        print ("run on gpu")
    
    # Initialize model
    with nvtx.range("define model"):
        model = BasicsTransformerLM(
            vocab_size=config["vocab_size"],
            context_length=config["context_length"],
            d_model=config["d_model"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            d_ff=config["d_ff"],
            rope_theta=config["rope_theta"],
        ).to(config["device"])
    
    # Create random batch
    with nvtx.range("define input"):
        batch = create_random_batch(
            config["batch_size"],
            config["context_length"],
            config["vocab_size"],
            config["device"]
        )
    
    # Run benchmark
    with nvtx.range("bench mark"):
        benchmark_model_with_nvtx(
            model,
            batch,
            config["warmup_steps"],
            config["benchmark_steps"],
            config["forward_only"],
            config["device"]
        )
    
if __name__ == "__main__":
    main() 