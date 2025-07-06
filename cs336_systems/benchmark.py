from __future__ import annotations

import timeit
import torch
import torch.nn as nn
from typing import Optional, Tuple
import yaml
from pathlib import Path
from cs336_basics.model import BasicsTransformerLM # type: ignore
from cs336_basics.optimizer import AdamW, get_cosine_lr
import torch.cuda.nvtx as nvtx
import os
from contextlib import nullcontext

print (os.getcwd())

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_random_batch(batch_size: int, context_length: int, vocab_size: int, device: str) -> torch.Tensor:
    """Create a random batch of token IDs for benchmarking."""
    return torch.randint(0, vocab_size, (batch_size, context_length), device=device)


def benchmark_model(
    model: nn.Module,
    batch: torch.Tensor,
    warmup_steps: int,
    benchmark_steps: int,
    forward_only: bool,
    device: str,
    autocast: bool
):
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
    print ("warmup")
    optimizer = AdamW(model.parameters(), lr=1e-3)
    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if autocast else nullcontext()
    # Warmup
    with ctx: 
        for _ in range(warmup_steps):
            lr = 0.1
            for group in optimizer.param_groups:
                group['lr'] = lr
            outputs = model(batch)
            if not forward_only:
                optimizer.zero_grad()
                loss = outputs.mean()
                loss.backward()
                optimizer.step()
            if device == "cuda":
                torch.cuda.synchronize()
    
    # Benchmark
    print ("benchmark:")
    forward_times = []
    backward_times = []
    with ctx: 
        for _ in range(benchmark_steps):
            lr = 0.1
            for group in optimizer.param_groups:
                group["lr"] = lr
            # Forward pass
            start_time = timeit.default_timer()
            outputs = model(batch)
            if device == "cuda":
                torch.cuda.synchronize()
            forward_time = timeit.default_timer() - start_time
            forward_times.append(forward_time)
            
            if not forward_only:
                # Backward pass
                start_time = timeit.default_timer()
                optimizer.zero_grad()
                loss = outputs.mean()
                loss.backward()
                optimizer.step()
                if device == "cuda":
                    torch.cuda.synchronize()
                backward_time = timeit.default_timer() - start_time
                backward_times.append(backward_time)

        torch.cuda.reset_peak_memory_stats()
        outputs = model(batch)
        torch.cuda.synchronize()
        memory_before_backward = torch.cuda.max_memory_allocated()/(1024**3)

        memory_backward=0
        if not forward_only:
            torch.cuda.reset_peak_memory_stats()
            optimizer.zero_grad()
            loss = outputs.mean()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            memory_backward = torch.cuda.max_memory_allocated()/(1024**3)
    
    avg_forward_time = sum(forward_times) / len(forward_times)
    avg_backward_time = sum(backward_times) / len(backward_times) if not forward_only else 0.0
    
    return avg_forward_time, avg_backward_time, memory_before_backward, memory_backward




def main():
    # Load configuration
    config_path = Path("configures/2.7B.yaml")
    config = load_config(config_path)
    
    # Initialize model
    model = BasicsTransformerLM(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=config["rope_theta"],
    ).to(config["device"])

    #model = torch.compile(model)
    
    # Create random batch
    batch = create_random_batch(
        config["batch_size"],
        config["context_length"],
        config["vocab_size"],
        config["device"]
    )
    
    # Run benchmarkl save
    for autocast in [False]:
        forward_time, backward_time, forward_memory, backward_memory = benchmark_model(
            model,
            batch,
            config["warmup_steps"],
            config["benchmark_steps"],
            config["forward_only"],
            config["device"],
            autocast
        )
        
        # Print results
        print(f"\nBenchmark Results:")
        print(f"Model Configuration:")
        print(f"autocast to bf16: {autocast}")
        # print(f"  - Layers: {config['num_layers']}")
        # print(f"  - Model Dimension: {config['d_model']}")
        # print(f"  - Heads: {config['num_heads']}")
        # print(f"  - FF Dimension: {config['d_ff']}")
        # print(f"  - Context Length: {config['context_length']}")
        # print(f"  - Batch Size: {config['batch_size']}")
        print(f"\nTiming Results:")
        print(f"  - Average Forward Time: {forward_time*1000:.2f} ms")
        print(f"forward peak memory: {forward_memory}")
        print(f"backward memory:{backward_memory}")
        if not config["forward_only"]:
            print(f"  - Average Backward Time: {backward_time*1000:.2f} ms")
            print(f"  - Total Time: {(forward_time + backward_time)*1000:.2f} ms")


if __name__ == "__main__":
    main() 