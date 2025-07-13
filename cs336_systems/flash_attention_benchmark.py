import torch
import triton
import pandas as pd
from typing import Tuple
import logging

logging.basicConfig(level=logging.DEBUG, force=True)
torch._dynamo.config.verbose = True
torch.autograd.set_detect_anomaly(True)
torch._inductor.config.debug = True

from cs336_systems.flash_attention import TritonFlashAttention2
from cs336_basics.model import scaled_dot_product_attention

def generate_inputs(batch_size: int, seq_len: int, dim: int, dtype: torch.dtype, device: str = "cuda") -> Tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    q = torch.randn(batch_size, seq_len, dim, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, seq_len, dim, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, seq_len, dim, device=device, dtype=dtype, requires_grad=True)
    do = torch.randn(batch_size, seq_len, dim, device=device, dtype=dtype)
    return q, k, v, do

def triton_flash_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = True) -> torch.Tensor:
    return TritonFlashAttention2.apply(q, k, v, is_causal)

def pytorch_vanilla_attn(q, k, v, is_causal=True):
    if is_causal:
        mask = torch.triu(torch.ones(q.shape[-2], k.shape[-2], device=q.device, dtype=torch.bool), diagonal=1)
        return scaled_dot_product_attention(q, k, v, mask=mask)
    else:
        return scaled_dot_product_attention(q, k, v, mask=None)

def pytorch_vanilla_bwd(q, k, v, is_causal, do):
    out = pytorch_vanilla_attn(q, k, v, is_causal)
    out.backward(do)

def pytorch_flash_bwd(q, k, v, is_causal, do):
    out = triton_flash_attn(q, k, v, is_causal)
    out.backward(do)

def benchmark_configs():
    seq_lengths = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    dims = [16, 32, 64, 128]
    dtypes = [torch.bfloat16, torch.float32]

    results_by_dtype = {dtype: [] for dtype in dtypes}

    for dtype in dtypes:
        for seq_len in seq_lengths:
            if dtype == torch.float32 and seq_len == 65536:
                continue

            for dim in dims:
                # Skip configurations that would exceed GPU memory
                if seq_len * dim * (8 if dtype == torch.float32 else 4) > 2**31:
                    continue

                print(f"Benchmarking seq_len={seq_len}, dim={dim}, dtype={dtype}")

                # Generate inputs
                q, k, v, do = generate_inputs(1, seq_len, dim, dtype)
                print("Generated inputs")

                # These are not actually being used
                if seq_len <= 2048:
                    q_tile_size = 64
                    k_tile_size = 64
                elif seq_len <= 8192:
                    q_tile_size = 32
                    k_tile_size = 32
                else:
                    q_tile_size = 16
                    k_tile_size = 16

                # Benchmark vanilla pytorch attention
                pytorch_fwd = triton.testing.do_bench(lambda: pytorch_vanilla_attn(q, k, v, False))
                pytorch_both = triton.testing.do_bench(lambda: (pytorch_vanilla_bwd(q, k, v, False, do), torch.cuda.synchronize()))
                pytorch_bwd = pytorch_both - pytorch_fwd
                # Benchmark partial Triton
                triton_fwd = triton.testing.do_bench(lambda: triton_flash_attn(q, k, v, False))
                triton_both = triton.testing.do_bench(lambda: (pytorch_flash_bwd(q, k, v, False, do), torch.cuda.synchronize()))
                triton_bwd = triton_both - triton_fwd
                results_by_dtype[dtype].append({
                    'seq_len': seq_len,
                    'dim': dim,
                    'pytorch_forward_ms': pytorch_fwd,
                    'pytorch_backward_ms': pytorch_bwd,
                    'pytorch_total_ms': pytorch_both,
                    'triton_forward_ms': triton_fwd,
                    'triton_backward_ms': triton_bwd,
                    'triton_total_ms': triton_both,
                })

                print(f"pytorch_vanilla_attn: {pytorch_fwd} ms, {pytorch_bwd} ms, {pytorch_fwd + pytorch_bwd} ms")
                print(f"triton_flash_attn: {triton_fwd} ms, {triton_bwd} ms, {triton_fwd + triton_bwd} ms")

                torch.cuda.empty_cache()
                print("Cleared GPU memory")

    # Create DataFrames and convert to LaTeX tables
    latex_tables = {}
    for dtype, results in results_by_dtype.items():
        df = pd.DataFrame(results)
        latex_table = df.to_latex(index=False, float_format=lambda x: '{:.2f}'.format(x))
        latex_tables[str(dtype).split('.')[-1]] = latex_table
        print("\nLaTeX Table:")
        print(latex_table)


if __name__ == "__main__":
    benchmark_configs()