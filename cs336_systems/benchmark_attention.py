from __future__ import annotations

import timeit
import pandas as pd
import torch
import torch.nn as nn
from typing import Optional, Tuple
import yaml
from pathlib import Path
from cs336_basics.model import CausalMultiHeadSelfAttention # type: ignore
import torch.cuda.nvtx as nvtx
from torch import Tensor
from einops import rearrange, reduce, einsum, repeat
from jaxtyping import Float, Int
import os
from contextlib import nullcontext

print (os.getcwd())

batch_size = 16
d_model_list = [128] # [16, 32, 64, 128]
seq_len_list = [256, 1024, 4096, 8192, 16384]
n_warmup = 5
n_repeat = 100
result = []
device = "cuda"

def softmax(x:Float[Tensor, "b n s s"], dim=-1):
    x_max = reduce(x, "b s1 s2->b s1 1", "max")
    x = torch.exp(x-x_max)
    x_sum = reduce(x, "b s1 s2->b s1 1", "sum")
    return x/x_sum

def scaled_dot_product_attention_multihead(Q, K, V, mask):
    d_k = Q.shape[-1]
    QK = einsum(Q, K, "b n s1 d_k, b n s2 d_k->b n s1 s2")
    QK = QK / (d_k**0.5)
    if mask is not None:
        QK = QK.masked_fill(mask==0, -torch.inf)
    attention = softmax(QK)
    atten_v = einsum(attention, V, "b n s s, b n s d_k->b n s d_k")
    output = rearrange(atten_v, "b n s d_k->b s (n d_k)")
    return output


def scaled_dot_product_attention(Q, K, V, mask):
    d_k = Q.shape[-1]
    QK = einsum(Q, K, "b s1 d_k, b s2 d_k->b s1 s2")
    QK = QK / (d_k**0.5)
    if mask is not None:
        QK = QK.masked_fill(mask==0, -torch.inf)
    attention = torch.softmax(QK, dim=-1)
    atten_v = einsum(attention, V, "b s s, b s d_k->b s d_k")
    return atten_v

class MHA(nn.Module):
    def __init__(self, d_model:int, num_heads:int, max_seq_len:int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.WQ = nn.Linear(d_model, num_heads * self.d_k)
        self.WK = nn.Linear(d_model, num_heads * self.d_k)
        self.WV = nn.Linear(d_model, num_heads * self.d_k)
        self.WO = nn.Linear(num_heads * self.d_k, d_model)

        self.max_seq_len = max_seq_len
        self.register_buffer("mask", torch.tril(torch.ones(max_seq_len, max_seq_len)))

    def forward(self, x: Float[Tensor, "batch seq d_k"]):
        batch, seq, d_model = x.shape
        Q = self.WQ(x)
        K = self.WK(x)
        V = self.WV(x)

        Q = rearrange(Q, "b s (n d_k)->b n s d_k", n = self.num_heads)
        K = rearrange(K, "b s (n d_k)->b n s d_k", n = self.num_heads)
        V = rearrange(V, "b s (n d_k)->b n s d_k", n = self.num_heads)

        mask = self.mask[:seq, :seq]
        mask = rearrange(mask, "s1 s2->1 1 s1 s2")

        compiled_attention = torch.compile(scaled_dot_product_attention)
        attn_output = compiled_attention(Q, K, V, self.mask)

        return self.WO(attn_output)

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def benchmark_attention(d_model, seq_len):
    compiled = torch.compile(scaled_dot_product_attention)
    print(f"Benchmarking d_model={d_model}, seq_len={seq_len}...")
    q = torch.randn(batch_size, seq_len, d_model, device=device)
    k = torch.randn(batch_size, seq_len, d_model, device=device)
    v = torch.randn(batch_size, seq_len, d_model, device=device)
    for i in range(n_warmup):
        output = compiled(q,k,v,mask=None)
        
    torch.cuda.synchronize()

    forward_times = []
    for i in range(n_repeat):
        start = timeit.default_timer()
        output = compiled(q,k,v,mask=None)
        torch.cuda.synchronize()
        forward_times.append(timeit.default_timer() - start)

    # print (forward_times)
    torch.cuda.reset_peak_memory_stats()
    _ = compiled(q,k,v,mask=None)
    torch.cuda.synchronize()
    memory_before_backward = torch.cuda.max_memory_allocated()/(1024**3)

    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)

    for i in range(n_warmup):
        atten = compiled(q,k,v,mask=None)
        atten.mean().backward()
    torch.cuda.synchronize()

    backward_times = []
    for i in range(n_repeat):
        q.grad = k.grad = v.grad = None
        start = timeit.default_timer()
        atten = compiled(q,k,v,mask=None)
        atten.mean().backward()
        torch.cuda.synchronize()
        backward_times.append(timeit.default_timer() - start)

    torch.cuda.reset_peak_memory_stats()
    atten = compiled(q,k,v,mask=None)
    atten.mean().backward()
    torch.cuda.synchronize()
    memory_in_backward = torch.cuda.max_memory_allocated()/(1024**3)

    avg_forward = round(sum(forward_times)*1000/len(forward_times), 2)
    avg_backward = round(sum(backward_times)*1000/len(backward_times), 2) 

    memory_usage = round(memory_before_backward, 4)
    memory_backward = round(memory_in_backward, 4)

    result.append({
        "d_model": d_model,
        "seq_len": seq_len,
        "forward_time(ms)": avg_forward,
        "backward_time(ms)": avg_backward,
        "memory_before_backward(GB)": memory_usage,
        "memory_in_backward(GB)":memory_backward,
        "status": "Success"
    })

def main():
    from itertools import product
    # Load configuration
    for d_model, seq_len in product(d_model_list, seq_len_list):
        try:
            benchmark_attention(d_model, seq_len)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                mem_atten = batch_size * seq_len * seq_len * 4/(1024**3)
                mem_total = mem_atten + (3 * batch_size * seq_len * d_model * 4)/(1024**3)
                result.append({
                "d_model": d_model,
                "seq_len": seq_len,
                "forward_time(ms)": "OOM",
                "backward_time(ms)": "OOM",
                "memory_before_backward(GB)": "OOM",
                "memory_in_backward(GB)":"OOM",
                "status": "OOM"
            })
                torch.cuda.empty_cache()
                continue
            else:
                raise e
    df = pd.DataFrame(result)
    print ("bench mark result:")
    print (df)
    df.to_csv("./profile/attention_good_softmax_compiled.csv", index=False, sep="\t")

    

if __name__ == "__main__":
    main() 