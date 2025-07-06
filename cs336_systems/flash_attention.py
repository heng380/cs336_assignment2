import torch
import triton
import triton.language as tl
import math
from einops import rearrange
from typing import Tuple, Optional

#############################################
# Part (a): PyTorch FlashAttention-2 Forward Pass
#############################################

class FlashAttentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_casual=False, scale=None):
        ctx.save_for_backward(q,k,v)
        