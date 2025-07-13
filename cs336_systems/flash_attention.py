import torch
import triton
import triton.language as tl
from math import ceil
from einops import rearrange, einsum
from typing import Tuple, Optional
from torch.autograd import Function

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal:tl.constexpr
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1,0)
    )

    kt_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index*stride_kb,
        shape=(D, N_KEYS),
        strides=(stride_kd, stride_kk),
        offsets=(0,0),
        block_shape=(D,K_TILE_SIZE),
        order=(0,1)
    )

    v_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index*stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0,0),
        block_shape=(K_TILE_SIZE, D),
        order=(1,0)
    )

    q_tile = tl.load(Q_block_ptr, boundary_check=(0,1), padding_option="zero")
    original_q_tile_dtype = q_tile.dtype

    q_tile = (q_tile * scale).to(original_q_tile_dtype)

    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    max_i = tl.full((Q_TILE_SIZE,), -float('inf'), dtype=tl.float32)
    o_i = tl.zeros((Q_TILE_SIZE,D), dtype=tl.float32)

    max_tiles = (query_tile_index+1) * Q_TILE_SIZE if is_causal else N_KEYS

    for k_tile_index in range(tl.cdiv(max_tiles, K_TILE_SIZE)):
        kt_tile = tl.load(kt_block_ptr, boundary_check=(0,1), padding_option="zero")
        v_tile = tl.load(v_block_ptr, boundary_check=(0,1), padding_option="zero")

        s_ij = tl.zeros((Q_TILE_SIZE, K_TILE_SIZE), dtype=tl.float32)
        s_ij = tl.dot(q_tile, kt_tile, acc=s_ij)

        if is_causal:
            q_range = query_tile_index * Q_TILE_SIZE + tl.arange(0,Q_TILE_SIZE)
            k_range = k_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            causal_mask = q_range[:, None] < k_range[None, :]
            s_ij = tl.where(causal_mask, -1e6, s_ij)

        new_max_i = tl.maximum(max_i, tl.max(s_ij, axis=-1))
        p_tilde = tl.math.exp(s_ij - new_max_i[:, None])
        exp_mi_diff = tl.math.exp(max_i - new_max_i)
        l_i = l_i*exp_mi_diff + tl.sum(p_tilde, -1)
        o_i = exp_mi_diff[:, None] * o_i
        o_i = tl.dot(p_tilde.to(v_tile.dtype), v_tile, acc=o_i)

        max_i = new_max_i

        kt_block_ptr = kt_block_ptr.advance((0, K_TILE_SIZE))
        v_block_ptr = v_block_ptr.advance((K_TILE_SIZE, 0))

    o_i /= l_i[:, None]
    l_i = tl.log(l_i) + max_i

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1,0)
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index*stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index*Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )

    tl.store(O_block_ptr, o_i, boundary_check=(0,1))
    tl.store(L_block_ptr, l_i, boundary_check=(0,))
#%%

#%%

class TritonFlashAttention2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, mask=None):
        b, n_q, d = q.shape
        b, n_k, d = k.shape
        assert q.is_cuda and k.is_cuda and v.is_cuda, "Expected CUDA tensors"
        assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous(), "Our pointer arithmetic will assume contiguous"

        ctx.is_causal = mask
        ctx.Q_TILE_SIZE = 16
        ctx.K_TILE_SIZE = 16

        o = torch.empty(b, n_q, d, device=q.device)
        l = torch.empty(b, n_q, device=q.device)

        flash_fwd_kernel[(triton.cdiv(n_q, ctx.Q_TILE_SIZE), b)](
            q, k, v,
            o, l,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            l.stride(0), l.stride(1),
            N_QUERIES=n_q, N_KEYS=n_k,
            D=d, scale=1/d**0.5,
            Q_TILE_SIZE=ctx.Q_TILE_SIZE,
            K_TILE_SIZE=ctx.K_TILE_SIZE,
            is_causal=ctx.is_causal
        )

        ctx.save_for_backward(o, l, q, k, v)
        return o
    @staticmethod
    def backward(ctx, dO):
        return FlashAttention2Forward.backward(ctx, dO)


#############################################
# Part (a): PyTorch FlashAttention-2 Forward Pass
#############################################
class FlashAttention2Forward(Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        """
        FlashAttention-2前向传播实现
        参数:
            query: [batch_size, seq_len_q, head_dim]
            key: [batch_size, seq_len_k, head_dim]
            value: [batch_size, seq_len_k, head_dim]
            is_causal: 是否使用因果掩码(本任务可忽略)
        返回:
            output: [batch_size, seq_len_q, head_dim]
            logsumexp: [batch_size, seq_len_q]  # 用于反向传播的logsumexp值
        """
        # 保存输入供反向传播使用
        b_q, b_k = 16, 16
        b, n_q, d = q.shape
        b, n_k, d = k.shape

        t_q_range = ceil(n_q / b_q)
        t_k_range = ceil(n_k / b_k)

        O = torch.zeros(b, n_q, d, device=q.device, dtype=q.dtype)
        L = torch.zeros(b, n_q, device=q.device, dtype=q.dtype)

        ctx.is_causal = is_causal
        
        for i in range(t_q_range):
            start_i = i*b_q
            end_i = min(start_i + b_q, n_q)

            q_i = q[:, start_i:end_i, :]
            
            prev_max_i = torch.full((b, b_q), -float('inf'), dtype=q.dtype, device=q.device)  # max 
            prev_sumexp_i = torch.zeros((b, b_q), dtype=q.dtype, device=q.device)                # logsumexp
            prev_o_i = torch.zeros((b, b_q, d), dtype=q.dtype, device=q.device)             # output

            for j in range(t_k_range):
                start_k = j*b_k
                end_k = min(start_k+b_k, n_k)

                k_j = k[:, start_k:end_k, :]
                v_j = v[:, start_k:end_k, :]

                s_ij = einsum(q_i, k_j, "b s1 d, b s2 d->b s1 s2") / d**0.5

                max_i = torch.max(prev_max_i, torch.amax(s_ij, dim=-1))
                exp_sij_re_max = torch.exp(s_ij - max_i.unsqueeze(-1))
                exp_mi_factor = torch.exp(prev_max_i - max_i)
                sumexp_i = exp_mi_factor * prev_sumexp_i + torch.sum(exp_sij_re_max, dim=-1)
                o_i = exp_mi_factor.unsqueeze(-1) * prev_o_i + einsum(exp_sij_re_max, v_j, 'b s1 s, b s d->b s1 d')

                prev_max_i = max_i
                prev_sumexp_i = sumexp_i
                prev_o_i = o_i

            O[:,start_i:end_i,:] = einsum(sumexp_i.reciprocal(), o_i, '... k, ... k d -> ... k d')
            L[:,start_i:end_i] = torch.log(sumexp_i) + max_i
        ctx.save_for_backward(O, L, q, k, v)
        return O

    @staticmethod
    @torch.compile(fullgraph=True)
    def backward(ctx, dO):
        O, L, Q, K, V = ctx.saved_tensors
        d = K.shape[-1]
        S = einsum(Q, K, "b s1 d, b s2 d->b s1 s2")/d**0.5
        if ctx.is_causal:
            mask = torch.tril(torch.ones(S.shape[-2], S.shape[-1], device=S.device)) # 下三角为1
            S = S.masked_fill(mask==0, -torch.inf)
        P = torch.exp(S - L.unsqueeze(-1))
        dV = einsum(P, dO, 'b s1 s2, b s1 d->b s2 d')
        dP = einsum(V, dO, 'b s2 d, b s1 d->b s1 s2')

        D = (O * dO).sum(dim=-1)
        dS = P * (dP - D.unsqueeze(-1))

        dQ = einsum(dS, K, 'b s1 s2, b s2 d->b s1 d')/d**0.5
        dK = einsum(dS, Q, 'b s1 s2, b s1 d->b s2 d')/d**0.5

        return dQ, dK, dV, None





