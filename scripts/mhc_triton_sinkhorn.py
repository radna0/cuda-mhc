import torch
import triton
import triton.language as tl

@triton.jit
def sinkhorn_kernel_fwd(
    log_alpha_ptr, p_ptr,
    stride_b, stride_l, stride_n,
    N_VAL: tl.constexpr,
    ITERS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # This kernel performs Sinkhorn-Knopp in log-space on-chip
    token_idx = tl.program_id(0)
    
    # Offsets for N x N matrix
    offs_n = tl.arange(0, BLOCK_SIZE)
    offs_m = tl.arange(0, BLOCK_SIZE)
    
    # Load log_alpha
    # We assume N=4, BLOCK_SIZE=4
    mask = (offs_n < N_VAL)[:, None] & (offs_m < N_VAL)[None, :]
    log_alpha = tl.load(log_alpha_ptr + token_idx * stride_l + offs_n[:, None] * stride_n + offs_m[None, :], mask=mask, other=-1e9)
    
    # Sinkhorn Iterations
    curr_log_alpha = log_alpha
    for _ in range(ITERS):
        # Row norm
        row_lse = tl.log(tl.sum(tl.exp(curr_log_alpha), axis=1))
        curr_log_alpha = curr_log_alpha - row_lse[:, None]
        # Col norm
        col_lse = tl.log(tl.sum(tl.exp(curr_log_alpha), axis=0))
        curr_log_alpha = curr_log_alpha - col_lse[None, :]
        
    p = tl.exp(curr_log_alpha)
    tl.store(p_ptr + token_idx * stride_l + offs_n[:, None] * stride_n + offs_m[None, :], p, mask=mask)

def triton_sinkhorn(log_alpha, iters=20):
    B, L, N, _ = log_alpha.shape
    log_alpha_flat = log_alpha.reshape(-1, N, N)
    num_tokens = log_alpha_flat.shape[0]
    out = torch.empty_like(log_alpha_flat)
    
    grid = (num_tokens,)
    sinkhorn_kernel_fwd[grid](
        log_alpha_flat, out,
        num_tokens, N*N, N,
        N, iters, 4
    )
    return out.reshape(B, L, N, N)
