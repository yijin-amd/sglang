from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.fla.utils import input_guard

def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def get_cuda_autotune_config():
    return [
        triton.Config({'BV': 8}, num_stages=3, num_warps=1),
    ]


def get_hip_autotune_config():
    return [
        triton.Config({'BV': 64}, num_stages=1, num_warps=4),
    ]

def get_autotune_config():
    if is_cuda():
        return get_cuda_autotune_config()
    else:
        return get_hip_autotune_config()

@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0_source"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=get_autotune_config(),
    key=['K', 'V'],
)
@triton.jit(do_not_specialize=["T"])
def fused_sigmoid_gating_delta_rule_update_kernel(
    A_log,
    a,
    dt_bias,
    softplus_beta,
    softplus_threshold,
    q,
    k,
    v,
    b,
    o,
    h0_source,
    h0_indices,
    cu_seqlens,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Fused kernel that combines sigmoid gating computation with recurrent delta rule update.
    """
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    p_b = b + bos * HV + i_hv
    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

    # Gating computation pointers
    p_A_log = A_log + i_hv
    p_a = a + bos * HV + i_hv
    p_dt_bias = dt_bias + i_hv

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        idx = tl.load(h0_indices + i_n)
        if idx >= 0:
            p_h0 = (
                h0_source
                + idx * HV * K * V
                + i_hv * K * V
                + o_k[:, None] * V
                + o_v[None, :]
            )
            b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for _ in range(0, T):
        # Load inputs
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_b = tl.load(p_b).to(tl.float32)

        # Compute sigmoid gating
        # Load gating parameters
        b_A_log = tl.load(p_A_log).to(tl.float32)
        b_a = tl.load(p_a).to(tl.float32)
        b_dt_bias = tl.load(p_dt_bias).to(tl.float32)

        # Compute g = -exp(A_log) * softplus(a + dt_bias)
        x = b_a + b_dt_bias
        beta_x = softplus_beta * x
        # Apply softplus with numerical stability
        softplus_x = tl.where(
            beta_x <= softplus_threshold,
            (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(beta_x)),
            x,
        )
        b_g = -tl.exp(b_A_log) * softplus_x

        # Compute beta = sigmoid(b)
        b_beta = 1.0 / (1.0 + tl.exp(-b_b))

        # Apply L2 normalization if enabled
        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / (tl.sqrt(tl.sum(b_q * b_q) + 1e-6))
            b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k) + 1e-6))

        b_q = b_q * scale

        # Apply gating to hidden state: h *= exp(g)
        b_h *= tl.exp(b_g)

        # Delta rule: v -= sum(h * k, dim=0)
        b_v -= tl.sum(b_h * b_k[:, None], 0)

        # Apply beta gating: v *= beta
        b_v *= b_beta

        # Update hidden state: h += k[:, None] * v[None, :]
        b_h += b_k[:, None] * b_v[None, :]

        # Compute output: o = sum(h * q, dim=0)
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        # Update pointers for next timestep
        p_q += H * K
        p_k += H * K
        p_o += HV * V
        p_v += HV * V
        p_b += HV
        p_a += HV

    # Store final state back to h0_source with bounds checking
    if USE_INITIAL_STATE:
        idx = tl.load(h0_indices + i_n)
        if idx >= 0:
            p_h0 = (
                h0_source
                + idx * HV * K * V
                + i_hv * K * V
                + o_k[:, None] * V
                + o_v[None, :]
            )
            tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)


@input_guard
def fused_sigmoid_gating_delta_rule_update(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float,
    softplus_threshold: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
):
    """
    Fused triton implementation of sigmoid gating delta rule update.
    This function uses a single fused kernel that combines both sigmoid gating computation
    and the recurrent delta rule update for better performance.
    """
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK = triton.next_power_of_2(K)
    NK = triton.cdiv(K, BK)
    assert NK == 1, "NK > 1 is not supported yet"

    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        assert scale > 0, "scale must be positive"

    o = q.new_empty(NK, *v.shape)
    grid = lambda META: (NK, triton.cdiv(V, META['BV']), N * HV)

    fused_sigmoid_gating_delta_rule_update_kernel[grid](
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        q=q,
        k=k,
        v=v,
        b=b,
        o=o,
        h0_source=initial_state_source,
        h0_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
        scale=scale,
        T=T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
    )
    o = o.squeeze(0)
    return o


# =============================================================================
# Fused Split GDR Update - combines split operation with delta rule update
# =============================================================================

@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0_source"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=get_autotune_config(),
    key=['K', 'V'],
)
@triton.jit(do_not_specialize=["T"])
def fused_split_gdr_update_kernel(
    # Input: mixed QKV tensor (batch, dim, seqlen) where dim = 2*key_dim + value_dim
    mixed_qkv,
    # Gating inputs
    A_log,
    a,
    dt_bias,
    softplus_beta,
    softplus_threshold,
    b,
    # Output
    o,
    # State
    h0_source,
    h0_indices,
    cu_seqlens,
    # Parameters
    scale,
    T,
    # Dimensions
    key_dim: tl.constexpr,
    value_dim: tl.constexpr,
    # Strides for mixed_qkv
    stride_x_batch: tl.constexpr,
    stride_x_dim: tl.constexpr,
    stride_x_seq: tl.constexpr,
    # Meta-parameters
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Fused kernel that combines:
    1. Split mixed_qkv into Q, K, V
    2. Apply activation (silu) 
    3. Sigmoid gating computation
    4. Recurrent delta rule update
    
    mixed_qkv layout: (batch, dim, seqlen) where dim = [Q: key_dim | K: key_dim | V: value_dim]
    """
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    
    # Compute corresponding Q/K head for this V head
    GROUP_SIZE: tl.constexpr = HV // H
    i_h = i_hv // GROUP_SIZE

    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        all = T
        T = eos - bos
        # For varlen mode with mixed_qkv (1, dim, total_tokens):
        # batch dimension is always 0, token position is bos
        batch_idx = 0
        token_start = bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T
        # For non-varlen mode with mixed_qkv (batch, dim, seqlen):
        # batch dimension is i_n, token position starts at 0
        batch_idx = i_n
        token_start = 0

    # Offset ranges for K and V dimensions
    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    
    # Compute SCALAR feature offset bases within mixed_qkv (like original kernel)
    # Layout: [Q: 0:key_dim | K: key_dim:2*key_dim | V: 2*key_dim:2*key_dim+value_dim]
    q_dim_start = i_h * K
    k_dim_start = key_dim + i_h * K
    v_dim_start = 2 * key_dim + i_hv * V

    # Gating parameter pointers
    p_A_log = A_log + i_hv
    p_dt_bias = dt_bias + i_hv
    
    # Load time-invariant gating parameters (optimization: load once outside loop)
    b_A_log = tl.load(p_A_log).to(tl.float32)
    b_dt_bias = tl.load(p_dt_bias).to(tl.float32)

    # Masks
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    # Initialize hidden state
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        idx = tl.load(h0_indices + i_n)
        if idx >= 0:
            p_h0 = (
                h0_source
                + idx * HV * K * V
                + i_hv * K * V
                + o_k[:, None] * V
                + o_v[None, :]
            )
            b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    # Pre-compute base pointers using SCALAR offsets (like original kernel)
    # mixed_qkv layout: (batch, dim, seqlen)
    # For varlen: batch=1, seqlen=total_tokens, token_start=bos
    # For non-varlen: batch=batch_size, seqlen=T, token_start=0
    base_ptr = mixed_qkv + batch_idx * stride_x_batch + token_start * stride_x_seq
    
    # Compute scalar feature offsets, then add vector offset
    q_base = base_ptr + q_dim_start * stride_x_dim
    k_base = base_ptr + k_dim_start * stride_x_dim
    v_base = base_ptr + v_dim_start * stride_x_dim
    
    # Now add vector offsets (o_k, o_v) - this is like original kernel: scalar + vector
    p_q = q_base + o_k * stride_x_dim
    p_k = k_base + o_k * stride_x_dim
    p_v = v_base + o_v * stride_x_dim
    
    # Pre-compute gating and output pointers
    p_a = a + bos * HV + i_hv
    p_b = b + bos * HV + i_hv
    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

    # Main token processing loop
    for _ in range(0, T):
        # Load Q, K, V from mixed_qkv (already activated by conv1d)
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        
        # Load gating parameters
        b_a = tl.load(p_a).to(tl.float32)
        b_b = tl.load(p_b).to(tl.float32)

        # Compute g = -exp(A_log) * softplus(a + dt_bias)
        x = b_a + b_dt_bias
        beta_x = softplus_beta * x
        softplus_x = tl.where(
            beta_x <= softplus_threshold,
            (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(beta_x)),
            x,
        )
        b_g = -tl.exp(b_A_log) * softplus_x

        # Compute beta = sigmoid(b)
        b_beta = 1.0 / (1.0 + tl.exp(-b_b))

        # Apply L2 normalization if enabled
        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / (tl.sqrt(tl.sum(b_q * b_q) + 1e-6))
            b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k) + 1e-6))

        b_q = b_q * scale

        # Apply gating to hidden state: h *= exp(g)
        b_h *= tl.exp(b_g)

        # Delta rule: v -= sum(h * k, dim=0)
        b_v -= tl.sum(b_h * b_k[:, None], 0)

        # Apply beta gating: v *= beta
        b_v *= b_beta

        # Update hidden state: h += k[:, None] * v[None, :]
        b_h += b_k[:, None] * b_v[None, :]

        # Compute output: o = sum(h * q, dim=0)
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
        
        # Update pointers for next timestep (simple addition like original kernel)
        p_q += stride_x_seq
        p_k += stride_x_seq
        p_v += stride_x_seq
        p_a += HV
        p_b += HV
        p_o += HV * V

    # Store final state back to h0_source
    if USE_INITIAL_STATE:
        idx = tl.load(h0_indices + i_n)
        if idx >= 0:
            p_h0 = (
                h0_source
                + idx * HV * K * V
                + i_hv * K * V
                + o_k[:, None] * V
                + o_v[None, :]
            )
            tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)


@input_guard
def fused_split_gdr_update(
    mixed_qkv: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    key_dim: int,
    value_dim: int,
    num_heads_qk: int,
    num_heads_v: int,
    head_dim: int,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = True,
    cu_seqlens: Optional[torch.Tensor] = None,
):
    """
    Fused triton implementation that combines split operation with sigmoid gating delta rule update.
    
    This function expects mixed_qkv to already have activation applied (e.g., from conv1d).
    It performs:
    1. Splits mixed_qkv into Q, K, V (implicit, via pointer arithmetic)
    2. Performs sigmoid gating computation
    3. Performs recurrent delta rule update
    
    Args:
        mixed_qkv: Input tensor of shape (batch, dim, seqlen) where dim = 2*key_dim + value_dim
                   NOTE: Activation (silu) should already be applied by conv1d
        A_log: Log of A parameter, shape (num_heads_v,)
        a: Time-variant gating parameter, shape (batch*seqlen, num_heads_v)
        dt_bias: Bias for dt, shape (num_heads_v,)
        b: Beta gating parameter, shape (batch*seqlen, num_heads_v)
        initial_state_source: SSM state tensor, shape (num_states, num_heads_v, head_dim, head_dim)
        initial_state_indices: Indices into initial_state_source, shape (batch,)
        key_dim: Key dimension (num_heads_qk * head_dim)
        value_dim: Value dimension (num_heads_v * head_dim)
        num_heads_qk: Number of Q/K heads
        num_heads_v: Number of V heads
        head_dim: Head dimension
        softplus_beta: Beta parameter for softplus
        softplus_threshold: Threshold for softplus
        scale: Scaling factor for Q (default: head_dim ** -0.5)
        use_qk_l2norm_in_kernel: Whether to use L2 normalization for Q and K
        cu_seqlens: Cumulative sequence lengths for variable length mode
        
    Returns:
        Output tensor of shape (batch, seqlen, num_heads_v, head_dim)
    """
    batch, dim, seqlen = mixed_qkv.shape
    assert dim == 2 * key_dim + value_dim
    assert key_dim == num_heads_qk * head_dim
    assert value_dim == num_heads_v * head_dim
    
    H = num_heads_qk
    HV = num_heads_v
    K = head_dim
    V = head_dim
    T = seqlen
    B = batch
    N = batch if cu_seqlens is None else len(cu_seqlens) - 1
    
    BK = triton.next_power_of_2(K)
    NK = triton.cdiv(K, BK)
    assert NK == 1, "NK > 1 is not supported yet"
    
    if scale is None:
        scale = K ** -0.5
    else:
        assert scale > 0, "scale must be positive"
    
    # Output shape: (batch, seqlen, num_heads_v, head_dim)
    o = mixed_qkv.new_empty(NK, B, T, HV, V)
    grid = lambda META: (NK, triton.cdiv(V, META['BV']), N * HV)
    
    fused_split_gdr_update_kernel[grid](
        mixed_qkv=mixed_qkv,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        b=b,
        o=o,
        h0_source=initial_state_source,
        h0_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
        scale=scale,
        T=T,
        key_dim=key_dim,
        value_dim=value_dim,
        stride_x_batch=mixed_qkv.stride(0),
        stride_x_dim=mixed_qkv.stride(1),
        stride_x_seq=mixed_qkv.stride(2),
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
    )
    o = o.squeeze(0)
    return o


# =============================================================================
# Optimized v2 kernel with:
# 1. Loop-invariant code motion: -exp(A_log) precomputed
# 2. rsqrt optimization: use rsqrt instead of 1/sqrt
# 3. Softplus optimization: 1/softplus_beta precomputed
# =============================================================================

@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0_source"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=get_autotune_config(),
    key=['K', 'V'],
)
@triton.jit(do_not_specialize=["T"])
def fused_split_gdr_update_kernel_v2(
    # Input: mixed QKV tensor (batch, dim, seqlen) where dim = 2*key_dim + value_dim
    mixed_qkv,
    # Gating inputs
    A_log,
    a,
    dt_bias,
    softplus_beta,
    softplus_threshold,
    b,
    # Output
    o,
    # State
    h0_source,
    h0_indices,
    cu_seqlens,
    # Parameters
    scale,
    T,
    # Dimensions
    key_dim: tl.constexpr,
    value_dim: tl.constexpr,
    # Strides for mixed_qkv
    stride_x_batch: tl.constexpr,
    stride_x_dim: tl.constexpr,
    stride_x_seq: tl.constexpr,
    # Meta-parameters
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Optimized v2 kernel with:
    1. Loop-invariant code motion: -exp(A_log) precomputed outside loop
    2. rsqrt optimization: use rsqrt instead of 1/sqrt for L2 norm
    3. Softplus optimization: 1/softplus_beta precomputed outside loop
    """
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    
    # Compute corresponding Q/K head for this V head
    GROUP_SIZE: tl.constexpr = HV // H
    i_h = i_hv // GROUP_SIZE

    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        all = T
        T = eos - bos
        batch_idx = 0
        token_start = bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T
        batch_idx = i_n
        token_start = 0

    # Offset ranges for K and V dimensions
    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    
    # Compute SCALAR feature offset bases within mixed_qkv
    q_dim_start = i_h * K
    k_dim_start = key_dim + i_h * K
    v_dim_start = 2 * key_dim + i_hv * V

    # Gating parameter pointers
    p_A_log = A_log + i_hv
    p_dt_bias = dt_bias + i_hv
    
    # Load time-invariant gating parameters
    b_A_log = tl.load(p_A_log).to(tl.float32)
    b_dt_bias = tl.load(p_dt_bias).to(tl.float32)
    
    # =========================================================================
    # Optimization 1: Precompute loop-invariant -exp(A_log)
    # =========================================================================
    neg_exp_A_log = -tl.exp(b_A_log)
    
    # =========================================================================
    # Optimization 3: Precompute 1/softplus_beta to avoid division in loop
    # =========================================================================
    inv_softplus_beta = 1.0 / softplus_beta

    # Masks
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    # Initialize hidden state
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        idx = tl.load(h0_indices + i_n)
        if idx >= 0:
            p_h0 = (
                h0_source
                + idx * HV * K * V
                + i_hv * K * V
                + o_k[:, None] * V
                + o_v[None, :]
            )
            b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    # Pre-compute base pointers
    base_ptr = mixed_qkv + batch_idx * stride_x_batch + token_start * stride_x_seq
    
    q_base = base_ptr + q_dim_start * stride_x_dim
    k_base = base_ptr + k_dim_start * stride_x_dim
    v_base = base_ptr + v_dim_start * stride_x_dim
    
    p_q = q_base + o_k * stride_x_dim
    p_k = k_base + o_k * stride_x_dim
    p_v = v_base + o_v * stride_x_dim
    
    # Pre-compute gating and output pointers
    p_a = a + bos * HV + i_hv
    p_b = b + bos * HV + i_hv
    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

    # Main token processing loop
    for _ in range(0, T):
        # Load Q, K, V from mixed_qkv (already activated by conv1d)
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        
        # Load gating parameters
        b_a = tl.load(p_a).to(tl.float32)
        b_b = tl.load(p_b).to(tl.float32)

        # Compute g = -exp(A_log) * softplus(a + dt_bias)
        x = b_a + b_dt_bias
        beta_x = softplus_beta * x
        # Optimization 3: use precomputed inv_softplus_beta
        softplus_x = tl.where(
            beta_x <= softplus_threshold,
            inv_softplus_beta * tl.log(1.0 + tl.exp(beta_x)),
            x,
        )
        # Optimization 1: use precomputed neg_exp_A_log
        b_g = neg_exp_A_log * softplus_x

        # Compute beta = sigmoid(b)
        b_beta = 1.0 / (1.0 + tl.exp(-b_b))

        # =====================================================================
        # Optimization 2: Use rsqrt instead of 1/sqrt for L2 normalization
        # rsqrt is a single GPU instruction, faster than sqrt + division
        # =====================================================================
        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q * tl.rsqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k * tl.rsqrt(tl.sum(b_k * b_k) + 1e-6)

        b_q = b_q * scale

        # Apply gating to hidden state: h *= exp(g)
        b_h *= tl.exp(b_g)

        # Delta rule: v -= sum(h * k, dim=0)
        b_v -= tl.sum(b_h * b_k[:, None], 0)

        # Apply beta gating: v *= beta
        b_v *= b_beta

        # Update hidden state: h += k[:, None] * v[None, :]
        b_h += b_k[:, None] * b_v[None, :]

        # Compute output: o = sum(h * q, dim=0)
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
        
        # Update pointers for next timestep
        p_q += stride_x_seq
        p_k += stride_x_seq
        p_v += stride_x_seq
        p_a += HV
        p_b += HV
        p_o += HV * V

    # Store final state back to h0_source
    if USE_INITIAL_STATE:
        idx = tl.load(h0_indices + i_n)
        if idx >= 0:
            p_h0 = (
                h0_source
                + idx * HV * K * V
                + i_hv * K * V
                + o_k[:, None] * V
                + o_v[None, :]
            )
            tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)


@input_guard
def fused_split_gdr_update_v2(
    mixed_qkv: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    key_dim: int,
    value_dim: int,
    num_heads_qk: int,
    num_heads_v: int,
    head_dim: int,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = True,
    cu_seqlens: Optional[torch.Tensor] = None,
):
    """
    Optimized v2 wrapper for fused_split_gdr_update_kernel_v2.
    
    Optimizations applied:
    1. Loop-invariant code motion: -exp(A_log) precomputed
    2. rsqrt optimization: use rsqrt instead of 1/sqrt
    3. Softplus optimization: 1/softplus_beta precomputed
    """
    # mixed_qkv: (batch, dim, seqlen) where dim = 2*key_dim + value_dim
    if mixed_qkv.dim() == 2:
        mixed_qkv = mixed_qkv.unsqueeze(0)
    
    batch, dim, seqlen = mixed_qkv.shape
    
    H = num_heads_qk
    HV = num_heads_v
    K = head_dim
    V = head_dim
    
    T = seqlen
    B = batch
    N = batch if cu_seqlens is None else len(cu_seqlens) - 1
    
    BK = triton.next_power_of_2(K)
    NK = triton.cdiv(K, BK)
    assert NK == 1, "NK > 1 is not supported yet"
    
    if scale is None:
        scale = K ** -0.5
    else:
        assert scale > 0, "scale must be positive"
    
    # Output shape: (batch, seqlen, num_heads_v, head_dim)
    o = mixed_qkv.new_empty(NK, B, T, HV, V)
    grid = lambda META: (NK, triton.cdiv(V, META['BV']), N * HV)
    
    fused_split_gdr_update_kernel_v2[grid](
        mixed_qkv=mixed_qkv,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        b=b,
        o=o,
        h0_source=initial_state_source,
        h0_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
        scale=scale,
        T=T,
        key_dim=key_dim,
        value_dim=value_dim,
        stride_x_batch=mixed_qkv.stride(0),
        stride_x_dim=mixed_qkv.stride(1),
        stride_x_seq=mixed_qkv.stride(2),
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
    )
    o = o.squeeze(0)
    return o


# =============================================================================
# Optimized v3 kernel with:
# 1. All v2 optimizations (loop-invariant code motion, rsqrt, softplus)
# 2. Pre-allocated output buffer support (avoids tensor allocation overhead)
# 3. Stride-based output access (enables flexible output buffer layout)
# 4. tl.sigmoid optimization (use built-in sigmoid instead of manual computation)
#
# Performance: ~5-6% speedup with pre-allocated output buffer
# =============================================================================

def get_autotune_config_v3():
    if is_cuda():
        return [triton.Config({'BV': 8}, num_stages=3, num_warps=1)]
    else:
        return [triton.Config({'BV': 64}, num_stages=1, num_warps=4)]


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0_source"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=get_autotune_config_v3(),
    key=['K', 'V'],
)
@triton.jit(do_not_specialize=["T"])
def fused_split_gdr_update_kernel_v3(
    # Input: mixed QKV tensor (batch, dim, seqlen) where dim = 2*key_dim + value_dim
    mixed_qkv,
    # Gating inputs
    A_log,
    a,
    dt_bias,
    softplus_beta,
    softplus_threshold,
    b,
    # Output
    o,
    # State
    h0_source,
    h0_indices,
    cu_seqlens,
    # Parameters
    scale,
    T,
    # Dimensions
    key_dim: tl.constexpr,
    value_dim: tl.constexpr,
    # Strides for mixed_qkv
    stride_x_batch: tl.constexpr,
    stride_x_dim: tl.constexpr,
    stride_x_seq: tl.constexpr,
    # Strides for output (enables pre-allocated buffer)
    stride_o_batch: tl.constexpr,
    stride_o_seq: tl.constexpr,
    stride_o_head: tl.constexpr,
    stride_o_dim: tl.constexpr,
    # Meta-parameters
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Optimized v3 kernel with:
    1. Loop-invariant code motion: -exp(A_log) precomputed outside loop
    2. rsqrt optimization: use rsqrt instead of 1/sqrt for L2 norm
    3. Softplus optimization: 1/softplus_beta precomputed outside loop
    4. tl.sigmoid: use built-in sigmoid instead of manual 1/(1+exp(-x))
    5. Stride-based output: supports pre-allocated output buffer
    """
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    
    # Compute corresponding Q/K head for this V head
    GROUP_SIZE: tl.constexpr = HV // H
    i_h = i_hv // GROUP_SIZE

    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        all_tokens = T
        T_local = eos - bos
        batch_idx = 0
        token_start = bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all_tokens = B * T
        batch_idx = i_n
        token_start = 0
        T_local = T

    # Offset ranges for K and V dimensions
    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    
    # Compute SCALAR feature offset bases within mixed_qkv
    q_dim_start = i_h * K
    k_dim_start = key_dim + i_h * K
    v_dim_start = 2 * key_dim + i_hv * V

    # Gating parameter pointers
    p_A_log = A_log + i_hv
    p_dt_bias = dt_bias + i_hv
    
    # Load time-invariant gating parameters
    b_A_log = tl.load(p_A_log).to(tl.float32)
    b_dt_bias = tl.load(p_dt_bias).to(tl.float32)
    
    # Precompute loop-invariants
    neg_exp_A_log = -tl.exp(b_A_log)
    inv_softplus_beta = 1.0 / softplus_beta

    # Masks
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    # Initialize hidden state
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        idx = tl.load(h0_indices + i_n)
        if idx >= 0:
            p_h0 = (
                h0_source
                + idx * HV * K * V
                + i_hv * K * V
                + o_k[:, None] * V
                + o_v[None, :]
            )
            b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    # Pre-compute base pointers
    base_ptr = mixed_qkv + batch_idx * stride_x_batch + token_start * stride_x_seq
    
    q_base = base_ptr + q_dim_start * stride_x_dim
    k_base = base_ptr + k_dim_start * stride_x_dim
    v_base = base_ptr + v_dim_start * stride_x_dim
    
    p_q = q_base + o_k * stride_x_dim
    p_k = k_base + o_k * stride_x_dim
    p_v = v_base + o_v * stride_x_dim
    
    # Pre-compute gating pointers
    p_a = a + bos * HV + i_hv
    p_b = b + bos * HV + i_hv
    
    # Output pointer using strides (supports pre-allocated buffer)
    # o[batch, seq, head, dim] layout
    p_o = o + i_n * stride_o_batch + i_hv * stride_o_head + o_v * stride_o_dim

    # Main token processing loop
    for _ in range(0, T_local):
        # Load Q, K, V from mixed_qkv (already activated by conv1d)
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        
        # Load gating parameters
        b_a = tl.load(p_a).to(tl.float32)
        b_b = tl.load(p_b).to(tl.float32)

        # Compute g = -exp(A_log) * softplus(a + dt_bias)
        x = b_a + b_dt_bias
        beta_x = softplus_beta * x
        softplus_x = tl.where(
            beta_x <= softplus_threshold,
            inv_softplus_beta * tl.log(1.0 + tl.exp(beta_x)),
            x,
        )
        b_g = neg_exp_A_log * softplus_x

        # Compute beta = sigmoid(b) using built-in tl.sigmoid
        b_beta = tl.sigmoid(b_b)

        # L2 normalization with rsqrt
        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q * tl.rsqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k * tl.rsqrt(tl.sum(b_k * b_k) + 1e-6)

        b_q = b_q * scale

        # Apply gating to hidden state: h *= exp(g)
        b_h *= tl.exp(b_g)

        # Delta rule: v -= sum(h * k, dim=0)
        b_v -= tl.sum(b_h * b_k[:, None], 0)

        # Apply beta gating: v *= beta
        b_v *= b_beta

        # Update hidden state: h += k[:, None] * v[None, :]
        b_h += b_k[:, None] * b_v[None, :]

        # Compute output: o = sum(h * q, dim=0)
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
        
        # Update pointers for next timestep
        p_q += stride_x_seq
        p_k += stride_x_seq
        p_v += stride_x_seq
        p_a += HV
        p_b += HV
        p_o += stride_o_seq

    # Store final state back to h0_source
    if USE_INITIAL_STATE:
        idx = tl.load(h0_indices + i_n)
        if idx >= 0:
            p_h0 = (
                h0_source
                + idx * HV * K * V
                + i_hv * K * V
                + o_k[:, None] * V
                + o_v[None, :]
            )
            tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)


@input_guard
def fused_split_gdr_update_v3(
    mixed_qkv: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    key_dim: int,
    value_dim: int,
    num_heads_qk: int,
    num_heads_v: int,
    head_dim: int,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = True,
    cu_seqlens: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,  # Pre-allocated output buffer
):
    """
    Optimized v3 wrapper with pre-allocated output buffer support.
    
    Optimizations:
    1. All v2 optimizations (loop-invariant, rsqrt, softplus)
    2. Pre-allocated output buffer: avoids tensor allocation overhead (~5-6% speedup)
    3. tl.sigmoid: use built-in sigmoid instead of manual computation
    
    Args:
        output: Optional pre-allocated output tensor of shape (batch, seqlen, num_heads_v, head_dim).
                If None, a new tensor is allocated.
    """
    # mixed_qkv: (batch, dim, seqlen) where dim = 2*key_dim + value_dim
    if mixed_qkv.dim() == 2:
        mixed_qkv = mixed_qkv.unsqueeze(0)
    
    batch, dim, seqlen = mixed_qkv.shape
    
    H = num_heads_qk
    HV = num_heads_v
    K = head_dim
    V = head_dim
    
    T = seqlen
    B = batch
    N = batch if cu_seqlens is None else len(cu_seqlens) - 1
    
    BK = triton.next_power_of_2(K)
    NK = triton.cdiv(K, BK)
    assert NK == 1, "NK > 1 is not supported yet"
    
    if scale is None:
        scale = K ** -0.5
    else:
        assert scale > 0, "scale must be positive"
    
    # Output: use pre-allocated buffer if provided, otherwise allocate new
    if output is None:
        o = mixed_qkv.new_empty(B, T, HV, V)
    else:
        expected_shape = (B, T, HV, V)
        assert output.shape == expected_shape, \
            f"Output shape mismatch: expected {expected_shape}, got {output.shape}"
        o = output
    
    grid = lambda META: (NK, triton.cdiv(V, META['BV']), N * HV)
    
    fused_split_gdr_update_kernel_v3[grid](
        mixed_qkv=mixed_qkv,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        b=b,
        o=o,
        h0_source=initial_state_source,
        h0_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
        scale=scale,
        T=T,
        key_dim=key_dim,
        value_dim=value_dim,
        stride_x_batch=mixed_qkv.stride(0),
        stride_x_dim=mixed_qkv.stride(1),
        stride_x_seq=mixed_qkv.stride(2),
        stride_o_batch=o.stride(0),
        stride_o_seq=o.stride(1),
        stride_o_head=o.stride(2),
        stride_o_dim=o.stride(3),
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
    )
    return o


# =============================================================================
# v4 kernel with bilinear output decomposition:
# 1. Keep v3 optimizations (loop-invariant code motion, rsqrt, softplus, tl.sigmoid)
# 2. Compute output from previous state without waiting for state write-back:
#      o_t = decay * (h_{t-1}^T q_t) + v_hat_t * (k_t^T q_t)
#    where:
#      v_hat_t = beta_t * (v_t - decay * (h_{t-1}^T k_t))
# 3. Preserve the original state update semantics:
#      h_t = decay * h_{t-1} + k_t v_hat_t^T
# =============================================================================

def get_autotune_config_v4():
    if is_cuda():
        return [triton.Config({'BV': 8}, num_stages=3, num_warps=1)]
    else:
        return [triton.Config({'BV': 64}, num_stages=1, num_warps=4)]


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0_source"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=get_autotune_config_v4(),
    key=['K', 'V'],
)
@triton.jit(do_not_specialize=["T"])
def fused_split_gdr_update_kernel_v4(
    mixed_qkv,
    A_log,
    a,
    dt_bias,
    softplus_beta,
    softplus_threshold,
    b,
    o,
    h0_source,
    h0_indices,
    cu_seqlens,
    scale,
    T,
    key_dim: tl.constexpr,
    value_dim: tl.constexpr,
    stride_x_batch: tl.constexpr,
    stride_x_dim: tl.constexpr,
    stride_x_seq: tl.constexpr,
    stride_o_batch: tl.constexpr,
    stride_o_seq: tl.constexpr,
    stride_o_head: tl.constexpr,
    stride_o_dim: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV

    GROUP_SIZE: tl.constexpr = HV // H
    i_h = i_hv // GROUP_SIZE

    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        T_local = eos - bos
        batch_idx = 0
        token_start = bos
    else:
        bos, eos = i_n * T, i_n * T + T
        batch_idx = i_n
        token_start = 0
        T_local = T

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    q_dim_start = i_h * K
    k_dim_start = key_dim + i_h * K
    v_dim_start = 2 * key_dim + i_hv * V

    p_A_log = A_log + i_hv
    p_dt_bias = dt_bias + i_hv

    b_A_log = tl.load(p_A_log).to(tl.float32)
    b_dt_bias = tl.load(p_dt_bias).to(tl.float32)

    neg_exp_A_log = -tl.exp(b_A_log)
    inv_softplus_beta = 1.0 / softplus_beta

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        idx = tl.load(h0_indices + i_n)
        if idx >= 0:
            p_h0 = (
                h0_source
                + idx * HV * K * V
                + i_hv * K * V
                + o_k[:, None] * V
                + o_v[None, :]
            )
            b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    base_ptr = mixed_qkv + batch_idx * stride_x_batch + token_start * stride_x_seq

    q_base = base_ptr + q_dim_start * stride_x_dim
    k_base = base_ptr + k_dim_start * stride_x_dim
    v_base = base_ptr + v_dim_start * stride_x_dim

    p_q = q_base + o_k * stride_x_dim
    p_k = k_base + o_k * stride_x_dim
    p_v = v_base + o_v * stride_x_dim

    p_a = a + bos * HV + i_hv
    p_b = b + bos * HV + i_hv

    p_o = o + i_n * stride_o_batch + i_hv * stride_o_head + o_v * stride_o_dim

    for _ in range(0, T_local):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)

        b_a = tl.load(p_a).to(tl.float32)
        b_b = tl.load(p_b).to(tl.float32)

        x = b_a + b_dt_bias
        beta_x = softplus_beta * x
        softplus_x = tl.where(
            beta_x <= softplus_threshold,
            inv_softplus_beta * tl.log(1.0 + tl.exp(beta_x)),
            x,
        )
        b_g = neg_exp_A_log * softplus_x
        decay = tl.exp(b_g)
        b_beta = tl.sigmoid(b_b)

        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q * tl.rsqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k * tl.rsqrt(tl.sum(b_k * b_k) + 1e-6)

        b_q = b_q * scale

        # Projections on previous state
        hk_proj = tl.sum(b_h * b_k[:, None], 0)
        hq_proj = tl.sum(b_h * b_q[:, None], 0)
        kq_dot = tl.sum(b_k * b_q)

        # v_hat = beta * (v - decay * (h^T k))
        b_v = (b_v - decay * hk_proj) * b_beta

        # Output from bilinear decomposition (no dependency on h write-back)
        b_o = decay * hq_proj + b_v * kq_dot
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        # Preserve original state transition
        b_h = decay * b_h + b_k[:, None] * b_v[None, :]

        p_q += stride_x_seq
        p_k += stride_x_seq
        p_v += stride_x_seq
        p_a += HV
        p_b += HV
        p_o += stride_o_seq

    if USE_INITIAL_STATE:
        idx = tl.load(h0_indices + i_n)
        if idx >= 0:
            p_h0 = (
                h0_source
                + idx * HV * K * V
                + i_hv * K * V
                + o_k[:, None] * V
                + o_v[None, :]
            )
            tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)


@input_guard
def fused_split_gdr_update_v4(
    mixed_qkv: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    key_dim: int,
    value_dim: int,
    num_heads_qk: int,
    num_heads_v: int,
    head_dim: int,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = True,
    cu_seqlens: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
):
    if mixed_qkv.dim() == 2:
        mixed_qkv = mixed_qkv.unsqueeze(0)

    batch, dim, seqlen = mixed_qkv.shape

    H = num_heads_qk
    HV = num_heads_v
    K = head_dim
    V = head_dim

    T = seqlen
    B = batch
    N = batch if cu_seqlens is None else len(cu_seqlens) - 1

    BK = triton.next_power_of_2(K)
    NK = triton.cdiv(K, BK)
    assert NK == 1, "NK > 1 is not supported yet"

    if scale is None:
        scale = K ** -0.5
    else:
        assert scale > 0, "scale must be positive"

    if output is None:
        o = mixed_qkv.new_empty(B, T, HV, V)
    else:
        expected_shape = (B, T, HV, V)
        assert output.shape == expected_shape, (
            f"Output shape mismatch: expected {expected_shape}, got {output.shape}"
        )
        o = output

    grid = lambda META: (NK, triton.cdiv(V, META['BV']), N * HV)

    fused_split_gdr_update_kernel_v4[grid](
        mixed_qkv=mixed_qkv,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        b=b,
        o=o,
        h0_source=initial_state_source,
        h0_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
        scale=scale,
        T=T,
        key_dim=key_dim,
        value_dim=value_dim,
        stride_x_batch=mixed_qkv.stride(0),
        stride_x_dim=mixed_qkv.stride(1),
        stride_x_seq=mixed_qkv.stride(2),
        stride_o_batch=o.stride(0),
        stride_o_seq=o.stride(1),
        stride_o_head=o.stride(2),
        stride_o_dim=o.stride(3),
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
    )
    return o


# =============================================================================
# v3_seqlen1: Specialized kernel for seqlen=1 (decode) scenario
#
# Optimizations over v3:
# 1. No loop overhead - single token processing without for loop
# 2. No pointer update code - removed stride increments
# 3. Simplified control flow - no T_local variable
# 4. All v3 optimizations retained (rsqrt, tl.sigmoid, pre-allocated output)
#
# Target: batch=64, seqlen=1 decode scenario on AMD MI300X
# =============================================================================

def get_autotune_config_v3_seqlen1():
    if is_cuda():
        return [triton.Config({'BV': 8}, num_stages=3, num_warps=1)]
    else:
        # AMD MI300X optimized config
        return [triton.Config({'BV': 64}, num_stages=1, num_warps=4)]


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0_source"] is not None,
    }
)
@triton.autotune(
    configs=get_autotune_config_v3_seqlen1(),
    key=['K', 'V'],
)
@triton.jit
def fused_split_gdr_update_kernel_v3_seqlen1(
    # Input: mixed QKV tensor (batch, dim, 1) where dim = 2*key_dim + value_dim
    mixed_qkv,
    # Gating inputs
    A_log,
    a,
    dt_bias,
    softplus_beta,
    softplus_threshold,
    b,
    # Output
    o,
    # State
    h0_source,
    h0_indices,
    # Parameters
    scale,
    # Dimensions
    key_dim: tl.constexpr,
    value_dim: tl.constexpr,
    # Strides for mixed_qkv
    stride_x_batch: tl.constexpr,
    stride_x_dim: tl.constexpr,
    # Strides for output
    stride_o_batch: tl.constexpr,
    stride_o_head: tl.constexpr,
    stride_o_dim: tl.constexpr,
    # Meta-parameters
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
):
    """
    Specialized kernel for seqlen=1 decode scenario.
    
    Key optimizations:
    - No for loop (single token)
    - No pointer increments
    - Simplified control flow
    """
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    
    # Compute corresponding Q/K head for this V head
    GROUP_SIZE: tl.constexpr = HV // H
    i_h = i_hv // GROUP_SIZE

    # Offset ranges for K and V dimensions
    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    
    # Compute feature offset bases within mixed_qkv
    q_dim_start = i_h * K
    k_dim_start = key_dim + i_h * K
    v_dim_start = 2 * key_dim + i_hv * V

    # Load time-invariant gating parameters
    b_A_log = tl.load(A_log + i_hv).to(tl.float32)
    b_dt_bias = tl.load(dt_bias + i_hv).to(tl.float32)
    
    # Precompute constants
    neg_exp_A_log = -tl.exp(b_A_log)
    inv_softplus_beta = 1.0 / softplus_beta

    # Masks
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    # Load hidden state
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        idx = tl.load(h0_indices + i_n)
        if idx >= 0:
            p_h0 = (
                h0_source
                + idx * HV * K * V
                + i_hv * K * V
                + o_k[:, None] * V
                + o_v[None, :]
            )
            b_h = tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    # Compute pointers for single token (seqlen=1, token at position 0)
    base_ptr = mixed_qkv + i_n * stride_x_batch
    
    p_q = base_ptr + (q_dim_start + o_k) * stride_x_dim
    p_k = base_ptr + (k_dim_start + o_k) * stride_x_dim
    p_v = base_ptr + (v_dim_start + o_v) * stride_x_dim

    # Load Q, K, V (single token)
    b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
    b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
    b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
    
    # Load gating parameters (single token at position i_n)
    b_a = tl.load(a + i_n * HV + i_hv).to(tl.float32)
    b_b = tl.load(b + i_n * HV + i_hv).to(tl.float32)

    # Compute g = -exp(A_log) * softplus(a + dt_bias)
    x = b_a + b_dt_bias
    beta_x = softplus_beta * x
    softplus_x = tl.where(
        beta_x <= softplus_threshold,
        inv_softplus_beta * tl.log(1.0 + tl.exp(beta_x)),
        x,
    )
    b_g = neg_exp_A_log * softplus_x

    # Compute beta = sigmoid(b)
    b_beta = tl.sigmoid(b_b)

    # L2 normalization with rsqrt
    if USE_QK_L2NORM_IN_KERNEL:
        b_q = b_q * tl.rsqrt(tl.sum(b_q * b_q) + 1e-6)
        b_k = b_k * tl.rsqrt(tl.sum(b_k * b_k) + 1e-6)

    b_q = b_q * scale

    # Apply gating to hidden state: h *= exp(g)
    b_h *= tl.exp(b_g)

    # Delta rule: v -= sum(h * k, dim=0)
    b_v -= tl.sum(b_h * b_k[:, None], 0)

    # Apply beta gating: v *= beta
    b_v *= b_beta

    # Update hidden state: h += k[:, None] * v[None, :]
    b_h += b_k[:, None] * b_v[None, :]

    # Compute output: o = sum(h * q, dim=0)
    b_o = tl.sum(b_h * b_q[:, None], 0)
    
    # Store output (seqlen=1, so seq stride not needed)
    p_o = o + i_n * stride_o_batch + i_hv * stride_o_head + o_v * stride_o_dim
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

    # Store final state back to h0_source
    if USE_INITIAL_STATE:
        idx = tl.load(h0_indices + i_n)
        if idx >= 0:
            p_h0 = (
                h0_source
                + idx * HV * K * V
                + i_hv * K * V
                + o_k[:, None] * V
                + o_v[None, :]
            )
            tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)


@input_guard
def fused_split_gdr_update_v3_seqlen1(
    mixed_qkv: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    key_dim: int,
    value_dim: int,
    num_heads_qk: int,
    num_heads_v: int,
    head_dim: int,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = True,
    output: Optional[torch.Tensor] = None,
):
    """
    Specialized wrapper for seqlen=1 decode scenario.
    
    This function is optimized for the decode phase where each request
    processes exactly one token at a time.
    
    Args:
        mixed_qkv: Input tensor of shape (batch, dim, 1) where dim = 2*key_dim + value_dim
        output: Optional pre-allocated output tensor of shape (batch, 1, num_heads_v, head_dim)
    """
    if mixed_qkv.dim() == 2:
        mixed_qkv = mixed_qkv.unsqueeze(0)
    
    batch, dim, seqlen = mixed_qkv.shape
    assert seqlen == 1, f"This kernel is specialized for seqlen=1, got seqlen={seqlen}"
    
    H = num_heads_qk
    HV = num_heads_v
    K = head_dim
    V = head_dim
    B = batch
    
    BK = triton.next_power_of_2(K)
    NK = triton.cdiv(K, BK)
    assert NK == 1, "NK > 1 is not supported yet"
    
    if scale is None:
        scale = K ** -0.5
    else:
        assert scale > 0, "scale must be positive"
    
    # Output: use pre-allocated buffer if provided
    if output is None:
        o = mixed_qkv.new_empty(B, 1, HV, V)
    else:
        expected_shape = (B, 1, HV, V)
        assert output.shape == expected_shape, \
            f"Output shape mismatch: expected {expected_shape}, got {output.shape}"
        o = output
    
    grid = lambda META: (NK, triton.cdiv(V, META['BV']), B * HV)
    
    fused_split_gdr_update_kernel_v3_seqlen1[grid](
        mixed_qkv=mixed_qkv,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        b=b,
        o=o,
        h0_source=initial_state_source,
        h0_indices=initial_state_indices,
        scale=scale,
        key_dim=key_dim,
        value_dim=value_dim,
        stride_x_batch=mixed_qkv.stride(0),
        stride_x_dim=mixed_qkv.stride(1),
        stride_o_batch=o.stride(0),
        stride_o_head=o.stride(2),
        stride_o_dim=o.stride(3),
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
    )
    return o
