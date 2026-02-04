"""
Fused Gating Delta Network (GDN) Forward Decode Kernel.

This module implements a fused Triton kernel that combines:
1. Causal Conv1D update with split Q/K/V
2. Sigmoid gating delta rule update

By fusing these operations, we avoid intermediate memory reads/writes
and achieve better performance for decode phase.
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
import triton.experimental.gluon.language as gl

PAD_SLOT_ID = -1


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def get_cuda_autotune_config():
    return [
        triton.Config({'BV': 8}, num_stages=3, num_warps=1),
    ]


def get_hip_autotune_config():
    return [
        triton.Config({'BV': 128}, num_stages=1, num_warps=4),
    ]


def get_autotune_config():
    if is_cuda():
        return get_cuda_autotune_config()
    else:
        return get_hip_autotune_config()


@tl.core.builtin
def tuple_combine(a: tl.tuple, b: tl.tensor, _semantic=None) -> tl.tuple:
    """Helper function to combine a tuple with a new tensor element."""
    return tl.tuple([*a.values, b])


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
def fused_gdn_fwd_decode_kernel_v2(
    # Conv1D inputs
    x_ptr,  # (batch, dim, seqlen) where dim = 2*key_dim + value_dim
    conv_w_ptr,  # (dim, conv_width)
    conv_bias_ptr,
    conv_state_ptr,
    conv_state_indices_ptr,
    # Gating inputs
    A_log,
    a,
    dt_bias,
    b,
    # SSM state
    h0_source,
    h0_indices,
    cu_seqlens,
    # Output
    o,
    # Dimensions
    key_dim: tl.constexpr,
    value_dim: tl.constexpr,
    batch: int,
    dim: tl.constexpr,
    seqlen: tl.constexpr,
    conv_state_len: tl.constexpr,
    num_cache_lines: tl.constexpr,
    T: int,
    # Gating parameters
    softplus_beta: float,
    softplus_threshold: float,
    scale: float,
    # Strides for conv
    stride_x_seq: tl.constexpr,
    stride_x_dim: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_conv_w_dim: tl.constexpr,
    stride_conv_w_width: tl.constexpr,
    stride_conv_state_seq: tl.constexpr,
    stride_conv_state_dim: tl.constexpr,
    stride_conv_state_tok: tl.constexpr,
    stride_state_indices: tl.constexpr,
    # Others
    pad_slot_id: tl.constexpr,
    # Meta-parameters
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    HAS_CONV_BIAS: tl.constexpr,
    CONV_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Simplified fused kernel where each block processes one V head.
    
    Grid layout: (num_k_blocks, num_v_blocks, batch * num_heads_v)
    
    Key differences from v1:
    - Grid indexed by V heads instead of Q/K heads
    - No GROUP_SIZE loop - each block handles one V head
    - Simpler logic, potentially better parallelism
    """
    # Get program IDs - indexed by V heads
    i_k, i_v, i_nhv = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nhv // HV, i_nhv % HV
    
    # Compute corresponding Q/K head
    GROUP_SIZE: tl.constexpr = HV // H
    i_h = i_hv // GROUP_SIZE
    
    # Handle variable length sequences
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        all = T
        T = eos - bos
        idx_seq = bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T
        idx_seq = i_n
    
    if idx_seq >= batch:
        return
    
    # Get conv state batch coordinate
    if IS_CONTINUOUS_BATCHING:
        conv_state_batch_coord = tl.load(
            conv_state_indices_ptr + idx_seq * stride_state_indices
        ).to(tl.int64)
    else:
        conv_state_batch_coord = idx_seq
        
    if USE_PAD_SLOT:
        if conv_state_batch_coord == pad_slot_id:
            return
    
    # Define offset ranges
    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    
    # Load gating parameters for this V head
    b_A_log = tl.load(A_log + i_hv).to(tl.float32)
    b_dt_bias = tl.load(dt_bias + i_hv).to(tl.float32)
    
    # Define feature offsets for Q, K, V
    q_dim_start = i_h * K
    q_feats = q_dim_start + o_k
    k_dim_start = key_dim + i_h * K
    k_feats = k_dim_start + o_k
    v_dim_start = 2 * key_dim + i_hv * V
    v_feats = v_dim_start + o_v
    
    if USE_INITIAL_STATE:
        idx = tl.load(h0_indices + i_n)
        if idx >= 0:
            # Load initial hidden state for this V head
            p_h = (
                h0_source
                + idx * HV * K * V
                + i_hv * K * V
                + o_k[:, None] * V
                + o_v[None, :]
            )
            b_h = tl.load(p_h).to(tl.float32)

            # gl.amd.cdna3.sched_barrier(0)
            
            # Pre-load conv_states and weights for Q, K, V
            b_q_conv_states = ()
            q_weights = (tl.load(conv_w_ptr + q_feats * stride_conv_w_dim),)
            for i in tl.static_range(CONV_WIDTH-1):
                b_q_conv_state = tl.load(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (q_feats * stride_conv_state_dim) + i * stride_conv_state_tok)
                w_val = tl.load(conv_w_ptr + q_feats * stride_conv_w_dim + (i+1) * stride_conv_w_width)
                q_weights = tuple_combine(q_weights, w_val)
                b_q_conv_states = tuple_combine(b_q_conv_states, b_q_conv_state)
            
            b_k_conv_states = ()
            k_weights = (tl.load(conv_w_ptr + k_feats * stride_conv_w_dim),)
            for i in tl.static_range(CONV_WIDTH-1):
                b_k_conv_state = tl.load(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (k_feats * stride_conv_state_dim) + i * stride_conv_state_tok)
                w_val = tl.load(conv_w_ptr + k_feats * stride_conv_w_dim + (i+1) * stride_conv_w_width)
                k_weights = tuple_combine(k_weights, w_val)
                b_k_conv_states = tuple_combine(b_k_conv_states, b_k_conv_state)
            
            b_v_conv_states = ()
            v_weights = (tl.load(conv_w_ptr + v_feats * stride_conv_w_dim),)
            for i in tl.static_range(CONV_WIDTH-1):
                b_v_conv_state = tl.load(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (v_feats * stride_conv_state_dim) + i * stride_conv_state_tok)
                w_val = tl.load(conv_w_ptr + v_feats * stride_conv_w_dim + (i+1) * stride_conv_w_width)
                v_weights = tuple_combine(v_weights, w_val)
                b_v_conv_states = tuple_combine(b_v_conv_states, b_v_conv_state)
            
            # Main token processing loop
            for idx_token in tl.static_range(seqlen):
                # Conv1D for K
                k_conv_acc = tl.load(conv_bias_ptr + k_feats).to(tl.float32) if HAS_CONV_BIAS else tl.zeros([BK], dtype=tl.float32)
                k_ptrs = x_ptr + idx_seq * stride_x_seq + k_feats * stride_x_dim + idx_token * stride_x_token
                b_k_conv_states = tuple_combine(b_k_conv_states, tl.load(k_ptrs))
                for j in tl.static_range(CONV_WIDTH):
                    k_conv_acc += b_k_conv_states[j] * k_weights[j]
                b_k_conv_states = b_k_conv_states[1:]
                
                if SILU_ACTIVATION:
                    k_conv_acc = k_conv_acc / (1 + tl.exp(-k_conv_acc))
                b_k = k_conv_acc.to(tl.float32)
                if USE_QK_L2NORM_IN_KERNEL:
                    b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k) + 1e-6))
                
                # Conv1D for V
                v_conv_acc = tl.load(conv_bias_ptr + v_feats).to(tl.float32) if HAS_CONV_BIAS else tl.zeros([BV], dtype=tl.float32)
                v_ptrs = x_ptr + idx_seq * stride_x_seq + v_feats * stride_x_dim + idx_token * stride_x_token
                b_v_conv_states = tuple_combine(b_v_conv_states, tl.load(v_ptrs))
                for j in tl.static_range(CONV_WIDTH):
                    v_conv_acc += b_v_conv_states[j] * v_weights[j]
                b_v_conv_states = b_v_conv_states[1:]
                
                if SILU_ACTIVATION:
                    v_conv_acc = v_conv_acc / (1 + tl.exp(-v_conv_acc))
                b_v = v_conv_acc.to(tl.float32)
                
                # Conv1D for Q
                q_conv_acc = tl.load(conv_bias_ptr + q_feats).to(tl.float32) if HAS_CONV_BIAS else tl.zeros([BK], dtype=tl.float32)
                q_ptrs = x_ptr + idx_seq * stride_x_seq + q_feats * stride_x_dim + idx_token * stride_x_token
                b_q_conv_states = tuple_combine(b_q_conv_states, tl.load(q_ptrs))
                for j in tl.static_range(CONV_WIDTH):
                    q_conv_acc += b_q_conv_states[j] * q_weights[j]
                b_q_conv_states = b_q_conv_states[1:]
                
                if SILU_ACTIVATION:
                    q_conv_acc = q_conv_acc / (1 + tl.exp(-q_conv_acc))
                b_q = q_conv_acc.to(tl.float32)
                if USE_QK_L2NORM_IN_KERNEL:
                    b_q_scale = scale / (tl.sqrt(tl.sum(b_q * b_q) + 1e-6))
                else:
                    b_q_scale = scale
                b_q = b_q * b_q_scale
                
                # Load time-variant gating parameters
                p_a = a + (bos + idx_token) * HV + i_hv
                p_b = b + (bos + idx_token) * HV + i_hv
                b_a = tl.load(p_a).to(tl.float32)
                b_b = tl.load(p_b).to(tl.float32)
                
                # Compute gating factor
                x = b_a + b_dt_bias
                beta_x = softplus_beta * x
                softplus_x = tl.where(
                    beta_x <= softplus_threshold,
                    (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(beta_x)),
                    x,
                )
                b_g = -tl.exp(b_A_log) * softplus_x
                b_beta = 1.0 / (1.0 + tl.exp(-b_b))
                
                # Delta rule recurrent update
                b_h *= tl.exp(b_g)
                b_v -= tl.sum(b_h * b_k[:, None], 0)
                b_v *= b_beta
                b_h += b_k[:, None] * b_v[None, :]
                
                # Compute and store output
                b_o = tl.sum(b_h * b_q[:, None], 0)
                p_o = o + ((i_k * all + bos + idx_token) * HV + i_hv) * V + o_v
                tl.store(p_o, b_o.to(p_o.dtype.element_ty))
            
            # Write back final hidden state
            tl.store(p_h, b_h.to(p_h.dtype.element_ty))
            
            # Write back conv_states
            for i in tl.static_range(CONV_WIDTH-1):
                tl.store(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (q_feats * stride_conv_state_dim) + i * stride_conv_state_tok, b_q_conv_states[i])
                tl.store(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (k_feats * stride_conv_state_dim) + i * stride_conv_state_tok, b_k_conv_states[i])
                tl.store(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (v_feats * stride_conv_state_dim) + i * stride_conv_state_tok, b_v_conv_states[i])
            
            return
    
    # Initialize zero hidden state
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    
    # Pre-load conv_states and weights
    b_q_conv_states = ()
    q_weights = (tl.load(conv_w_ptr + q_feats * stride_conv_w_dim),)
    for i in tl.static_range(CONV_WIDTH-1):
        b_q_conv_state = tl.load(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (q_feats * stride_conv_state_dim) + i * stride_conv_state_tok)
        w_val = tl.load(conv_w_ptr + q_feats * stride_conv_w_dim + (i+1) * stride_conv_w_width)
        q_weights = tuple_combine(q_weights, w_val)
        b_q_conv_states = tuple_combine(b_q_conv_states, b_q_conv_state)
    
    b_k_conv_states = ()
    k_weights = (tl.load(conv_w_ptr + k_feats * stride_conv_w_dim),)
    for i in tl.static_range(CONV_WIDTH-1):
        b_k_conv_state = tl.load(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (k_feats * stride_conv_state_dim) + i * stride_conv_state_tok)
        w_val = tl.load(conv_w_ptr + k_feats * stride_conv_w_dim + (i+1) * stride_conv_w_width)
        k_weights = tuple_combine(k_weights, w_val)
        b_k_conv_states = tuple_combine(b_k_conv_states, b_k_conv_state)
    
    b_v_conv_states = ()
    v_weights = (tl.load(conv_w_ptr + v_feats * stride_conv_w_dim),)
    for i in tl.static_range(CONV_WIDTH-1):
        b_v_conv_state = tl.load(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (v_feats * stride_conv_state_dim) + i * stride_conv_state_tok)
        w_val = tl.load(conv_w_ptr + v_feats * stride_conv_w_dim + (i+1) * stride_conv_w_width)
        v_weights = tuple_combine(v_weights, w_val)
        b_v_conv_states = tuple_combine(b_v_conv_states, b_v_conv_state)

    # Main token processing loop
    for idx_token in tl.static_range(seqlen):
        
        # Conv1D for K
        k_conv_acc = tl.load(conv_bias_ptr + k_feats).to(tl.float32) if HAS_CONV_BIAS else tl.zeros([BK], dtype=tl.float32)
        k_ptrs = x_ptr + idx_seq * stride_x_seq + k_feats * stride_x_dim + idx_token * stride_x_token
        b_k_conv_states = tuple_combine(b_k_conv_states, tl.load(k_ptrs))
        for j in tl.static_range(CONV_WIDTH):
            k_conv_acc += b_k_conv_states[j] * k_weights[j]
        b_k_conv_states = b_k_conv_states[1:]
        
        if SILU_ACTIVATION:
            k_conv_acc = k_conv_acc / (1 + tl.exp(-k_conv_acc))
        b_k = k_conv_acc.to(tl.float32)
        if USE_QK_L2NORM_IN_KERNEL:
            b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k) + 1e-6))
        
        # Conv1D for V
        v_conv_acc = tl.load(conv_bias_ptr + v_feats).to(tl.float32) if HAS_CONV_BIAS else tl.zeros([BV], dtype=tl.float32)
        v_ptrs = x_ptr + idx_seq * stride_x_seq + v_feats * stride_x_dim + idx_token * stride_x_token
        b_v_conv_states = tuple_combine(b_v_conv_states, tl.load(v_ptrs))
        for j in tl.static_range(CONV_WIDTH):
            v_conv_acc += b_v_conv_states[j] * v_weights[j]
        b_v_conv_states = b_v_conv_states[1:]
        
        if SILU_ACTIVATION:
            v_conv_acc = v_conv_acc / (1 + tl.exp(-v_conv_acc))
        b_v = v_conv_acc.to(tl.float32)
        
        # Conv1D for Q
        q_conv_acc = tl.load(conv_bias_ptr + q_feats).to(tl.float32) if HAS_CONV_BIAS else tl.zeros([BK], dtype=tl.float32)
        q_ptrs = x_ptr + idx_seq * stride_x_seq + q_feats * stride_x_dim + idx_token * stride_x_token
        b_q_conv_states = tuple_combine(b_q_conv_states, tl.load(q_ptrs))
        for j in tl.static_range(CONV_WIDTH):
            q_conv_acc += b_q_conv_states[j] * q_weights[j]
        b_q_conv_states = b_q_conv_states[1:]
        
        if SILU_ACTIVATION:
            q_conv_acc = q_conv_acc / (1 + tl.exp(-q_conv_acc))
        b_q = q_conv_acc.to(tl.float32)
        if USE_QK_L2NORM_IN_KERNEL:
            b_q_scale = scale / (tl.sqrt(tl.sum(b_q * b_q) + 1e-6))
        else:
            b_q_scale = scale
        b_q = b_q * b_q_scale
        
        # Load time-variant gating parameters
        p_a = a + (bos + idx_token) * HV + i_hv
        p_b = b + (bos + idx_token) * HV + i_hv
        b_a = tl.load(p_a).to(tl.float32)
        b_b = tl.load(p_b).to(tl.float32)
        
        # Compute gating factor
        x = b_a + b_dt_bias
        beta_x = softplus_beta * x
        softplus_x = tl.where(
            beta_x <= softplus_threshold,
            (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(beta_x)),
            x,
        )
        b_g = -tl.exp(b_A_log) * softplus_x
        b_beta = 1.0 / (1.0 + tl.exp(-b_b))
        
        # Delta rule recurrent update
        b_h *= tl.exp(b_g)
        b_v -= tl.sum(b_h * b_k[:, None], 0)
        b_v *= b_beta
        b_h += b_k[:, None] * b_v[None, :]
        
        # Compute and store output
        b_o = tl.sum(b_h * b_q[:, None], 0)
        p_o = o + ((i_k * all + bos + idx_token) * HV + i_hv) * V + o_v
        tl.store(p_o, b_o.to(p_o.dtype.element_ty))
    
    # Write back conv_states
    for i in tl.static_range(CONV_WIDTH-1):
        tl.store(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (q_feats * stride_conv_state_dim) + i * stride_conv_state_tok, b_q_conv_states[i])
        tl.store(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (k_feats * stride_conv_state_dim) + i * stride_conv_state_tok, b_k_conv_states[i])
        tl.store(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (v_feats * stride_conv_state_dim) + i * stride_conv_state_tok, b_v_conv_states[i])


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0_source"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=get_autotune_config(),
    key=['K', 'V']
)
@triton.jit(do_not_specialize=["T"])
def fused_gdn_fwd_decode_kernel(
    # Conv1D inputs
    x_ptr,  # (batch, dim, seqlen) where dim = 2*key_dim + value_dim
    conv_w_ptr,  # (dim, conv_width)
    conv_bias_ptr,
    conv_state_ptr,
    conv_state_indices_ptr,
    # Gating inputs
    A_log,
    a,
    dt_bias,
    b,
    # SSM state
    h0_source,
    h0_indices,
    cu_seqlens,
    # Output
    o,
    # Dimensions
    key_dim: tl.constexpr,
    value_dim: tl.constexpr,
    batch: int,
    dim: tl.constexpr,
    seqlen: tl.constexpr,
    conv_state_len: tl.constexpr,
    num_cache_lines: tl.constexpr,
    T: int,
    # Gating parameters
    softplus_beta: float,
    softplus_threshold: float,
    scale: float,
    # Strides for conv
    stride_x_seq: tl.constexpr,
    stride_x_dim: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_conv_w_dim: tl.constexpr,
    stride_conv_w_width: tl.constexpr,
    stride_conv_state_seq: tl.constexpr,
    stride_conv_state_dim: tl.constexpr,
    stride_conv_state_tok: tl.constexpr,
    stride_state_indices: tl.constexpr,
    # Others
    pad_slot_id: tl.constexpr,
    # Meta-parameters
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    HAS_CONV_BIAS: tl.constexpr,
    CONV_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Optimized fused kernel with Q/K reuse and batched V head processing.
    
    Key optimizations:
    1. Each block processes GROUP_SIZE (HV//H) value heads simultaneously
    2. Q/K are computed once per group and broadcast across all V heads
    3. V heads are represented as tensor dimensions for efficient batching
    4. Delta Rule updates use broadcasting instead of loops
    
    Processing flow:
    1. Load gating parameters for all V heads: [GROUP_SIZE]
    2. Load/initialize hidden states for all V heads: [GROUP_SIZE, BK, BV]
    3. For each token:
       a. Conv1D for K: [BK] (shared across all V heads)
       b. Conv1D for V: [GROUP_SIZE, BV] (batched for all V heads)
       c. Conv1D for Q: [BK] (shared across all V heads)
       d. Batched Delta Rule update with broadcasting:
          - b_h: [GROUP_SIZE, BK, BV]
          - b_k: [BK] -> broadcast to [1, BK, 1]
          - b_q: [BK] -> broadcast to [1, BK, 1]
          - b_v: [GROUP_SIZE, BV]
          - b_g, b_beta: [GROUP_SIZE] -> broadcast to [GROUP_SIZE, 1, 1] or [GROUP_SIZE, 1]
       e. Store outputs for all V heads
    4. Write back conv states
    """
    # Get program IDs - now indexed by Q/K heads (not V heads)
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_h = i_nh // H, i_nh % H
    
    # Number of V heads per Q/K head (group size)
    GROUP_SIZE: tl.constexpr = HV // H
    
    # Handle variable length sequences
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        all = T
        T = eos - bos
        idx_seq = bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T
        idx_seq = i_n
    
    if idx_seq >= batch:
        return
    
    # Get conv state batch coordinate
    if IS_CONTINUOUS_BATCHING:
        conv_state_batch_coord = tl.load(
            conv_state_indices_ptr + idx_seq * stride_state_indices
        ).to(tl.int64)
    else:
        conv_state_batch_coord = idx_seq
        
    if USE_PAD_SLOT:
        if conv_state_batch_coord == pad_slot_id:
            return
    
    # ============================================================================
    # Initialization: Setup dimensions and load gating parameters
    # ============================================================================
    
    # Define offset ranges for tensor blocks
    # o_k: [BK] - Offsets for K dimension (shared by Q and K)
    # o_v: [BV] - Offsets for V dimension
    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    # Define V head indices for this Q/K head group
    # i_hv: [GROUP_SIZE] - Absolute indices of V heads in this group
    # Example: if i_h=1, HV=8, H=4, then GROUP_SIZE=2, i_hv=[2, 3]
    i_hv = i_h * GROUP_SIZE + tl.arange(0, GROUP_SIZE)

    # Load time-invariant gating parameters for all V heads in this group
    # b_A_log: [GROUP_SIZE] - Log of recurrent matrix eigenvalues
    # b_dt_bias: [GROUP_SIZE] - Time step bias parameters
    b_A_log = tl.load(A_log + i_hv).to(tl.float32)
    b_dt_bias = tl.load(dt_bias + i_hv).to(tl.float32)

    if USE_INITIAL_STATE:
        idx = tl.load(h0_indices + i_n)
        if idx >= 0:
            # ====================================================================
            # Load initial hidden states for all V heads in this Q/K head group
            # Shape: [GROUP_SIZE, BK, BV]
            # - GROUP_SIZE dimension: all V heads managed by this Q/K head
            # - BK dimension: key/query embedding dimension
            # - BV dimension: value embedding dimension
            # ====================================================================
            # Calculate memory address with broadcasting:
            # i_hv[:, None, None]: [GROUP_SIZE, 1, 1]
            # o_k[None, :, None]: [1, BK, 1]
            # o_v[None, None, :]: [1, 1, BV]
            # Result: p_h has shape [GROUP_SIZE, BK, BV]
            p_h = (
                h0_source
                + idx * HV * K * V
                + i_hv[:, None, None] * K * V
                + o_k[None, :, None] * V
                + o_v[None, None, :]
            )
            b_h = tl.load(p_h).to(tl.float32)  # [GROUP_SIZE, BK, BV]

            # ====================================================================
            # Pre-load conv_state sliding windows and weights for K, V, Q
            # Window size = CONV_WIDTH - 1 (historical states)
            # Conv states are stored as tuples for sliding window updates
            # This avoids repeated memory loads in the token processing loop
            # ====================================================================
            
            # K conv setup (shared across all V heads)
            # k_feats: [BK] - Feature indices for K in mixed_qkv tensor
            k_dim_start = key_dim + i_h * K
            k_feats = k_dim_start + o_k
            
            # Load K conv weights and states
            # k_weights: tuple of [BK] tensors, length CONV_WIDTH
            # b_k_conv_states: tuple of [BK] tensors, length CONV_WIDTH-1
            b_k_conv_states = ()
            k_weights = (tl.load(conv_w_ptr + k_feats * stride_conv_w_dim),)
            for i in tl.static_range(CONV_WIDTH-1):
                b_k_conv_state = tl.load(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (k_feats * stride_conv_state_dim) + i * stride_conv_state_tok)
                w_val = tl.load(conv_w_ptr + k_feats * stride_conv_w_dim + (i+1) * stride_conv_w_width)
                k_weights = tuple_combine(k_weights, w_val)
                b_k_conv_states = tuple_combine(b_k_conv_states, b_k_conv_state)
            
            # V conv setup (batched for all GROUP_SIZE V heads)
            # v_feats: [GROUP_SIZE, BV] - Feature indices for all V heads
            # Broadcasting: i_hv[:, None] * V creates [GROUP_SIZE, 1], then + o_v[None, :] creates [GROUP_SIZE, BV]
            v_dim_start = 2 * key_dim + i_hv[:, None] * V
            v_feats = v_dim_start + o_v[None, :]
            
            # Load V conv weights and states for all V heads
            # v_weights: tuple of [GROUP_SIZE, BV] tensors, length CONV_WIDTH
            # b_v_conv_states: tuple of [GROUP_SIZE, BV] tensors, length CONV_WIDTH-1
            b_v_conv_states = ()
            v_weights = (tl.load(conv_w_ptr + v_feats * stride_conv_w_dim),)
            for j in tl.static_range(CONV_WIDTH-1):
                b_v_conv_state = tl.load(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (v_feats * stride_conv_state_dim) + j * stride_conv_state_tok)
                w_val = tl.load(conv_w_ptr + v_feats * stride_conv_w_dim + (j+1) * stride_conv_w_width)
                v_weights = tuple_combine(v_weights, w_val)
                b_v_conv_states = tuple_combine(b_v_conv_states, b_v_conv_state)

            # Q conv setup (shared across all V heads)
            # q_feats: [BK] - Feature indices for Q in mixed_qkv tensor
            q_dim_start = i_h * K
            q_feats = q_dim_start + o_k
            
            # Load Q conv weights and states
            # q_weights: tuple of [BK] tensors, length CONV_WIDTH
            # b_q_conv_states: tuple of [BK] tensors, length CONV_WIDTH-1
            b_q_conv_states = ()
            q_weights = (tl.load(conv_w_ptr + q_feats * stride_conv_w_dim),)
            for i in tl.static_range(CONV_WIDTH-1):
                b_q_conv_state = tl.load(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (q_feats * stride_conv_state_dim) + i * stride_conv_state_tok)
                w_val = tl.load(conv_w_ptr + q_feats * stride_conv_w_dim + (i+1) * stride_conv_w_width)
                q_weights = tuple_combine(q_weights, w_val)
                b_q_conv_states = tuple_combine(b_q_conv_states, b_q_conv_state)
            
            # Memory barrier for AMD GPUs to ensure all loads complete
            # gl.amd.cdna3.sched_barrier(0)
            # ====================================================================
            # Main token processing loop
            # Processing order: K → V (all heads) → Q → Delta Rule (all heads)
            # This order hides memory latency by interleaving loads with computation
            # ====================================================================
            for idx_token in tl.static_range(seqlen):
                # ================================================================
                # Step 1: Conv1D for K
                # Shape: [BK]
                # K is shared across all V heads, so computed only once per token
                # ================================================================
                # Initialize accumulator with bias
                k_conv_acc = tl.load(conv_bias_ptr + k_feats).to(tl.float32) if HAS_CONV_BIAS else tl.zeros([BK], dtype=tl.float32)
                
                # Load current token's K values and add to sliding window
                k_ptrs = (
                    x_ptr + idx_seq * stride_x_seq 
                    + k_feats * stride_x_dim 
                    + idx_token * stride_x_token
                )
                b_k_conv_states = tuple_combine(b_k_conv_states, tl.load(k_ptrs))
                
                # Causal convolution: weighted sum over sliding window
                # b_k_conv_states now has length CONV_WIDTH (history + current)
                for j in tl.static_range(CONV_WIDTH):
                    k_conv_acc += b_k_conv_states[j] * k_weights[j]
                
                # Slide window: drop oldest state
                b_k_conv_states = b_k_conv_states[1:]
                
                # Apply SiLU activation if enabled
                if SILU_ACTIVATION:
                    k_conv_acc = k_conv_acc / (1 + tl.exp(-k_conv_acc))
                
                b_k = k_conv_acc.to(tl.float32)  # [BK]
                
                # Apply L2 normalization to K if enabled
                if USE_QK_L2NORM_IN_KERNEL:
                    b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k) + 1e-6))
                
                # ================================================================
                # Step 2: Conv1D for all V heads
                # Shape: [GROUP_SIZE, BV]
                # All V heads processed in parallel using batched operations
                # Maintains K→V→Q load order for optimal memory access
                # ================================================================
                # Initialize accumulator with bias for all V heads
                v_conv_acc = tl.load(conv_bias_ptr + v_feats).to(tl.float32) if HAS_CONV_BIAS else tl.zeros([GROUP_SIZE, BV], dtype=tl.float32)
                
                # Load current token's V values for all heads and add to sliding window
                v_ptrs = (
                    x_ptr + idx_seq * stride_x_seq 
                    + v_feats * stride_x_dim 
                    + idx_token * stride_x_token
                )
                b_v_conv_states = tuple_combine(b_v_conv_states, tl.load(v_ptrs))
                
                # Causal convolution: weighted sum over sliding window
                # Operations broadcast across [GROUP_SIZE, BV]
                for j in tl.static_range(CONV_WIDTH):
                    v_conv_acc += b_v_conv_states[j] * v_weights[j]
                
                # Slide window: drop oldest states
                b_v_conv_states = b_v_conv_states[1:]
                
                # Apply SiLU activation if enabled
                if SILU_ACTIVATION:
                    v_conv_acc = v_conv_acc / (1 + tl.exp(-v_conv_acc))
                
                b_v = v_conv_acc.to(tl.float32)  # [GROUP_SIZE, BV]
                
                # ================================================================
                # Step 3: Conv1D for Q
                # Shape: [BK]
                # Q is shared across all V heads, so computed only once per token
                # This maintains K→V→Q load order for optimal memory access
                # ================================================================
                # Initialize accumulator with bias
                q_conv_acc = tl.load(conv_bias_ptr + q_feats).to(tl.float32) if HAS_CONV_BIAS else tl.zeros([BK], dtype=tl.float32)
                
                # Load current token's Q values and add to sliding window
                q_ptrs = (
                    x_ptr + idx_seq * stride_x_seq 
                    + q_feats * stride_x_dim 
                    + idx_token * stride_x_token
                )
                b_q_conv_states = tuple_combine(b_q_conv_states, tl.load(q_ptrs))
                
                # Causal convolution: weighted sum over sliding window
                for j in tl.static_range(CONV_WIDTH):
                    q_conv_acc += b_q_conv_states[j] * q_weights[j]
                
                # Slide window: drop oldest state
                b_q_conv_states = b_q_conv_states[1:]
                
                # Apply SiLU activation if enabled
                if SILU_ACTIVATION:
                    q_conv_acc = q_conv_acc / (1 + tl.exp(-q_conv_acc))
                
                b_q = q_conv_acc.to(tl.float32)  # [BK]
                
                # Apply L2 normalization and scaling to Q
                if USE_QK_L2NORM_IN_KERNEL:
                    b_q_scale = scale / (tl.sqrt(tl.sum(b_q * b_q) + 1e-6))
                else:
                    b_q_scale = scale
                b_q = b_q * b_q_scale  # [BK]
                
                # ================================================================
                # Step 4: Batched Delta Rule updates for all V heads
                # Using broadcasting for efficient parallel processing
                # All GROUP_SIZE V heads updated simultaneously
                # ================================================================
                
                # Load time-variant gating parameters for all V heads
                # p_a, p_b: [GROUP_SIZE] - Memory addresses for parameters
                # b_a, b_b: [GROUP_SIZE] - Gating parameters for current token
                p_a = a + (bos + idx_token) * HV + i_hv
                p_b = b + (bos + idx_token) * HV + i_hv
                b_a = tl.load(p_a).to(tl.float32)
                b_b = tl.load(p_b).to(tl.float32)
                
                # Compute gating factors for all V heads using broadcasting
                # x, beta_x, softplus_x, b_g, b_beta: all have shape [GROUP_SIZE]
                x = b_a + b_dt_bias  # [GROUP_SIZE] + [GROUP_SIZE]
                beta_x = softplus_beta * x
                softplus_x = tl.where(
                    beta_x <= softplus_threshold,
                    (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(beta_x)),
                    x,
                )
                b_g = -tl.exp(b_A_log) * softplus_x  # [GROUP_SIZE]
                b_beta = 1.0 / (1.0 + tl.exp(-b_b))  # [GROUP_SIZE]
                
                # Batched Delta Rule recurrent update using broadcasting
                # Input shapes:
                #   b_h: [GROUP_SIZE, BK, BV]
                #   b_k: [BK]
                #   b_q: [BK]
                #   b_v: [GROUP_SIZE, BV]
                #   b_g: [GROUP_SIZE]
                #   b_beta: [GROUP_SIZE]
                #
                # Broadcasting:
                #   b_g[:, None, None] -> [GROUP_SIZE, 1, 1]
                #   b_k[None, :, None] -> [1, BK, 1]
                #   b_q[None, :, None] -> [1, BK, 1]
                #   b_v[:, None, :] -> [GROUP_SIZE, 1, BV]
                #   b_beta[:, None] -> [GROUP_SIZE, 1]
                
                # Step 4a: Apply exponential decay to hidden states
                b_h *= tl.exp(b_g[:, None, None])  # [GROUP_SIZE, BK, BV]
                
                # Step 4b: Delta rule correction
                # tl.sum(..., 1) sums over dimension 1 (BK), result: [GROUP_SIZE, BV]
                b_v -= tl.sum(b_h * b_k[None, :, None], 1)
                
                # Step 4c: Apply beta gating
                b_v *= b_beta[:, None]  # [GROUP_SIZE, BV]
                
                # Step 4d: Update hidden states with outer product
                b_h += b_k[None, :, None] * b_v[:, None, :]  # [GROUP_SIZE, BK, BV]
                
                # Step 4e: Compute outputs for all V heads
                # tl.sum(..., 1) sums over dimension 1 (BK), result: [GROUP_SIZE, BV]
                b_o = tl.sum(b_h * b_q[None, :, None], 1)
                
                # Step 4f: Store outputs for all V heads
                # p_o: [GROUP_SIZE, BV] - Memory addresses with broadcasting
                p_o = o + ((i_k * all + bos + idx_token) * HV + i_hv[:, None]) * V + o_v[None, :]
                tl.store(p_o, b_o.to(p_o.dtype.element_ty))
                
                # Step 4g: Store updated hidden states for all V heads
                # p_h0: [GROUP_SIZE, BK, BV] - Memory addresses with broadcasting
                p_h0 = (
                    h0_source
                    + idx * HV * K * V
                    + i_hv[:, None, None] * K * V
                    + o_k[None, :, None] * V
                    + o_v[None, None, :]
                )
                tl.store(p_h0, b_h.to(p_h0.dtype.element_ty))
                

            # ====================================================================
            # Write back final conv_state sliding windows to memory
            # After processing all tokens, store the last CONV_WIDTH-1 historical states
            # These will be used as initial states for the next decoding step
            # ====================================================================
            
            # Write back Q conv_states: [BK] for each of CONV_WIDTH-1 steps
            for i in tl.static_range(CONV_WIDTH-1):
                tl.store(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (q_feats * stride_conv_state_dim) + i * stride_conv_state_tok, b_q_conv_states[i])
            
            # Write back K conv_states: [BK] for each of CONV_WIDTH-1 steps
            for i in tl.static_range(CONV_WIDTH-1):
                tl.store(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (k_feats * stride_conv_state_dim) + i * stride_conv_state_tok, b_k_conv_states[i])
            
            # Write back V conv_states: [GROUP_SIZE, BV] for each of CONV_WIDTH-1 steps
            # All V heads' states are written in parallel
            for i in tl.static_range(CONV_WIDTH-1): 
                tl.store(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (v_feats * stride_conv_state_dim) + i * stride_conv_state_tok, b_v_conv_states[i])
            
            return

    # ============================================================================
    # Non-initial-state branch: Zero initialization
    # This branch handles cases where no initial hidden state is provided
    # Processing is identical to initial-state branch except h0 starts at zero
    # ============================================================================
    
    # Initialize zero hidden states for all V heads in the group
    # Shape: [GROUP_SIZE, BK, BV]
    b_h = tl.zeros([GROUP_SIZE, BK, BV], dtype=tl.float32)

    # ====================================================================
    # Pre-load conv_state sliding windows and weights for K, V, Q
    # Same structure as initial-state branch
    # ====================================================================
    
    # K conv setup (shared across all V heads)
    k_dim_start = key_dim + i_h * K
    k_feats = k_dim_start + o_k  # [BK]
    
    b_k_conv_states = ()
    k_weights = (tl.load(conv_w_ptr + k_feats * stride_conv_w_dim),)
    for i in tl.static_range(CONV_WIDTH-1):
        b_k_conv_state = tl.load(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (k_feats * stride_conv_state_dim) + i * stride_conv_state_tok)
        w_val = tl.load(conv_w_ptr + k_feats * stride_conv_w_dim + (i+1) * stride_conv_w_width)
        k_weights = tuple_combine(k_weights, w_val)
        b_k_conv_states = tuple_combine(b_k_conv_states, b_k_conv_state)
    
    # V conv setup (batched for all GROUP_SIZE V heads)
    v_dim_start = 2 * key_dim + i_hv[:, None] * V
    v_feats = v_dim_start + o_v[None, :]  # [GROUP_SIZE, BV]
    
    b_v_conv_states = ()
    v_weights = (tl.load(conv_w_ptr + v_feats * stride_conv_w_dim),)
    for j in tl.static_range(CONV_WIDTH-1):
        b_v_conv_state = tl.load(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (v_feats * stride_conv_state_dim) + j * stride_conv_state_tok)
        w_val = tl.load(conv_w_ptr + v_feats * stride_conv_w_dim + (j+1) * stride_conv_w_width)
        v_weights = tuple_combine(v_weights, w_val)
        b_v_conv_states = tuple_combine(b_v_conv_states, b_v_conv_state)

    # Q conv setup (shared across all V heads)
    q_dim_start = i_h * K
    q_feats = q_dim_start + o_k  # [BK]
    
    b_q_conv_states = ()
    q_weights = (tl.load(conv_w_ptr + q_feats * stride_conv_w_dim),)
    for i in tl.static_range(CONV_WIDTH-1):
        b_q_conv_state = tl.load(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (q_feats * stride_conv_state_dim) + i * stride_conv_state_tok)
        w_val = tl.load(conv_w_ptr + q_feats * stride_conv_w_dim + (i+1) * stride_conv_w_width)
        q_weights = tuple_combine(q_weights, w_val)
        b_q_conv_states = tuple_combine(b_q_conv_states, b_q_conv_state)
    # ====================================================================
    # Main token processing loop (identical to initial-state branch)
    # ====================================================================
    for idx_token in tl.static_range(seqlen):
        # ================================================================
        # Step 1: Conv1D for K (shared across all V heads)
        # Shape: [BK]
        # ================================================================
        k_conv_acc = tl.load(conv_bias_ptr + k_feats).to(tl.float32) if HAS_CONV_BIAS else tl.zeros([BK], dtype=tl.float32)
        k_ptrs = (
            x_ptr + idx_seq * stride_x_seq 
            + k_feats * stride_x_dim 
            + idx_token * stride_x_token
        )
        b_k_conv_states = tuple_combine(b_k_conv_states, tl.load(k_ptrs))
        for j in tl.static_range(CONV_WIDTH):
            k_conv_acc += b_k_conv_states[j] * k_weights[j]
        b_k_conv_states = b_k_conv_states[1:]
        
        if SILU_ACTIVATION:
            k_conv_acc = k_conv_acc / (1 + tl.exp(-k_conv_acc))
        
        b_k = k_conv_acc.to(tl.float32)  # [BK]
        
        if USE_QK_L2NORM_IN_KERNEL:
            b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k) + 1e-6))
        
        # ================================================================
        # Step 2: Conv1D for all V heads (batched processing)
        # Shape: [GROUP_SIZE, BV]
        # ================================================================
        v_conv_acc = tl.load(conv_bias_ptr + v_feats).to(tl.float32) if HAS_CONV_BIAS else tl.zeros([GROUP_SIZE, BV], dtype=tl.float32)
        v_ptrs = (
            x_ptr + idx_seq * stride_x_seq 
            + v_feats * stride_x_dim 
            + idx_token * stride_x_token
        )
        
        b_v_conv_states = tuple_combine(b_v_conv_states, tl.load(v_ptrs))
        for j in tl.static_range(CONV_WIDTH):
            v_conv_acc += b_v_conv_states[j] * v_weights[j]
        b_v_conv_states = b_v_conv_states[1:]
        
        if SILU_ACTIVATION:
            v_conv_acc = v_conv_acc / (1 + tl.exp(-v_conv_acc))
        b_v = v_conv_acc.to(tl.float32)  # [GROUP_SIZE, BV]
        
        # ================================================================
        # Step 3: Conv1D for Q (shared across all V heads)
        # Shape: [BK]
        # ================================================================
        q_conv_acc = tl.load(conv_bias_ptr + q_feats).to(tl.float32) if HAS_CONV_BIAS else tl.zeros([BK], dtype=tl.float32)
        q_ptrs = (
            x_ptr + idx_seq * stride_x_seq 
            + q_feats * stride_x_dim 
            + idx_token * stride_x_token
        )
        b_q_conv_states = tuple_combine(b_q_conv_states, tl.load(q_ptrs))
        for j in tl.static_range(CONV_WIDTH):
            q_conv_acc += b_q_conv_states[j] * q_weights[j]
        b_q_conv_states = b_q_conv_states[1:]
        
        if SILU_ACTIVATION:
            q_conv_acc = q_conv_acc / (1 + tl.exp(-q_conv_acc))
        
        b_q = q_conv_acc.to(tl.float32)  # [BK]
        if USE_QK_L2NORM_IN_KERNEL:
            b_q_scale = scale / (tl.sqrt(tl.sum(b_q * b_q) + 1e-6))
        else:
            b_q_scale = scale
        b_q = b_q * b_q_scale  # [BK]
        
        # ================================================================
        # Step 4: Batched Delta Rule updates for all V heads
        # Using broadcasting for efficient parallel processing
        # ================================================================
        p_a = a + (bos + idx_token) * HV + i_hv
        p_b = b + (bos + idx_token) * HV + i_hv
        b_a = tl.load(p_a).to(tl.float32)  # [GROUP_SIZE]
        b_b = tl.load(p_b).to(tl.float32)  # [GROUP_SIZE]
        x = b_a + b_dt_bias  # [GROUP_SIZE]
        beta_x = softplus_beta * x
        softplus_x = tl.where(
            beta_x <= softplus_threshold,
            (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(beta_x)),
            x,
        )
        b_g = -tl.exp(b_A_log) * softplus_x  # [GROUP_SIZE]
        b_beta = 1.0 / (1.0 + tl.exp(-b_b))  # [GROUP_SIZE]
        
        # Batched Delta Rule update with broadcasting
        b_h *= tl.exp(b_g[:, None, None])  # [GROUP_SIZE, BK, BV]
        b_v -= tl.sum(b_h * b_k[None, :, None], 1)  # [GROUP_SIZE, BV]
        b_v *= b_beta[:, None]  # [GROUP_SIZE, BV]
        b_h += b_k[None, :, None] * b_v[:, None, :]  # [GROUP_SIZE, BK, BV]
        
        b_o = tl.sum(b_h * b_q[None, :, None], 1)  # [GROUP_SIZE, BV]
        p_o = o + ((i_k * all + bos + idx_token) * HV + i_hv[:, None]) * V + o_v[None, :]
        tl.store(p_o, b_o.to(p_o.dtype.element_ty))
        
    # ====================================================================
    # Write back final conv_state sliding windows to memory
    # ====================================================================
    for i in tl.static_range(CONV_WIDTH-1):
        tl.store(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (q_feats * stride_conv_state_dim) + i * stride_conv_state_tok, b_q_conv_states[i])
    
    for i in tl.static_range(CONV_WIDTH-1):
        tl.store(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (k_feats * stride_conv_state_dim) + i * stride_conv_state_tok, b_k_conv_states[i])
    
    for j in tl.static_range(CONV_WIDTH-1):
        tl.store(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (v_feats * stride_conv_state_dim) + j * stride_conv_state_tok, b_v_conv_states[j])

@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0_source"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=get_autotune_config(),
    key=['K', 'V']
)
@triton.jit(do_not_specialize=["T"])
def fused_gdn_fwd_decode_kernel_v3(
    # Conv1D inputs
    x_ptr,  # (batch, dim, seqlen) where dim = 2*key_dim + value_dim
    conv_w_ptr,  # (dim, conv_width)
    conv_bias_ptr,
    conv_state_ptr,
    conv_state_indices_ptr,  # Note: same as h0_indices, loaded only once
    # Gating inputs
    A_log,
    a,
    dt_bias,
    b,
    # SSM state
    h0_source,
    h0_indices,  # Note: same as conv_state_indices_ptr, shared to avoid redundant loads
    cu_seqlens,
    # Output
    o,
    # Dimensions
    key_dim: tl.constexpr,
    value_dim: tl.constexpr,
    batch: int,
    dim: tl.constexpr,
    seqlen: tl.constexpr,
    conv_state_len: tl.constexpr,
    num_cache_lines: tl.constexpr,
    T: int,
    # Gating parameters
    softplus_beta: float,
    softplus_threshold: float,
    scale: float,
    # Strides for conv
    stride_x_seq: tl.constexpr,
    stride_x_dim: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_conv_w_dim: tl.constexpr,
    stride_conv_w_width: tl.constexpr,
    stride_conv_state_seq: tl.constexpr,
    stride_conv_state_dim: tl.constexpr,
    stride_conv_state_tok: tl.constexpr,
    stride_state_indices: tl.constexpr,
    # Others
    pad_slot_id: tl.constexpr,
    # Meta-parameters
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    HAS_CONV_BIAS: tl.constexpr,
    CONV_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Optimized fused kernel with Q/K reuse and batched V head processing.
    
    Key optimizations:
    1. Each block processes GROUP_SIZE (HV//H) value heads simultaneously
    2. Q/K are computed once per group and broadcast across all V heads
    3. V heads are represented as tensor dimensions for efficient batching
    4. Delta Rule updates use broadcasting instead of loops
    
    Processing flow:
    1. Load gating parameters for all V heads: [GROUP_SIZE]
    2. Load/initialize hidden states for all V heads: [GROUP_SIZE, BK, BV]
    3. For each token:
       a. Conv1D for K: [BK] (shared across all V heads)
       b. Conv1D for V: [GROUP_SIZE, BV] (batched for all V heads)
       c. Conv1D for Q: [BK] (shared across all V heads)
       d. Batched Delta Rule update with broadcasting:
          - b_h: [GROUP_SIZE, BK, BV]
          - b_k: [BK] -> broadcast to [1, BK, 1]
          - b_q: [BK] -> broadcast to [1, BK, 1]
          - b_v: [GROUP_SIZE, BV]
          - b_g, b_beta: [GROUP_SIZE] -> broadcast to [GROUP_SIZE, 1, 1] or [GROUP_SIZE, 1]
       e. Store outputs for all V heads
    4. Write back conv states
    """
    # Get program IDs - now indexed by Q/K heads (not V heads)
    i_k, i_h, i_n = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    
    # Number of V heads per Q/K head (group size)
    GROUP_SIZE: tl.constexpr = HV // H
    
    # Handle variable length sequences
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        all = T
        T = eos - bos
        idx_seq = bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T
        idx_seq = i_n
    
    if idx_seq >= batch:
        return
    
    if IS_CONTINUOUS_BATCHING:
        idx = tl.load(
            conv_state_indices_ptr + idx_seq * stride_state_indices
        ).to(tl.int64)
        conv_state_batch_coord = idx
    else:
        idx = idx_seq
        conv_state_batch_coord = idx_seq
        
    if USE_PAD_SLOT:
        if conv_state_batch_coord == pad_slot_id:
            return
    
    # ============================================================================
    # Initialization: Setup dimensions and load gating parameters
    # ============================================================================
    
    # Define offset ranges for tensor blocks
    # o_k: [BK] - Offsets for K dimension (shared by Q and K)
    # o_v: [BV] - Offsets for V dimension

    i_v = 0

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    # Define V head indices for this Q/K head group
    # i_hv: [GROUP_SIZE] - Absolute indices of V heads in this group
    # Example: if i_h=1, HV=8, H=4, then GROUP_SIZE=2, i_hv=[2, 3]
    i_hv = i_h * GROUP_SIZE + tl.arange(0, GROUP_SIZE)

    # Load time-invariant gating parameters for all V heads in this group
    # b_A_log: [GROUP_SIZE] - Log of recurrent matrix eigenvalues
    # b_dt_bias: [GROUP_SIZE] - Time step bias parameters
    b_A_log = tl.load(A_log + i_hv).to(tl.float32)
    b_dt_bias = tl.load(dt_bias + i_hv).to(tl.float32)

    b_h = tl.zeros([GROUP_SIZE, BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        # idx was already loaded above (same as conv_state_batch_coord in continuous batching mode)
        if idx >= 0:
            p_h = (
                h0_source
                + idx * HV * K * V
                + i_hv[:, None, None] * K * V
                + o_k[None, :, None] * V
                + o_v[None, None, :]
            )
            b_h = tl.load(p_h).to(tl.float32)  # [GROUP_SIZE, BK, BV]

    # K conv setup (shared across all V heads)
    # k_feats: [BK] - Feature indices for K in mixed_qkv tensor
    k_dim_start = key_dim + i_h * K
    k_feats = k_dim_start + o_k
    
    # Load K conv weights and states
    # k_weights: tuple of [BK] tensors, length CONV_WIDTH
    # b_k_conv_states: tuple of [BK] tensors, length CONV_WIDTH-1
    b_k_conv_states = ()
    k_weights = (tl.load(conv_w_ptr + k_feats * stride_conv_w_dim),)
    for i in tl.static_range(CONV_WIDTH-1):
        b_k_conv_state = tl.load(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (k_feats * stride_conv_state_dim) + i * stride_conv_state_tok)
        w_val = tl.load(conv_w_ptr + k_feats * stride_conv_w_dim + (i+1) * stride_conv_w_width)
        k_weights = tuple_combine(k_weights, w_val)
        b_k_conv_states = tuple_combine(b_k_conv_states, b_k_conv_state)
    
    # V conv setup (batched for all GROUP_SIZE V heads)
    # v_feats: [GROUP_SIZE, BV] - Feature indices for all V heads
    # Broadcasting: i_hv[:, None] * V creates [GROUP_SIZE, 1], then + o_v[None, :] creates [GROUP_SIZE, BV]
    v_dim_start = 2 * key_dim + i_hv[:, None] * V
    v_feats = v_dim_start + o_v[None, :]
    
    # Load V conv weights and states for all V heads
    # v_weights: tuple of [GROUP_SIZE, BV] tensors, length CONV_WIDTH
    # b_v_conv_states: tuple of [GROUP_SIZE, BV] tensors, length CONV_WIDTH-1
    b_v_conv_states = ()
    v_weights = (tl.load(conv_w_ptr + v_feats * stride_conv_w_dim),)
    for j in tl.static_range(CONV_WIDTH-1):
        b_v_conv_state = tl.load(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (v_feats * stride_conv_state_dim) + j * stride_conv_state_tok)
        w_val = tl.load(conv_w_ptr + v_feats * stride_conv_w_dim + (j+1) * stride_conv_w_width)
        v_weights = tuple_combine(v_weights, w_val)
        b_v_conv_states = tuple_combine(b_v_conv_states, b_v_conv_state)

    # Q conv setup (shared across all V heads)
    # q_feats: [BK] - Feature indices for Q in mixed_qkv tensor
    q_dim_start = i_h * K
    q_feats = q_dim_start + o_k
    
    # Load Q conv weights and states
    # q_weights: tuple of [BK] tensors, length CONV_WIDTH
    # b_q_conv_states: tuple of [BK] tensors, length CONV_WIDTH-1
    b_q_conv_states = ()
    q_weights = (tl.load(conv_w_ptr + q_feats * stride_conv_w_dim),)
    for i in tl.static_range(CONV_WIDTH-1):
        b_q_conv_state = tl.load(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (q_feats * stride_conv_state_dim) + i * stride_conv_state_tok)
        w_val = tl.load(conv_w_ptr + q_feats * stride_conv_w_dim + (i+1) * stride_conv_w_width)
        q_weights = tuple_combine(q_weights, w_val)
        b_q_conv_states = tuple_combine(b_q_conv_states, b_q_conv_state)
    
    # Memory barrier for AMD GPUs to ensure all loads complete
    gl.amd.cdna3.sched_barrier(0)
    # ====================================================================
    # Main token processing loop
    # Processing order: K → V (all heads) → Q → Delta Rule (all heads)
    # This order hides memory latency by interleaving loads with computation
    # ====================================================================
    for idx_token in tl.static_range(seqlen):
        # ================================================================
        # Step 1: Conv1D for K
        # Shape: [BK]
        # K is shared across all V heads, so computed only once per token
        # ================================================================
        # Initialize accumulator with bias
        k_conv_acc = tl.load(conv_bias_ptr + k_feats).to(tl.float32) if HAS_CONV_BIAS else tl.zeros([BK], dtype=tl.float32)
        
        # Load current token's K values and add to sliding window
        k_ptrs = (
            x_ptr + idx_seq * stride_x_seq 
            + k_feats * stride_x_dim 
            + idx_token * stride_x_token
        )
        b_k_conv_states = tuple_combine(b_k_conv_states, tl.load(k_ptrs))
        
        # Causal convolution: weighted sum over sliding window
        # b_k_conv_states now has length CONV_WIDTH (history + current)
        for j in tl.static_range(CONV_WIDTH):
            k_conv_acc += b_k_conv_states[j] * k_weights[j]
        
        # Slide window: drop oldest state
        b_k_conv_states = b_k_conv_states[1:]
        
        # Apply SiLU activation if enabled
        if SILU_ACTIVATION:
            k_conv_acc = k_conv_acc / (1 + tl.exp(-k_conv_acc))
        
        b_k = k_conv_acc.to(tl.float32)  # [BK]
        
        # Apply L2 normalization to K if enabled
        if USE_QK_L2NORM_IN_KERNEL:
            b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k) + 1e-6))
        
        # ================================================================
        # Step 2: Conv1D for all V heads
        # Shape: [GROUP_SIZE, BV]
        # All V heads processed in parallel using batched operations
        # Maintains K→V→Q load order for optimal memory access
        # ================================================================
        # Initialize accumulator with bias for all V heads
        v_conv_acc = tl.load(conv_bias_ptr + v_feats).to(tl.float32) if HAS_CONV_BIAS else tl.zeros([GROUP_SIZE, BV], dtype=tl.float32)
        
        # Load current token's V values for all heads and add to sliding window
        v_ptrs = (
            x_ptr + idx_seq * stride_x_seq 
            + v_feats * stride_x_dim 
            + idx_token * stride_x_token
        )
        b_v_conv_states = tuple_combine(b_v_conv_states, tl.load(v_ptrs))
        
        # Causal convolution: weighted sum over sliding window
        # Operations broadcast across [GROUP_SIZE, BV]
        for j in tl.static_range(CONV_WIDTH):
            v_conv_acc += b_v_conv_states[j] * v_weights[j]
        
        # Slide window: drop oldest states
        b_v_conv_states = b_v_conv_states[1:]
        
        # Apply SiLU activation if enabled
        if SILU_ACTIVATION:
            v_conv_acc = v_conv_acc / (1 + tl.exp(-v_conv_acc))
        
        b_v = v_conv_acc.to(tl.float32)  # [GROUP_SIZE, BV]
        
        # ================================================================
        # Step 3: Conv1D for Q
        # Shape: [BK]
        # Q is shared across all V heads, so computed only once per token
        # This maintains K→V→Q load order for optimal memory access
        # ================================================================
        # Initialize accumulator with bias
        q_conv_acc = tl.load(conv_bias_ptr + q_feats).to(tl.float32) if HAS_CONV_BIAS else tl.zeros([BK], dtype=tl.float32)
        
        # Load current token's Q values and add to sliding window
        q_ptrs = (
            x_ptr + idx_seq * stride_x_seq 
            + q_feats * stride_x_dim 
            + idx_token * stride_x_token
        )
        b_q_conv_states = tuple_combine(b_q_conv_states, tl.load(q_ptrs))
        
        # Causal convolution: weighted sum over sliding window
        for j in tl.static_range(CONV_WIDTH):
            q_conv_acc += b_q_conv_states[j] * q_weights[j]
        
        # Slide window: drop oldest state
        b_q_conv_states = b_q_conv_states[1:]
        
        # Apply SiLU activation if enabled
        if SILU_ACTIVATION:
            q_conv_acc = q_conv_acc / (1 + tl.exp(-q_conv_acc))
        
        b_q = q_conv_acc.to(tl.float32)  # [BK]
        
        # Apply L2 normalization and scaling to Q
        if USE_QK_L2NORM_IN_KERNEL:
            b_q_scale = scale / (tl.sqrt(tl.sum(b_q * b_q) + 1e-6))
        else:
            b_q_scale = scale
        b_q = b_q * b_q_scale  # [BK]
        
        # ================================================================
        # Step 4: Batched Delta Rule updates for all V heads
        # Using broadcasting for efficient parallel processing
        # All GROUP_SIZE V heads updated simultaneously
        # ================================================================
        
        # Load time-variant gating parameters for all V heads
        # p_a, p_b: [GROUP_SIZE] - Memory addresses for parameters
        # b_a, b_b: [GROUP_SIZE] - Gating parameters for current token
        p_a = a + (bos + idx_token) * HV + i_hv
        p_b = b + (bos + idx_token) * HV + i_hv
        b_a = tl.load(p_a).to(tl.float32)
        b_b = tl.load(p_b).to(tl.float32)
        
        # Compute gating factors for all V heads using broadcasting
        # x, beta_x, softplus_x, b_g, b_beta: all have shape [GROUP_SIZE]
        x = b_a + b_dt_bias  # [GROUP_SIZE] + [GROUP_SIZE]
        beta_x = softplus_beta * x
        softplus_x = tl.where(
            beta_x <= softplus_threshold,
            (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(beta_x)),
            x,
        )
        b_g = -tl.exp(b_A_log) * softplus_x  # [GROUP_SIZE]
        b_beta = 1.0 / (1.0 + tl.exp(-b_b))  # [GROUP_SIZE]
        
        # Batched Delta Rule recurrent update using broadcasting
        # Input shapes:
        #   b_h: [GROUP_SIZE, BK, BV]
        #   b_k: [BK]
        #   b_q: [BK]
        #   b_v: [GROUP_SIZE, BV]
        #   b_g: [GROUP_SIZE]
        #   b_beta: [GROUP_SIZE]
        #
        # Broadcasting:
        #   b_g[:, None, None] -> [GROUP_SIZE, 1, 1]
        #   b_k[None, :, None] -> [1, BK, 1]
        #   b_q[None, :, None] -> [1, BK, 1]
        #   b_v[:, None, :] -> [GROUP_SIZE, 1, BV]
        #   b_beta[:, None] -> [GROUP_SIZE, 1]
        
        # Step 4a: Apply exponential decay to hidden states
        b_h *= tl.exp(b_g[:, None, None])  # [GROUP_SIZE, BK, BV]
        
        # Step 4b: Delta rule correction
        # tl.sum(..., 1) sums over dimension 1 (BK), result: [GROUP_SIZE, BV]
        b_v -= tl.sum(b_h * b_k[None, :, None], 1)
        
        # Step 4c: Apply beta gating
        b_v *= b_beta[:, None]  # [GROUP_SIZE, BV]
        
        # Step 4d: Update hidden states with outer product
        b_h += b_k[None, :, None] * b_v[:, None, :]  # [GROUP_SIZE, BK, BV]
        
        # Step 4e: Compute outputs for all V heads
        # tl.sum(..., 1) sums over dimension 1 (BK), result: [GROUP_SIZE, BV]
        b_o = tl.sum(b_h * b_q[None, :, None], 1)
        
        # Step 4f: Store outputs for all V heads
        # p_o: [GROUP_SIZE, BV] - Memory addresses with broadcasting
        p_o = o + ((i_k * all + bos + idx_token) * HV + i_hv[:, None]) * V + o_v[None, :]
        tl.store(p_o, b_o.to(p_o.dtype.element_ty))
        
        # Step 4g: Store updated hidden states for all V heads
        # p_h0: [GROUP_SIZE, BK, BV] - Memory addresses with broadcasting
        # Note: idx is the same as conv_state_batch_coord (loaded once at the start)
        p_h0 = (
            h0_source
            + idx * HV * K * V
            + i_hv[:, None, None] * K * V
            + o_k[None, :, None] * V
            + o_v[None, None, :]
        )
        tl.store(p_h0, b_h.to(p_h0.dtype.element_ty))
        

    # ====================================================================
    # Write back final conv_state sliding windows to memory
    # After processing all tokens, store the last CONV_WIDTH-1 historical states
    # These will be used as initial states for the next decoding step
    # ====================================================================
    
    # Write back Q conv_states: [BK] for each of CONV_WIDTH-1 steps
    for i in tl.static_range(CONV_WIDTH-1):
        tl.store(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (q_feats * stride_conv_state_dim) + i * stride_conv_state_tok, b_q_conv_states[i])
    
    # Write back K conv_states: [BK] for each of CONV_WIDTH-1 steps
    for i in tl.static_range(CONV_WIDTH-1):
        tl.store(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (k_feats * stride_conv_state_dim) + i * stride_conv_state_tok, b_k_conv_states[i])
    
    # Write back V conv_states: [GROUP_SIZE, BV] for each of CONV_WIDTH-1 steps
    # All V heads' states are written in parallel
    for i in tl.static_range(CONV_WIDTH-1): 
        tl.store(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (v_feats * stride_conv_state_dim) + i * stride_conv_state_tok, b_v_conv_states[i])


def fused_gdn_fwd_decode_v2(
    mixed_qkv: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    ssm_state: torch.Tensor,
    key_dim: int,
    value_dim: int,
    num_heads_qk: int,
    num_heads_v: int,
    head_dim: int,
    conv_bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
    conv_state_indices: Optional[torch.Tensor] = None,
    ssm_state_indices: Optional[torch.Tensor] = None,
    pad_slot_id: int = PAD_SLOT_ID,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = True,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Fused Gating Delta Network forward decode operation (V2 - indexed by V heads).
    
    This version uses a simpler grid layout where each block processes one V head,
    potentially offering better parallelism at the cost of redundant Q/K computation.
    
    Args:
        mixed_qkv: Input tensor (batch, dim, seqlen) where dim = 2*key_dim + value_dim
        conv_state: Convolution state (num_cache_lines, dim, state_len)
        conv_weight: Convolution weights (dim, width)
        A_log: Gating parameter A (num_heads_v * head_dim,)
        a: Gating parameter a (batch, num_heads_v * head_dim)
        dt_bias: Gating parameter dt_bias (num_heads_v * head_dim,)
        b: Gating parameter b (batch, num_heads_v * head_dim)
        ssm_state: SSM state pool (num_cache_lines, num_heads_v * head_dim, head_dim, head_dim)
        key_dim: Dimension of query and key (= num_heads_qk * head_dim)
        value_dim: Dimension of value (= num_heads_v * head_dim)
        num_heads_qk: Number of query/key heads
        num_heads_v: Number of value heads
        head_dim: Dimension per head
        conv_bias: Optional convolution bias (dim,)
        activation: Activation function ("silu" or None)
        conv_state_indices: Optional batch indices for continuous batching
        ssm_state_indices: Optional batch indices for SSM state
        pad_slot_id: ID for padded slots
        scale: Query scaling factor
        use_qk_l2norm_in_kernel: Whether to apply L2 normalization to Q/K
        softplus_beta: Beta parameter for softplus
        softplus_threshold: Threshold for softplus
        cu_seqlens: Cumulative sequence lengths for variable length sequences
    
    Returns:
        Output tensor
    """
    batch, dim, seqlen = mixed_qkv.shape
    assert dim == 2 * key_dim + value_dim, f"dim {dim} != 2*{key_dim} + {value_dim}"
    assert key_dim == num_heads_qk * head_dim, f"key_dim {key_dim} != {num_heads_qk} * {head_dim}"
    assert value_dim == num_heads_v * head_dim, f"value_dim {value_dim} != {num_heads_v} * {head_dim}"
    
    _, conv_width = conv_weight.shape
    num_cache_lines, _, conv_state_len = conv_state.size()
    
    # Head dimensions
    HV = num_heads_v
    K = head_dim
    V = head_dim
    H = num_heads_qk
    
    if scale is None:
        scale = K ** -0.5
    
    T = seqlen
    N = batch if cu_seqlens is None else len(cu_seqlens) - 1
    B = batch
    
    BK = triton.next_power_of_2(K)
    NK = triton.cdiv(K, BK)
    assert NK == 1, "NK > 1 is not supported yet"
    
    # Create output tensor
    o = mixed_qkv.new_empty(NK, B, T, HV, V)
    
    # Grid configuration: launch N * HV blocks (one per V head)
    # This is simpler than v1 but may have redundant Q/K computation
    grid = lambda META: (NK, triton.cdiv(V, META['BV']), N * HV)
    
    # Launch kernel
    stride_state_indices = (
        conv_state_indices.stride(0) if conv_state_indices is not None else 0
    )
    np2_statelen = triton.next_power_of_2(conv_state_len)
    
    fused_gdn_fwd_decode_kernel_v2[grid](
        x_ptr=mixed_qkv,
        conv_w_ptr=conv_weight,
        conv_bias_ptr=conv_bias,
        conv_state_ptr=conv_state,
        conv_state_indices_ptr=conv_state_indices,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        h0_source=ssm_state,
        h0_indices=ssm_state_indices,
        cu_seqlens=cu_seqlens,
        o=o,
        key_dim=key_dim,
        value_dim=value_dim,
        batch=batch,
        dim=dim,
        seqlen=seqlen,
        conv_state_len=conv_state_len,
        num_cache_lines=num_cache_lines,
        T=T,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        scale=scale,
        stride_x_seq=mixed_qkv.stride(0),
        stride_x_dim=mixed_qkv.stride(1),
        stride_x_token=mixed_qkv.stride(2),
        stride_conv_w_dim=conv_weight.stride(0),
        stride_conv_w_width=conv_weight.stride(1),
        stride_conv_state_seq=conv_state.stride(0),
        stride_conv_state_dim=conv_state.stride(1),
        stride_conv_state_tok=conv_state.stride(2),
        stride_state_indices=stride_state_indices,
        pad_slot_id=pad_slot_id,
        B=N,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        HAS_CONV_BIAS=conv_bias is not None,
        CONV_WIDTH=conv_width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        IS_CONTINUOUS_BATCHING=conv_state_indices is not None,
        NP2_STATELEN=np2_statelen,
        USE_PAD_SLOT=pad_slot_id is not None,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
    )
    
    # Squeeze first dimension (NK) to get (B, T, HV, V)
    o = o.squeeze(0)
    return o


def fused_gdn_fwd_decode(
    mixed_qkv: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    ssm_state: torch.Tensor,
    key_dim: int,
    value_dim: int,
    num_heads_qk: int,
    num_heads_v: int,
    head_dim: int,
    conv_bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
    conv_state_indices: Optional[torch.Tensor] = None,
    ssm_state_indices: Optional[torch.Tensor] = None,
    pad_slot_id: int = PAD_SLOT_ID,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = True,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Fused Gating Delta Network forward decode operation (V1 - indexed by Q/K heads).
    
    This version uses Q/K head indexing with GROUP_SIZE loop to process multiple V heads
    per block, reducing redundant Q/K computation but with more complex logic.
    
    Args:
        mixed_qkv: Input tensor (batch, dim, seqlen) where dim = 2*key_dim + value_dim
        conv_state: Convolution state (num_cache_lines, dim, state_len)
        conv_weight: Convolution weights (dim, width)
        A_log: Gating parameter A (num_heads_v * head_dim,)
        a: Gating parameter a (batch, num_heads_v * head_dim)
        dt_bias: Gating parameter dt_bias (num_heads_v * head_dim,)
        b: Gating parameter b (batch, num_heads_v * head_dim)
        ssm_state: SSM state pool (num_cache_lines, num_heads_v * head_dim, head_dim, head_dim)
        key_dim: Dimension of query and key (= num_heads_qk * head_dim)
        value_dim: Dimension of value (= num_heads_v * head_dim)
        num_heads_qk: Number of query/key heads
        num_heads_v: Number of value heads
        head_dim: Dimension per head
        conv_bias: Optional convolution bias (dim,)
        activation: Activation function ("silu" or None)
        conv_state_indices: Optional batch indices for continuous batching
        ssm_state_indices: Optional batch indices for SSM state
        pad_slot_id: ID for padded slots
        scale: Query scaling factor
        use_qk_l2norm_in_kernel: Whether to apply L2 normalization to Q/K
        softplus_beta: Beta parameter for softplus
        softplus_threshold: Threshold for softplus
        cu_seqlens: Cumulative sequence lengths for variable length sequences
    
    Returns:
        Output tensor
    """
    batch, dim, seqlen = mixed_qkv.shape
    assert dim == 2 * key_dim + value_dim, f"dim {dim} != 2*{key_dim} + {value_dim}"
    assert key_dim == num_heads_qk * head_dim, f"key_dim {key_dim} != {num_heads_qk} * {head_dim}"
    assert value_dim == num_heads_v * head_dim, f"value_dim {value_dim} != {num_heads_v} * {head_dim}"
    
    _, conv_width = conv_weight.shape
    num_cache_lines, _, conv_state_len = conv_state.size()
    
    # Head dimensions
    # HV is the number of value heads (not total value dimension)
    HV = num_heads_v
    K = head_dim
    V = head_dim
    H = num_heads_qk
    
    if scale is None:
        scale = K ** -0.5
    
    T = seqlen
    N = batch if cu_seqlens is None else len(cu_seqlens) - 1
    B = batch
    
    BK = triton.next_power_of_2(K)
    NK = triton.cdiv(K, BK)
    assert NK == 1, "NK > 1 is not supported yet"
    
    # Create output tensor
    # Shape should match the gating delta rule output: (NK, B, T, HV, V)
    o = mixed_qkv.new_empty(NK, B, T, HV, V)
    
    # Grid configuration: launch N * H blocks (one per Q/K head, not per V head)
    # Each block processes GROUP_SIZE (HV//H) V heads
    grid = lambda META: (NK, triton.cdiv(V, META['BV']), N * H)
    
    # Launch kernel
    stride_state_indices = (
        conv_state_indices.stride(0) if conv_state_indices is not None else 0
    )
    np2_statelen = triton.next_power_of_2(conv_state_len)
    
    fused_gdn_fwd_decode_kernel[grid](
        x_ptr=mixed_qkv,
        conv_w_ptr=conv_weight,
        conv_bias_ptr=conv_bias,
        conv_state_ptr=conv_state,
        conv_state_indices_ptr=conv_state_indices,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        h0_source=ssm_state,
        h0_indices=ssm_state_indices,
        cu_seqlens=cu_seqlens,
        o=o,
        key_dim=key_dim,
        value_dim=value_dim,
        batch=batch,
        dim=dim,
        seqlen=seqlen,
        conv_state_len=conv_state_len,
        num_cache_lines=num_cache_lines,
        T=T,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        scale=scale,
        stride_x_seq=mixed_qkv.stride(0),
        stride_x_dim=mixed_qkv.stride(1),
        stride_x_token=mixed_qkv.stride(2),
        stride_conv_w_dim=conv_weight.stride(0),
        stride_conv_w_width=conv_weight.stride(1),
        stride_conv_state_seq=conv_state.stride(0),
        stride_conv_state_dim=conv_state.stride(1),
        stride_conv_state_tok=conv_state.stride(2),
        stride_state_indices=stride_state_indices,
        pad_slot_id=pad_slot_id,
        B=N,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        HAS_CONV_BIAS=conv_bias is not None,
        CONV_WIDTH=conv_width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        IS_CONTINUOUS_BATCHING=conv_state_indices is not None,
        NP2_STATELEN=np2_statelen,
        USE_PAD_SLOT=pad_slot_id is not None,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
    )
    
    # Squeeze first dimension (NK) to get (B, T, HV, V)
    o = o.squeeze(0)
    return o

def fused_gdn_fwd_decode_v3(
    mixed_qkv: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    ssm_state: torch.Tensor,
    key_dim: int,
    value_dim: int,
    num_heads_qk: int,
    num_heads_v: int,
    head_dim: int,
    conv_bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
    conv_state_indices: Optional[torch.Tensor] = None,
    ssm_state_indices: Optional[torch.Tensor] = None,
    pad_slot_id: int = PAD_SLOT_ID,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = True,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Fused Gating Delta Network forward decode operation (V1 - indexed by Q/K heads).
    
    This version uses Q/K head indexing with GROUP_SIZE loop to process multiple V heads
    per block, reducing redundant Q/K computation but with more complex logic.
    
    Args:
        mixed_qkv: Input tensor (batch, dim, seqlen) where dim = 2*key_dim + value_dim
        conv_state: Convolution state (num_cache_lines, dim, state_len)
        conv_weight: Convolution weights (dim, width)
        A_log: Gating parameter A (num_heads_v * head_dim,)
        a: Gating parameter a (batch, num_heads_v * head_dim)
        dt_bias: Gating parameter dt_bias (num_heads_v * head_dim,)
        b: Gating parameter b (batch, num_heads_v * head_dim)
        ssm_state: SSM state pool (num_cache_lines, num_heads_v * head_dim, head_dim, head_dim)
        key_dim: Dimension of query and key (= num_heads_qk * head_dim)
        value_dim: Dimension of value (= num_heads_v * head_dim)
        num_heads_qk: Number of query/key heads
        num_heads_v: Number of value heads
        head_dim: Dimension per head
        conv_bias: Optional convolution bias (dim,)
        activation: Activation function ("silu" or None)
        conv_state_indices: Optional batch indices for continuous batching
        ssm_state_indices: Optional batch indices for SSM state
        pad_slot_id: ID for padded slots
        scale: Query scaling factor
        use_qk_l2norm_in_kernel: Whether to apply L2 normalization to Q/K
        softplus_beta: Beta parameter for softplus
        softplus_threshold: Threshold for softplus
        cu_seqlens: Cumulative sequence lengths for variable length sequences
    
    Returns:
        Output tensor
    """
    batch, dim, seqlen = mixed_qkv.shape
    assert dim == 2 * key_dim + value_dim, f"dim {dim} != 2*{key_dim} + {value_dim}"
    assert key_dim == num_heads_qk * head_dim, f"key_dim {key_dim} != {num_heads_qk} * {head_dim}"
    assert value_dim == num_heads_v * head_dim, f"value_dim {value_dim} != {num_heads_v} * {head_dim}"
    
    _, conv_width = conv_weight.shape
    num_cache_lines, _, conv_state_len = conv_state.size()
    
    # Head dimensions
    # HV is the number of value heads (not total value dimension)
    HV = num_heads_v
    K = head_dim
    V = head_dim
    H = num_heads_qk
    
    if scale is None:
        scale = K ** -0.5
    
    T = seqlen
    N = batch if cu_seqlens is None else len(cu_seqlens) - 1
    B = batch
    
    BK = triton.next_power_of_2(K)
    NK = triton.cdiv(K, BK)
    assert NK == 1, "NK > 1 is not supported yet"
    
    # Create output tensor
    # Shape should match the gating delta rule output: (NK, B, T, HV, V)
    o = mixed_qkv.new_empty(NK, B, T, HV, V)
    
    # Grid configuration: launch N * H blocks (one per Q/K head, not per V head)
    # Each block processes GROUP_SIZE (HV//H) V heads
    grid = lambda META: (NK, H, N)
    
    # Launch kernel
    stride_state_indices = (
        conv_state_indices.stride(0) if conv_state_indices is not None else 0
    )
    np2_statelen = triton.next_power_of_2(conv_state_len)
    
    fused_gdn_fwd_decode_kernel_v3[grid](
        x_ptr=mixed_qkv,
        conv_w_ptr=conv_weight,
        conv_bias_ptr=conv_bias,
        conv_state_ptr=conv_state,
        conv_state_indices_ptr=conv_state_indices,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        h0_source=ssm_state,
        h0_indices=ssm_state_indices,
        cu_seqlens=cu_seqlens,
        o=o,
        key_dim=key_dim,
        value_dim=value_dim,
        batch=batch,
        dim=dim,
        seqlen=seqlen,
        conv_state_len=conv_state_len,
        num_cache_lines=num_cache_lines,
        T=T,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        scale=scale,
        stride_x_seq=mixed_qkv.stride(0),
        stride_x_dim=mixed_qkv.stride(1),
        stride_x_token=mixed_qkv.stride(2),
        stride_conv_w_dim=conv_weight.stride(0),
        stride_conv_w_width=conv_weight.stride(1),
        stride_conv_state_seq=conv_state.stride(0),
        stride_conv_state_dim=conv_state.stride(1),
        stride_conv_state_tok=conv_state.stride(2),
        stride_state_indices=stride_state_indices,
        pad_slot_id=pad_slot_id,
        B=N,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        HAS_CONV_BIAS=conv_bias is not None,
        CONV_WIDTH=conv_width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        IS_CONTINUOUS_BATCHING=conv_state_indices is not None,
        NP2_STATELEN=np2_statelen,
        USE_PAD_SLOT=pad_slot_id is not None,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
    )
    
    # Squeeze first dimension (NK) to get (B, T, HV, V)
    o = o.squeeze(0)
    return o
