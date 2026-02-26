"""
Unit tests for Fused Sigmoid Gating Delta Rule Update Kernel.

Tests correctness and performance of fused_sigmoid_gating_delta_rule_update kernel
using parameters based on Qwen3Next configuration.

Qwen3Next Linear Attention Config:
    - linear_key_head_dim: 128
    - linear_value_head_dim: 128
    - linear_num_key_heads: 16 (4 per TP4 shard)
    - linear_num_value_heads: 32 (8 per TP4 shard)
"""

import pytest
import torch
import math

from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
    fused_split_gdr_update,
    fused_split_gdr_update_v2,
    fused_split_gdr_update_v3,
    fused_split_gdr_update_v4,
    fused_split_gdr_update_v3_seqlen1,
)


def sigmoid_gating_delta_rule_ref(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float,
    softplus_threshold: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    initial_state: torch.Tensor,
    scale: float = None,
    use_qk_l2norm_in_kernel: bool = False,
):
    """
    Pure PyTorch reference implementation for sigmoid gating delta rule update.
    
    This serves as ground truth for correctness verification.
    
    Args:
        A_log: Log of A parameter, shape (num_heads_v,)
        a: Time-variant gating parameter, shape (batch, seqlen, num_heads_v)
        dt_bias: Bias for dt, shape (num_heads_v,)
        softplus_beta: Beta parameter for softplus
        softplus_threshold: Threshold for softplus
        q: Query tensor, shape (batch, seqlen, num_heads_qk, head_dim)
        k: Key tensor, shape (batch, seqlen, num_heads_qk, head_dim)
        v: Value tensor, shape (batch, seqlen, num_heads_v, head_dim)
        b: Beta gating parameter, shape (batch, seqlen, num_heads_v)
        initial_state: Initial hidden state, shape (batch, num_heads_v, head_dim, head_dim)
        scale: Scaling factor for attention
        use_qk_l2norm_in_kernel: Whether to use L2 normalization for Q and K
    
    Returns:
        output: Output tensor, shape (batch, seqlen, num_heads_v, head_dim)
    """
    batch, seqlen, num_heads_qk, head_dim_k = q.shape
    _, _, num_heads_v, head_dim_v = v.shape
    
    if scale is None:
        scale = head_dim_k ** -0.5
    
    # Group size for QK heads to V heads mapping
    group_size = num_heads_v // num_heads_qk
    
    # Clone initial state to avoid modifying input
    h = initial_state.clone().float()  # (batch, num_heads_v, K, V)
    
    outputs = []
    
    for t in range(seqlen):
        # Get current timestep inputs
        q_t = q[:, t, :, :].float()  # (batch, num_heads_qk, K)
        k_t = k[:, t, :, :].float()  # (batch, num_heads_qk, K)
        v_t = v[:, t, :, :].float()  # (batch, num_heads_v, V)
        a_t = a[:, t, :].float()     # (batch, num_heads_v)
        b_t = b[:, t, :].float()     # (batch, num_heads_v)
        
        # Compute gating: g = -exp(A_log) * softplus(a + dt_bias)
        x = a_t + dt_bias.unsqueeze(0)  # (batch, num_heads_v)
        beta_x = softplus_beta * x
        softplus_x = torch.where(
            beta_x <= softplus_threshold,
            (1.0 / softplus_beta) * torch.log(1.0 + torch.exp(beta_x)),
            x
        )
        g = -torch.exp(A_log.unsqueeze(0)) * softplus_x  # (batch, num_heads_v)
        
        # Compute beta from b using sigmoid
        beta = torch.sigmoid(b_t)  # (batch, num_heads_v)
        
        output_t = []
        
        for hv in range(num_heads_v):
            h_idx = hv // group_size  # Corresponding QK head index
            
            q_h = q_t[:, h_idx, :]  # (batch, K)
            k_h = k_t[:, h_idx, :]  # (batch, K)
            v_h = v_t[:, hv, :]     # (batch, V)
            g_h = g[:, hv]          # (batch,)
            beta_h = beta[:, hv]    # (batch,)
            
            # Apply L2 normalization if enabled
            if use_qk_l2norm_in_kernel:
                q_h = q_h / (torch.norm(q_h, dim=-1, keepdim=True) + 1e-6)
                k_h = k_h / (torch.norm(k_h, dim=-1, keepdim=True) + 1e-6)
            
            # Scale query
            q_h = q_h * scale
            
            # Get hidden state for this head: h[batch, hv, K, V]
            h_hv = h[:, hv, :, :]  # (batch, K, V)
            
            # Delta rule update
            # h = h * exp(g) + k[:, None] * (beta * (v - sum(h * k[:, None], dim=0)))
            h_hv = h_hv * torch.exp(g_h).unsqueeze(-1).unsqueeze(-1)  # (batch, K, V)
            
            # v_update = v - sum(h * k, dim=K)
            v_update = v_h - torch.einsum('bkv,bk->bv', h_hv, k_h)  # (batch, V)
            v_update = v_update * beta_h.unsqueeze(-1)  # (batch, V)
            
            # h += k[:, None] * v_update[None, :]
            h_hv = h_hv + torch.einsum('bk,bv->bkv', k_h, v_update)  # (batch, K, V)
            
            # Store updated hidden state
            h[:, hv, :, :] = h_hv
            
            # Compute output: o = sum(h * q, dim=K)
            o_h = torch.einsum('bkv,bk->bv', h_hv, q_h)  # (batch, V)
            output_t.append(o_h)
        
        # Stack outputs for all V heads
        output_t = torch.stack(output_t, dim=1)  # (batch, num_heads_v, V)
        outputs.append(output_t)
    
    # Stack outputs for all timesteps
    output = torch.stack(outputs, dim=1)  # (batch, seqlen, num_heads_v, V)
    
    return output


class TestFusedSigmoidGatingDeltaRuleUpdate:
    """Test class for fused_sigmoid_gating_delta_rule_update kernel."""
    
    @pytest.fixture
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    @pytest.fixture
    def dtype(self):
        return torch.bfloat16
    
    def create_inputs(
        self,
        batch_size: int,
        seqlen: int,
        num_heads_qk: int,
        num_heads_v: int,
        head_dim: int,
        device: str,
        dtype: torch.dtype,
        use_varlen: bool = False,
    ):
        """
        Create test inputs based on Qwen3Next configuration.
        
        Args:
            use_varlen: If True, use varlen mode matching the actual inference pattern.
                        In varlen mode: batch=1, T=batch_size*seqlen, with cu_seqlens.
                        This matches how hybrid_linear_attn_backend.py calls the kernel.
        """
        if use_varlen:
            # Varlen mode: matches actual inference in hybrid_linear_attn_backend.py
            # All batch items packed into a single "batch" with cu_seqlens marking boundaries
            total_tokens = batch_size * seqlen
            
            # Q, K: (1, total_tokens, num_heads_qk, head_dim)
            # Use .contiguous() to ensure memory layout matches inference
            q = torch.randn(1, total_tokens, num_heads_qk, head_dim, device=device, dtype=dtype).contiguous()
            k = torch.randn(1, total_tokens, num_heads_qk, head_dim, device=device, dtype=dtype).contiguous()
            
            # V: (1, total_tokens, num_heads_v, head_dim)
            v = torch.randn(1, total_tokens, num_heads_v, head_dim, device=device, dtype=dtype).contiguous()
            
            # Time-variant gating: (total_tokens, num_heads_v) - 2D tensor!
            # This matches the shape in qwen3_next.py fix_query_key_value_ordering():
            #   a = a.reshape(a.size(0), self.num_v_heads // self.attn_tp_size)
            # The kernel accesses: p_a = a + bos * HV + i_hv (assumes 2D layout)
            a = torch.randn(total_tokens, num_heads_v, device=device, dtype=dtype).contiguous()
            b = torch.randn(total_tokens, num_heads_v, device=device, dtype=dtype).contiguous()
            
            # cu_seqlens: [0, seqlen, 2*seqlen, ..., batch_size*seqlen]
            cu_seqlens = torch.arange(
                0, total_tokens + 1, seqlen, 
                device=device, dtype=torch.int32
            )
            
            # SSM state indices: one per sequence in cu_seqlens
            ssm_state_indices = torch.arange(batch_size, device=device, dtype=torch.int32)
            
            # SSM state: (batch_size + padding, num_heads_v, head_dim, head_dim)
            # Ensure contiguous memory for state
            ssm_state = torch.randn(
                batch_size + 10, num_heads_v, head_dim, head_dim, 
                device=device, dtype=torch.float32
            ).contiguous()
        else:
            # Standard batch mode
            # Q, K: (batch, seqlen, num_heads_qk, head_dim)
            q = torch.randn(batch_size, seqlen, num_heads_qk, head_dim, device=device, dtype=dtype)
            k = torch.randn(batch_size, seqlen, num_heads_qk, head_dim, device=device, dtype=dtype)
            
            # V: (batch, seqlen, num_heads_v, head_dim)
            v = torch.randn(batch_size, seqlen, num_heads_v, head_dim, device=device, dtype=dtype)
            
            # Time-variant gating: (batch, seqlen, num_heads_v)
            a = torch.randn(batch_size, seqlen, num_heads_v, device=device, dtype=dtype)
            b = torch.randn(batch_size, seqlen, num_heads_v, device=device, dtype=dtype)
            
            cu_seqlens = None
            
            # SSM state indices: simple 1:1 mapping
            ssm_state_indices = torch.arange(batch_size, device=device, dtype=torch.int32)
            
            # SSM state: (batch + padding, num_heads_v, head_dim, head_dim)
            ssm_state = torch.randn(
                batch_size + 10, num_heads_v, head_dim, head_dim, 
                device=device, dtype=torch.float32
            )
        
        # Gating parameters (same for both modes)
        A_log = torch.randn(num_heads_v, device=device, dtype=dtype)
        dt_bias = torch.randn(num_heads_v, device=device, dtype=dtype)
        
        return {
            "q": q,
            "k": k,
            "v": v,
            "A_log": A_log,
            "dt_bias": dt_bias,
            "a": a,
            "b": b,
            "ssm_state": ssm_state,
            "ssm_state_indices": ssm_state_indices,
            "cu_seqlens": cu_seqlens,
            "batch_size": batch_size,  # Store for reference
        }
    
    @pytest.mark.parametrize("batch_size", [64])
    @pytest.mark.parametrize("seqlen", [1])
    @pytest.mark.parametrize("num_heads_qk", [4])
    @pytest.mark.parametrize("num_heads_v", [8])
    @pytest.mark.parametrize("head_dim", [128])
    def test_correctness(
        self, 
        batch_size, 
        seqlen, 
        num_heads_qk, 
        num_heads_v, 
        head_dim,
        device, 
        dtype
    ):
        """Test correctness of fused kernel against reference implementation."""
        torch.manual_seed(42)
        
        inputs = self.create_inputs(
            batch_size, seqlen, num_heads_qk, num_heads_v, head_dim, device, dtype
        )
        
        softplus_beta = 1.0
        softplus_threshold = 20.0
        scale = head_dim ** -0.5
        use_qk_l2norm_in_kernel = True
        
        # Get initial state for reference
        initial_state = inputs["ssm_state"][inputs["ssm_state_indices"]].clone()
        
        # Reference implementation
        ref_output = sigmoid_gating_delta_rule_ref(
            A_log=inputs["A_log"],
            a=inputs["a"],
            dt_bias=inputs["dt_bias"],
            softplus_beta=softplus_beta,
            softplus_threshold=softplus_threshold,
            q=inputs["q"],
            k=inputs["k"],
            v=inputs["v"],
            b=inputs["b"],
            initial_state=initial_state,
            scale=scale,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
        
        # Fused kernel
        fused_output = fused_sigmoid_gating_delta_rule_update(
            A_log=inputs["A_log"],
            a=inputs["a"],
            dt_bias=inputs["dt_bias"],
            softplus_beta=softplus_beta,
            softplus_threshold=softplus_threshold,
            q=inputs["q"],
            k=inputs["k"],
            v=inputs["v"],
            b=inputs["b"],
            initial_state_source=inputs["ssm_state"],
            initial_state_indices=inputs["ssm_state_indices"],
            scale=scale,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
        
        # Compare outputs
        ref_output = ref_output.to(dtype)
        
        # Check shapes match
        assert fused_output.shape == ref_output.shape, (
            f"Shape mismatch: fused={fused_output.shape}, ref={ref_output.shape}"
        )
        
        # Check values are close
        rtol = 1e-2
        atol = 1e-2
        
        max_diff = (fused_output - ref_output).abs().max().item()
        mean_diff = (fused_output - ref_output).abs().mean().item()
        
        print(f"\n{'='*70}")
        print(f"Correctness Test: batch={batch_size}, seqlen={seqlen}")
        print(f"  num_heads_qk={num_heads_qk}, num_heads_v={num_heads_v}, head_dim={head_dim}")
        print(f"  Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")
        print(f"{'='*70}")
        
        assert torch.allclose(fused_output, ref_output, rtol=rtol, atol=atol), (
            f"Output mismatch! Max diff: {max_diff}, Mean diff: {mean_diff}"
        )
    
    @pytest.mark.parametrize("batch_size", [64])
    @pytest.mark.parametrize("seqlen", [1])
    @pytest.mark.parametrize("num_heads_qk", [4])
    @pytest.mark.parametrize("num_heads_v", [8])
    @pytest.mark.parametrize("head_dim", [128])
    def test_performance_varlen(
        self, 
        batch_size, 
        seqlen, 
        num_heads_qk, 
        num_heads_v, 
        head_dim,
        device, 
        dtype
    ):
        """
        Benchmark performance using varlen mode (matching actual inference pattern).
        
        This test uses the same calling convention as hybrid_linear_attn_backend.py:
        - batch=1, T=batch_size (all tokens packed into one sequence)
        - cu_seqlens to mark sequence boundaries
        """
        torch.manual_seed(42)
        
        inputs = self.create_inputs(
            batch_size, seqlen, num_heads_qk, num_heads_v, head_dim, device, dtype,
            use_varlen=True  # Use varlen mode!
        )
        
        softplus_beta = 1.0
        softplus_threshold = 20.0
        scale = head_dim ** -0.5
        use_qk_l2norm_in_kernel = True
        
        # Warmup - use more iterations to ensure kernel is fully compiled
        for _ in range(20):
            _ = fused_sigmoid_gating_delta_rule_update(
                A_log=inputs["A_log"],
                a=inputs["a"],
                dt_bias=inputs["dt_bias"],
                softplus_beta=softplus_beta,
                softplus_threshold=softplus_threshold,
                q=inputs["q"],
                k=inputs["k"],
                v=inputs["v"],
                b=inputs["b"],
                initial_state_source=inputs["ssm_state"],
                initial_state_indices=inputs["ssm_state_indices"],
                scale=scale,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                cu_seqlens=inputs["cu_seqlens"],  # Use varlen mode!
            )
        torch.cuda.synchronize()
        
        # Benchmark
        num_iters = 100
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        torch.cuda.synchronize()
        start_event.record()
        
        for _ in range(num_iters):
            _ = fused_sigmoid_gating_delta_rule_update(
                A_log=inputs["A_log"],
                a=inputs["a"],
                dt_bias=inputs["dt_bias"],
                softplus_beta=softplus_beta,
                softplus_threshold=softplus_threshold,
                q=inputs["q"],
                k=inputs["k"],
                v=inputs["v"],
                b=inputs["b"],
                initial_state_source=inputs["ssm_state"],
                initial_state_indices=inputs["ssm_state_indices"],
                scale=scale,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                cu_seqlens=inputs["cu_seqlens"],
            )
        
        end_event.record()
        torch.cuda.synchronize()
        
        elapsed_time = start_event.elapsed_time(end_event) / num_iters  # ms
        elapsed_us = elapsed_time * 1000  # Convert to microseconds
        
        # Calculate memory bandwidth
        # Hidden state: batch_size * num_heads_v * head_dim * head_dim * 4 bytes (fp32)
        # Read + Write = 2x
        hidden_state_bytes = batch_size * num_heads_v * head_dim * head_dim * 4 * 2
        
        # Q, K, V reads
        total_tokens = batch_size * seqlen
        qkv_bytes = total_tokens * (
            2 * num_heads_qk * head_dim +  # Q, K
            num_heads_v * head_dim          # V
        ) * 2  # bf16 = 2 bytes
        
        # Output write
        output_bytes = total_tokens * num_heads_v * head_dim * 2  # bf16
        
        # Gating parameters
        gating_bytes = total_tokens * num_heads_v * 2 * 2  # a, b in bf16
        
        total_bytes = hidden_state_bytes + qkv_bytes + output_bytes + gating_bytes
        bandwidth_gb_s = (total_bytes / 1e9) / (elapsed_time / 1e3)
        
        print(f"\n{'='*70}")
        print(f"Performance Benchmark (VARLEN mode - matches inference)")
        print(f"  batch={batch_size}, seqlen={seqlen}")
        print(f"  num_heads_qk={num_heads_qk}, num_heads_v={num_heads_v}, head_dim={head_dim}")
        print(f"{'='*70}")
        print(f"  Kernel time: {elapsed_time:.4f} ms ({elapsed_us:.2f} us)")
        print(f"  Throughput: {1000/elapsed_time:.2f} iterations/sec")
        print(f"  Memory accessed: {total_bytes/1e6:.2f} MB")
        print(f"  Effective bandwidth: {bandwidth_gb_s:.2f} GB/s")
        print(f"{'='*70}")
    
    @pytest.mark.parametrize("batch_size", [64])
    @pytest.mark.parametrize("seqlen", [1])
    @pytest.mark.parametrize("num_heads_qk", [4])
    @pytest.mark.parametrize("num_heads_v", [8])
    @pytest.mark.parametrize("head_dim", [128])
    def test_performance(
        self, 
        batch_size, 
        seqlen, 
        num_heads_qk, 
        num_heads_v, 
        head_dim,
        device, 
        dtype
    ):
        """Benchmark performance of fused kernel (standard batch mode)."""
        torch.manual_seed(42)
        
        inputs = self.create_inputs(
            batch_size, seqlen, num_heads_qk, num_heads_v, head_dim, device, dtype,
            use_varlen=False
        )
        
        softplus_beta = 1.0
        softplus_threshold = 20.0
        scale = head_dim ** -0.5
        use_qk_l2norm_in_kernel = True
        
        # Warmup
        for _ in range(5):
            _ = fused_sigmoid_gating_delta_rule_update(
                A_log=inputs["A_log"],
                a=inputs["a"],
                dt_bias=inputs["dt_bias"],
                softplus_beta=softplus_beta,
                softplus_threshold=softplus_threshold,
                q=inputs["q"],
                k=inputs["k"],
                v=inputs["v"],
                b=inputs["b"],
                initial_state_source=inputs["ssm_state"],
                initial_state_indices=inputs["ssm_state_indices"],
                scale=scale,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )
        torch.cuda.synchronize()
        
        # Benchmark
        num_iters = 100
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        torch.cuda.synchronize()
        start_event.record()
        
        for _ in range(num_iters):
            _ = fused_sigmoid_gating_delta_rule_update(
                A_log=inputs["A_log"],
                a=inputs["a"],
                dt_bias=inputs["dt_bias"],
                softplus_beta=softplus_beta,
                softplus_threshold=softplus_threshold,
                q=inputs["q"],
                k=inputs["k"],
                v=inputs["v"],
                b=inputs["b"],
                initial_state_source=inputs["ssm_state"],
                initial_state_indices=inputs["ssm_state_indices"],
                scale=scale,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )
        
        end_event.record()
        torch.cuda.synchronize()
        
        elapsed_time = start_event.elapsed_time(end_event) / num_iters  # ms
        
        # Calculate memory bandwidth
        # Hidden state: batch * num_heads_v * head_dim * head_dim * 4 bytes (fp32)
        # Read + Write = 2x
        hidden_state_bytes = batch_size * num_heads_v * head_dim * head_dim * 4 * 2
        
        # Q, K, V reads
        qkv_bytes = batch_size * seqlen * (
            2 * num_heads_qk * head_dim +  # Q, K
            num_heads_v * head_dim          # V
        ) * 2  # bf16 = 2 bytes
        
        # Output write
        output_bytes = batch_size * seqlen * num_heads_v * head_dim * 2  # bf16
        
        # Gating parameters
        gating_bytes = batch_size * seqlen * num_heads_v * 2 * 2  # a, b in bf16
        
        total_bytes = hidden_state_bytes + qkv_bytes + output_bytes + gating_bytes
        bandwidth_gb_s = (total_bytes / 1e9) / (elapsed_time / 1e3)
        
        print(f"\n{'='*70}")
        print(f"Performance Benchmark: batch={batch_size}, seqlen={seqlen}")
        print(f"  num_heads_qk={num_heads_qk}, num_heads_v={num_heads_v}, head_dim={head_dim}")
        print(f"{'='*70}")
        print(f"  Kernel time: {elapsed_time:.4f} ms")
        print(f"  Throughput: {1000/elapsed_time:.2f} iterations/sec")
        print(f"  Memory accessed: {total_bytes/1e6:.2f} MB")
        print(f"  Effective bandwidth: {bandwidth_gb_s:.2f} GB/s")
        print(f"{'='*70}")
    
    @pytest.mark.parametrize("batch_size", [64, 128])
    @pytest.mark.parametrize("seqlen", [1])
    def test_state_update(self, batch_size, seqlen, device, dtype):
        """Test that hidden state is correctly updated after kernel execution."""
        torch.manual_seed(42)
        
        num_heads_qk = 4
        num_heads_v = 8
        head_dim = 128
        
        inputs = self.create_inputs(
            batch_size, seqlen, num_heads_qk, num_heads_v, head_dim, device, dtype
        )
        
        # Clone initial state to check for updates
        initial_state_copy = inputs["ssm_state"].clone()
        
        softplus_beta = 1.0
        softplus_threshold = 20.0
        
        # Run kernel
        _ = fused_sigmoid_gating_delta_rule_update(
            A_log=inputs["A_log"],
            a=inputs["a"],
            dt_bias=inputs["dt_bias"],
            softplus_beta=softplus_beta,
            softplus_threshold=softplus_threshold,
            q=inputs["q"],
            k=inputs["k"],
            v=inputs["v"],
            b=inputs["b"],
            initial_state_source=inputs["ssm_state"],
            initial_state_indices=inputs["ssm_state_indices"],
            use_qk_l2norm_in_kernel=True,
        )
        
        # Check that state was updated
        state_diff = (inputs["ssm_state"] - initial_state_copy).abs().max().item()
        
        print(f"\n{'='*70}")
        print(f"State Update Test: batch={batch_size}, seqlen={seqlen}")
        print(f"  Max state change: {state_diff:.6f}")
        print(f"{'='*70}")
        
        # State should have been updated (non-zero difference for valid indices)
        assert state_diff > 0, "State was not updated!"


class TestFusedSplitGDRUpdate:
    """Test class for fused_split_gdr_update kernel (combined split + GDR)."""

    @pytest.fixture
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    @pytest.fixture
    def dtype(self):
        return torch.bfloat16

    def create_inputs(
        self,
        batch_size: int,
        seqlen: int,
        num_heads_qk: int,
        num_heads_v: int,
        head_dim: int,
        device: str,
        dtype: torch.dtype,
    ):
        """
        Create test inputs for fused_split_gdr_update.
        
        mixed_qkv: (batch, dim, seqlen) where dim = 2*key_dim + value_dim
        """
        key_dim = num_heads_qk * head_dim
        value_dim = num_heads_v * head_dim
        dim = 2 * key_dim + value_dim
        
        # mixed_qkv: (batch, dim, seqlen)
        mixed_qkv = torch.randn(batch_size, dim, seqlen, device=device, dtype=dtype)
        
        # Gating parameters
        A_log = torch.randn(num_heads_v, device=device, dtype=torch.float32)
        dt_bias = torch.randn(num_heads_v, device=device, dtype=dtype)
        
        # Time-variant gating: (batch * seqlen, num_heads_v)
        a = torch.randn(batch_size * seqlen, num_heads_v, device=device, dtype=dtype)
        b = torch.randn(batch_size * seqlen, num_heads_v, device=device, dtype=dtype)
        
        # SSM state: (batch + padding, num_heads_v, head_dim, head_dim)
        ssm_state = torch.randn(
            batch_size + 10, num_heads_v, head_dim, head_dim,
            device=device, dtype=torch.float32
        )
        
        ssm_state_indices = torch.arange(batch_size, device=device, dtype=torch.int32)
        
        return {
            "mixed_qkv": mixed_qkv,
            "A_log": A_log,
            "a": a,
            "dt_bias": dt_bias,
            "b": b,
            "ssm_state": ssm_state,
            "ssm_state_indices": ssm_state_indices,
            "key_dim": key_dim,
            "value_dim": value_dim,
            "num_heads_qk": num_heads_qk,
            "num_heads_v": num_heads_v,
            "head_dim": head_dim,
        }

    @pytest.mark.parametrize("batch_size", [64])
    @pytest.mark.parametrize("seqlen", [1])
    @pytest.mark.parametrize("num_heads_qk", [4])
    @pytest.mark.parametrize("num_heads_v", [8])
    @pytest.mark.parametrize("head_dim", [128])
    def test_split_gdr_correctness(
        self,
        batch_size,
        seqlen,
        num_heads_qk,
        num_heads_v,
        head_dim,
        device,
        dtype,
    ):
        """
        Test correctness of fused_split_gdr_update against reference.
        
        Reference: split mixed_qkv manually + fused_sigmoid_gating_delta_rule_update
        """
        torch.manual_seed(42)
        
        inputs = self.create_inputs(
            batch_size, seqlen, num_heads_qk, num_heads_v, head_dim, device, dtype
        )
        
        key_dim = inputs["key_dim"]
        value_dim = inputs["value_dim"]
        
        softplus_beta = 1.0
        softplus_threshold = 20.0
        scale = head_dim ** -0.5
        use_qk_l2norm_in_kernel = True
        
        # Clone ssm_state for separate runs
        ssm_state_ref = inputs["ssm_state"].clone()
        ssm_state_fused = inputs["ssm_state"].clone()
        
        # ============================================================
        # Reference: manual split + fused_sigmoid_gating_delta_rule_update
        # ============================================================
        mixed_qkv = inputs["mixed_qkv"]
        batch, dim, T = mixed_qkv.shape
        
        # Apply silu activation to mixed_qkv first (simulating conv1d output)
        mixed_qkv_activated = mixed_qkv * torch.sigmoid(mixed_qkv)
        
        # Split activated mixed_qkv into Q, K, V for reference
        q = mixed_qkv_activated[:, :key_dim, :]
        k = mixed_qkv_activated[:, key_dim:2*key_dim, :]
        v = mixed_qkv_activated[:, 2*key_dim:, :]
        
        # Reshape for fused_sigmoid_gating_delta_rule_update
        # (batch, dim, seqlen) -> (batch, seqlen, heads, head_dim)
        q = q.view(batch, num_heads_qk, head_dim, T).permute(0, 3, 1, 2).contiguous()
        k = k.view(batch, num_heads_qk, head_dim, T).permute(0, 3, 1, 2).contiguous()
        v = v.view(batch, num_heads_v, head_dim, T).permute(0, 3, 1, 2).contiguous()
        
        # Reference: fused_sigmoid_gating_delta_rule_update
        output_ref = fused_sigmoid_gating_delta_rule_update(
            A_log=inputs["A_log"],
            a=inputs["a"],
            dt_bias=inputs["dt_bias"],
            softplus_beta=softplus_beta,
            softplus_threshold=softplus_threshold,
            q=q,
            k=k,
            v=v,
            b=inputs["b"],
            initial_state_source=ssm_state_ref,
            initial_state_indices=inputs["ssm_state_indices"],
            scale=scale,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
        
        # ============================================================
        # Fused: fused_split_gdr_update (expects pre-activated input)
        # ============================================================
        output_fused = fused_split_gdr_update(
            mixed_qkv=mixed_qkv_activated,  # Use activated version
            A_log=inputs["A_log"],
            a=inputs["a"],
            dt_bias=inputs["dt_bias"],
            b=inputs["b"],
            initial_state_source=ssm_state_fused,
            initial_state_indices=inputs["ssm_state_indices"],
            key_dim=key_dim,
            value_dim=value_dim,
            num_heads_qk=num_heads_qk,
            num_heads_v=num_heads_v,
            head_dim=head_dim,
            softplus_beta=softplus_beta,
            softplus_threshold=softplus_threshold,
            scale=scale,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
        
        # Compare outputs
        rtol, atol = 1e-2, 5e-2
        max_diff = (output_fused - output_ref).abs().max().item()
        mean_diff = (output_fused - output_ref).abs().mean().item()
        
        print(f"\n{'='*70}")
        print(f"Split GDR Correctness Test: batch={batch_size}, seqlen={seqlen}")
        print(f"  num_heads_qk={num_heads_qk}, num_heads_v={num_heads_v}, head_dim={head_dim}")
        print(f"  Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")
        print(f"{'='*70}")
        
        assert torch.allclose(output_fused, output_ref, rtol=rtol, atol=atol), (
            f"Output mismatch! Max diff: {max_diff}, Mean diff: {mean_diff}"
        )
        
        # Also check SSM state was updated consistently
        state_diff = (ssm_state_fused - ssm_state_ref).abs().max().item()
        print(f"  SSM state max diff: {state_diff:.6f}")
        
        assert torch.allclose(ssm_state_fused, ssm_state_ref, rtol=rtol, atol=atol), (
            f"SSM state mismatch! Max diff: {state_diff}"
        )
        
        print(f"  ✓ Split GDR correctness test passed!")

    @pytest.mark.parametrize("batch_size", [64])
    @pytest.mark.parametrize("seqlen", [1])
    @pytest.mark.parametrize("num_heads_qk", [4])
    @pytest.mark.parametrize("num_heads_v", [8])
    @pytest.mark.parametrize("head_dim", [128])
    def test_split_gdr_performance(
        self,
        batch_size,
        seqlen,
        num_heads_qk,
        num_heads_v,
        head_dim,
        device,
        dtype,
    ):
        """Benchmark performance of fused_split_gdr_update vs fused_sigmoid_gating_delta_rule_update."""
        torch.manual_seed(42)
        
        inputs = self.create_inputs(
            batch_size, seqlen, num_heads_qk, num_heads_v, head_dim, device, dtype
        )
        
        key_dim = inputs["key_dim"]
        value_dim = inputs["value_dim"]
        
        softplus_beta = 1.0
        softplus_threshold = 20.0
        scale = head_dim ** -0.5
        
        num_iters = 1000
        
        # Prepare q, k, v for fused_sigmoid_gating_delta_rule_update (baseline)
        mixed_qkv = inputs["mixed_qkv"]
        batch, dim, T = mixed_qkv.shape
        
        # Apply silu activation to mixed_qkv (simulating conv1d output)
        mixed_qkv_activated = mixed_qkv * torch.sigmoid(mixed_qkv)
        
        # Split activated mixed_qkv into Q, K, V
        q = mixed_qkv_activated[:, :key_dim, :]
        k = mixed_qkv_activated[:, key_dim:2*key_dim, :]
        v = mixed_qkv_activated[:, 2*key_dim:, :]
        
        # Reshape for fused_sigmoid_gating_delta_rule_update
        # (batch, dim, seqlen) -> (batch, seqlen, heads, head_dim)
        q = q.view(batch, num_heads_qk, head_dim, T).permute(0, 3, 1, 2).contiguous()
        k = k.view(batch, num_heads_qk, head_dim, T).permute(0, 3, 1, 2).contiguous()
        v = v.view(batch, num_heads_v, head_dim, T).permute(0, 3, 1, 2).contiguous()
        
        # ============================================================
        # Benchmark baseline: fused_sigmoid_gating_delta_rule_update
        # ============================================================
        for _ in range(10):
            _ = fused_sigmoid_gating_delta_rule_update(
                A_log=inputs["A_log"],
                a=inputs["a"],
                dt_bias=inputs["dt_bias"],
                softplus_beta=softplus_beta,
                softplus_threshold=softplus_threshold,
                q=q,
                k=k,
                v=v,
                b=inputs["b"],
                initial_state_source=inputs["ssm_state"],
                initial_state_indices=inputs["ssm_state_indices"],
                scale=scale,
                use_qk_l2norm_in_kernel=True,
            )
        torch.cuda.synchronize()
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        torch.cuda.synchronize()
        start_event.record()
        
        for _ in range(num_iters):
            _ = fused_sigmoid_gating_delta_rule_update(
                A_log=inputs["A_log"],
                a=inputs["a"],
                dt_bias=inputs["dt_bias"],
                softplus_beta=softplus_beta,
                softplus_threshold=softplus_threshold,
                q=q,
                k=k,
                v=v,
                b=inputs["b"],
                initial_state_source=inputs["ssm_state"],
                initial_state_indices=inputs["ssm_state_indices"],
                scale=scale,
                use_qk_l2norm_in_kernel=True,
            )
        
        end_event.record()
        torch.cuda.synchronize()
        
        baseline_time_us = start_event.elapsed_time(end_event) / num_iters * 1000
        
        # ============================================================
        # Benchmark fused_split_gdr_update
        # ============================================================
        for _ in range(10):
            _ = fused_split_gdr_update(
                mixed_qkv=inputs["mixed_qkv"],
                A_log=inputs["A_log"],
                a=inputs["a"],
                dt_bias=inputs["dt_bias"],
                b=inputs["b"],
                initial_state_source=inputs["ssm_state"],
                initial_state_indices=inputs["ssm_state_indices"],
                key_dim=key_dim,
                value_dim=value_dim,
                num_heads_qk=num_heads_qk,
                num_heads_v=num_heads_v,
                head_dim=head_dim,
                softplus_beta=softplus_beta,
                softplus_threshold=softplus_threshold,
                scale=scale,
                use_qk_l2norm_in_kernel=True,
            )
        torch.cuda.synchronize()
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        torch.cuda.synchronize()
        start_event.record()
        
        for _ in range(num_iters):
            _ = fused_split_gdr_update(
                mixed_qkv=inputs["mixed_qkv"],
                A_log=inputs["A_log"],
                a=inputs["a"],
                dt_bias=inputs["dt_bias"],
                b=inputs["b"],
                initial_state_source=inputs["ssm_state"],
                initial_state_indices=inputs["ssm_state_indices"],
                key_dim=key_dim,
                value_dim=value_dim,
                num_heads_qk=num_heads_qk,
                num_heads_v=num_heads_v,
                head_dim=head_dim,
                softplus_beta=softplus_beta,
                softplus_threshold=softplus_threshold,
                scale=scale,
                use_qk_l2norm_in_kernel=True,
            )
        
        end_event.record()
        torch.cuda.synchronize()
        
        split_gdr_time_us = start_event.elapsed_time(end_event) / num_iters * 1000
        
        # Calculate speedup
        speedup = baseline_time_us / split_gdr_time_us
        
        print(f"\n{'='*70}")
        print(f"Split GDR Performance: batch={batch_size}, seqlen={seqlen}")
        print(f"  num_heads_qk={num_heads_qk}, num_heads_v={num_heads_v}, head_dim={head_dim}")
        print(f"{'='*70}")
        print(f"  fused_sigmoid_gating_delta_rule_update (baseline): {baseline_time_us:.2f} us")
        print(f"  fused_split_gdr_update:                           {split_gdr_time_us:.2f} us (speedup: {speedup:.2f}x)")
        print(f"{'='*70}")


class TestFusedSplitGDRUpdateOpt:
    """Test class for fused_split_gdr_update_v2 kernel (optimized version)."""

    @pytest.fixture
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    @pytest.fixture
    def dtype(self):
        return torch.bfloat16

    def create_inputs(
        self,
        batch_size: int,
        seqlen: int,
        num_heads_qk: int,
        num_heads_v: int,
        head_dim: int,
        device: str,
        dtype: torch.dtype,
    ):
        """Create test inputs for fused_split_gdr_update_v2.
        """
        key_dim = num_heads_qk * head_dim
        value_dim = num_heads_v * head_dim
        dim = 2 * key_dim + value_dim
        
        # mixed_qkv: (batch, dim, seqlen)
        mixed_qkv = torch.randn(batch_size, dim, seqlen, device=device, dtype=dtype)
        
        # Gating parameters - A_log must be float32 to match v1 test
        A_log = torch.randn(num_heads_v, device=device, dtype=torch.float32)
        dt_bias = torch.randn(num_heads_v, device=device, dtype=dtype)
        
        # Time-variant gating: (batch * seqlen, num_heads_v)
        a = torch.randn(batch_size * seqlen, num_heads_v, device=device, dtype=dtype)
        b = torch.randn(batch_size * seqlen, num_heads_v, device=device, dtype=dtype)
        
        # SSM state must be float32 to match v1 test
        # Shape: (batch + padding, num_heads_v, head_dim, head_dim)
        ssm_state = torch.randn(
            batch_size + 10, num_heads_v, head_dim, head_dim,
            device=device, dtype=torch.float32
        )
        ssm_state_indices = torch.arange(batch_size, device=device, dtype=torch.int32)
        
        return {
            "mixed_qkv": mixed_qkv,
            "A_log": A_log,
            "a": a,
            "dt_bias": dt_bias,
            "b": b,
            "ssm_state": ssm_state,
            "ssm_state_indices": ssm_state_indices,
            "key_dim": key_dim,
            "value_dim": value_dim,
        }

    @pytest.mark.parametrize("batch_size", [64])
    @pytest.mark.parametrize("seqlen", [1])
    @pytest.mark.parametrize("num_heads_qk", [4])
    @pytest.mark.parametrize("num_heads_v", [8])
    @pytest.mark.parametrize("head_dim", [128])
    def test_split_gdr_v2_correctness(
        self,
        batch_size,
        seqlen,
        num_heads_qk,
        num_heads_v,
        head_dim,
        device,
        dtype,
    ):
        """Test correctness of fused_split_gdr_update_v2 against v1."""
        from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent import (
            fused_split_gdr_update,
            fused_split_gdr_update_v2,
        )
        
        torch.manual_seed(42)
        
        inputs = self.create_inputs(
            batch_size, seqlen, num_heads_qk, num_heads_v, head_dim, device, dtype
        )
        
        key_dim = inputs["key_dim"]
        value_dim = inputs["value_dim"]
        
        softplus_beta = 1.0
        softplus_threshold = 20.0
        scale = head_dim ** -0.5
        
        # Clone state for v1
        ssm_state_v1 = inputs["ssm_state"].clone()
        
        # Run v1
        output_v1 = fused_split_gdr_update(
            mixed_qkv=inputs["mixed_qkv"],
            A_log=inputs["A_log"],
            a=inputs["a"],
            dt_bias=inputs["dt_bias"],
            b=inputs["b"],
            initial_state_source=ssm_state_v1,
            initial_state_indices=inputs["ssm_state_indices"],
            key_dim=key_dim,
            value_dim=value_dim,
            num_heads_qk=num_heads_qk,
            num_heads_v=num_heads_v,
            head_dim=head_dim,
            softplus_beta=softplus_beta,
            softplus_threshold=softplus_threshold,
            scale=scale,
            use_qk_l2norm_in_kernel=True,
        )
        
        # Clone state for v2
        ssm_state_v2 = inputs["ssm_state"].clone()
        
        # Run v2
        output_v2 = fused_split_gdr_update_v2(
            mixed_qkv=inputs["mixed_qkv"],
            A_log=inputs["A_log"],
            a=inputs["a"],
            dt_bias=inputs["dt_bias"],
            b=inputs["b"],
            initial_state_source=ssm_state_v2,
            initial_state_indices=inputs["ssm_state_indices"],
            key_dim=key_dim,
            value_dim=value_dim,
            num_heads_qk=num_heads_qk,
            num_heads_v=num_heads_v,
            head_dim=head_dim,
            softplus_beta=softplus_beta,
            softplus_threshold=softplus_threshold,
            scale=scale,
            use_qk_l2norm_in_kernel=True,
        )
        
        # Compare outputs
        output_diff = (output_v1 - output_v2).abs().max().item()
        state_diff = (ssm_state_v1 - ssm_state_v2).abs().max().item()
        
        print(f"\n{'='*70}")
        print(f"Split GDR v2 Correctness Test: batch={batch_size}, seqlen={seqlen}")
        print(f"{'='*70}")
        print(f"  Output max diff: {output_diff:.6f}")
        print(f"  State max diff: {state_diff:.6f}")
        print(f"{'='*70}")
        
        # Allow slightly larger tolerance due to rsqrt vs 1/sqrt numerical differences
        assert output_diff < 5e-3, f"Output diff too large: {output_diff}"
        assert state_diff < 5e-3, f"State diff too large: {state_diff}"
        
        print(f"  ✓ Split GDR v2 correctness test passed!")

    @pytest.mark.parametrize("batch_size", [64])
    @pytest.mark.parametrize("seqlen", [1])
    @pytest.mark.parametrize("num_heads_qk", [4])
    @pytest.mark.parametrize("num_heads_v", [8])
    @pytest.mark.parametrize("head_dim", [128])
    def test_split_gdr_v2_performance(
        self,
        batch_size,
        seqlen,
        num_heads_qk,
        num_heads_v,
        head_dim,
        device,
        dtype,
    ):
        """Benchmark performance comparison of v1 vs v2."""
        from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent import (
            fused_split_gdr_update,
            fused_split_gdr_update_v2,
        )
        
        torch.manual_seed(42)
        
        inputs = self.create_inputs(
            batch_size, seqlen, num_heads_qk, num_heads_v, head_dim, device, dtype
        )
        
        key_dim = inputs["key_dim"]
        value_dim = inputs["value_dim"]
        
        softplus_beta = 1.0
        softplus_threshold = 20.0
        scale = head_dim ** -0.5
        
        # Warmup v1
        for _ in range(10):
            _ = fused_split_gdr_update(
                mixed_qkv=inputs["mixed_qkv"],
                A_log=inputs["A_log"],
                a=inputs["a"],
                dt_bias=inputs["dt_bias"],
                b=inputs["b"],
                initial_state_source=inputs["ssm_state"],
                initial_state_indices=inputs["ssm_state_indices"],
                key_dim=key_dim,
                value_dim=value_dim,
                num_heads_qk=num_heads_qk,
                num_heads_v=num_heads_v,
                head_dim=head_dim,
                softplus_beta=softplus_beta,
                softplus_threshold=softplus_threshold,
                scale=scale,
                use_qk_l2norm_in_kernel=True,
            )
        torch.cuda.synchronize()
        
        # Benchmark v1
        num_iters = 1000
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        torch.cuda.synchronize()
        start_event.record()
        
        for _ in range(num_iters):
            _ = fused_split_gdr_update(
                mixed_qkv=inputs["mixed_qkv"],
                A_log=inputs["A_log"],
                a=inputs["a"],
                dt_bias=inputs["dt_bias"],
                b=inputs["b"],
                initial_state_source=inputs["ssm_state"],
                initial_state_indices=inputs["ssm_state_indices"],
                key_dim=key_dim,
                value_dim=value_dim,
                num_heads_qk=num_heads_qk,
                num_heads_v=num_heads_v,
                head_dim=head_dim,
                softplus_beta=softplus_beta,
                softplus_threshold=softplus_threshold,
                scale=scale,
                use_qk_l2norm_in_kernel=True,
            )
        
        end_event.record()
        torch.cuda.synchronize()
        
        v1_time_us = start_event.elapsed_time(end_event) / num_iters * 1000
        
        # Warmup v2
        for _ in range(10):
            _ = fused_split_gdr_update_v2(
                mixed_qkv=inputs["mixed_qkv"],
                A_log=inputs["A_log"],
                a=inputs["a"],
                dt_bias=inputs["dt_bias"],
                b=inputs["b"],
                initial_state_source=inputs["ssm_state"],
                initial_state_indices=inputs["ssm_state_indices"],
                key_dim=key_dim,
                value_dim=value_dim,
                num_heads_qk=num_heads_qk,
                num_heads_v=num_heads_v,
                head_dim=head_dim,
                softplus_beta=softplus_beta,
                softplus_threshold=softplus_threshold,
                scale=scale,
                use_qk_l2norm_in_kernel=True,
            )
        torch.cuda.synchronize()
        
        # Benchmark v2
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        torch.cuda.synchronize()
        start_event.record()
        
        for _ in range(num_iters):
            _ = fused_split_gdr_update_v2(
                mixed_qkv=inputs["mixed_qkv"],
                A_log=inputs["A_log"],
                a=inputs["a"],
                dt_bias=inputs["dt_bias"],
                b=inputs["b"],
                initial_state_source=inputs["ssm_state"],
                initial_state_indices=inputs["ssm_state_indices"],
                key_dim=key_dim,
                value_dim=value_dim,
                num_heads_qk=num_heads_qk,
                num_heads_v=num_heads_v,
                head_dim=head_dim,
                softplus_beta=softplus_beta,
                softplus_threshold=softplus_threshold,
                scale=scale,
                use_qk_l2norm_in_kernel=True,
            )
        
        end_event.record()
        torch.cuda.synchronize()
        
        v2_time_us = start_event.elapsed_time(end_event) / num_iters * 1000
        
        speedup = v1_time_us / v2_time_us
        
        print(f"\n{'='*70}")
        print(f"Split GDR v1 vs v2 Performance: batch={batch_size}, seqlen={seqlen}")
        print(f"{'='*70}")
        print(f"  v1 (baseline):  {v1_time_us:.2f} us")
        print(f"  v2 (optimized): {v2_time_us:.2f} us")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"{'='*70}")

    # ========================================================================
    # v3 Tests: Pre-allocated output buffer + tl.sigmoid optimization
    # ========================================================================

    @pytest.mark.parametrize("batch_size", [64])
    @pytest.mark.parametrize("seqlen", [1])
    @pytest.mark.parametrize("num_heads_qk", [4])
    @pytest.mark.parametrize("num_heads_v", [8])
    @pytest.mark.parametrize("head_dim", [128])
    def test_split_gdr_v3_correctness(
        self,
        batch_size,
        seqlen,
        num_heads_qk,
        num_heads_v,
        head_dim,
        device,
        dtype,
    ):
        """Test correctness of fused_split_gdr_update_v3 against v2."""
        torch.manual_seed(42)
        
        inputs = self.create_inputs(
            batch_size, seqlen, num_heads_qk, num_heads_v, head_dim, device, dtype
        )
        
        key_dim = inputs["key_dim"]
        value_dim = inputs["value_dim"]
        
        softplus_beta = 1.0
        softplus_threshold = 20.0
        scale = head_dim ** -0.5
        
        # Clone state for v2
        ssm_state_v2 = inputs["ssm_state"].clone()
        
        # Run v2
        output_v2 = fused_split_gdr_update_v2(
            mixed_qkv=inputs["mixed_qkv"],
            A_log=inputs["A_log"],
            a=inputs["a"],
            dt_bias=inputs["dt_bias"],
            b=inputs["b"],
            initial_state_source=ssm_state_v2,
            initial_state_indices=inputs["ssm_state_indices"],
            key_dim=key_dim,
            value_dim=value_dim,
            num_heads_qk=num_heads_qk,
            num_heads_v=num_heads_v,
            head_dim=head_dim,
            softplus_beta=softplus_beta,
            softplus_threshold=softplus_threshold,
            scale=scale,
            use_qk_l2norm_in_kernel=True,
        )
        
        # Clone state for v3
        ssm_state_v3 = inputs["ssm_state"].clone()
        
        # Run v3 (without pre-allocated output)
        output_v3 = fused_split_gdr_update_v3(
            mixed_qkv=inputs["mixed_qkv"],
            A_log=inputs["A_log"],
            a=inputs["a"],
            dt_bias=inputs["dt_bias"],
            b=inputs["b"],
            initial_state_source=ssm_state_v3,
            initial_state_indices=inputs["ssm_state_indices"],
            key_dim=key_dim,
            value_dim=value_dim,
            num_heads_qk=num_heads_qk,
            num_heads_v=num_heads_v,
            head_dim=head_dim,
            softplus_beta=softplus_beta,
            softplus_threshold=softplus_threshold,
            scale=scale,
            use_qk_l2norm_in_kernel=True,
        )
        
        # Compare outputs
        output_diff = (output_v2 - output_v3).abs().max().item()
        state_diff = (ssm_state_v2 - ssm_state_v3).abs().max().item()
        
        print(f"\n{'='*70}")
        print(f"Split GDR v3 Correctness Test: batch={batch_size}, seqlen={seqlen}")
        print(f"{'='*70}")
        print(f"  Output max diff: {output_diff:.6f}")
        print(f"  State max diff: {state_diff:.6f}")
        print(f"{'='*70}")
        
        # Allow slightly larger tolerance due to tl.sigmoid vs manual sigmoid
        assert output_diff < 5e-3, f"Output diff too large: {output_diff}"
        assert state_diff < 5e-3, f"State diff too large: {state_diff}"
        
        print(f"  ✓ Split GDR v3 correctness test passed!")



    @pytest.mark.parametrize("batch_size", [64])
    @pytest.mark.parametrize("seqlen", [1])
    @pytest.mark.parametrize("num_heads_qk", [4])
    @pytest.mark.parametrize("num_heads_v", [8])
    @pytest.mark.parametrize("head_dim", [128])
    def test_split_gdr_v3_performance(
        self,
        batch_size,
        seqlen,
        num_heads_qk,
        num_heads_v,
        head_dim,
        device,
        dtype,
    ):
        """Benchmark performance comparison of v2 vs v3 (with and without pre-allocated buffer)."""
        torch.manual_seed(42)
        
        inputs = self.create_inputs(
            batch_size, seqlen, num_heads_qk, num_heads_v, head_dim, device, dtype
        )
        
        key_dim = inputs["key_dim"]
        value_dim = inputs["value_dim"]
        
        softplus_beta = 1.0
        softplus_threshold = 20.0
        scale = head_dim ** -0.5
        
        # Pre-allocate output buffer for v3 prealloc test
        output_buffer = torch.empty(
            batch_size, seqlen, num_heads_v, head_dim,
            device=device, dtype=dtype
        )
        
        num_iters = 1000
        
        # Prepare q, k, v for fused_sigmoid_gating_delta_rule_update (base)
        mixed_qkv = inputs["mixed_qkv"]
        batch, dim, T = mixed_qkv.shape
        
        # Apply silu activation to mixed_qkv (simulating conv1d output)
        mixed_qkv_activated = mixed_qkv * torch.sigmoid(mixed_qkv)
        
        # Split activated mixed_qkv into Q, K, V
        q = mixed_qkv_activated[:, :key_dim, :]
        k = mixed_qkv_activated[:, key_dim:2*key_dim, :]
        v = mixed_qkv_activated[:, 2*key_dim:, :]
        
        # Reshape for fused_sigmoid_gating_delta_rule_update
        # (batch, dim, seqlen) -> (batch, seqlen, heads, head_dim)
        q = q.view(batch, num_heads_qk, head_dim, T).permute(0, 3, 1, 2).contiguous()
        k = k.view(batch, num_heads_qk, head_dim, T).permute(0, 3, 1, 2).contiguous()
        v = v.view(batch, num_heads_v, head_dim, T).permute(0, 3, 1, 2).contiguous()
        
        # ============================================================
        # Benchmark base: fused_sigmoid_gating_delta_rule_update
        # ============================================================
        for _ in range(10):
            _ = fused_sigmoid_gating_delta_rule_update(
                A_log=inputs["A_log"],
                a=inputs["a"],
                dt_bias=inputs["dt_bias"],
                softplus_beta=softplus_beta,
                softplus_threshold=softplus_threshold,
                q=q,
                k=k,
                v=v,
                b=inputs["b"],
                initial_state_source=inputs["ssm_state"],
                initial_state_indices=inputs["ssm_state_indices"],
                scale=scale,
                use_qk_l2norm_in_kernel=True,
            )
        torch.cuda.synchronize()
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        torch.cuda.synchronize()
        start_event.record()
        
        for _ in range(num_iters):
            _ = fused_sigmoid_gating_delta_rule_update(
                A_log=inputs["A_log"],
                a=inputs["a"],
                dt_bias=inputs["dt_bias"],
                softplus_beta=softplus_beta,
                softplus_threshold=softplus_threshold,
                q=q,
                k=k,
                v=v,
                b=inputs["b"],
                initial_state_source=inputs["ssm_state"],
                initial_state_indices=inputs["ssm_state_indices"],
                scale=scale,
                use_qk_l2norm_in_kernel=True,
            )
        
        end_event.record()
        torch.cuda.synchronize()
        
        base_time_us = start_event.elapsed_time(end_event) / num_iters * 1000
        
        # ============================================================
        # Benchmark v1 (original fused_split_gdr_update)
        # ============================================================
        for _ in range(10):
            _ = fused_split_gdr_update(
                mixed_qkv=inputs["mixed_qkv"],
                A_log=inputs["A_log"],
                a=inputs["a"],
                dt_bias=inputs["dt_bias"],
                b=inputs["b"],
                initial_state_source=inputs["ssm_state"],
                initial_state_indices=inputs["ssm_state_indices"],
                key_dim=key_dim,
                value_dim=value_dim,
                num_heads_qk=num_heads_qk,
                num_heads_v=num_heads_v,
                head_dim=head_dim,
                softplus_beta=softplus_beta,
                softplus_threshold=softplus_threshold,
                scale=scale,
                use_qk_l2norm_in_kernel=True,
            )
        torch.cuda.synchronize()
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        torch.cuda.synchronize()
        start_event.record()
        
        for _ in range(num_iters):
            _ = fused_split_gdr_update(
                mixed_qkv=inputs["mixed_qkv"],
                A_log=inputs["A_log"],
                a=inputs["a"],
                dt_bias=inputs["dt_bias"],
                b=inputs["b"],
                initial_state_source=inputs["ssm_state"],
                initial_state_indices=inputs["ssm_state_indices"],
                key_dim=key_dim,
                value_dim=value_dim,
                num_heads_qk=num_heads_qk,
                num_heads_v=num_heads_v,
                head_dim=head_dim,
                softplus_beta=softplus_beta,
                softplus_threshold=softplus_threshold,
                scale=scale,
                use_qk_l2norm_in_kernel=True,
            )
        
        end_event.record()
        torch.cuda.synchronize()
        
        v1_time_us = start_event.elapsed_time(end_event) / num_iters * 1000
        
        # ============================================================
        # Benchmark v2
        # ============================================================
        for _ in range(10):
            _ = fused_split_gdr_update_v2(
                mixed_qkv=inputs["mixed_qkv"],
                A_log=inputs["A_log"],
                a=inputs["a"],
                dt_bias=inputs["dt_bias"],
                b=inputs["b"],
                initial_state_source=inputs["ssm_state"],
                initial_state_indices=inputs["ssm_state_indices"],
                key_dim=key_dim,
                value_dim=value_dim,
                num_heads_qk=num_heads_qk,
                num_heads_v=num_heads_v,
                head_dim=head_dim,
                softplus_beta=softplus_beta,
                softplus_threshold=softplus_threshold,
                scale=scale,
                use_qk_l2norm_in_kernel=True,
            )
        torch.cuda.synchronize()
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        torch.cuda.synchronize()
        start_event.record()
        
        for _ in range(num_iters):
            _ = fused_split_gdr_update_v2(
                mixed_qkv=inputs["mixed_qkv"],
                A_log=inputs["A_log"],
                a=inputs["a"],
                dt_bias=inputs["dt_bias"],
                b=inputs["b"],
                initial_state_source=inputs["ssm_state"],
                initial_state_indices=inputs["ssm_state_indices"],
                key_dim=key_dim,
                value_dim=value_dim,
                num_heads_qk=num_heads_qk,
                num_heads_v=num_heads_v,
                head_dim=head_dim,
                softplus_beta=softplus_beta,
                softplus_threshold=softplus_threshold,
                scale=scale,
                use_qk_l2norm_in_kernel=True,
            )
        
        end_event.record()
        torch.cuda.synchronize()
        
        v2_time_us = start_event.elapsed_time(end_event) / num_iters * 1000
        
        # ============================================================
        # Benchmark v3
        # ============================================================
        for _ in range(10):
            _ = fused_split_gdr_update_v3(
                mixed_qkv=inputs["mixed_qkv"],
                A_log=inputs["A_log"],
                a=inputs["a"],
                dt_bias=inputs["dt_bias"],
                b=inputs["b"],
                initial_state_source=inputs["ssm_state"],
                initial_state_indices=inputs["ssm_state_indices"],
                key_dim=key_dim,
                value_dim=value_dim,
                num_heads_qk=num_heads_qk,
                num_heads_v=num_heads_v,
                head_dim=head_dim,
                softplus_beta=softplus_beta,
                softplus_threshold=softplus_threshold,
                scale=scale,
                use_qk_l2norm_in_kernel=True,
                output=output_buffer,
            )
        torch.cuda.synchronize()
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        torch.cuda.synchronize()
        start_event.record()
        
        for _ in range(num_iters):
            _ = fused_split_gdr_update_v3(
                mixed_qkv=inputs["mixed_qkv"],
                A_log=inputs["A_log"],
                a=inputs["a"],
                dt_bias=inputs["dt_bias"],
                b=inputs["b"],
                initial_state_source=inputs["ssm_state"],
                initial_state_indices=inputs["ssm_state_indices"],
                key_dim=key_dim,
                value_dim=value_dim,
                num_heads_qk=num_heads_qk,
                num_heads_v=num_heads_v,
                head_dim=head_dim,
                softplus_beta=softplus_beta,
                softplus_threshold=softplus_threshold,
                scale=scale,
                use_qk_l2norm_in_kernel=True,
                output=output_buffer,
            )
        
        end_event.record()
        torch.cuda.synchronize()
        
        v3_time_us = start_event.elapsed_time(end_event) / num_iters * 1000
        
        # Calculate speedups (base as baseline)
        v1_speedup = base_time_us / v1_time_us
        v2_speedup = base_time_us / v2_time_us
        v3_speedup = base_time_us / v3_time_us
        
        print(f"\n{'='*70}")
        print(f"Split GDR Performance: batch={batch_size}, seqlen={seqlen}")
        print(f"{'='*70}")
        print(f"  base (fused_sigmoid_gating_delta_rule_update): {base_time_us:.2f} us")
        print(f"  v1 (fused_split_gdr_update):       {v1_time_us:.2f} us (speedup: {v1_speedup:.2f}x)")
        print(f"  v2 (fused_split_gdr_update_v2):    {v2_time_us:.2f} us (speedup: {v2_speedup:.2f}x)")
        print(f"  v3 (fused_split_gdr_update_v3):    {v3_time_us:.2f} us (speedup: {v3_speedup:.2f}x)")
        print(f"{'='*70}")

    # ========================================================================
    # v3_seqlen1 Tests: Specialized decode kernel
    # ========================================================================

    @pytest.mark.parametrize("batch_size", [64, 128])
    @pytest.mark.parametrize("num_heads_qk", [4])
    @pytest.mark.parametrize("num_heads_v", [8])
    @pytest.mark.parametrize("head_dim", [128])
    def test_split_gdr_v3_seqlen1_correctness(
        self,
        batch_size,
        num_heads_qk,
        num_heads_v,
        head_dim,
        device,
        dtype,
    ):
        """Test correctness of v3_seqlen1 against v3."""
        torch.manual_seed(42)
        
        seqlen = 1  # Fixed for this specialized kernel
        
        inputs = self.create_inputs(
            batch_size, seqlen, num_heads_qk, num_heads_v, head_dim, device, dtype
        )
        
        key_dim = inputs["key_dim"]
        value_dim = inputs["value_dim"]
        
        softplus_beta = 1.0
        softplus_threshold = 20.0
        scale = head_dim ** -0.5
        
        # Run v3 (reference)
        ssm_state_v3 = inputs["ssm_state"].clone()
        output_v3 = fused_split_gdr_update_v3(
            mixed_qkv=inputs["mixed_qkv"],
            A_log=inputs["A_log"],
            a=inputs["a"],
            dt_bias=inputs["dt_bias"],
            b=inputs["b"],
            initial_state_source=ssm_state_v3,
            initial_state_indices=inputs["ssm_state_indices"],
            key_dim=key_dim,
            value_dim=value_dim,
            num_heads_qk=num_heads_qk,
            num_heads_v=num_heads_v,
            head_dim=head_dim,
            softplus_beta=softplus_beta,
            softplus_threshold=softplus_threshold,
            scale=scale,
            use_qk_l2norm_in_kernel=True,
        )
        
        # Run v3_seqlen1 (specialized)
        ssm_state_seqlen1 = inputs["ssm_state"].clone()
        output_seqlen1 = fused_split_gdr_update_v3_seqlen1(
            mixed_qkv=inputs["mixed_qkv"],
            A_log=inputs["A_log"],
            a=inputs["a"],
            dt_bias=inputs["dt_bias"],
            b=inputs["b"],
            initial_state_source=ssm_state_seqlen1,
            initial_state_indices=inputs["ssm_state_indices"],
            key_dim=key_dim,
            value_dim=value_dim,
            num_heads_qk=num_heads_qk,
            num_heads_v=num_heads_v,
            head_dim=head_dim,
            softplus_beta=softplus_beta,
            softplus_threshold=softplus_threshold,
            scale=scale,
            use_qk_l2norm_in_kernel=True,
        )
        
        output_diff = (output_v3 - output_seqlen1).abs().max().item()
        state_diff = (ssm_state_v3 - ssm_state_seqlen1).abs().max().item()
        
        print(f"\n{'='*70}")
        print(f"v3_seqlen1 Correctness Test: batch={batch_size}, seqlen=1")
        print(f"{'='*70}")
        print(f"  Output max diff: {output_diff:.6f}")
        print(f"  State max diff:  {state_diff:.6f}")
        
        assert output_diff < 1e-4, f"Output diff too large: {output_diff}"
        assert state_diff < 5e-3, f"State diff too large: {state_diff}"
        
        print(f"  ✓ v3_seqlen1 correctness test passed!")

    @pytest.mark.parametrize("batch_size", [64])
    @pytest.mark.parametrize("num_heads_qk", [4])
    @pytest.mark.parametrize("num_heads_v", [8])
    @pytest.mark.parametrize("head_dim", [128])
    def test_split_gdr_v3_seqlen1_performance(
        self,
        batch_size,
        num_heads_qk,
        num_heads_v,
        head_dim,
        device,
        dtype,
    ):
        """Benchmark v3 vs v3_seqlen1 for decode scenario."""
        torch.manual_seed(42)
        
        seqlen = 1
        
        inputs = self.create_inputs(
            batch_size, seqlen, num_heads_qk, num_heads_v, head_dim, device, dtype
        )
        
        key_dim = inputs["key_dim"]
        value_dim = inputs["value_dim"]
        
        softplus_beta = 1.0
        softplus_threshold = 20.0
        scale = head_dim ** -0.5
        
        # Pre-allocate output buffers
        output_v3 = torch.empty(batch_size, seqlen, num_heads_v, head_dim, device=device, dtype=dtype)
        output_seqlen1 = torch.empty(batch_size, seqlen, num_heads_v, head_dim, device=device, dtype=dtype)
        
        num_iters = 1000
        
        # Warmup v3
        for _ in range(10):
            fused_split_gdr_update_v3(
                mixed_qkv=inputs["mixed_qkv"], A_log=inputs["A_log"],
                a=inputs["a"], dt_bias=inputs["dt_bias"], b=inputs["b"],
                initial_state_source=inputs["ssm_state"],
                initial_state_indices=inputs["ssm_state_indices"],
                key_dim=key_dim, value_dim=value_dim,
                num_heads_qk=num_heads_qk, num_heads_v=num_heads_v, head_dim=head_dim,
                softplus_beta=softplus_beta, softplus_threshold=softplus_threshold,
                scale=scale, use_qk_l2norm_in_kernel=True, output=output_v3,
            )
        torch.cuda.synchronize()
        
        # Benchmark v3
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(num_iters):
            fused_split_gdr_update_v3(
                mixed_qkv=inputs["mixed_qkv"], A_log=inputs["A_log"],
                a=inputs["a"], dt_bias=inputs["dt_bias"], b=inputs["b"],
                initial_state_source=inputs["ssm_state"],
                initial_state_indices=inputs["ssm_state_indices"],
                key_dim=key_dim, value_dim=value_dim,
                num_heads_qk=num_heads_qk, num_heads_v=num_heads_v, head_dim=head_dim,
                softplus_beta=softplus_beta, softplus_threshold=softplus_threshold,
                scale=scale, use_qk_l2norm_in_kernel=True, output=output_v3,
            )
        end.record()
        torch.cuda.synchronize()
        v3_time_us = start.elapsed_time(end) / num_iters * 1000
        
        # Warmup v3_seqlen1
        for _ in range(10):
            fused_split_gdr_update_v3_seqlen1(
                mixed_qkv=inputs["mixed_qkv"], A_log=inputs["A_log"],
                a=inputs["a"], dt_bias=inputs["dt_bias"], b=inputs["b"],
                initial_state_source=inputs["ssm_state"],
                initial_state_indices=inputs["ssm_state_indices"],
                key_dim=key_dim, value_dim=value_dim,
                num_heads_qk=num_heads_qk, num_heads_v=num_heads_v, head_dim=head_dim,
                softplus_beta=softplus_beta, softplus_threshold=softplus_threshold,
                scale=scale, use_qk_l2norm_in_kernel=True, output=output_seqlen1,
            )
        torch.cuda.synchronize()
        
        # Benchmark v3_seqlen1
        start.record()
        for _ in range(num_iters):
            fused_split_gdr_update_v3_seqlen1(
                mixed_qkv=inputs["mixed_qkv"], A_log=inputs["A_log"],
                a=inputs["a"], dt_bias=inputs["dt_bias"], b=inputs["b"],
                initial_state_source=inputs["ssm_state"],
                initial_state_indices=inputs["ssm_state_indices"],
                key_dim=key_dim, value_dim=value_dim,
                num_heads_qk=num_heads_qk, num_heads_v=num_heads_v, head_dim=head_dim,
                softplus_beta=softplus_beta, softplus_threshold=softplus_threshold,
                scale=scale, use_qk_l2norm_in_kernel=True, output=output_seqlen1,
            )
        end.record()
        torch.cuda.synchronize()
        seqlen1_time_us = start.elapsed_time(end) / num_iters * 1000
        
        speedup = v3_time_us / seqlen1_time_us
        
        print(f"\n{'='*70}")
        print(f"v3 vs v3_seqlen1 Performance: batch={batch_size}, seqlen=1")
        print(f"{'='*70}")
        print(f"  v3 (general):      {v3_time_us:.2f} us")
        print(f"  v3_seqlen1:        {seqlen1_time_us:.2f} us (speedup: {speedup:.2f}x, {(speedup-1)*100:+.1f}%)")
        print(f"{'='*70}")

    @pytest.mark.parametrize("batch_size", [64])
    @pytest.mark.parametrize("seqlen", [1, 4])
    @pytest.mark.parametrize("num_heads_qk", [4])
    @pytest.mark.parametrize("num_heads_v", [8])
    @pytest.mark.parametrize("head_dim", [128])
    def test_split_gdr_v4_correctness(
        self,
        batch_size,
        seqlen,
        num_heads_qk,
        num_heads_v,
        head_dim,
        device,
        dtype,
    ):
        """Test v4 bilinear-output decomposition against v3."""
        torch.manual_seed(42)

        inputs = self.create_inputs(
            batch_size, seqlen, num_heads_qk, num_heads_v, head_dim, device, dtype
        )

        key_dim = inputs["key_dim"]
        value_dim = inputs["value_dim"]
        softplus_beta = 1.0
        softplus_threshold = 20.0
        scale = head_dim ** -0.5

        ssm_state_v3 = inputs["ssm_state"].clone()
        output_v3 = fused_split_gdr_update_v3(
            mixed_qkv=inputs["mixed_qkv"],
            A_log=inputs["A_log"],
            a=inputs["a"],
            dt_bias=inputs["dt_bias"],
            b=inputs["b"],
            initial_state_source=ssm_state_v3,
            initial_state_indices=inputs["ssm_state_indices"],
            key_dim=key_dim,
            value_dim=value_dim,
            num_heads_qk=num_heads_qk,
            num_heads_v=num_heads_v,
            head_dim=head_dim,
            softplus_beta=softplus_beta,
            softplus_threshold=softplus_threshold,
            scale=scale,
            use_qk_l2norm_in_kernel=True,
        )

        ssm_state_v4 = inputs["ssm_state"].clone()
        output_v4 = fused_split_gdr_update_v4(
            mixed_qkv=inputs["mixed_qkv"],
            A_log=inputs["A_log"],
            a=inputs["a"],
            dt_bias=inputs["dt_bias"],
            b=inputs["b"],
            initial_state_source=ssm_state_v4,
            initial_state_indices=inputs["ssm_state_indices"],
            key_dim=key_dim,
            value_dim=value_dim,
            num_heads_qk=num_heads_qk,
            num_heads_v=num_heads_v,
            head_dim=head_dim,
            softplus_beta=softplus_beta,
            softplus_threshold=softplus_threshold,
            scale=scale,
            use_qk_l2norm_in_kernel=True,
        )

        output_diff = (output_v3 - output_v4).abs().max().item()
        state_diff = (ssm_state_v3 - ssm_state_v4).abs().max().item()

        print(f"\n{'='*70}")
        print(f"v4 Correctness Test: batch={batch_size}, seqlen={seqlen}")
        print(f"{'='*70}")
        print(f"  Output max diff: {output_diff:.6f}")
        print(f"  State max diff:  {state_diff:.6f}")

        # Reordered arithmetic introduces tiny rounding differences.
        assert output_diff < 1e-3, f"Output diff too large: {output_diff}"
        assert state_diff < 1e-3, f"State diff too large: {state_diff}"

    @pytest.mark.parametrize("batch_size", [64])
    @pytest.mark.parametrize("num_heads_qk", [4])
    @pytest.mark.parametrize("num_heads_v", [8])
    @pytest.mark.parametrize("head_dim", [128])
    def test_split_gdr_v4_performance(
        self,
        batch_size,
        num_heads_qk,
        num_heads_v,
        head_dim,
        device,
        dtype,
    ):
        """Benchmark v3 vs v4 on decode-shaped workload (qwen3next)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for performance benchmark")

        torch.manual_seed(42)
        seqlen = 1

        inputs = self.create_inputs(
            batch_size, seqlen, num_heads_qk, num_heads_v, head_dim, device, dtype
        )

        key_dim = inputs["key_dim"]
        value_dim = inputs["value_dim"]
        softplus_beta = 1.0
        softplus_threshold = 20.0
        scale = head_dim ** -0.5

        output_v3 = torch.empty(batch_size, seqlen, num_heads_v, head_dim, device=device, dtype=dtype)
        output_v4 = torch.empty(batch_size, seqlen, num_heads_v, head_dim, device=device, dtype=dtype)

        num_iters = 1000

        for _ in range(10):
            fused_split_gdr_update_v3(
                mixed_qkv=inputs["mixed_qkv"],
                A_log=inputs["A_log"],
                a=inputs["a"],
                dt_bias=inputs["dt_bias"],
                b=inputs["b"],
                initial_state_source=inputs["ssm_state"],
                initial_state_indices=inputs["ssm_state_indices"],
                key_dim=key_dim,
                value_dim=value_dim,
                num_heads_qk=num_heads_qk,
                num_heads_v=num_heads_v,
                head_dim=head_dim,
                softplus_beta=softplus_beta,
                softplus_threshold=softplus_threshold,
                scale=scale,
                use_qk_l2norm_in_kernel=True,
                output=output_v3,
            )
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(num_iters):
            fused_split_gdr_update_v3(
                mixed_qkv=inputs["mixed_qkv"],
                A_log=inputs["A_log"],
                a=inputs["a"],
                dt_bias=inputs["dt_bias"],
                b=inputs["b"],
                initial_state_source=inputs["ssm_state"],
                initial_state_indices=inputs["ssm_state_indices"],
                key_dim=key_dim,
                value_dim=value_dim,
                num_heads_qk=num_heads_qk,
                num_heads_v=num_heads_v,
                head_dim=head_dim,
                softplus_beta=softplus_beta,
                softplus_threshold=softplus_threshold,
                scale=scale,
                use_qk_l2norm_in_kernel=True,
                output=output_v3,
            )
        end.record()
        torch.cuda.synchronize()
        v3_time_us = start.elapsed_time(end) / num_iters * 1000

        for _ in range(10):
            fused_split_gdr_update_v4(
                mixed_qkv=inputs["mixed_qkv"],
                A_log=inputs["A_log"],
                a=inputs["a"],
                dt_bias=inputs["dt_bias"],
                b=inputs["b"],
                initial_state_source=inputs["ssm_state"],
                initial_state_indices=inputs["ssm_state_indices"],
                key_dim=key_dim,
                value_dim=value_dim,
                num_heads_qk=num_heads_qk,
                num_heads_v=num_heads_v,
                head_dim=head_dim,
                softplus_beta=softplus_beta,
                softplus_threshold=softplus_threshold,
                scale=scale,
                use_qk_l2norm_in_kernel=True,
                output=output_v4,
            )
        torch.cuda.synchronize()

        start.record()
        for _ in range(num_iters):
            fused_split_gdr_update_v4(
                mixed_qkv=inputs["mixed_qkv"],
                A_log=inputs["A_log"],
                a=inputs["a"],
                dt_bias=inputs["dt_bias"],
                b=inputs["b"],
                initial_state_source=inputs["ssm_state"],
                initial_state_indices=inputs["ssm_state_indices"],
                key_dim=key_dim,
                value_dim=value_dim,
                num_heads_qk=num_heads_qk,
                num_heads_v=num_heads_v,
                head_dim=head_dim,
                softplus_beta=softplus_beta,
                softplus_threshold=softplus_threshold,
                scale=scale,
                use_qk_l2norm_in_kernel=True,
                output=output_v4,
            )
        end.record()
        torch.cuda.synchronize()
        v4_time_us = start.elapsed_time(end) / num_iters * 1000

        speedup = v3_time_us / v4_time_us

        print(f"\n{'='*70}")
        print(f"v3 vs v4 Performance: batch={batch_size}, seqlen=1")
        print(f"{'='*70}")
        print(f"  v3: {v3_time_us:.2f} us")
        print(f"  v4: {v4_time_us:.2f} us (speedup: {speedup:.2f}x, {(speedup - 1) * 100:+.1f}%)")
        print(f"{'='*70}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
