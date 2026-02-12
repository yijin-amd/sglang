"""
Unit tests for Fused Gating Delta Network (GDN) Forward Decode Kernel.

Tests correctness and performance of the fused kernel against the reference
implementation that runs causal_conv1d_update_split_qkv and 
fused_sigmoid_gating_delta_rule_update separately.
"""

import pytest
import torch
import time

from sglang.srt.layers.attention.mamba.causal_conv1d_split_qkv import (
    causal_conv1d_update_split_qkv,
)
from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)
from sglang.srt.layers.attention.fla.fused_gdn_fwd_decode import (
    fused_gdn_fwd_decode, fused_gdn_fwd_decode_v2, fused_gdn_fwd_decode_v3,
    PAD_SLOT_ID,
)
from sglang.srt.layers.attention.fla.fused_gdn_fwd_decode_gluon import (
    fused_gdn_fwd_decode_gluon, fused_gdn_fwd_decode_gluon_v2, fused_gdn_fwd_decode_gluon_v3, 
    fused_gdn_fwd_decode_gluon_v4, fused_gdn_fwd_decode_gluon_v5, fused_gdn_fwd_decode_gluon_v6,
    split_gdn_fwd_decode_gluon_v5,
)
import triton


# =============================================================================
# Qwen3Next model configuration with Tensor Parallelism
# =============================================================================
NUM_HEADS_QK = 16       # Total Q/K heads across all GPUs
NUM_HEADS_V = 32        # Total V heads across all GPUs
DEFAULT_TP = 8           # Default tensor parallelism degree
NUM_HEADS_QK_PER_GPU = NUM_HEADS_QK // DEFAULT_TP  # = 4
NUM_HEADS_V_PER_GPU = NUM_HEADS_V // DEFAULT_TP    # = 8


def gdn_fwd_decode_ref(
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
    conv_bias=None,
    activation="silu",
    conv_state_indices=None,
    ssm_state_indices=None,
    scale=None,
    use_qk_l2norm_in_kernel=True,
    softplus_beta=1.0,
    softplus_threshold=20.0,
    cu_seqlens=None,
):
    """
    Reference implementation using separate kernels.
    
    This mimics the behavior in hybrid_linear_attn_backend.py lines 278-326.
    """
    # Step 1: Causal Conv1D with split Q/K/V
    query, key, value = causal_conv1d_update_split_qkv(
        mixed_qkv,
        conv_state,
        conv_weight,
        key_dim=key_dim,
        value_dim=value_dim,
        bias=conv_bias,
        activation=activation,
        conv_state_indices=conv_state_indices,
        use_gluon=False,
    )
    
    # Reshape to match expected input format for gating delta rule
    batch, _, seqlen = query.shape
    
    query = query.view(batch, seqlen, num_heads_qk, head_dim)
    key = key.view(batch, seqlen, num_heads_qk, head_dim)
    value = value.view(batch, seqlen, num_heads_v, head_dim)
    
    # Step 2: Sigmoid gating delta rule update
    output = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        q=query,
        k=key,
        v=value,
        b=b,
        initial_state_source=ssm_state,
        initial_state_indices=ssm_state_indices,
        scale=scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        cu_seqlens=cu_seqlens,
    )
    
    return output

def get_copy_size(batch_size, head_dim, seqlen, conv_width, num_heads_v):
    return (
        (batch_size * head_dim * num_heads_v * seqlen * 3  * conv_width * 2 # qkv + conv_states + conv_weights
        + batch_size * head_dim * seqlen * num_heads_v  # conv_bias
        + batch_size * num_heads_v * 4 # A_log + a + dt_bias + b
        ) * 2 # bf16
        + batch_size * num_heads_v * head_dim * head_dim # ssm_state
        * 4 # fp32
    )

class TestFusedGDNFwdDecode:
    """Test suite for fused GDN forward decode kernel."""
    
    @pytest.fixture
    def device(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA device not available")
        return "cuda"
    
    @pytest.fixture
    def dtype(self):
        return torch.bfloat16
    
    def create_test_inputs(
        self,
        batch_size,
        key_dim,
        value_dim,
        num_heads_qk,
        num_heads_v,
        head_dim,
        seqlen,
        conv_width,
        device,
        dtype,
        has_bias=True,
    ):
        """Create test inputs for the GDN forward decode operation."""
        assert key_dim == num_heads_qk * head_dim
        assert value_dim == num_heads_v * head_dim
        
        dim = 2 * key_dim + value_dim
        HV = num_heads_v * head_dim
        
        # Conv1D inputs
        mixed_qkv = torch.randn(batch_size, dim, seqlen, device=device, dtype=dtype)
        conv_state = torch.randn(
            batch_size + 10, conv_width - 1, dim, device=device, dtype=dtype
        ).transpose(1, 2)
        conv_weight = torch.randn(dim, conv_width, device=device, dtype=dtype)
        conv_bias = torch.randn(dim, device=device, dtype=dtype) if has_bias else None
        
        # Gating inputs
        A_log = torch.randn(HV, device=device, dtype=torch.float32)
        a = torch.randn(batch_size, HV, device=device, dtype=dtype)
        dt_bias = torch.randn(HV, device=device, dtype=dtype)
        b = torch.randn(batch_size, HV, device=device, dtype=dtype)
        
        # SSM state
        ssm_state = torch.randn(
            batch_size + 10, HV, head_dim, head_dim,
            device=device, dtype=torch.float32
        )
        
        # Indices for continuous batching
        conv_state_indices = torch.arange(batch_size, device=device, dtype=torch.int32)
        ssm_state_indices = torch.arange(batch_size, device=device, dtype=torch.int32)
        
        return {
            "mixed_qkv": mixed_qkv,
            "conv_state": conv_state,
            "conv_weight": conv_weight,
            "conv_bias": conv_bias,
            "A_log": A_log,
            "a": a,
            "dt_bias": dt_bias,
            "b": b,
            "ssm_state": ssm_state,
            "conv_state_indices": conv_state_indices,
            "ssm_state_indices": ssm_state_indices,
            "key_dim": key_dim,
            "value_dim": value_dim,
            "num_heads_qk": num_heads_qk,
            "num_heads_v": num_heads_v,
            "head_dim": head_dim,
        }
    
    @pytest.mark.parametrize("batch_size", [1])
    @pytest.mark.parametrize("num_heads_qk", [NUM_HEADS_QK_PER_GPU])
    @pytest.mark.parametrize("num_heads_v", [NUM_HEADS_V_PER_GPU])
    @pytest.mark.parametrize("head_dim", [128])
    @pytest.mark.parametrize("seqlen", [2])
    @pytest.mark.parametrize("conv_width", [4])
    @pytest.mark.parametrize("has_bias", [True])
    def test_correctness(
        self,
        batch_size,
        num_heads_qk,
        num_heads_v,
        head_dim,
        seqlen,
        conv_width,
        has_bias,
        device,
        dtype,
    ):
        """Test that fused kernel produces the same results as reference."""
        key_dim = num_heads_qk * head_dim
        value_dim = num_heads_v * head_dim
        torch.cuda.manual_seed(0)
        
        # Create inputs
        inputs = self.create_test_inputs(
            batch_size, key_dim, value_dim, num_heads_qk, num_heads_v, head_dim,
            seqlen, conv_width, device, dtype, has_bias
        )
        
        # Clone states for separate runs
        conv_state_ref = inputs["conv_state"].clone()
        conv_state_fused = inputs["conv_state"].clone()
        ssm_state_ref = inputs["ssm_state"].clone()
        ssm_state_fused = inputs["ssm_state"].clone()
        
        # Run reference
        output_ref = gdn_fwd_decode_ref(
            mixed_qkv=inputs["mixed_qkv"].clone(),
            conv_state=conv_state_ref,
            conv_weight=inputs["conv_weight"],
            A_log=inputs["A_log"],
            a=inputs["a"],
            dt_bias=inputs["dt_bias"],
            b=inputs["b"],
            ssm_state=ssm_state_ref,
            key_dim=key_dim,
            value_dim=value_dim,
            num_heads_qk=num_heads_qk,
            num_heads_v=num_heads_v,
            head_dim=head_dim,
            conv_bias=inputs["conv_bias"],
            activation="silu",
            conv_state_indices=inputs["conv_state_indices"],
            ssm_state_indices=inputs["ssm_state_indices"],
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
        )
        
        # Run fused kernel
        output_fused = fused_gdn_fwd_decode(
            mixed_qkv=inputs["mixed_qkv"].clone(),
            conv_state=conv_state_fused,
            conv_weight=inputs["conv_weight"],
            A_log=inputs["A_log"],
            a=inputs["a"],
            dt_bias=inputs["dt_bias"],
            b=inputs["b"],
            ssm_state=ssm_state_fused,
            key_dim=key_dim,
            value_dim=value_dim,
            num_heads_qk=num_heads_qk,
            num_heads_v=num_heads_v,
            head_dim=head_dim,
            conv_bias=inputs["conv_bias"],
            activation="silu",
            conv_state_indices=inputs["conv_state_indices"],
            ssm_state_indices=inputs["ssm_state_indices"],
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
        )
        
        # Compare outputs
        rtol, atol = 1e-2, 5e-2 if dtype == torch.bfloat16 else (3e-3, 5e-3)
        
        # Check output match
        output_diff = (output_fused - output_ref).abs().max().item()
        print(f"\n[B={batch_size}, H_qk={num_heads_qk}, H_v={num_heads_v}, D={head_dim}]")
        print(f"  Output max diff: {output_diff:.6e}")
        
        assert torch.allclose(output_fused, output_ref, rtol=rtol, atol=atol), \
            f"Output mismatch: max diff = {output_diff}"
        
        # Check SSM state match
        ssm_state_diff = (ssm_state_fused - ssm_state_ref).abs().max().item()
        print(f"  SSM state max diff: {ssm_state_diff:.6e}")
        
        assert torch.allclose(ssm_state_fused, ssm_state_ref, rtol=rtol, atol=atol), \
            f"SSM state mismatch: max diff = {ssm_state_diff}"
        
        # Check conv_state match
        conv_state_diff = (conv_state_fused - conv_state_ref).abs().max().item()
        print(f"  Conv state max diff: {conv_state_diff:.6e}")
        
        assert torch.allclose(conv_state_fused, conv_state_ref, rtol=rtol, atol=atol), \
            f"Conv state mismatch: max diff = {conv_state_diff}"
        
        print(f"  ✓ Correctness test passed!")
    


    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 32, 64])
    def test_decode_throughput(self, batch_size, device, dtype):
        """Test decode throughput with various batch sizes."""
        num_heads_qk = NUM_HEADS_QK_PER_GPU
        num_heads_v = NUM_HEADS_V_PER_GPU
        head_dim = 128
        key_dim = num_heads_qk * head_dim
        value_dim = num_heads_v * head_dim
        seqlen = 1
        conv_width = 4
     
        # Create inputs
        inputs = self.create_test_inputs(
            batch_size, key_dim, value_dim, num_heads_qk, num_heads_v, head_dim,
            seqlen, conv_width, device, dtype, has_bias=True
        )
        
        # ====================================================================
        # Benchmark Reference (Separate Kernels)
        # ====================================================================
        
        # Prepare inputs outside timing loop (not caring about result correctness)
        mixed_qkv_ref = inputs["mixed_qkv"]
        conv_state_ref = inputs["conv_state"]
        ssm_state_ref = inputs["ssm_state"]
        
        # Warmup
        for _ in range(3):
            _ = gdn_fwd_decode_ref(
                mixed_qkv=mixed_qkv_ref,
                conv_state=conv_state_ref,
                conv_weight=inputs["conv_weight"],
                A_log=inputs["A_log"],
                a=inputs["a"],
                dt_bias=inputs["dt_bias"],
                b=inputs["b"],
                ssm_state=ssm_state_ref,
                key_dim=key_dim,
                value_dim=value_dim,
                num_heads_qk=num_heads_qk,
                num_heads_v=num_heads_v,
                head_dim=head_dim,
                conv_bias=inputs["conv_bias"],
                activation="silu",
                conv_state_indices=inputs["conv_state_indices"],
                ssm_state_indices=inputs["ssm_state_indices"],
                use_qk_l2norm_in_kernel=True,
            )
        torch.cuda.synchronize()
        
        # Benchmark
        num_iters = 1000
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_iters):
            _ = gdn_fwd_decode_ref(
                mixed_qkv=mixed_qkv_ref,
                conv_state=conv_state_ref,
                conv_weight=inputs["conv_weight"],
                A_log=inputs["A_log"],
                a=inputs["a"],
                dt_bias=inputs["dt_bias"],
                b=inputs["b"],
                ssm_state=ssm_state_ref,
                key_dim=key_dim,
                value_dim=value_dim,
                num_heads_qk=num_heads_qk,
                num_heads_v=num_heads_v,
                head_dim=head_dim,
                conv_bias=inputs["conv_bias"],
                activation="silu",
                conv_state_indices=inputs["conv_state_indices"],
                ssm_state_indices=inputs["ssm_state_indices"],
                use_qk_l2norm_in_kernel=True,
            )
        torch.cuda.synchronize()
        ref_time = (time.time() - start) / num_iters * 1000  # ms
        
        # ====================================================================
        # Benchmark Fused Kernel
        # ====================================================================
        
        # Prepare inputs outside timing loop (not caring about result correctness)
        mixed_qkv_fused = inputs["mixed_qkv"]
        conv_state_fused = inputs["conv_state"]
        ssm_state_fused = inputs["ssm_state"]
        
        # Warmup
        for _ in range(3):
            _ = fused_gdn_fwd_decode_gluon_v2( # fused_gdn_fwd_decode_gluon_v2
                mixed_qkv=mixed_qkv_fused,
                conv_state=conv_state_fused,
                conv_weight=inputs["conv_weight"],
                A_log=inputs["A_log"],
                a=inputs["a"],
                dt_bias=inputs["dt_bias"],
                b=inputs["b"],
                ssm_state=ssm_state_fused,
                key_dim=key_dim,
                value_dim=value_dim,
                num_heads_qk=num_heads_qk,
                num_heads_v=num_heads_v,
                head_dim=head_dim,
                conv_bias=inputs["conv_bias"],
                activation="silu",
                conv_state_indices=inputs["conv_state_indices"],
                ssm_state_indices=inputs["ssm_state_indices"],
                use_qk_l2norm_in_kernel=True,
            )
        torch.cuda.synchronize()
        
        # Benchmark
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_iters):
            _ = fused_gdn_fwd_decode_gluon_v2(  # fused_gdn_fwd_decode_gluon_v2
                mixed_qkv=mixed_qkv_fused,
                conv_state=conv_state_fused,
                conv_weight=inputs["conv_weight"],
                A_log=inputs["A_log"],
                a=inputs["a"],
                dt_bias=inputs["dt_bias"],
                b=inputs["b"],
                ssm_state=ssm_state_fused,
                key_dim=key_dim,
                value_dim=value_dim,
                num_heads_qk=num_heads_qk,
                num_heads_v=num_heads_v,
                head_dim=head_dim,
                conv_bias=inputs["conv_bias"],
                activation="silu",
                conv_state_indices=inputs["conv_state_indices"],
                ssm_state_indices=inputs["ssm_state_indices"],
                use_qk_l2norm_in_kernel=True,
            )
        torch.cuda.synchronize()
        fused_time = (time.time() - start) / num_iters * 1000  # ms
        
        copy_size = get_copy_size(batch_size, head_dim, seqlen, conv_width, num_heads_v)
        # Calculate metrics
        speedup = ref_time / fused_time
        throughput_ref = (num_iters * copy_size) / (ref_time * num_iters / 1000)
        throughput_fused = (num_iters * copy_size) / (fused_time * num_iters / 1000)
        print()
        
        # print(f"\n{'='*70}")
        # print(f"Decode Throughput Test: batch_size={batch_size}")
        # print(f"{'='*70}")
        # print(f"Configuration:")
        # print(f"  - num_heads_qk: {num_heads_qk}")
        # print(f"  - num_heads_v:  {num_heads_v}")
        # print(f"  - head_dim:     {head_dim}")
        # print(f"  - seqlen:       {seqlen}")
        # print(f"  - dtype:        {dtype}")
        # print(f"\nPerformance Results (averaged over {num_iters} iterations):")
        print(f"\n  Reference (Separate Kernels):")
        print(f"    - Time per iteration:  {ref_time=:.4f} ms")
        print(f"    - Throughput:          {throughput_ref:.2f} tokens/s")
        print(f"\n  Fused Kernel:")
        print(f"    - Time per iteration:  {fused_time=:.4f} ms")
        print(f"    - Throughput:          {throughput_fused:.2f} tokens/s")
        # print(f"\n  Performance Comparison:")
        # print(f"    - Speedup (Fused/Reference): {speedup:.2f}x")
        # print(f"    - Time saved:                {ref_time - fused_time:.4f} ms")
        
        if speedup > 1.05:
            print(f"    - Status:                    ✓ Fused kernel is {speedup:.2f}x FASTER")
        elif speedup < 0.95:
            print(f"    - Status:                    ⚠ Reference is {1/speedup:.2f}x FASTER")
        else:
            print(f"    - Status:                    ≈ Performance is similar")
        
        print(f"{'='*70}")
    
    @pytest.mark.parametrize("activation", ["silu", None])
    @pytest.mark.parametrize("use_qk_l2norm", [True, False])
    def test_different_configs(self, activation, use_qk_l2norm, device, dtype):
        """Test different activation and normalization configurations."""
        batch_size = 16
        num_heads_qk = NUM_HEADS_QK_PER_GPU
        num_heads_v = NUM_HEADS_V_PER_GPU
        head_dim = 128
        key_dim = num_heads_qk * head_dim
        value_dim = num_heads_v * head_dim
        seqlen = 1
        conv_width = 4
        
        # Create inputs
        inputs = self.create_test_inputs(
            batch_size, key_dim, value_dim, num_heads_qk, num_heads_v, head_dim,
            seqlen, conv_width, device, dtype, has_bias=True
        )
        
        # Clone states
        conv_state_ref = inputs["conv_state"].clone()
        conv_state_fused = inputs["conv_state"].clone()
        ssm_state_ref = inputs["ssm_state"].clone()
        ssm_state_fused = inputs["ssm_state"].clone()
        
        # Run reference
        output_ref = gdn_fwd_decode_ref(
            mixed_qkv=inputs["mixed_qkv"].clone(),
            conv_state=conv_state_ref,
            conv_weight=inputs["conv_weight"],
            A_log=inputs["A_log"],
            a=inputs["a"],
            dt_bias=inputs["dt_bias"],
            b=inputs["b"],
            ssm_state=ssm_state_ref,
            key_dim=key_dim,
            value_dim=value_dim,
            num_heads_qk=num_heads_qk,
            num_heads_v=num_heads_v,
            head_dim=head_dim,
            conv_bias=inputs["conv_bias"],
            activation=activation,
            conv_state_indices=inputs["conv_state_indices"],
            ssm_state_indices=inputs["ssm_state_indices"],
            use_qk_l2norm_in_kernel=use_qk_l2norm,
        )
        
        # Run fused
        output_fused = fused_gdn_fwd_decode(
            mixed_qkv=inputs["mixed_qkv"].clone(),
            conv_state=conv_state_fused,
            conv_weight=inputs["conv_weight"],
            A_log=inputs["A_log"],
            a=inputs["a"],
            dt_bias=inputs["dt_bias"],
            b=inputs["b"],
            ssm_state=ssm_state_fused,
            key_dim=key_dim,
            value_dim=value_dim,
            num_heads_qk=num_heads_qk,
            num_heads_v=num_heads_v,
            head_dim=head_dim,
            conv_bias=inputs["conv_bias"],
            activation=activation,
            conv_state_indices=inputs["conv_state_indices"],
            ssm_state_indices=inputs["ssm_state_indices"],
            use_qk_l2norm_in_kernel=use_qk_l2norm,
        )
        
        # Compare
        rtol, atol = 1e-2, 5e-2
        assert torch.allclose(output_fused, output_ref, rtol=rtol, atol=atol)
        
        print(f"✓ Config test passed: activation={activation}, l2norm={use_qk_l2norm}")

class TestGluonFusedGDNFwdDecode:
    """Test suite for Gluon version of Fused GDN Forward Decode kernel."""
    
    @pytest.fixture
    def device(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA device not available")
        return "cuda"
    
    @pytest.fixture
    def dtype(self):
        return torch.bfloat16
    
    def create_test_inputs(
        self,
        batch_size,
        key_dim,
        value_dim,
        num_heads_qk,
        num_heads_v,
        head_dim,
        seqlen,
        conv_width,
        device,
        dtype,
        has_bias=True,
    ):
        """Create test inputs for the GDN forward decode operation."""
        assert key_dim == num_heads_qk * head_dim
        assert value_dim == num_heads_v * head_dim
        
        dim = 2 * key_dim + value_dim
        HV = num_heads_v * head_dim
        
        mixed_qkv = torch.randn(batch_size, dim, seqlen, device=device, dtype=dtype)
        conv_state = torch.randn(
            batch_size + 10, conv_width - 1, dim, device=device, dtype=dtype
        ).transpose(1, 2)
        conv_weight = torch.randn(dim, conv_width, device=device, dtype=dtype)
        conv_bias = torch.randn(dim, device=device, dtype=dtype) if has_bias else None
        
        A_log = torch.randn(HV, device=device, dtype=torch.float32)
        a = torch.randn(batch_size, HV, device=device, dtype=dtype)
        dt_bias = torch.randn(HV, device=device, dtype=dtype)
        b = torch.randn(batch_size, HV, device=device, dtype=dtype)
        
        ssm_state = torch.randn(
            batch_size + 10, HV, head_dim, head_dim,
            device=device, dtype=torch.float32
        )
        
        conv_state_indices = torch.arange(batch_size, device=device, dtype=torch.int32)
        ssm_state_indices = torch.arange(batch_size, device=device, dtype=torch.int32)
        
        return {
            "mixed_qkv": mixed_qkv,
            "conv_state": conv_state,
            "conv_weight": conv_weight,
            "conv_bias": conv_bias,
            "A_log": A_log,
            "a": a,
            "dt_bias": dt_bias,
            "b": b,
            "ssm_state": ssm_state,
            "conv_state_indices": conv_state_indices,
            "ssm_state_indices": ssm_state_indices,
            "key_dim": key_dim,
            "value_dim": value_dim,
            "num_heads_qk": num_heads_qk,
            "num_heads_v": num_heads_v,
            "head_dim": head_dim,
        }
    
    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 32, 64, 128])
    @pytest.mark.parametrize("num_heads_qk", [NUM_HEADS_QK_PER_GPU])
    @pytest.mark.parametrize("num_heads_v", [NUM_HEADS_V_PER_GPU])
    @pytest.mark.parametrize("head_dim", [128])
    @pytest.mark.parametrize("seqlen", [1])
    @pytest.mark.parametrize("conv_width", [4])
    @pytest.mark.parametrize("has_bias", [True])
    def test_gluon_vs_reference(
        self,
        batch_size,
        num_heads_qk,
        num_heads_v,
        head_dim,
        seqlen,
        conv_width,
        has_bias,
        device,
        dtype,
    ):
        """Test that Gluon kernel produces the same results as reference."""
        key_dim = num_heads_qk * head_dim
        value_dim = num_heads_v * head_dim
        
        inputs = self.create_test_inputs(
            batch_size, key_dim, value_dim, num_heads_qk, num_heads_v, head_dim,
            seqlen, conv_width, device, dtype, has_bias
        )
        
        # Clone inputs for each run
        inputs_ref = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        inputs_gluon = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Run reference implementation
        output_ref = gdn_fwd_decode_ref(**inputs_ref)
        conv_state_ref = inputs_ref["conv_state"]
        
        # Run Gluon implementation
        output_gluon = fused_gdn_fwd_decode_gluon(**inputs_gluon)
        conv_state_gluon = inputs_gluon["conv_state"]
        
        # Compare outputs
        rtol, atol = 1e-2, 5e-2
        assert torch.allclose(output_gluon, output_ref, rtol=rtol, atol=atol), \
            f"Output mismatch: max diff = {(output_gluon - output_ref).abs().max()}"
        
        # Compare conv_states
        assert torch.allclose(conv_state_gluon, conv_state_ref, rtol=rtol, atol=atol), \
            f"Conv state mismatch: max diff = {(conv_state_gluon - conv_state_ref).abs().max()}"
        
        print(f"✓ Gluon kernel test passed")
    
    @pytest.mark.parametrize("batch_size", [1])
    @pytest.mark.parametrize("num_heads_qk", [NUM_HEADS_QK_PER_GPU])
    @pytest.mark.parametrize("num_heads_v", [NUM_HEADS_V_PER_GPU])
    @pytest.mark.parametrize("head_dim", [128])
    @pytest.mark.parametrize("seqlen", [1])
    @pytest.mark.parametrize("conv_width", [4])
    @pytest.mark.parametrize("has_bias", [True])
    def test_gluon_vs_triton(
        self,
        batch_size,
        num_heads_qk,
        num_heads_v,
        head_dim,
        seqlen,
        conv_width,
        has_bias,
        device,
        dtype,
    ):
        """Test that Gluon kernel produces the same results as Triton kernel."""
        key_dim = num_heads_qk * head_dim
        value_dim = num_heads_v * head_dim
        
        inputs = self.create_test_inputs(
            batch_size, key_dim, value_dim, num_heads_qk, num_heads_v, head_dim,
            seqlen, conv_width, device, dtype, has_bias
        )
        
        # Clone inputs for each run
        inputs_triton = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        inputs_gluon = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Run Triton implementation
        output_triton = fused_gdn_fwd_decode(**inputs_triton)
        conv_state_triton = inputs_triton["conv_state"]
        
        # Run Gluon implementation
        output_gluon = fused_gdn_fwd_decode_gluon(**inputs_gluon)
        conv_state_gluon = inputs_gluon["conv_state"]
        
        # Compare outputs
        rtol, atol = 1e-3, 1e-3
        assert torch.allclose(output_gluon, output_triton, rtol=rtol, atol=atol), \
            f"Output mismatch: max diff = {(output_gluon - output_triton).abs().max()}"
        
        # Compare conv_states
        assert torch.allclose(conv_state_gluon, conv_state_triton, rtol=rtol, atol=atol), \
            f"Conv state mismatch: max diff = {(conv_state_gluon - conv_state_triton).abs().max()}"
        
        print(f"✓ Gluon vs Triton test passed")
    
    @pytest.mark.parametrize("batch_size", [64])
    def test_gluon_vs_reference_throughput(self, batch_size, device, dtype):
        """Benchmark performance of Gluon kernel vs reference implementation."""
        import os
        
        num_heads_qk = NUM_HEADS_QK_PER_GPU
        num_heads_v = NUM_HEADS_V_PER_GPU
        head_dim = 128
        key_dim = num_heads_qk * head_dim
        value_dim = num_heads_v * head_dim
        seqlen = 1
        conv_width = 4
        
        # Create inputs
        inputs = self.create_test_inputs(
            batch_size, key_dim, value_dim, num_heads_qk, num_heads_v, head_dim,
            seqlen, conv_width, device, dtype, has_bias=True
        )
        
        # ====================================================================
        # Benchmark Reference (Separate Kernels)
        # ====================================================================
        
        # Prepare inputs outside timing loop
        mixed_qkv_ref = inputs["mixed_qkv"]
        conv_state_ref = inputs["conv_state"]
        ssm_state_ref = inputs["ssm_state"]
        
        # Warmup
        for _ in range(3):
            _ = gdn_fwd_decode_ref(
                mixed_qkv=mixed_qkv_ref,
                conv_state=conv_state_ref,
                conv_weight=inputs["conv_weight"],
                A_log=inputs["A_log"],
                a=inputs["a"],
                dt_bias=inputs["dt_bias"],
                b=inputs["b"],
                ssm_state=ssm_state_ref,
                key_dim=key_dim,
                value_dim=value_dim,
                num_heads_qk=num_heads_qk,
                num_heads_v=num_heads_v,
                head_dim=head_dim,
                conv_bias=inputs["conv_bias"],
                activation="silu",
                conv_state_indices=inputs["conv_state_indices"],
                ssm_state_indices=inputs["ssm_state_indices"],
                use_qk_l2norm_in_kernel=True,
            )
        torch.cuda.synchronize()
        
        # Benchmark
        num_iters = 1000
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_iters):
            _ = gdn_fwd_decode_ref(
                mixed_qkv=mixed_qkv_ref,
                conv_state=conv_state_ref,
                conv_weight=inputs["conv_weight"],
                A_log=inputs["A_log"],
                a=inputs["a"],
                dt_bias=inputs["dt_bias"],
                b=inputs["b"],
                ssm_state=ssm_state_ref,
                key_dim=key_dim,
                value_dim=value_dim,
                num_heads_qk=num_heads_qk,
                num_heads_v=num_heads_v,
                head_dim=head_dim,
                conv_bias=inputs["conv_bias"],
                activation="silu",
                conv_state_indices=inputs["conv_state_indices"],
                ssm_state_indices=inputs["ssm_state_indices"],
                use_qk_l2norm_in_kernel=True,
            )
        torch.cuda.synchronize()
        ref_time = (time.time() - start) / num_iters * 1000  # ms
        
        # ====================================================================
        # Benchmark Gluon Kernel
        # ====================================================================
        
        # Prepare inputs outside timing loop
        mixed_qkv_gluon = inputs["mixed_qkv"]
        conv_state_gluon = inputs["conv_state"]
        ssm_state_gluon = inputs["ssm_state"]
        
        # Warmup
        for _ in range(3):
            _ = fused_gdn_fwd_decode_gluon_v2(
                mixed_qkv=mixed_qkv_gluon,
                conv_state=conv_state_gluon,
                conv_weight=inputs["conv_weight"],
                A_log=inputs["A_log"],
                a=inputs["a"],
                dt_bias=inputs["dt_bias"],
                b=inputs["b"],
                ssm_state=ssm_state_gluon,
                key_dim=key_dim,
                value_dim=value_dim,
                num_heads_qk=num_heads_qk,
                num_heads_v=num_heads_v,
                head_dim=head_dim,
                conv_bias=inputs["conv_bias"],
                activation="silu",
                conv_state_indices=inputs["conv_state_indices"],
                ssm_state_indices=inputs["ssm_state_indices"],
                use_qk_l2norm_in_kernel=True,
            )
        torch.cuda.synchronize()
        
        # Benchmark
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_iters):
            _ = fused_gdn_fwd_decode_gluon_v2(
                mixed_qkv=mixed_qkv_gluon,
                conv_state=conv_state_gluon,
                conv_weight=inputs["conv_weight"],
                A_log=inputs["A_log"],
                a=inputs["a"],
                dt_bias=inputs["dt_bias"],
                b=inputs["b"],
                ssm_state=ssm_state_gluon,
                key_dim=key_dim,
                value_dim=value_dim,
                num_heads_qk=num_heads_qk,
                num_heads_v=num_heads_v,
                head_dim=head_dim,
                conv_bias=inputs["conv_bias"],
                activation="silu",
                conv_state_indices=inputs["conv_state_indices"],
                ssm_state_indices=inputs["ssm_state_indices"],
                use_qk_l2norm_in_kernel=True,
            )
        torch.cuda.synchronize()
        gluon_time = (time.time() - start) / num_iters * 1000  # ms
        
        # Calculate metrics
        speedup = ref_time / gluon_time
        throughput_ref = (num_iters * batch_size) / (ref_time * num_iters / 1000)
        throughput_gluon = (num_iters * batch_size) / (gluon_time * num_iters / 1000)
        print()
        
        print(f"    - Reference time per iteration: {ref_time=:.4f} ms")
        print(f"    - Gluon time per iteration:     {gluon_time=:.4f} ms")
        
        if speedup > 1.05:
            print(f"    - Status: ✓ Gluon kernel is {speedup:.2f}x FASTER")
        elif speedup < 0.95:
            print(f"    - Status: ⚠ Reference is {1/speedup:.2f}x FASTER")
        else:
            print(f"    - Status: ≈ Performance is similar")
        
        print(f"{'='*70}")
    
    @pytest.mark.parametrize("batch_size", [64])
    @pytest.mark.parametrize("num_iters", [100])
    def test_generate_profiler_traces(self, batch_size, num_iters, device, dtype):
        """
        Generate PyTorch profiler traces for Reference and Gluon v2 implementations.
        
        This test creates separate trace files for detailed performance analysis:
        - ~/trace_gdn_fwd_decode_ref.json: Reference implementation trace
        - ~/trace_fused_gdn_fwd_decode_gluon_v2.json: Gluon v2 implementation trace
        
        Use chrome://tracing or https://ui.perfetto.dev/ to visualize the traces.
        """
        import os
        
        num_heads_qk = NUM_HEADS_QK_PER_GPU
        num_heads_v = NUM_HEADS_V_PER_GPU
        head_dim = 128
        key_dim = num_heads_qk * head_dim
        value_dim = num_heads_v * head_dim
        seqlen = 1
        conv_width = 4
        
        # Create inputs
        inputs = self.create_test_inputs(
            batch_size, key_dim, value_dim, num_heads_qk, num_heads_v, head_dim,
            seqlen, conv_width, device, dtype, has_bias=True
        )
        
        print(f"\n{'='*70}")
        print(f"Generating Profiler Traces")
        print(f"{'='*70}")
        print(f"Configuration:")
        print(f"  - batch_size: {batch_size}")
        print(f"  - num_iters: {num_iters}")
        print(f"  - num_heads_qk: {num_heads_qk}")
        print(f"  - num_heads_v: {num_heads_v}")
        print(f"  - head_dim: {head_dim}")
        print(f"  - seqlen: {seqlen}")
        print(f"{'='*70}\n")
        
        # ====================================================================
        # Profile Reference Implementation
        # ====================================================================
        
        print("Profiling Reference implementation...")
        
        # Prepare inputs
        mixed_qkv_ref = inputs["mixed_qkv"]
        conv_state_ref = inputs["conv_state"]
        ssm_state_ref = inputs["ssm_state"]
        
        # Warmup
        for _ in range(3):
            _ = gdn_fwd_decode_ref(
                mixed_qkv=mixed_qkv_ref,
                conv_state=conv_state_ref,
                conv_weight=inputs["conv_weight"],
                A_log=inputs["A_log"],
                a=inputs["a"],
                dt_bias=inputs["dt_bias"],
                b=inputs["b"],
                ssm_state=ssm_state_ref,
                key_dim=key_dim,
                value_dim=value_dim,
                num_heads_qk=num_heads_qk,
                num_heads_v=num_heads_v,
                head_dim=head_dim,
                conv_bias=inputs["conv_bias"],
                activation="silu",
                conv_state_indices=inputs["conv_state_indices"],
                ssm_state_indices=inputs["ssm_state_indices"],
                use_qk_l2norm_in_kernel=True,
            )
        torch.cuda.synchronize()
        
        # Profile
        trace_path_ref = os.path.expanduser("~/trace_gdn_fwd_decode_ref.json")
        
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof_ref:
            for _ in range(num_iters):
                _ = gdn_fwd_decode_ref(
                    mixed_qkv=mixed_qkv_ref,
                    conv_state=conv_state_ref,
                    conv_weight=inputs["conv_weight"],
                    A_log=inputs["A_log"],
                    a=inputs["a"],
                    dt_bias=inputs["dt_bias"],
                    b=inputs["b"],
                    ssm_state=ssm_state_ref,
                    key_dim=key_dim,
                    value_dim=value_dim,
                    num_heads_qk=num_heads_qk,
                    num_heads_v=num_heads_v,
                    head_dim=head_dim,
                    conv_bias=inputs["conv_bias"],
                    activation="silu",
                    conv_state_indices=inputs["conv_state_indices"],
                    ssm_state_indices=inputs["ssm_state_indices"],
                    use_qk_l2norm_in_kernel=True,
                )
        
        torch.cuda.synchronize()
        
        # Export trace
        prof_ref.export_chrome_trace(trace_path_ref)
        print(f"  ✓ Reference trace saved: {trace_path_ref}")
        
        # ====================================================================
        # Profile Gluon v2 Implementation
        # ====================================================================
        
        print("\nProfiling Gluon v2 implementation...")
        
        # Prepare inputs
        mixed_qkv_gluon = inputs["mixed_qkv"]
        conv_state_gluon = inputs["conv_state"]
        ssm_state_gluon = inputs["ssm_state"]
        
        # Warmup
        for _ in range(3):
            _ = fused_gdn_fwd_decode_gluon_v2(
                mixed_qkv=mixed_qkv_gluon,
                conv_state=conv_state_gluon,
                conv_weight=inputs["conv_weight"],
                A_log=inputs["A_log"],
                a=inputs["a"],
                dt_bias=inputs["dt_bias"],
                b=inputs["b"],
                ssm_state=ssm_state_gluon,
                key_dim=key_dim,
                value_dim=value_dim,
                num_heads_qk=num_heads_qk,
                num_heads_v=num_heads_v,
                head_dim=head_dim,
                conv_bias=inputs["conv_bias"],
                activation="silu",
                conv_state_indices=inputs["conv_state_indices"],
                ssm_state_indices=inputs["ssm_state_indices"],
                use_qk_l2norm_in_kernel=True,
            )
        torch.cuda.synchronize()
        
        # Profile
        trace_path_gluon = os.path.expanduser("~/trace_fused_gdn_fwd_decode_gluon_v2.json")
        
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof_gluon:
            for _ in range(num_iters):
                _ = fused_gdn_fwd_decode_gluon_v2(
                    mixed_qkv=mixed_qkv_gluon,
                    conv_state=conv_state_gluon,
                    conv_weight=inputs["conv_weight"],
                    A_log=inputs["A_log"],
                    a=inputs["a"],
                    dt_bias=inputs["dt_bias"],
                    b=inputs["b"],
                    ssm_state=ssm_state_gluon,
                    key_dim=key_dim,
                    value_dim=value_dim,
                    num_heads_qk=num_heads_qk,
                    num_heads_v=num_heads_v,
                    head_dim=head_dim,
                    conv_bias=inputs["conv_bias"],
                    activation="silu",
                    conv_state_indices=inputs["conv_state_indices"],
                    ssm_state_indices=inputs["ssm_state_indices"],
                    use_qk_l2norm_in_kernel=True,
                )
        
        torch.cuda.synchronize()
        
        # Export trace
        prof_gluon.export_chrome_trace(trace_path_gluon)
        print(f"  ✓ Gluon v2 trace saved: {trace_path_gluon}")
        
        print(f"\n{'='*70}")
        print("Profiler traces generated successfully!")
        print(f"{'='*70}")
        print("\nView traces using:")
        print("  - Chrome: chrome://tracing")
        print("  - Perfetto: https://ui.perfetto.dev/")
        print(f"{'='*70}\n")
    
    @pytest.mark.parametrize("batch_size", [64])
    def test_three_way_performance_comparison(self, batch_size, device, dtype):
        """
        Benchmark performance comparison of three implementations:
        1. gdn_fwd_decode_ref (Python reference)
        2. fused_gdn_fwd_decode_gluon (Gluon v1, Q/K-indexed)
        3. fused_gdn_fwd_decode_gluon_v2 (Gluon v2, V-indexed)
        """
        num_heads_qk = NUM_HEADS_QK_PER_GPU
        num_heads_v = NUM_HEADS_V_PER_GPU
        head_dim = 128
        key_dim = num_heads_qk * head_dim
        value_dim = num_heads_v * head_dim
        seqlen = 1
        conv_width = 4
        
        # Create inputs
        inputs = self.create_test_inputs(
            batch_size, key_dim, value_dim, num_heads_qk, num_heads_v, head_dim,
            seqlen, conv_width, device, dtype, has_bias=True
        )
        
        # Prepare inputs outside timing loop (shared across all implementations)
        mixed_qkv = inputs["mixed_qkv"]
        conv_state = inputs["conv_state"]
        ssm_state = inputs["ssm_state"]
        
        num_iters = 1000
        
        # ====================================================================
        # Benchmark 1: Reference Implementation (Python)
        # ====================================================================
        
        # Warmup
        for _ in range(3):
            _ = gdn_fwd_decode_ref(
                mixed_qkv=mixed_qkv,
                conv_state=conv_state,
                conv_weight=inputs["conv_weight"],
                A_log=inputs["A_log"],
                a=inputs["a"],
                dt_bias=inputs["dt_bias"],
                b=inputs["b"],
                ssm_state=ssm_state,
                key_dim=key_dim,
                value_dim=value_dim,
                num_heads_qk=num_heads_qk,
                num_heads_v=num_heads_v,
                head_dim=head_dim,
                conv_bias=inputs["conv_bias"],
                activation="silu",
                conv_state_indices=inputs["conv_state_indices"],
                ssm_state_indices=inputs["ssm_state_indices"],
                use_qk_l2norm_in_kernel=True,
            )
        torch.cuda.synchronize()
        
        # Benchmark
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_iters):
            _ = gdn_fwd_decode_ref(
                mixed_qkv=mixed_qkv,
                conv_state=conv_state,
                conv_weight=inputs["conv_weight"],
                A_log=inputs["A_log"],
                a=inputs["a"],
                dt_bias=inputs["dt_bias"],
                b=inputs["b"],
                ssm_state=ssm_state,
                key_dim=key_dim,
                value_dim=value_dim,
                num_heads_qk=num_heads_qk,
                num_heads_v=num_heads_v,
                head_dim=head_dim,
                conv_bias=inputs["conv_bias"],
                activation="silu",
                conv_state_indices=inputs["conv_state_indices"],
                ssm_state_indices=inputs["ssm_state_indices"],
                use_qk_l2norm_in_kernel=True,
            )
        torch.cuda.synchronize()
        ref_time = (time.time() - start) / num_iters * 1000  # ms
        
        # ====================================================================
        # Benchmark 2: Gluon v1 (Q/K-indexed)
        # ====================================================================
        
        # Warmup
        for _ in range(3):
            _ = fused_gdn_fwd_decode_gluon(
                mixed_qkv=mixed_qkv,
                conv_state=conv_state,
                conv_weight=inputs["conv_weight"],
                A_log=inputs["A_log"],
                a=inputs["a"],
                dt_bias=inputs["dt_bias"],
                b=inputs["b"],
                ssm_state=ssm_state,
                key_dim=key_dim,
                value_dim=value_dim,
                num_heads_qk=num_heads_qk,
                num_heads_v=num_heads_v,
                head_dim=head_dim,
                conv_bias=inputs["conv_bias"],
                activation="silu",
                conv_state_indices=inputs["conv_state_indices"],
                ssm_state_indices=inputs["ssm_state_indices"],
                use_qk_l2norm_in_kernel=True,
            )
        torch.cuda.synchronize()
        
        # Benchmark
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_iters):
            _ = fused_gdn_fwd_decode_gluon(
                mixed_qkv=mixed_qkv,
                conv_state=conv_state,
                conv_weight=inputs["conv_weight"],
                A_log=inputs["A_log"],
                a=inputs["a"],
                dt_bias=inputs["dt_bias"],
                b=inputs["b"],
                ssm_state=ssm_state,
                key_dim=key_dim,
                value_dim=value_dim,
                num_heads_qk=num_heads_qk,
                num_heads_v=num_heads_v,
                head_dim=head_dim,
                conv_bias=inputs["conv_bias"],
                activation="silu",
                conv_state_indices=inputs["conv_state_indices"],
                ssm_state_indices=inputs["ssm_state_indices"],
                use_qk_l2norm_in_kernel=True,
            )
        torch.cuda.synchronize()
        gluon_v1_time = (time.time() - start) / num_iters * 1000  # ms
        
        # ====================================================================
        # Benchmark 3: Gluon v2 (V-indexed)
        # ====================================================================
        
        # Warmup
        for _ in range(3):
            _ = fused_gdn_fwd_decode_gluon_v6(
                mixed_qkv=mixed_qkv,
                conv_state=conv_state,
                conv_weight=inputs["conv_weight"],
                A_log=inputs["A_log"],
                a=inputs["a"],
                dt_bias=inputs["dt_bias"],
                b=inputs["b"],
                ssm_state=ssm_state,
                key_dim=key_dim,
                value_dim=value_dim,
                num_heads_qk=num_heads_qk,
                num_heads_v=num_heads_v,
                head_dim=head_dim,
                conv_bias=inputs["conv_bias"],
                activation="silu",
                conv_state_indices=inputs["conv_state_indices"],
                ssm_state_indices=inputs["ssm_state_indices"],
                use_qk_l2norm_in_kernel=True,
            )
        torch.cuda.synchronize()
        
        # Benchmark
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_iters):
            _ = fused_gdn_fwd_decode_gluon_v6(
                mixed_qkv=mixed_qkv,
                conv_state=conv_state,
                conv_weight=inputs["conv_weight"],
                A_log=inputs["A_log"],
                a=inputs["a"],
                dt_bias=inputs["dt_bias"],
                b=inputs["b"],
                ssm_state=ssm_state,
                key_dim=key_dim,
                value_dim=value_dim,
                num_heads_qk=num_heads_qk,
                num_heads_v=num_heads_v,
                head_dim=head_dim,
                conv_bias=inputs["conv_bias"],
                activation="silu",
                conv_state_indices=inputs["conv_state_indices"],
                ssm_state_indices=inputs["ssm_state_indices"],
                use_qk_l2norm_in_kernel=True,
            )
        torch.cuda.synchronize()
        gluon_v2_time = (time.time() - start) / num_iters * 1000  # ms
        
        # ====================================================================
        # Calculate and Display Results
        # ====================================================================
        
        speedup_v1_vs_ref = ref_time / gluon_v1_time
        speedup_v2_vs_ref = ref_time / gluon_v2_time
        speedup_v2_vs_v1 = gluon_v1_time / gluon_v2_time
        
        copy_size = get_copy_size(batch_size, head_dim, seqlen, conv_width, num_heads_v) / 1024 / 1024 / 1024 # GB

        bandwidth_ref = copy_size / (ref_time / 1000)
        bandwidth_v1 = copy_size / (gluon_v1_time / 1000)
        bandwidth_v2 = copy_size / (gluon_v2_time / 1000)

        print()
        print(f"{'='*70}")
        print(f"Three-Way Performance Comparison (batch_size={batch_size})")
        print(f"{'='*70}")
        print(f"  Reference (Python):     {ref_time:.4f} ms, {bandwidth_ref:.2f} GB/s")
        print(f"  Gluon v1 (Q/K-indexed): {gluon_v1_time:.4f} ms, {bandwidth_v1:.2f} GB/s  (vs ref: {speedup_v1_vs_ref:.2f}x)")
        print(f"  Gluon v2 (V-indexed):   {gluon_v2_time:.4f} ms, {bandwidth_v2:.2f} GB/s  (vs ref: {speedup_v2_vs_ref:.2f}x, vs v1: {speedup_v2_vs_v1:.2f}x)")
        print(f"{'='*70}")
        
        # Determine the fastest
        times = {
            "Reference": ref_time,
            "Gluon v1": gluon_v1_time,
            "Gluon v2": gluon_v2_time,
        }
        fastest = min(times, key=times.get)
        print(f"  ✓ Fastest: {fastest} ({times[fastest]:.4f} ms)")
        print(f"{'='*70}")


class TestKernelComparison:
    """Performance and accuracy comparison between v1 (Q/K-indexed) and v2 (V-indexed) kernels."""
    
    @pytest.mark.parametrize("has_initial_state", [True, False])
    @pytest.mark.parametrize("batch", [1, 4])
    @pytest.mark.parametrize("seqlen", [1, 16])
    @pytest.mark.parametrize("head_dim", [128])
    @pytest.mark.parametrize("num_heads_v", [NUM_HEADS_V_PER_GPU])
    @pytest.mark.parametrize("num_heads_qk", [NUM_HEADS_QK_PER_GPU])
    @pytest.mark.parametrize("conv_width", [4])
    def test_v1_v2_accuracy(
        self,
        has_initial_state,
        batch,
        seqlen,
        head_dim,
        num_heads_v,
        num_heads_qk,
        conv_width,
    ):
        """Test that v1 and v2 produce identical results."""
        torch.manual_seed(42)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            pytest.skip("CUDA required for Triton kernels")
        
        # Setup dimensions
        key_dim = num_heads_qk * head_dim
        value_dim = num_heads_v * head_dim
        dim = 2 * key_dim + value_dim
        
        # Create input tensors
        mixed_qkv = torch.randn(batch, dim, seqlen, device=device, dtype=torch.float32)
        conv_state = torch.randn(batch, conv_width - 1, dim, device=device, dtype=torch.float32).transpose(1, 2)
        conv_weight = torch.randn(dim, conv_width, device=device, dtype=torch.float32)
        conv_bias = torch.randn(dim, device=device, dtype=torch.float32)
        
        # Gating parameters
        A_log = torch.randn(value_dim, device=device, dtype=torch.float32)
        a = torch.randn(batch * seqlen, value_dim, device=device, dtype=torch.float32)
        dt_bias = torch.randn(value_dim, device=device, dtype=torch.float32)
        b = torch.randn(batch * seqlen, value_dim, device=device, dtype=torch.float32)
        
        # SSM state
        if has_initial_state:
            ssm_state = torch.randn(
                batch, value_dim, head_dim, head_dim, device=device, dtype=torch.float32
            )
            ssm_state_indices = torch.arange(batch, device=device, dtype=torch.int32)
        else:
            ssm_state = None
            ssm_state_indices = torch.full((batch,), -1, device=device, dtype=torch.int32)
        
        # Make copies for v2
        conv_state_v1 = conv_state.clone()
        conv_state_v2 = conv_state.clone()
        ssm_state_v1 = ssm_state.clone() if ssm_state is not None else None
        ssm_state_v2 = ssm_state.clone() if ssm_state is not None else None
        
        # Run v1 (Q/K-indexed)
        from sglang.srt.layers.attention.fla.fused_gdn_fwd_decode import (
            fused_gdn_fwd_decode,
        )
        
        output_v1 = fused_gdn_fwd_decode(
            mixed_qkv=mixed_qkv,
            conv_state=conv_state_v1,
            conv_weight=conv_weight,
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            b=b,
            ssm_state=ssm_state_v1,
            key_dim=key_dim,
            value_dim=value_dim,
            num_heads_qk=num_heads_qk,
            num_heads_v=num_heads_v,
            head_dim=head_dim,
            conv_bias=conv_bias,
            activation="silu",
            ssm_state_indices=ssm_state_indices,
        )
        
        output_v2 = fused_gdn_fwd_decode_v2(
            mixed_qkv=mixed_qkv,
            conv_state=conv_state_v2,
            conv_weight=conv_weight,
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            b=b,
            ssm_state=ssm_state_v2,
            key_dim=key_dim,
            value_dim=value_dim,
            num_heads_qk=num_heads_qk,
            num_heads_v=num_heads_v,
            head_dim=head_dim,
            conv_bias=conv_bias,
            activation="silu",
            ssm_state_indices=ssm_state_indices,
        )
        
        # Compare outputs
        max_diff = (output_v1 - output_v2).abs().max().item()
        print(f"\nOutput max diff (v1 vs v2): {max_diff}")
        assert torch.allclose(output_v1, output_v2, atol=1e-4, rtol=1e-3), \
            f"Output mismatch: max diff = {max_diff}"
        
        # Compare conv states
        max_conv_diff = (conv_state_v1 - conv_state_v2).abs().max().item()
        print(f"Conv state max diff (v1 vs v2): {max_conv_diff}")
        assert torch.allclose(conv_state_v1, conv_state_v2, atol=1e-4, rtol=1e-3), \
            f"Conv state mismatch: max diff = {max_conv_diff}"
        
        # Compare SSM states if present
        if has_initial_state:
            max_ssm_diff = (ssm_state_v1 - ssm_state_v2).abs().max().item()
            print(f"SSM state max diff (v1 vs v2): {max_ssm_diff}")
            assert torch.allclose(ssm_state_v1, ssm_state_v2, atol=1e-4, rtol=1e-3), \
                f"SSM state mismatch: max diff = {max_ssm_diff}"
        
        print(f"✓ V1 vs V2 accuracy test passed")
    
    @pytest.mark.parametrize("batch", [1, 4, 16])
    @pytest.mark.parametrize("seqlen", [1, 16, 64])
    @pytest.mark.parametrize("head_dim", [128])
    @pytest.mark.parametrize("num_heads_v", [NUM_HEADS_V_PER_GPU])
    @pytest.mark.parametrize("num_heads_qk", [NUM_HEADS_QK_PER_GPU])
    def test_v1_v2_performance(
        self,
        batch,
        seqlen,
        head_dim,
        num_heads_v,
        num_heads_qk,
    ):
        """Benchmark performance of v1 vs v2 kernels."""
        torch.manual_seed(42)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            pytest.skip("CUDA required for Triton kernels")
        
        # Setup dimensions
        key_dim = num_heads_qk * head_dim
        value_dim = num_heads_v * head_dim
        dim = 2 * key_dim + value_dim
        conv_width = 4
        
        # Create input tensors
        mixed_qkv = torch.randn(batch, dim, seqlen, device=device, dtype=torch.float32)
        conv_state = torch.randn(batch, conv_width - 1, dim, device=device, dtype=torch.float32).transpose(1, 2)
        conv_weight = torch.randn(dim, conv_width, device=device, dtype=torch.float32)
        conv_bias = torch.randn(dim, device=device, dtype=torch.float32)
        
        # Gating parameters
        A_log = torch.randn(value_dim, device=device, dtype=torch.float32)
        a = torch.randn(batch * seqlen, value_dim, device=device, dtype=torch.float32)
        dt_bias = torch.randn(value_dim, device=device, dtype=torch.float32)
        b = torch.randn(batch * seqlen, value_dim, device=device, dtype=torch.float32)
        
        # SSM state
        ssm_state = torch.randn(
            batch, value_dim, head_dim, head_dim, device=device, dtype=torch.float32
        )
        ssm_state_indices = torch.arange(batch, device=device, dtype=torch.int32)
        
        # Warmup
        for _ in range(10):
            _ = fused_gdn_fwd_decode(
                mixed_qkv=mixed_qkv,
                conv_state=conv_state,
                conv_weight=conv_weight,
                A_log=A_log,
                a=a,
                dt_bias=dt_bias,
                b=b,
                ssm_state=ssm_state,
                key_dim=key_dim,
                value_dim=value_dim,
                num_heads_qk=num_heads_qk,
                num_heads_v=num_heads_v,
                head_dim=head_dim,
                conv_bias=conv_bias,
                activation="silu",
                ssm_state_indices=ssm_state_indices,
            )
            _ = fused_gdn_fwd_decode_v2(
                mixed_qkv=mixed_qkv,
                conv_state=conv_state,
                conv_weight=conv_weight,
                A_log=A_log,
                a=a,
                dt_bias=dt_bias,
                b=b,
                ssm_state=ssm_state,
                key_dim=key_dim,
                value_dim=value_dim,
                num_heads_qk=num_heads_qk,
                num_heads_v=num_heads_v,
                head_dim=head_dim,
                conv_bias=conv_bias,
                activation="silu",
                ssm_state_indices=ssm_state_indices,
            )
        torch.cuda.synchronize()
        
        # Benchmark v1
        import time
        n_iters = 100
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_iters):
            _ = fused_gdn_fwd_decode(
                mixed_qkv=mixed_qkv,
                conv_state=conv_state,
                conv_weight=conv_weight,
                A_log=A_log,
                a=a,
                dt_bias=dt_bias,
                b=b,
                ssm_state=ssm_state,
                key_dim=key_dim,
                value_dim=value_dim,
                num_heads_qk=num_heads_qk,
                num_heads_v=num_heads_v,
                head_dim=head_dim,
                conv_bias=conv_bias,
                activation="silu",
                ssm_state_indices=ssm_state_indices,
            )
        torch.cuda.synchronize()
        time_v1 = (time.perf_counter() - start) / n_iters * 1000  # ms
        
        # Benchmark v2
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_iters):
            _ = fused_gdn_fwd_decode_v2(
                mixed_qkv=mixed_qkv,
                conv_state=conv_state,
                conv_weight=conv_weight,
                A_log=A_log,
                a=a,
                dt_bias=dt_bias,
                b=b,
                ssm_state=ssm_state,
                key_dim=key_dim,
                value_dim=value_dim,
                num_heads_qk=num_heads_qk,
                num_heads_v=num_heads_v,
                head_dim=head_dim,
                conv_bias=conv_bias,
                activation="silu",
                ssm_state_indices=ssm_state_indices,
            )
        torch.cuda.synchronize()
        time_v2 = (time.perf_counter() - start) / n_iters * 1000  # ms
        
        speedup = time_v1 / time_v2
        print(f"\n{'='*60}")
        print(f"Batch: {batch}, Seqlen: {seqlen}, Heads(QK/V): {num_heads_qk}/{num_heads_v}")
        print(f"V1 (Q/K-indexed): {time_v1:.3f} ms")
        print(f"V2 (V-indexed):   {time_v2:.3f} ms")
        print(f"Speedup (v2/v1):  {speedup:.2f}x")
        print(f"{'='*60}")


class TestGluonFusedGDNFwdDecodeV2:
    """Test suite for Gluon V2 (V-indexed) version of Fused GDN Forward Decode kernel."""
    
    @pytest.fixture
    def device(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA device not available")
        return "cuda"
    
    @pytest.fixture
    def dtype(self):
        return torch.bfloat16
    
    def create_test_inputs(
        self,
        batch_size,
        key_dim,
        value_dim,
        num_heads_qk,
        num_heads_v,
        head_dim,
        seqlen,
        conv_width,
        device,
        dtype,
        has_bias=True,
    ):
        """Create test inputs for the GDN forward decode operation."""
        assert key_dim == num_heads_qk * head_dim
        assert value_dim == num_heads_v * head_dim
        
        dim = 2 * key_dim + value_dim
        HV = num_heads_v
        
        mixed_qkv = torch.randn(batch_size, dim, seqlen, device=device, dtype=dtype)
        conv_state = torch.randn(
            batch_size + 10, conv_width - 1, dim, device=device, dtype=dtype
        ).transpose(1, 2)
        conv_weight = torch.randn(dim, conv_width, device=device, dtype=dtype)
        conv_bias = torch.randn(dim, device=device, dtype=dtype) if has_bias else None
        
        A_log = torch.randn(HV, device=device, dtype=torch.float32)
        a = torch.randn(batch_size, HV, device=device, dtype=dtype)
        dt_bias = torch.randn(HV, device=device, dtype=dtype)
        b = torch.randn(batch_size, HV, device=device, dtype=dtype)
        
        ssm_state = torch.randn(
            batch_size + 10, HV, head_dim, head_dim,
            device=device, dtype=torch.float32
        )
        
        conv_state_indices = torch.arange(batch_size, device=device, dtype=torch.int32)
        ssm_state_indices = torch.arange(batch_size, device=device, dtype=torch.int32)
        
        return {
            "mixed_qkv": mixed_qkv,
            "conv_state": conv_state,
            "conv_weight": conv_weight,
            "conv_bias": conv_bias,
            "A_log": A_log,
            "a": a,
            "dt_bias": dt_bias,
            "b": b,
            "ssm_state": ssm_state,
            "conv_state_indices": conv_state_indices,
            "ssm_state_indices": ssm_state_indices,
            "key_dim": key_dim,
            "value_dim": value_dim,
            "num_heads_qk": num_heads_qk,
            "num_heads_v": num_heads_v,
            "head_dim": head_dim,
        }
    
    @pytest.mark.parametrize("batch_size", [1])
    @pytest.mark.parametrize("num_heads_qk", [NUM_HEADS_QK_PER_GPU])
    @pytest.mark.parametrize("num_heads_v", [NUM_HEADS_V_PER_GPU])
    @pytest.mark.parametrize("head_dim", [128])
    @pytest.mark.parametrize("seqlen", [1])
    @pytest.mark.parametrize("conv_width", [4])
    @pytest.mark.parametrize("has_bias", [True])
    def test_gluon_v2_vs_reference(
        self,
        batch_size,
        num_heads_qk,
        num_heads_v,
        head_dim,
        seqlen,
        conv_width,
        has_bias,
        device,
        dtype,
    ):
        """Test that Gluon v2 kernel produces the same results as reference."""
        torch.cuda.manual_seed(42)
        key_dim = num_heads_qk * head_dim
        value_dim = num_heads_v * head_dim
        
        inputs = self.create_test_inputs(
            batch_size, key_dim, value_dim, num_heads_qk, num_heads_v, head_dim,
            seqlen, conv_width, device, dtype, has_bias
        )
        
        # Clone inputs for each run
        inputs_ref = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        inputs_gluon_v2 = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        # Run reference implementation
        output_ref = gdn_fwd_decode_ref(**inputs_ref)
        conv_state_ref = inputs_ref["conv_state"][:,:512,:]
        ssm_state_ref = inputs_ref["ssm_state"]
        
        print("gdn_fwd_decode_ref @@@@@@@@@@@@@ fused_gdn_fwd_decode_gluon_v2")

        # Run Gluon v2 implementation
        output_gluon_v2 = fused_gdn_fwd_decode_gluon_v6(**inputs_gluon_v2)
        conv_state_gluon_v2 = inputs_gluon_v2["conv_state"][:,:512,:]
        ssm_state_gluon_v2 = inputs_gluon_v2["ssm_state"]

        # print(f"{conv_state_gluon_v2.shape=}\n{conv_state_ref.shape=}")
        # print(f"{conv_state_gluon_v2=}\n{conv_state_ref=}")

        # print(f"{ssm_state_ref.shape=}\n{ssm_state_gluon_v2.shape=}")
        # print(f"{ssm_state_ref=}\n{ssm_state_gluon_v2=}")

        rtol, atol = 1e-2, 5e-2
        # Compare conv_states
        torch.testing.assert_close(conv_state_gluon_v2, conv_state_ref, rtol=rtol, atol=atol)
        torch.testing.assert_close(ssm_state_gluon_v2, ssm_state_ref, rtol=rtol, atol=atol)
        # Compare outputs
        torch.testing.assert_close(output_gluon_v2, output_ref, rtol=rtol, atol=atol)
        
        print(f"✓ Gluon v2 vs reference test passed (batch={batch_size}, seqlen={seqlen})")
    
    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("num_heads_qk", [NUM_HEADS_QK_PER_GPU])
    @pytest.mark.parametrize("num_heads_v", [NUM_HEADS_V_PER_GPU])
    @pytest.mark.parametrize("head_dim", [128])
    @pytest.mark.parametrize("seqlen", [1, 16])
    @pytest.mark.parametrize("conv_width", [4])
    @pytest.mark.parametrize("has_bias", [True])
    def test_gluon_v2_vs_triton_v2(
        self,
        batch_size,
        num_heads_qk,
        num_heads_v,
        head_dim,
        seqlen,
        conv_width,
        has_bias,
        device,
        dtype,
    ):
        """Test that Gluon v2 kernel produces the same results as Triton v2 kernel."""
        
        key_dim = num_heads_qk * head_dim
        value_dim = num_heads_v * head_dim
        
        inputs = self.create_test_inputs(
            batch_size, key_dim, value_dim, num_heads_qk, num_heads_v, head_dim,
            seqlen, conv_width, device, dtype, has_bias
        )
        
        # Clone inputs for each run
        inputs_triton_v2 = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        inputs_gluon_v2 = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Run Triton v2 implementation
        output_triton_v2 = fused_gdn_fwd_decode_v2(**inputs_triton_v2)
        conv_state_triton_v2 = inputs_triton_v2["conv_state"]
        
        # Run Gluon v2 implementation
        output_gluon_v2 = fused_gdn_fwd_decode_gluon_v2(**inputs_gluon_v2)
        conv_state_gluon_v2 = inputs_gluon_v2["conv_state"]
        
        # Compare outputs
        rtol, atol = 1e-3, 1e-3
        assert torch.allclose(output_gluon_v2, output_triton_v2, rtol=rtol, atol=atol), \
            f"Output mismatch: max diff = {(output_gluon_v2 - output_triton_v2).abs().max()}"
        
        # Compare conv_states
        assert torch.allclose(conv_state_gluon_v2, conv_state_triton_v2, rtol=rtol, atol=atol), \
            f"Conv state mismatch: max diff = {(conv_state_gluon_v2 - conv_state_triton_v2).abs().max()}"
        
        print(f"✓ Gluon v2 vs Triton v2 test passed (batch={batch_size}, seqlen={seqlen})")
    
    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("num_heads_qk", [NUM_HEADS_QK_PER_GPU])
    @pytest.mark.parametrize("num_heads_v", [NUM_HEADS_V_PER_GPU])
    @pytest.mark.parametrize("head_dim", [128])
    @pytest.mark.parametrize("seqlen", [1, 16])
    @pytest.mark.parametrize("conv_width", [4])
    @pytest.mark.parametrize("has_bias", [True])
    def test_gluon_v1_vs_v2(
        self,
        batch_size,
        num_heads_qk,
        num_heads_v,
        head_dim,
        seqlen,
        conv_width,
        has_bias,
        device,
        dtype,
    ):
        """Test that Gluon v1 and v2 produce the same results."""
        key_dim = num_heads_qk * head_dim
        value_dim = num_heads_v * head_dim
        
        inputs = self.create_test_inputs(
            batch_size, key_dim, value_dim, num_heads_qk, num_heads_v, head_dim,
            seqlen, conv_width, device, dtype, has_bias
        )
        
        # Clone inputs for each run
        inputs_gluon_v1 = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        inputs_gluon_v2 = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Run Gluon v1 implementation
        output_gluon_v1 = fused_gdn_fwd_decode_gluon(**inputs_gluon_v1)
        conv_state_gluon_v1 = inputs_gluon_v1["conv_state"]
        
        # Run Gluon v2 implementation
        output_gluon_v2 = fused_gdn_fwd_decode_gluon_v2(**inputs_gluon_v2)
        conv_state_gluon_v2 = inputs_gluon_v2["conv_state"]
        
        # Compare outputs
        rtol, atol = 1e-4, 1e-4
        assert torch.allclose(output_gluon_v2, output_gluon_v1, rtol=rtol, atol=atol), \
            f"Output mismatch: max diff = {(output_gluon_v2 - output_gluon_v1).abs().max()}"
        
        # Compare conv_states
        assert torch.allclose(conv_state_gluon_v2, conv_state_gluon_v1, rtol=rtol, atol=atol), \
            f"Conv state mismatch: max diff = {(conv_state_gluon_v2 - conv_state_gluon_v1).abs().max()}"
        
        print(f"✓ Gluon v1 vs v2 test passed (batch={batch_size}, seqlen={seqlen})")
    
    @pytest.mark.parametrize("batch", [1, 2, 4, 8, 16, 32, 64])
    @pytest.mark.parametrize("seqlen", [1])
    @pytest.mark.parametrize("head_dim", [128])
    @pytest.mark.parametrize("num_heads_v", [NUM_HEADS_V_PER_GPU])
    @pytest.mark.parametrize("num_heads_qk", [NUM_HEADS_QK_PER_GPU])
    def test_gluon_v1_v2_performance(
        self,
        batch,
        seqlen,
        head_dim,
        num_heads_v,
        num_heads_qk,
        device,
    ):
        """Benchmark performance of Gluon v1 vs v2 kernels."""
        torch.manual_seed(42)
        dtype = torch.float32
        
        # Setup dimensions
        key_dim = num_heads_qk * head_dim
        value_dim = num_heads_v * head_dim
        dim = 2 * key_dim + value_dim
        conv_width = 4
        
        # Create input tensors
        mixed_qkv = torch.randn(batch, dim, seqlen, device=device, dtype=dtype)
        conv_state = torch.randn(batch, conv_width - 1, dim, device=device, dtype=dtype).transpose(1, 2)
        conv_weight = torch.randn(dim, conv_width, device=device, dtype=dtype)
        conv_bias = torch.randn(dim, device=device, dtype=dtype)
        
        # Gating parameters
        HV = num_heads_v * head_dim
        A_log = torch.randn(HV, device=device, dtype=torch.float32)
        a = torch.randn(batch * seqlen, HV, device=device, dtype=dtype)
        dt_bias = torch.randn(HV, device=device, dtype=dtype)
        b = torch.randn(batch * seqlen, HV, device=device, dtype=dtype)
        
        # SSM state
        ssm_state = torch.randn(
            batch, HV, head_dim, head_dim, device=device, dtype=torch.float32
        )
        ssm_state_indices = torch.arange(batch, device=device, dtype=torch.int32)
        
        # Warmup
        n_warmup = 10
        for _ in range(n_warmup):
            _ = fused_gdn_fwd_decode_gluon(
                mixed_qkv=mixed_qkv,
                conv_state=conv_state,
                conv_weight=conv_weight,
                A_log=A_log,
                a=a,
                dt_bias=dt_bias,
                b=b,
                ssm_state=ssm_state,
                key_dim=key_dim,
                value_dim=value_dim,
                num_heads_qk=num_heads_qk,
                num_heads_v=num_heads_v,
                head_dim=head_dim,
                conv_bias=conv_bias,
                activation="silu",
                ssm_state_indices=ssm_state_indices,
            )
        
        # Benchmark v1
        n_iters = 100
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_iters):
            _ = fused_gdn_fwd_decode_gluon(
                mixed_qkv=mixed_qkv,
                conv_state=conv_state,
                conv_weight=conv_weight,
                A_log=A_log,
                a=a,
                dt_bias=dt_bias,
                b=b,
                ssm_state=ssm_state,
                key_dim=key_dim,
                value_dim=value_dim,
                num_heads_qk=num_heads_qk,
                num_heads_v=num_heads_v,
                head_dim=head_dim,
                conv_bias=conv_bias,
                activation="silu",
                ssm_state_indices=ssm_state_indices,
            )
        torch.cuda.synchronize()
        time_v1 = (time.perf_counter() - start) / n_iters * 1000  # ms
        
        # Benchmark v2
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_iters):
            _ = fused_gdn_fwd_decode_gluon_v2(
                mixed_qkv=mixed_qkv,
                conv_state=conv_state,
                conv_weight=conv_weight,
                A_log=A_log,
                a=a,
                dt_bias=dt_bias,
                b=b,
                ssm_state=ssm_state,
                key_dim=key_dim,
                value_dim=value_dim,
                num_heads_qk=num_heads_qk,
                num_heads_v=num_heads_v,
                head_dim=head_dim,
                conv_bias=conv_bias,
                activation="silu",
                ssm_state_indices=ssm_state_indices,
            )
        torch.cuda.synchronize()
        time_v2 = (time.perf_counter() - start) / n_iters * 1000  # ms
        
        speedup = time_v1 / time_v2
        print(f"\n{'='*70}")
        print(f"Gluon Kernel Performance Comparison")
        print(f"Batch: {batch}, Seqlen: {seqlen}, Heads(QK/V): {num_heads_qk}/{num_heads_v}")
        print(f"V1 (Q/K-indexed, batched): {time_v1:.3f} ms")
        print(f"V2 (V-indexed, simple):    {time_v2:.3f} ms")
        print(f"Speedup (v2/v1):           {speedup:.2f}x")
        print(f"{'='*70}")


class TestSplitGDNFwdDecodeV5:
    """
    Test suite for split GDN kernel (post-conv input, no conv logic).
    
    This kernel takes already conv'd mixed_qkv as input and only performs:
    1. Split into Q, K, V
    2. Apply activation
    3. Delta Rule computation
    """
    
    @pytest.fixture
    def device(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA device not available")
        return "cuda"
    
    @pytest.fixture
    def dtype(self):
        return torch.bfloat16
    
    def create_post_conv_inputs(
        self,
        batch_size,
        key_dim,
        value_dim,
        num_heads_qk,
        num_heads_v,
        head_dim,
        seqlen,
        device,
        dtype,
    ):
        """
        Create test inputs for split GDN kernel.
        The mixed_qkv here represents post-conv1d results.
        """
        assert key_dim == num_heads_qk * head_dim
        assert value_dim == num_heads_v * head_dim
        
        dim = 2 * key_dim + value_dim
        
        # This represents the post-conv mixed_qkv tensor
        mixed_qkv = torch.randn(batch_size, dim, seqlen, device=device, dtype=dtype)
        
        # Gating parameters
        # A_log and dt_bias: shape (num_heads_v,) - one per V head
        A_log = torch.randn(num_heads_v, device=device, dtype=torch.float32)
        dt_bias = torch.randn(num_heads_v, device=device, dtype=dtype)
        
        # a and b: shape (batch_size * seqlen, num_heads_v) - time-variant gating
        # Kernel accesses: p_a = a + (bos + 0) * HV + i_hv where HV = num_heads_v
        a = torch.randn(batch_size * seqlen, num_heads_v, device=device, dtype=dtype)
        b = torch.randn(batch_size * seqlen, num_heads_v, device=device, dtype=dtype)
        
        # SSM state: shape (batch + padding, num_heads_v, head_dim, head_dim)
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
    
    def split_gdn_reference(
        self,
        mixed_qkv,
        A_log,
        a,
        dt_bias,
        b,
        ssm_state,
        key_dim,
        value_dim,
        num_heads_qk,
        num_heads_v,
        head_dim,
        ssm_state_indices=None,
        scale=None,
        use_qk_l2norm_in_kernel=True,
        softplus_beta=1.0,
        softplus_threshold=20.0,
    ):
        """
        Reference implementation for split GDN (no conv, post-conv input).
        Uses the fused_sigmoid_gating_delta_rule_update function.
        """
        from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent import (
            fused_sigmoid_gating_delta_rule_update,
        )
        
        batch, dim, seqlen = mixed_qkv.shape
        
        # Split mixed_qkv into Q, K, V
        q = mixed_qkv[:, :key_dim, :]
        k = mixed_qkv[:, key_dim:2*key_dim, :]
        v = mixed_qkv[:, 2*key_dim:, :]
        
        # Apply silu activation
        q = q * torch.sigmoid(q)
        k = k * torch.sigmoid(k)
        v = v * torch.sigmoid(v)
        
        # Reshape
        q = q.view(batch, seqlen, num_heads_qk, head_dim)
        k = k.view(batch, seqlen, num_heads_qk, head_dim)
        v = v.view(batch, seqlen, num_heads_v, head_dim)
        
        # Call the delta rule update
        output = fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            softplus_beta=softplus_beta,
            softplus_threshold=softplus_threshold,
            q=q,
            k=k,
            v=v,
            b=b,
            initial_state_source=ssm_state,
            initial_state_indices=ssm_state_indices,
            scale=scale,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
        
        return output
    
    @pytest.mark.parametrize("batch_size", [64])
    @pytest.mark.parametrize("num_heads_qk", [NUM_HEADS_QK_PER_GPU])
    @pytest.mark.parametrize("num_heads_v", [NUM_HEADS_V_PER_GPU])
    @pytest.mark.parametrize("head_dim", [128])
    @pytest.mark.parametrize("seqlen", [1])
    def test_split_gdn_v5_correctness(
        self,
        batch_size,
        num_heads_qk,
        num_heads_v,
        head_dim,
        seqlen,
        device,
        dtype,
    ):
        """Test correctness of split GDN v5 kernel against reference."""
        torch.cuda.manual_seed(42)
        key_dim = num_heads_qk * head_dim
        value_dim = num_heads_v * head_dim
        
        inputs = self.create_post_conv_inputs(
            batch_size, key_dim, value_dim, num_heads_qk, num_heads_v, head_dim,
            seqlen, device, dtype
        )
        
        # Clone inputs for separate runs
        inputs_ref = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        inputs_split = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Run reference
        output_ref = self.split_gdn_reference(**inputs_ref)
        
        # Run split kernel
        output_split = split_gdn_fwd_decode_gluon_v5(
            mixed_qkv=inputs_split["mixed_qkv"],
            A_log=inputs_split["A_log"],
            a=inputs_split["a"],
            dt_bias=inputs_split["dt_bias"],
            b=inputs_split["b"],
            ssm_state=inputs_split["ssm_state"],
            key_dim=key_dim,
            value_dim=value_dim,
            num_heads_qk=num_heads_qk,
            num_heads_v=num_heads_v,
            head_dim=head_dim,
            ssm_state_indices=inputs_split["ssm_state_indices"],
        )
        
        # Compare outputs
        rtol, atol = 1e-2, 5e-2
        max_diff = (output_split - output_ref).abs().max().item()
        print(f"\n[batch={batch_size}] Output max diff: {max_diff:.6e}")
        
        assert torch.allclose(output_split, output_ref, rtol=rtol, atol=atol), \
            f"Output mismatch: max diff = {max_diff}"
        
        print(f"  ✓ Split GDN v5 correctness test passed (batch={batch_size})")
    
    @pytest.mark.parametrize("batch_size", [64])
    @pytest.mark.parametrize("seqlen", [1])
    def test_split_gdn_v5_performance(self, batch_size, seqlen, device, dtype):
        """Benchmark performance of split GDN v5 kernel."""
        num_heads_qk = NUM_HEADS_QK_PER_GPU
        num_heads_v = NUM_HEADS_V_PER_GPU
        head_dim = 128
        key_dim = num_heads_qk * head_dim
        value_dim = num_heads_v * head_dim
        
        inputs = self.create_post_conv_inputs(
            batch_size, key_dim, value_dim, num_heads_qk, num_heads_v, head_dim,
            seqlen, device, dtype
        )
        
        # Warmup for split GDN v5
        for _ in range(5):
            _ = split_gdn_fwd_decode_gluon_v5(
                mixed_qkv=inputs["mixed_qkv"],
                A_log=inputs["A_log"],
                a=inputs["a"],
                dt_bias=inputs["dt_bias"],
                b=inputs["b"],
                ssm_state=inputs["ssm_state"],
                key_dim=key_dim,
                value_dim=value_dim,
                num_heads_qk=num_heads_qk,
                num_heads_v=num_heads_v,
                head_dim=head_dim,
                ssm_state_indices=inputs["ssm_state_indices"],
            )
        torch.cuda.synchronize()
        
        # Benchmark split GDN v5
        num_iters = 100
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        torch.cuda.synchronize()
        start_event.record()
        for _ in range(num_iters):
            _ = split_gdn_fwd_decode_gluon_v5(
                mixed_qkv=inputs["mixed_qkv"],
                A_log=inputs["A_log"],
                a=inputs["a"],
                dt_bias=inputs["dt_bias"],
                b=inputs["b"],
                ssm_state=inputs["ssm_state"],
                key_dim=key_dim,
                value_dim=value_dim,
                num_heads_qk=num_heads_qk,
                num_heads_v=num_heads_v,
                head_dim=head_dim,
                ssm_state_indices=inputs["ssm_state_indices"],
            )
        end_event.record()
        torch.cuda.synchronize()
        split_time = start_event.elapsed_time(end_event) / num_iters  # ms

        print(f"\n{'='*70}")
        print(f"Split GDN v5 Performance Benchmark")
        print(f"Configuration: batch={batch_size}, seqlen={seqlen}")
        print(f"  num_heads_qk={num_heads_qk}, num_heads_v={num_heads_v}, head_dim={head_dim}")
        print(f"{'='*70}")
        print(f"  Split GDN v5 Time per iteration: {split_time:.4f} ms")
        print(f"{'='*70}")

    @pytest.mark.parametrize("batch_size", [64])
    @pytest.mark.parametrize("seqlen", [1])
    def test_fused_gdn_v5_performance(self, batch_size, seqlen, device, dtype):
        """Benchmark performance of fused GDN v5 kernel."""
        num_heads_qk = NUM_HEADS_QK_PER_GPU
        num_heads_v = NUM_HEADS_V_PER_GPU
        head_dim = 128
        key_dim = num_heads_qk * head_dim
        value_dim = num_heads_v * head_dim
        dim = 2 * key_dim + value_dim
        conv_width = 4
        
        inputs = self.create_post_conv_inputs(
            batch_size, key_dim, value_dim, num_heads_qk, num_heads_v, head_dim,
            seqlen, device, dtype
        )

        # Create additional inputs for fused kernel (with conv)
        conv_state = torch.randn(
            batch_size + 10, conv_width - 1, dim, device=device, dtype=dtype
        ).transpose(1, 2)
        conv_weight = torch.randn(dim, conv_width, device=device, dtype=dtype)
        conv_bias = torch.randn(dim, device=device, dtype=dtype)
        conv_state_indices = torch.arange(batch_size, device=device, dtype=torch.int32)
        
        # Warmup for fused GDN v5
        for _ in range(5):
             _ = fused_gdn_fwd_decode_gluon_v6(
                mixed_qkv=inputs["mixed_qkv"],
                conv_state=conv_state,
                conv_weight=conv_weight,
                A_log=inputs["A_log"],
                a=inputs["a"],
                dt_bias=inputs["dt_bias"],
                b=inputs["b"],
                ssm_state=inputs["ssm_state"],
                key_dim=key_dim,
                value_dim=value_dim,
                num_heads_qk=num_heads_qk,
                num_heads_v=num_heads_v,
                head_dim=head_dim,
                conv_bias=conv_bias,
                conv_state_indices=conv_state_indices,
                ssm_state_indices=inputs["ssm_state_indices"],
            )
        torch.cuda.synchronize()

        # Benchmark fused GDN v5
        num_iters = 100
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        torch.cuda.synchronize()
        start_event.record()
        for _ in range(num_iters):
             _ = fused_gdn_fwd_decode_gluon_v6(
                mixed_qkv=inputs["mixed_qkv"],
                conv_state=conv_state,
                conv_weight=conv_weight,
                A_log=inputs["A_log"],
                a=inputs["a"],
                dt_bias=inputs["dt_bias"],
                b=inputs["b"],
                ssm_state=inputs["ssm_state"],
                key_dim=key_dim,
                value_dim=value_dim,
                num_heads_qk=num_heads_qk,
                num_heads_v=num_heads_v,
                head_dim=head_dim,
                conv_bias=conv_bias,
                conv_state_indices=conv_state_indices,
                ssm_state_indices=inputs["ssm_state_indices"],
            )
        end_event.record()
        torch.cuda.synchronize()
        fused_time = start_event.elapsed_time(end_event) / num_iters  # ms
        
        print(f"\n{'='*70}")
        print(f"Fused GDN v5 Performance Benchmark")
        print(f"Configuration: batch={batch_size}, seqlen={seqlen}")
        print(f"  num_heads_qk={num_heads_qk}, num_heads_v={num_heads_v}, head_dim={head_dim}")
        print(f"{'='*70}")
        print(f"  Fused GDN v5 Time per iteration: {fused_time:.4f} ms")
        print(f"{'='*70}")
    
    @pytest.mark.parametrize("batch_size", [64])
    def test_split_gdn_v5_vs_fused_gdn_v5(self, batch_size, device, dtype):
        """
        Compare performance of split GDN v5 (no conv) vs fused GDN v5 (with conv).
        The split version should be faster as it skips conv computation.
        """
        num_heads_qk = NUM_HEADS_QK_PER_GPU
        num_heads_v = NUM_HEADS_V_PER_GPU
        head_dim = 128
        key_dim = num_heads_qk * head_dim
        value_dim = num_heads_v * head_dim
        dim = 2 * key_dim + value_dim
        seqlen = 1
        conv_width = 4
        
        # Create inputs for split kernel (post-conv)
        split_inputs = self.create_post_conv_inputs(
            batch_size, key_dim, value_dim, num_heads_qk, num_heads_v, head_dim,
            seqlen, device, dtype
        )
        
        # Create additional inputs for fused kernel (with conv)
        conv_state = torch.randn(
            batch_size + 10, conv_width - 1, dim, device=device, dtype=dtype
        ).transpose(1, 2)
        conv_weight = torch.randn(dim, conv_width, device=device, dtype=dtype)
        conv_bias = torch.randn(dim, device=device, dtype=dtype)
        conv_state_indices = torch.arange(batch_size, device=device, dtype=torch.int32)
        
        num_iters = 50
        
        # Benchmark split kernel
        for _ in range(3):
            _ = split_gdn_fwd_decode_gluon_v5(
                mixed_qkv=split_inputs["mixed_qkv"],
                A_log=split_inputs["A_log"],
                a=split_inputs["a"],
                dt_bias=split_inputs["dt_bias"],
                b=split_inputs["b"],
                ssm_state=split_inputs["ssm_state"],
                key_dim=key_dim,
                value_dim=value_dim,
                num_heads_qk=num_heads_qk,
                num_heads_v=num_heads_v,
                head_dim=head_dim,
                ssm_state_indices=split_inputs["ssm_state_indices"],
            )
        torch.cuda.synchronize()
        
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(num_iters):
            _ = split_gdn_fwd_decode_gluon_v5(
                mixed_qkv=split_inputs["mixed_qkv"],
                A_log=split_inputs["A_log"],
                a=split_inputs["a"],
                dt_bias=split_inputs["dt_bias"],
                b=split_inputs["b"],
                ssm_state=split_inputs["ssm_state"],
                key_dim=key_dim,
                value_dim=value_dim,
                num_heads_qk=num_heads_qk,
                num_heads_v=num_heads_v,
                head_dim=head_dim,
                ssm_state_indices=split_inputs["ssm_state_indices"],
            )
        end_event.record()
        torch.cuda.synchronize()
        split_time = start_event.elapsed_time(end_event) / num_iters
        
        # Benchmark fused kernel (v1 - which doesn't use sched_barrier)
        for _ in range(3):
            _ = fused_gdn_fwd_decode_gluon_v5(
                mixed_qkv=split_inputs["mixed_qkv"],
                conv_state=conv_state,
                conv_weight=conv_weight,
                A_log=split_inputs["A_log"],
                a=split_inputs["a"],
                dt_bias=split_inputs["dt_bias"],
                b=split_inputs["b"],
                ssm_state=split_inputs["ssm_state"],
                key_dim=key_dim,
                value_dim=value_dim,
                num_heads_qk=num_heads_qk,
                num_heads_v=num_heads_v,
                head_dim=head_dim,
                conv_bias=conv_bias,
                conv_state_indices=conv_state_indices,
                ssm_state_indices=split_inputs["ssm_state_indices"],
            )
        torch.cuda.synchronize()
        
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(num_iters):
            _ = fused_gdn_fwd_decode_gluon_v5(
                mixed_qkv=split_inputs["mixed_qkv"],
                conv_state=conv_state,
                conv_weight=conv_weight,
                A_log=split_inputs["A_log"],
                a=split_inputs["a"],
                dt_bias=split_inputs["dt_bias"],
                b=split_inputs["b"],
                ssm_state=split_inputs["ssm_state"],
                key_dim=key_dim,
                value_dim=value_dim,
                num_heads_qk=num_heads_qk,
                num_heads_v=num_heads_v,
                head_dim=head_dim,
                conv_bias=conv_bias,
                conv_state_indices=conv_state_indices,
                ssm_state_indices=split_inputs["ssm_state_indices"],
            )
        end_event.record()
        torch.cuda.synchronize()
        fused_time = start_event.elapsed_time(end_event) / num_iters
        
        speedup = fused_time / split_time
        
        print(f"\n[batch={batch_size}] Split: {split_time:.4f}ms | Fused: {fused_time:.4f}ms | Speedup: {speedup:.2f}x")


    
    @pytest.mark.parametrize("batch_size", [64])
    @pytest.mark.parametrize("seqlen", [1])
    def test_fused_gdn_performance(self, batch_size, seqlen, device, dtype):
        """Benchmark performance of fused GDN kernel."""
        num_heads_qk = NUM_HEADS_QK_PER_GPU
        num_heads_v = NUM_HEADS_V_PER_GPU
        head_dim = 128
        key_dim = num_heads_qk * head_dim
        value_dim = num_heads_v * head_dim
        dim = 2 * key_dim + value_dim
        conv_width = 4
        
        # Create inputs
        inputs = self.create_post_conv_inputs(
            batch_size, key_dim, value_dim, num_heads_qk, num_heads_v, head_dim,
            seqlen, device, dtype
        )
        
        conv_state = torch.randn(
            batch_size + 10, conv_width - 1, dim, device=device, dtype=dtype
        ).transpose(1, 2)
        conv_weight = torch.randn(dim, conv_width, device=device, dtype=dtype)
        conv_bias = torch.randn(dim, device=device, dtype=dtype)
        conv_state_indices = torch.arange(batch_size, device=device, dtype=torch.int32)
        
        # Warmup
        for _ in range(5):
            _ = fused_gdn_fwd_decode(
                mixed_qkv=inputs["mixed_qkv"],
                conv_state=conv_state,
                conv_weight=conv_weight,
                A_log=inputs["A_log"],
                a=inputs["a"],
                dt_bias=inputs["dt_bias"],
                b=inputs["b"],
                ssm_state=inputs["ssm_state"],
                key_dim=key_dim,
                value_dim=value_dim,
                num_heads_qk=num_heads_qk,
                num_heads_v=num_heads_v,
                head_dim=head_dim,
                conv_bias=conv_bias,
                conv_state_indices=conv_state_indices,
                ssm_state_indices=inputs["ssm_state_indices"],
            )
        torch.cuda.synchronize()
        
        # Benchmark
        num_iters = 100
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        torch.cuda.synchronize()
        start_event.record()
        for _ in range(num_iters):
            _ = fused_gdn_fwd_decode(
                mixed_qkv=inputs["mixed_qkv"],
                conv_state=conv_state,
                conv_weight=conv_weight,
                A_log=inputs["A_log"],
                a=inputs["a"],
                dt_bias=inputs["dt_bias"],
                b=inputs["b"],
                ssm_state=inputs["ssm_state"],
                key_dim=key_dim,
                value_dim=value_dim,
                num_heads_qk=num_heads_qk,
                num_heads_v=num_heads_v,
                head_dim=head_dim,
                conv_bias=conv_bias,
                conv_state_indices=conv_state_indices,
                ssm_state_indices=inputs["ssm_state_indices"],
            )
        end_event.record()
        torch.cuda.synchronize()
        fused_time = start_event.elapsed_time(end_event) / num_iters  # ms
        
        print(f"\n{'='*70}")
        print(f"Fused GDN Performance Benchmark")
        print(f"Configuration: batch={batch_size}, seqlen={seqlen}")
        print(f"  num_heads_qk={num_heads_qk}, num_heads_v={num_heads_v}, head_dim={head_dim}")
        print(f"{'='*70}")
        print(f"  Time per iteration: {fused_time:.4f} ms")
        print(f"{'='*70}")


class TestSplitGDNPipelined:
    """Test for the software-pipelined split GDN kernel."""
    
    @property
    def device(self):
        return "cuda"
    
    @property
    def dtype(self):
        return torch.bfloat16
    
    def create_post_conv_inputs(
        self,
        batch_size,
        key_dim,
        value_dim,
        num_heads_qk,
        num_heads_v,
        head_dim,
        seqlen,
        device,
        dtype,
    ):
        """Create test inputs for split GDN kernel."""
        assert key_dim == num_heads_qk * head_dim
        assert value_dim == num_heads_v * head_dim
        
        dim = 2 * key_dim + value_dim
        mixed_qkv = torch.randn(batch_size, dim, seqlen, device=device, dtype=dtype)
        
        A_log = torch.randn(num_heads_v, device=device, dtype=torch.float32)
        dt_bias = torch.randn(num_heads_v, device=device, dtype=dtype)
        a = torch.randn(batch_size * seqlen, num_heads_v, device=device, dtype=dtype)
        b = torch.randn(batch_size * seqlen, num_heads_v, device=device, dtype=dtype)
        
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
        }
    
    @pytest.mark.parametrize("head_dim,num_heads_v,num_heads_qk,seqlen,batch_size", [
        (128, NUM_HEADS_V_PER_GPU, NUM_HEADS_QK_PER_GPU, 1, 64),
    ])
    def test_split_gdn_v5_pipelined_correctness(self, head_dim, num_heads_v, num_heads_qk, seqlen, batch_size):
        """Test correctness of pipelined kernel against non-pipelined version."""
        from sglang.srt.layers.attention.fla.fused_gdn_fwd_decode_gluon import (
            split_gdn_fwd_decode_gluon_v5,
            split_gdn_fwd_decode_gluon_v5_pipelined,
        )
        
        key_dim = num_heads_qk * head_dim
        value_dim = num_heads_v * head_dim
        
        # Create post-conv inputs
        inputs = self.create_post_conv_inputs(
            batch_size=batch_size,
            seqlen=seqlen,
            num_heads_qk=num_heads_qk,
            num_heads_v=num_heads_v,
            head_dim=head_dim,
            key_dim=key_dim,
            value_dim=value_dim,
            device=self.device,
            dtype=self.dtype,
        )
        
        # Clone ssm_state for both tests
        ssm_state_ref = inputs["ssm_state"].clone()
        ssm_state_pipelined = inputs["ssm_state"].clone()
        
        # Run reference (non-pipelined)
        output_ref = split_gdn_fwd_decode_gluon_v5(
            mixed_qkv=inputs["mixed_qkv"],
            A_log=inputs["A_log"],
            a=inputs["a"],
            dt_bias=inputs["dt_bias"],
            b=inputs["b"],
            ssm_state=ssm_state_ref,
            key_dim=key_dim,
            value_dim=value_dim,
            num_heads_qk=num_heads_qk,
            num_heads_v=num_heads_v,
            head_dim=head_dim,
            ssm_state_indices=inputs["ssm_state_indices"],
        )
        
        # Run pipelined version
        output_pipelined = split_gdn_fwd_decode_gluon_v5_pipelined(
            mixed_qkv=inputs["mixed_qkv"],
            A_log=inputs["A_log"],
            a=inputs["a"],
            dt_bias=inputs["dt_bias"],
            b=inputs["b"],
            ssm_state=ssm_state_pipelined,
            key_dim=key_dim,
            value_dim=value_dim,
            num_heads_qk=num_heads_qk,
            num_heads_v=num_heads_v,
            head_dim=head_dim,
            ssm_state_indices=inputs["ssm_state_indices"],
        )
        
        # Compare outputs
        max_diff = (output_pipelined - output_ref).abs().max().item()
        mean_diff = (output_pipelined - output_ref).abs().mean().item()
        
        print(f"\n{'='*70}")
        print(f"Pipelined vs Non-pipelined Correctness Test")
        print(f"  batch={batch_size}, seqlen={seqlen}")
        print(f"  max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
        print(f"{'='*70}")
        
        assert max_diff < 1e-3, f"Output mismatch: max_diff={max_diff}"
        
        # Compare SSM states
        state_diff = (ssm_state_pipelined - ssm_state_ref).abs().max().item()
        print(f"  SSM state max_diff={state_diff:.6f}")
        assert state_diff < 1e-3, f"SSM state mismatch: max_diff={state_diff}"
        
        print("  ✓ Pipelined kernel correctness test PASSED!")
    
    @pytest.mark.parametrize("head_dim,num_heads_v,num_heads_qk,seqlen,batch_size", [
        (128, NUM_HEADS_V_PER_GPU, NUM_HEADS_QK_PER_GPU, 1, 64),
    ])
    def test_split_gdn_v5_pipelined_performance(self, head_dim, num_heads_v, num_heads_qk, seqlen, batch_size):
        """Benchmark pipelined vs non-pipelined kernel performance."""
        from sglang.srt.layers.attention.fla.fused_gdn_fwd_decode_gluon import (
            split_gdn_fwd_decode_gluon_v5,
            split_gdn_fwd_decode_gluon_v5_pipelined,
        )
        
        key_dim = num_heads_qk * head_dim
        value_dim = num_heads_v * head_dim
        
        inputs = self.create_post_conv_inputs(
            batch_size=batch_size,
            seqlen=seqlen,
            num_heads_qk=num_heads_qk,
            num_heads_v=num_heads_v,
            head_dim=head_dim,
            key_dim=key_dim,
            value_dim=value_dim,
            device=self.device,
            dtype=self.dtype,
        )
        
        num_warmup = 10
        num_iters = 100
        
        # Warmup and benchmark non-pipelined
        for _ in range(num_warmup):
            _ = split_gdn_fwd_decode_gluon_v5(
                mixed_qkv=inputs["mixed_qkv"],
                A_log=inputs["A_log"],
                a=inputs["a"],
                dt_bias=inputs["dt_bias"],
                b=inputs["b"],
                ssm_state=inputs["ssm_state"],
                key_dim=key_dim,
                value_dim=value_dim,
                num_heads_qk=num_heads_qk,
                num_heads_v=num_heads_v,
                head_dim=head_dim,
                ssm_state_indices=inputs["ssm_state_indices"],
            )
        torch.cuda.synchronize()
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(num_iters):
            _ = split_gdn_fwd_decode_gluon_v5(
                mixed_qkv=inputs["mixed_qkv"],
                A_log=inputs["A_log"],
                a=inputs["a"],
                dt_bias=inputs["dt_bias"],
                b=inputs["b"],
                ssm_state=inputs["ssm_state"],
                key_dim=key_dim,
                value_dim=value_dim,
                num_heads_qk=num_heads_qk,
                num_heads_v=num_heads_v,
                head_dim=head_dim,
                ssm_state_indices=inputs["ssm_state_indices"],
            )
        end.record()
        torch.cuda.synchronize()
        non_pipelined_time = start.elapsed_time(end) / num_iters * 1000  # us
        
        # Warmup and benchmark pipelined
        for _ in range(num_warmup):
            _ = split_gdn_fwd_decode_gluon_v5_pipelined(
                mixed_qkv=inputs["mixed_qkv"],
                A_log=inputs["A_log"],
                a=inputs["a"],
                dt_bias=inputs["dt_bias"],
                b=inputs["b"],
                ssm_state=inputs["ssm_state"],
                key_dim=key_dim,
                value_dim=value_dim,
                num_heads_qk=num_heads_qk,
                num_heads_v=num_heads_v,
                head_dim=head_dim,
                ssm_state_indices=inputs["ssm_state_indices"],
            )
        torch.cuda.synchronize()
        
        start.record()
        for _ in range(num_iters):
            _ = split_gdn_fwd_decode_gluon_v5_pipelined(
                mixed_qkv=inputs["mixed_qkv"],
                A_log=inputs["A_log"],
                a=inputs["a"],
                dt_bias=inputs["dt_bias"],
                b=inputs["b"],
                ssm_state=inputs["ssm_state"],
                key_dim=key_dim,
                value_dim=value_dim,
                num_heads_qk=num_heads_qk,
                num_heads_v=num_heads_v,
                head_dim=head_dim,
                ssm_state_indices=inputs["ssm_state_indices"],
            )
        end.record()
        torch.cuda.synchronize()
        pipelined_time = start.elapsed_time(end) / num_iters * 1000  # us
        
        speedup = non_pipelined_time / pipelined_time
        
        print(f"\n{'='*70}")
        print(f"Pipelined Performance Benchmark: batch={batch_size}, seqlen={seqlen}")
        print(f"{'='*70}")
        print(f"  Non-pipelined: {non_pipelined_time:.2f} us")
        print(f"  Pipelined:     {pipelined_time:.2f} us")
        print(f"  Speedup:       {speedup:.2f}x")
        print(f"{'='*70}")
    
    @pytest.mark.parametrize("head_dim,num_heads_v,num_heads_qk,seqlen,batch_size", [
        (128, NUM_HEADS_V_PER_GPU, NUM_HEADS_QK_PER_GPU, 1, 64),
    ])
    def test_split_gdn_v5_pipelined_v2_correctness(self, head_dim, num_heads_v, num_heads_qk, seqlen, batch_size):
        """Test correctness of pipelined v2 (store-compute overlap) kernel."""
        from sglang.srt.layers.attention.fla.fused_gdn_fwd_decode_gluon import (
            split_gdn_fwd_decode_gluon_v5,
            split_gdn_fwd_decode_gluon_v5_pipelined_v2,
        )
        
        key_dim = num_heads_qk * head_dim
        value_dim = num_heads_v * head_dim
        
        inputs = self.create_post_conv_inputs(
            batch_size=batch_size, seqlen=seqlen, num_heads_qk=num_heads_qk,
            num_heads_v=num_heads_v, head_dim=head_dim, key_dim=key_dim,
            value_dim=value_dim, device=self.device, dtype=self.dtype,
        )
        
        ssm_state_ref = inputs["ssm_state"].clone()
        ssm_state_v2 = inputs["ssm_state"].clone()
        
        output_ref = split_gdn_fwd_decode_gluon_v5(
            mixed_qkv=inputs["mixed_qkv"], A_log=inputs["A_log"], a=inputs["a"],
            dt_bias=inputs["dt_bias"], b=inputs["b"], ssm_state=ssm_state_ref,
            key_dim=key_dim, value_dim=value_dim, num_heads_qk=num_heads_qk,
            num_heads_v=num_heads_v, head_dim=head_dim,
            ssm_state_indices=inputs["ssm_state_indices"],
        )
        
        output_v2 = split_gdn_fwd_decode_gluon_v5_pipelined_v2(
            mixed_qkv=inputs["mixed_qkv"], A_log=inputs["A_log"], a=inputs["a"],
            dt_bias=inputs["dt_bias"], b=inputs["b"], ssm_state=ssm_state_v2,
            key_dim=key_dim, value_dim=value_dim, num_heads_qk=num_heads_qk,
            num_heads_v=num_heads_v, head_dim=head_dim,
            ssm_state_indices=inputs["ssm_state_indices"],
        )
        
        max_diff = (output_v2 - output_ref).abs().max().item()
        state_diff = (ssm_state_v2 - ssm_state_ref).abs().max().item()
        
        print(f"\n{'='*70}")
        print(f"Pipelined v2 (Store-Compute Overlap) Correctness Test")
        print(f"  batch={batch_size}, seqlen={seqlen}")
        print(f"  output max_diff={max_diff:.6f}, state max_diff={state_diff:.6f}")
        print(f"{'='*70}")
        
        assert max_diff < 1e-3, f"Output mismatch: max_diff={max_diff}"
        assert state_diff < 1e-3, f"State mismatch: max_diff={state_diff}"
        print("  ✓ Pipelined v2 correctness test PASSED!")
    
    @pytest.mark.parametrize("head_dim,num_heads_v,num_heads_qk,seqlen,batch_size", [
        (128, NUM_HEADS_V_PER_GPU, NUM_HEADS_QK_PER_GPU, 1, 64),
    ])
    def test_split_gdn_v5_pipelined_v2_performance(self, head_dim, num_heads_v, num_heads_qk, seqlen, batch_size):
        """Benchmark pipelined v2 (store-compute overlap) vs other versions."""
        from sglang.srt.layers.attention.fla.fused_gdn_fwd_decode_gluon import (
            split_gdn_fwd_decode_gluon_v5,
            split_gdn_fwd_decode_gluon_v5_pipelined,
            split_gdn_fwd_decode_gluon_v5_pipelined_v2,
        )
        
        key_dim = num_heads_qk * head_dim
        value_dim = num_heads_v * head_dim
        
        inputs = self.create_post_conv_inputs(
            batch_size=batch_size, seqlen=seqlen, num_heads_qk=num_heads_qk,
            num_heads_v=num_heads_v, head_dim=head_dim, key_dim=key_dim,
            value_dim=value_dim, device=self.device, dtype=self.dtype,
        )
        
        num_warmup, num_iters = 10, 100
        
        def bench(fn):
            for _ in range(num_warmup):
                _ = fn(mixed_qkv=inputs["mixed_qkv"], A_log=inputs["A_log"], a=inputs["a"],
                    dt_bias=inputs["dt_bias"], b=inputs["b"], ssm_state=inputs["ssm_state"],
                    key_dim=key_dim, value_dim=value_dim, num_heads_qk=num_heads_qk,
                    num_heads_v=num_heads_v, head_dim=head_dim,
                    ssm_state_indices=inputs["ssm_state_indices"])
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(num_iters):
                _ = fn(mixed_qkv=inputs["mixed_qkv"], A_log=inputs["A_log"], a=inputs["a"],
                    dt_bias=inputs["dt_bias"], b=inputs["b"], ssm_state=inputs["ssm_state"],
                    key_dim=key_dim, value_dim=value_dim, num_heads_qk=num_heads_qk,
                    num_heads_v=num_heads_v, head_dim=head_dim,
                    ssm_state_indices=inputs["ssm_state_indices"])
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end) / num_iters * 1000
        
        t_ref = bench(split_gdn_fwd_decode_gluon_v5)
        t_v1 = bench(split_gdn_fwd_decode_gluon_v5_pipelined)
        t_v2 = bench(split_gdn_fwd_decode_gluon_v5_pipelined_v2)
        
        print(f"\n{'='*70}")
        print(f"Pipelined v2 Performance: batch={batch_size}, seqlen={seqlen}")
        print(f"{'='*70}")
        print(f"  Non-pipelined:      {t_ref:.2f} us")
        print(f"  Pipelined v1:       {t_v1:.2f} us (speedup: {t_ref/t_v1:.2f}x)")
        print(f"  Pipelined v2:       {t_v2:.2f} us (speedup: {t_ref/t_v2:.2f}x)")
        print(f"{'='*70}")
    
    @pytest.mark.parametrize("head_dim,num_heads_v,num_heads_qk,seqlen,batch_size", [
        (128, NUM_HEADS_V_PER_GPU, NUM_HEADS_QK_PER_GPU, 1, 64),
    ])
    def test_split_gdn_v5_pipelined_v2_vtile64_correctness(self, head_dim, num_heads_v, num_heads_qk, seqlen, batch_size):
        """Test correctness of pipelined v2 with V-tiling (128x64) kernel."""
        from sglang.srt.layers.attention.fla.fused_gdn_fwd_decode_gluon import (
            split_gdn_fwd_decode_gluon_v5,
            split_gdn_fwd_decode_gluon_v5_pipelined_v2_vtile64,
        )
        
        key_dim = num_heads_qk * head_dim
        value_dim = num_heads_v * head_dim
        
        inputs = self.create_post_conv_inputs(
            batch_size=batch_size, seqlen=seqlen, num_heads_qk=num_heads_qk,
            num_heads_v=num_heads_v, head_dim=head_dim, key_dim=key_dim,
            value_dim=value_dim, device=self.device, dtype=self.dtype,
        )
        
        ssm_state_ref = inputs["ssm_state"].clone()
        ssm_state_vtile64 = inputs["ssm_state"].clone()
        
        output_ref = split_gdn_fwd_decode_gluon_v5(
            mixed_qkv=inputs["mixed_qkv"], A_log=inputs["A_log"], a=inputs["a"],
            dt_bias=inputs["dt_bias"], b=inputs["b"], ssm_state=ssm_state_ref,
            key_dim=key_dim, value_dim=value_dim, num_heads_qk=num_heads_qk,
            num_heads_v=num_heads_v, head_dim=head_dim,
            ssm_state_indices=inputs["ssm_state_indices"],
        )
        
        output_vtile64 = split_gdn_fwd_decode_gluon_v5_pipelined_v2_vtile64(
            mixed_qkv=inputs["mixed_qkv"], A_log=inputs["A_log"], a=inputs["a"],
            dt_bias=inputs["dt_bias"], b=inputs["b"], ssm_state=ssm_state_vtile64,
            key_dim=key_dim, value_dim=value_dim, num_heads_qk=num_heads_qk,
            num_heads_v=num_heads_v, head_dim=head_dim,
            ssm_state_indices=inputs["ssm_state_indices"],
        )
        
        max_diff = (output_vtile64 - output_ref).abs().max().item()
        state_diff = (ssm_state_vtile64 - ssm_state_ref).abs().max().item()
        
        print(f"\n{'='*70}")
        print(f"Pipelined v2 V-Tiling (128x64) Correctness Test")
        print(f"  batch={batch_size}, seqlen={seqlen}")
        print(f"  output max_diff={max_diff:.6f}, state max_diff={state_diff:.6f}")
        print(f"{'='*70}")
        
        assert max_diff < 1e-3, f"Output mismatch: max_diff={max_diff}"
        assert state_diff < 1e-3, f"State mismatch: max_diff={state_diff}"
        print("  V-Tiling correctness test PASSED!")
    
    @pytest.mark.parametrize("head_dim,num_heads_v,num_heads_qk,seqlen,batch_size", [
        (128, NUM_HEADS_V_PER_GPU, NUM_HEADS_QK_PER_GPU, 1, 64),
    ])
    def test_split_gdn_v5_pipelined_v2_vtile64_performance(self, head_dim, num_heads_v, num_heads_qk, seqlen, batch_size):
        """Benchmark pipelined v2 with V-tiling (BV=64) vs other versions."""
        from sglang.srt.layers.attention.fla.fused_gdn_fwd_decode_gluon import (
            split_gdn_fwd_decode_gluon_v5,
            split_gdn_fwd_decode_gluon_v5_pipelined,
            split_gdn_fwd_decode_gluon_v5_pipelined_v2,
            split_gdn_fwd_decode_gluon_v5_pipelined_v2_vtile64,
        )
        
        key_dim = num_heads_qk * head_dim
        value_dim = num_heads_v * head_dim
        
        inputs = self.create_post_conv_inputs(
            batch_size=batch_size, seqlen=seqlen, num_heads_qk=num_heads_qk,
            num_heads_v=num_heads_v, head_dim=head_dim, key_dim=key_dim,
            value_dim=value_dim, device=self.device, dtype=self.dtype,
        )
        
        num_warmup, num_iters = 10, 100
        
        def bench(fn):
            for _ in range(num_warmup):
                _ = fn(mixed_qkv=inputs["mixed_qkv"], A_log=inputs["A_log"], a=inputs["a"],
                    dt_bias=inputs["dt_bias"], b=inputs["b"], ssm_state=inputs["ssm_state"],
                    key_dim=key_dim, value_dim=value_dim, num_heads_qk=num_heads_qk,
                    num_heads_v=num_heads_v, head_dim=head_dim,
                    ssm_state_indices=inputs["ssm_state_indices"])
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(num_iters):
                _ = fn(mixed_qkv=inputs["mixed_qkv"], A_log=inputs["A_log"], a=inputs["a"],
                    dt_bias=inputs["dt_bias"], b=inputs["b"], ssm_state=inputs["ssm_state"],
                    key_dim=key_dim, value_dim=value_dim, num_heads_qk=num_heads_qk,
                    num_heads_v=num_heads_v, head_dim=head_dim,
                    ssm_state_indices=inputs["ssm_state_indices"])
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end) / num_iters * 1000
        
        t_ref = bench(split_gdn_fwd_decode_gluon_v5)
        t_v1 = bench(split_gdn_fwd_decode_gluon_v5_pipelined)
        t_v2 = bench(split_gdn_fwd_decode_gluon_v5_pipelined_v2)
        t_vtile64 = bench(split_gdn_fwd_decode_gluon_v5_pipelined_v2_vtile64)
        
        print(f"\n{'='*70}")
        print(f"VTile64 Performance: batch={batch_size}, seqlen={seqlen}")
        print(f"{'='*70}")
        print(f"  Non-pipelined (128x128):   {t_ref:.2f} us")
        print(f"  Pipelined v1 (128x128):    {t_v1:.2f} us (speedup: {t_ref/t_v1:.2f}x)")
        print(f"  Pipelined v2 (128x128):    {t_v2:.2f} us (speedup: {t_ref/t_v2:.2f}x)")
        print(f"  VTile64 (128x64, 160 blk): {t_vtile64:.2f} us (speedup: {t_ref/t_vtile64:.2f}x)")
        print(f"{'='*70}")
        print(f"  VTile64 vs v2 speedup:     {t_v2/t_vtile64:.2f}x")
        print(f"{'='*70}")
    
    @pytest.mark.parametrize("head_dim,num_heads_v,num_heads_qk,seqlen,batch_size", [
        (128, NUM_HEADS_V_PER_GPU, NUM_HEADS_QK_PER_GPU, 1, 64),
    ])
    def test_split_gdn_v5_pipelined_v2_vtile32_correctness(self, head_dim, num_heads_v, num_heads_qk, seqlen, batch_size):
        """Test correctness of vtile32 (BV=32, 320 blocks)."""
        from sglang.srt.layers.attention.fla.fused_gdn_fwd_decode_gluon import (
            split_gdn_fwd_decode_gluon_v5,
            split_gdn_fwd_decode_gluon_v5_pipelined_v2_vtile32,
        )
        
        key_dim = num_heads_qk * head_dim
        value_dim = num_heads_v * head_dim
        
        inputs = self.create_post_conv_inputs(
            batch_size=batch_size, seqlen=seqlen, num_heads_qk=num_heads_qk,
            num_heads_v=num_heads_v, head_dim=head_dim, key_dim=key_dim,
            value_dim=value_dim, device=self.device, dtype=self.dtype,
        )
        
        ssm_state_ref = inputs["ssm_state"].clone()
        ssm_state_vtile32 = inputs["ssm_state"].clone()
        
        output_ref = split_gdn_fwd_decode_gluon_v5(
            mixed_qkv=inputs["mixed_qkv"], A_log=inputs["A_log"], a=inputs["a"],
            dt_bias=inputs["dt_bias"], b=inputs["b"], ssm_state=ssm_state_ref,
            key_dim=key_dim, value_dim=value_dim, num_heads_qk=num_heads_qk,
            num_heads_v=num_heads_v, head_dim=head_dim,
            ssm_state_indices=inputs["ssm_state_indices"],
        )
        
        output_vtile32 = split_gdn_fwd_decode_gluon_v5_pipelined_v2_vtile32(
            mixed_qkv=inputs["mixed_qkv"], A_log=inputs["A_log"], a=inputs["a"],
            dt_bias=inputs["dt_bias"], b=inputs["b"], ssm_state=ssm_state_vtile32,
            key_dim=key_dim, value_dim=value_dim, num_heads_qk=num_heads_qk,
            num_heads_v=num_heads_v, head_dim=head_dim,
            ssm_state_indices=inputs["ssm_state_indices"],
        )
        
        max_diff = (output_ref - output_vtile32).abs().max().item()
        state_diff = (ssm_state_ref - ssm_state_vtile32).abs().max().item()
        
        print(f"\n{'='*70}")
        print(f"VTile32 (BV=32, 320 blocks) Correctness Test")
        print(f"  batch={batch_size}, seqlen={seqlen}")
        print(f"  output max_diff={max_diff:.6f}, state max_diff={state_diff:.6f}")
        print(f"{'='*70}")
        
        assert max_diff < 1e-3, f"Output mismatch: max_diff={max_diff}"
        assert state_diff < 1e-3, f"State mismatch: max_diff={state_diff}"
        print("  VTile32 correctness test PASSED!")
    
    @pytest.mark.parametrize("head_dim,num_heads_v,num_heads_qk,seqlen,batch_size", [
        (128, NUM_HEADS_V_PER_GPU, NUM_HEADS_QK_PER_GPU, 1, 64),
        # (128, NUM_HEADS_V_PER_GPU, NUM_HEADS_QK_PER_GPU, 1, 128),
    ])
    def test_split_gdn_v5_pipelined_v2_vtile32_performance(self, head_dim, num_heads_v, num_heads_qk, seqlen, batch_size):
        """Benchmark vtile32 (BV=32, 320 blocks) vs vtile64 (BV=64, 160 blocks)."""
        from sglang.srt.layers.attention.fla.fused_gdn_fwd_decode_gluon import (
            split_gdn_fwd_decode_gluon_v5,
            split_gdn_fwd_decode_gluon_v5_pipelined,
            split_gdn_fwd_decode_gluon_v5_pipelined_v2,
            split_gdn_fwd_decode_gluon_v5_pipelined_v2_vtile64,
            split_gdn_fwd_decode_gluon_v5_pipelined_v2_vtile32,
        )
        
        key_dim = num_heads_qk * head_dim
        value_dim = num_heads_v * head_dim
        
        inputs = self.create_post_conv_inputs(
            batch_size=batch_size, seqlen=seqlen, num_heads_qk=num_heads_qk,
            num_heads_v=num_heads_v, head_dim=head_dim, key_dim=key_dim,
            value_dim=value_dim, device=self.device, dtype=self.dtype,
        )
        
        num_warmup, num_iters = 10, 1000
        
        def bench(fn):
            for _ in range(num_warmup):
                _ = fn(mixed_qkv=inputs["mixed_qkv"], A_log=inputs["A_log"], a=inputs["a"],
                    dt_bias=inputs["dt_bias"], b=inputs["b"], ssm_state=inputs["ssm_state"],
                    key_dim=key_dim, value_dim=value_dim, num_heads_qk=num_heads_qk,
                    num_heads_v=num_heads_v, head_dim=head_dim,
                    ssm_state_indices=inputs["ssm_state_indices"])
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(num_iters):
                _ = fn(mixed_qkv=inputs["mixed_qkv"], A_log=inputs["A_log"], a=inputs["a"],
                    dt_bias=inputs["dt_bias"], b=inputs["b"], ssm_state=inputs["ssm_state"],
                    key_dim=key_dim, value_dim=value_dim, num_heads_qk=num_heads_qk,
                    num_heads_v=num_heads_v, head_dim=head_dim,
                    ssm_state_indices=inputs["ssm_state_indices"])
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end) / num_iters * 1000
        
        t_ref = bench(split_gdn_fwd_decode_gluon_v5)
        t_pipelined = bench(split_gdn_fwd_decode_gluon_v5_pipelined)
        t_v2 = bench(split_gdn_fwd_decode_gluon_v5_pipelined_v2)
        t_vtile64 = bench(split_gdn_fwd_decode_gluon_v5_pipelined_v2_vtile64)
        t_vtile32 = bench(split_gdn_fwd_decode_gluon_v5_pipelined_v2_vtile32)
        
        print(f"\n{'='*70}")
        print(f"VTile32 Performance: batch={batch_size}, seqlen={seqlen}")
        print(f"{'='*70}")
        print(f"  Non-pipelined (128x128):    {t_ref:.2f} us")
        print(f"  Pipelined (128x128):        {t_pipelined:.2f} us (speedup: {t_ref/t_pipelined:.2f}x)")
        print(f"  Pipelined v2 (128x128):     {t_v2:.2f} us (speedup: {t_ref/t_v2:.2f}x)")
        print(f"  VTile64 (128x64, 160 blk):  {t_vtile64:.2f} us (speedup: {t_ref/t_vtile64:.2f}x)")
        print(f"  VTile32 (128x32, 320 blk):  {t_vtile32:.2f} us (speedup: {t_ref/t_vtile32:.2f}x)")
        print(f"{'='*70}")
        print(f"  VTile32 vs VTile64 speedup: {t_vtile64/t_vtile32:.2f}x")
        print(f"{'='*70}")
    
    @pytest.mark.parametrize("head_dim,num_heads_v,num_heads_qk,seqlen,batch_size", [
        (128, NUM_HEADS_V_PER_GPU, NUM_HEADS_QK_PER_GPU, 1, 64),
    ])
    def test_split_gdn_v5_pipelined_v2_vtile_inloop_correctness(self, head_dim, num_heads_v, num_heads_qk, seqlen, batch_size):
        """Test correctness of vtile_inloop (persistent kernel with internal V-tile loop)."""
        from sglang.srt.layers.attention.fla.fused_gdn_fwd_decode_gluon import (
            split_gdn_fwd_decode_gluon_v5,
            split_gdn_fwd_decode_gluon_v5_pipelined_v2_vtile_inloop,
        )
        
        key_dim = num_heads_qk * head_dim
        value_dim = num_heads_v * head_dim
        
        inputs = self.create_post_conv_inputs(
            batch_size=batch_size, seqlen=seqlen, num_heads_qk=num_heads_qk,
            num_heads_v=num_heads_v, head_dim=head_dim, key_dim=key_dim,
            value_dim=value_dim, device=self.device, dtype=self.dtype,
        )
        
        ssm_state_ref = inputs["ssm_state"].clone()
        ssm_state_v3 = inputs["ssm_state"].clone()
        
        output_ref = split_gdn_fwd_decode_gluon_v5(
            mixed_qkv=inputs["mixed_qkv"], A_log=inputs["A_log"], a=inputs["a"],
            dt_bias=inputs["dt_bias"], b=inputs["b"], ssm_state=ssm_state_ref,
            key_dim=key_dim, value_dim=value_dim, num_heads_qk=num_heads_qk,
            num_heads_v=num_heads_v, head_dim=head_dim,
            ssm_state_indices=inputs["ssm_state_indices"],
        )
        
        output_v3 = split_gdn_fwd_decode_gluon_v5_pipelined_v2_vtile_inloop(
            mixed_qkv=inputs["mixed_qkv"], A_log=inputs["A_log"], a=inputs["a"],
            dt_bias=inputs["dt_bias"], b=inputs["b"], ssm_state=ssm_state_v3,
            key_dim=key_dim, value_dim=value_dim, num_heads_qk=num_heads_qk,
            num_heads_v=num_heads_v, head_dim=head_dim,
            ssm_state_indices=inputs["ssm_state_indices"],
        )
        
        max_diff = (output_v3 - output_ref).abs().max().item()
        state_diff = (ssm_state_v3 - ssm_state_ref).abs().max().item()
        
        print(f"\n{'='*70}")
        print(f"VTile InLoop (Persistent + Internal Loop) Correctness Test")
        print(f"  batch={batch_size}, seqlen={seqlen}")
        print(f"  output max_diff={max_diff:.6f}, state max_diff={state_diff:.6f}")
        print(f"{'='*70}")
        
        assert max_diff < 1e-3, f"Output mismatch: max_diff={max_diff}"
        assert state_diff < 1e-3, f"State mismatch: max_diff={state_diff}"
        print("  VTile InLoop correctness test PASSED!")
    
    @pytest.mark.parametrize("head_dim,num_heads_v,num_heads_qk,seqlen,batch_size", [
        (128, NUM_HEADS_V_PER_GPU, NUM_HEADS_QK_PER_GPU, 1, 64),
        (128, NUM_HEADS_V_PER_GPU, NUM_HEADS_QK_PER_GPU, 1, 128),
    ])
    def test_split_gdn_v5_pipelined_v2_vtile_inloop_performance(self, head_dim, num_heads_v, num_heads_qk, seqlen, batch_size):
        """Benchmark vtile_inloop (persistent + internal loop) vs other versions."""
        from sglang.srt.layers.attention.fla.fused_gdn_fwd_decode_gluon import (
            split_gdn_fwd_decode_gluon_v5,
            split_gdn_fwd_decode_gluon_v5_pipelined_v2,
            split_gdn_fwd_decode_gluon_v5_pipelined_v2_vtile64,
            split_gdn_fwd_decode_gluon_v5_pipelined_v2_vtile_inloop,
        )
        
        key_dim = num_heads_qk * head_dim
        value_dim = num_heads_v * head_dim
        
        inputs = self.create_post_conv_inputs(
            batch_size=batch_size, seqlen=seqlen, num_heads_qk=num_heads_qk,
            num_heads_v=num_heads_v, head_dim=head_dim, key_dim=key_dim,
            value_dim=value_dim, device=self.device, dtype=self.dtype,
        )
        
        num_warmup, num_iters = 10, 100
        
        def bench(fn):
            for _ in range(num_warmup):
                _ = fn(mixed_qkv=inputs["mixed_qkv"], A_log=inputs["A_log"], a=inputs["a"],
                    dt_bias=inputs["dt_bias"], b=inputs["b"], ssm_state=inputs["ssm_state"],
                    key_dim=key_dim, value_dim=value_dim, num_heads_qk=num_heads_qk,
                    num_heads_v=num_heads_v, head_dim=head_dim,
                    ssm_state_indices=inputs["ssm_state_indices"])
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(num_iters):
                _ = fn(mixed_qkv=inputs["mixed_qkv"], A_log=inputs["A_log"], a=inputs["a"],
                    dt_bias=inputs["dt_bias"], b=inputs["b"], ssm_state=inputs["ssm_state"],
                    key_dim=key_dim, value_dim=value_dim, num_heads_qk=num_heads_qk,
                    num_heads_v=num_heads_v, head_dim=head_dim,
                    ssm_state_indices=inputs["ssm_state_indices"])
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end) / num_iters * 1000
        
        t_ref = bench(split_gdn_fwd_decode_gluon_v5)
        t_v2 = bench(split_gdn_fwd_decode_gluon_v5_pipelined_v2)
        t_vtile64 = bench(split_gdn_fwd_decode_gluon_v5_pipelined_v2_vtile64)
        t_vtile_inloop = bench(split_gdn_fwd_decode_gluon_v5_pipelined_v2_vtile_inloop)
        
        print(f"\n{'='*70}")
        print(f"VTile InLoop Performance: batch={batch_size}, seqlen={seqlen}")
        print(f"{'='*70}")
        print(f"  Non-pipelined (128x128):         {t_ref:.2f} us")
        print(f"  Pipelined v2 (128x128):          {t_v2:.2f} us (speedup: {t_ref/t_v2:.2f}x)")
        print(f"  VTile64 (160 blk):               {t_vtile64:.2f} us (speedup: {t_ref/t_vtile64:.2f}x)")
        print(f"  VTile InLoop (80 blk, internal): {t_vtile_inloop:.2f} us (speedup: {t_ref/t_vtile_inloop:.2f}x)")
        print(f"{'='*70}")
        print(f"  VTile InLoop vs VTile64 speedup: {t_vtile64/t_vtile_inloop:.2f}x")
        print(f"{'='*70}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

