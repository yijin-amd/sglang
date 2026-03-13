"""
Unit tests for the HIP implementation of Fused Split GDR (Gating Delta Rule) Update Kernel.

Tests correctness of the HIP fused_split_gdr_update kernel by comparing its
output against a pure PyTorch CPU reference implementation.

The test logic mirrors TestFusedSplitGDRUpdateOpt.test_split_gdr_v3_correctness
from test_fused_sigmoid_gating_recurrent.py, but uses torch CPU as ground truth.

Qwen3Next Linear Attention Config (per TP4 shard):
    - linear_key_head_dim:   128
    - linear_value_head_dim: 128
    - linear_num_key_heads:  4
    - linear_num_value_heads: 8

State Layout:
    The HIP kernel uses a SWIZZLED state layout (N_states, HV, K/4, V, 4)
    for efficient float4 vectorized loads with cross-thread coalescing.
    
    Standard layout: (N_states, HV, K, V)
    Swizzled layout: (N_states, HV, K/4, V, 4)
    
    Conversion:
        swizzled = standard.reshape(N, HV, K//4, 4, V).permute(0,1,2,4,3).contiguous()
        standard = swizzled.permute(0,1,2,4,3).reshape(N, HV, K, V).contiguous()
"""

import os
import pytest
import torch
import math


# ---------------------------------------------------------------------------
# State layout conversion utilities
# ---------------------------------------------------------------------------

def to_swizzled_layout(state: torch.Tensor) -> torch.Tensor:
    """
    Convert state from standard (N, HV, K, V) to swizzled (N, HV, K/4, V, 4) layout.
    
    Swizzled layout enables float4 vectorized loads with cross-thread coalescing:
    - Each thread loads 4 consecutive K values as float4
    - All 64 threads access consecutive addresses (1024 bytes per load)
    
    Memory layout transformation:
        Standard:  h[n, hv, k, v] at address n*HV*K*V + hv*K*V + k*V + v
        Swizzled:  h[n, hv, kg, v, k4] at address n*HV*KG*V*4 + hv*KG*V*4 + kg*V*4 + v*4 + k4
                   where kg = k // 4, k4 = k % 4
    """
    N, HV, K, V = state.shape
    assert K % 4 == 0, f"K ({K}) must be divisible by 4 for swizzled layout"
    
    # Reshape: (N, HV, K, V) -> (N, HV, K/4, 4, V)
    state = state.reshape(N, HV, K // 4, 4, V)
    # Permute: (N, HV, K/4, 4, V) -> (N, HV, K/4, V, 4)
    state = state.permute(0, 1, 2, 4, 3)
    # Make contiguous
    return state.contiguous()


def from_swizzled_layout(state: torch.Tensor) -> torch.Tensor:
    """
    Convert state from swizzled (N, HV, K/4, V, 4) to standard (N, HV, K, V) layout.
    """
    N, HV, K4, V, four = state.shape
    assert four == 4, f"Last dimension must be 4, got {four}"
    K = K4 * 4
    
    # Permute: (N, HV, K/4, V, 4) -> (N, HV, K/4, 4, V)
    state = state.permute(0, 1, 2, 4, 3)
    # Reshape: (N, HV, K/4, 4, V) -> (N, HV, K, V)
    state = state.reshape(N, HV, K, V)
    # Make contiguous
    return state.contiguous()


def to_vsplit_layout(state: torch.Tensor) -> torch.Tensor:
    """
    Convert state from standard (N, HV, K, V) to vsplit (N, HV, V/4, K, 4) layout.
    
    Vsplit layout groups 4 contiguous V values together:
    - Each float4 = 4 V values for one K position
    - Optimized for the vsplit kernel's [8,8] thread layout
    """
    N, HV, K, V = state.shape
    assert V % 4 == 0, f"V ({V}) must be divisible by 4 for vsplit layout"
    return state.reshape(N, HV, K, V // 4, 4).permute(0, 1, 3, 2, 4).contiguous()


def from_vsplit_layout(state: torch.Tensor) -> torch.Tensor:
    """
    Convert state from vsplit (N, HV, V/4, K, 4) to standard (N, HV, K, V) layout.
    """
    N, HV, V4, K, four = state.shape
    assert four == 4, f"Last dimension must be 4, got {four}"
    V = V4 * 4
    return state.permute(0, 1, 3, 2, 4).reshape(N, HV, K, V).contiguous()

# ---------------------------------------------------------------------------
# Compile and load the HIP extension at import time
# ---------------------------------------------------------------------------
from torch.utils.cpp_extension import load

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_HIP_SRC = os.path.join(
    _THIS_DIR,
    "..", "..", "..", "..", "..",  # navigate to sglang/
    "python", "sglang", "srt", "layers", "attention", "fla",
    "split_gdr_decode_hip.hip",
)
_HIP_SRC = os.path.normpath(_HIP_SRC)

# Lazy-load so the module is compiled only when a test actually runs.
_split_gdr_hip = None


def _get_hip_module():
    """JIT-compile and cache the HIP extension module."""
    global _split_gdr_hip
    if _split_gdr_hip is None:
        _split_gdr_hip = load(
            name="split_gdr_hip",
            sources=[_HIP_SRC],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            verbose=True,
        )
    return _split_gdr_hip


# ---------------------------------------------------------------------------
# Pure PyTorch CPU reference implementation
# ---------------------------------------------------------------------------

def split_gdr_reference(
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
    scale: float = None,
    use_qk_l2norm_in_kernel: bool = True,
):
    """
    Pure PyTorch reference for fused_split_gdr_update_kernel_v3.

    Mirrors the Triton kernel logic step-by-step on CPU in float32.

    Args:
        mixed_qkv: (B, dim, T), bfloat16 — concatenated Q, K, V along dim axis.
                   Q region: [0, key_dim), K region: [key_dim, 2*key_dim),
                   V region: [2*key_dim, 2*key_dim + value_dim).
        A_log:     (HV,), float32
        a:         (B*T, HV), bfloat16 — time-variant gating parameter
        dt_bias:   (HV,), bfloat16
        b:         (B*T, HV), bfloat16 — beta gating parameter
        initial_state_source: (N_states, HV, K, V), float32
        initial_state_indices: (B,), int32
        key_dim:   total key dimension = num_heads_qk * head_dim
        value_dim: total value dimension = num_heads_v * head_dim
        num_heads_qk: number of QK heads (H)
        num_heads_v:  number of V heads  (HV)
        head_dim:     per-head dimension (K = V = head_dim)

    Returns:
        output: (B, T, HV, V), same dtype as mixed_qkv
    """
    B, dim, T = mixed_qkv.shape
    H = num_heads_qk
    HV = num_heads_v
    K = head_dim
    V = head_dim
    GROUP_SIZE = HV // H

    if scale is None:
        scale = K ** -0.5

    # Work in float32 on CPU
    mixed_qkv_f = mixed_qkv.float().cpu()
    A_log_f = A_log.float().cpu()
    dt_bias_f = dt_bias.float().cpu()
    a_f = a.float().cpu()          # (B*T, HV)
    b_f = b.float().cpu()          # (B*T, HV)

    # Reshape a, b to (B, T, HV)
    a_f = a_f.view(B, T, HV)
    b_f = b_f.view(B, T, HV)

    # Clone initial states — indexed by initial_state_indices
    # h: (B, HV, K, V) in float32
    h = torch.zeros(B, HV, K, V, dtype=torch.float32)
    indices = initial_state_indices.cpu()
    for n in range(B):
        idx = indices[n].item()
        if idx >= 0:
            h[n] = initial_state_source[idx].float().cpu()

    # Split mixed_qkv along dim axis
    # Q: (B, key_dim, T), K_tensor: (B, key_dim, T), V_tensor: (B, value_dim, T)
    Q_all = mixed_qkv_f[:, :key_dim, :]             # (B, key_dim, T)
    K_all = mixed_qkv_f[:, key_dim:2*key_dim, :]    # (B, key_dim, T)
    V_all = mixed_qkv_f[:, 2*key_dim:, :]           # (B, value_dim, T)

    output = torch.zeros(B, T, HV, V, dtype=torch.float32)

    for t in range(T):
        for hv in range(HV):
            i_h = hv // GROUP_SIZE  # corresponding QK head

            # Extract per-head Q, K, V for this timestep
            q_vec = Q_all[:, i_h * K:(i_h + 1) * K, t]   # (B, K)
            k_vec = K_all[:, i_h * K:(i_h + 1) * K, t]   # (B, K)
            v_vec = V_all[:, hv * V:(hv + 1) * V, t]      # (B, V)

            # Gating parameters for this timestep and head
            a_t = a_f[:, t, hv]    # (B,)
            b_t = b_f[:, t, hv]    # (B,)

            # g = -exp(A_log[hv]) * softplus(a_t + dt_bias[hv])
            x = a_t + dt_bias_f[hv]                       # (B,)
            beta_x = softplus_beta * x
            softplus_x = torch.where(
                beta_x <= softplus_threshold,
                (1.0 / softplus_beta) * torch.log(1.0 + torch.exp(beta_x)),
                x,
            )
            g = -torch.exp(A_log_f[hv]) * softplus_x      # (B,)

            # beta = sigmoid(b_t)
            beta = torch.sigmoid(b_t)                      # (B,)

            # L2 normalization
            if use_qk_l2norm_in_kernel:
                q_vec = q_vec / (torch.sqrt(torch.sum(q_vec * q_vec, dim=-1, keepdim=True) + 1e-6))
                k_vec = k_vec / (torch.sqrt(torch.sum(k_vec * k_vec, dim=-1, keepdim=True) + 1e-6))

            # Scale query
            q_vec = q_vec * scale                          # (B, K)

            # h *= exp(g)  — decay
            h[:, hv, :, :] *= torch.exp(g).unsqueeze(-1).unsqueeze(-1)  # (B, K, V)

            # v -= sum(h * k[:, :, None], dim=K)  — delta rule
            v_vec = v_vec - torch.einsum('bkv,bk->bv', h[:, hv, :, :], k_vec)  # (B, V)

            # v *= beta  — beta gating
            v_vec = v_vec * beta.unsqueeze(-1)             # (B, V)

            # h += k[:, :, None] * v[:, None, :]  — state update
            h[:, hv, :, :] += torch.einsum('bk,bv->bkv', k_vec, v_vec)  # (B, K, V)

            # o = sum(h * q[:, :, None], dim=K)  — output
            o_vec = torch.einsum('bkv,bk->bv', h[:, hv, :, :], q_vec)  # (B, V)
            output[:, t, hv, :] = o_vec

    # Write final state back to initial_state_source (in-place, on CPU copy)
    for n in range(B):
        idx = indices[n].item()
        if idx >= 0:
            initial_state_source[idx] = h[n].to(initial_state_source.dtype).to(initial_state_source.device)

    return output.to(mixed_qkv.dtype).to(mixed_qkv.device)


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------
class TestFusedSplitGDRHip:
    """Test HIP fused_split_gdr_update against pure PyTorch CPU reference."""

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
        """Create test inputs (identical to TestFusedSplitGDRUpdateOpt.create_inputs)."""
        key_dim = num_heads_qk * head_dim
        value_dim = num_heads_v * head_dim
        dim = 2 * key_dim + value_dim

        # mixed_qkv: (batch, dim, seqlen)
        mixed_qkv = torch.randn(batch_size, dim, seqlen, device=device, dtype=dtype)

        # Gating parameters — A_log must be float32
        A_log = torch.randn(num_heads_v, device=device, dtype=torch.float32)
        dt_bias = torch.randn(num_heads_v, device=device, dtype=dtype)

        # Time-variant gating: (batch * seqlen, num_heads_v)
        a = torch.randn(batch_size * seqlen, num_heads_v, device=device, dtype=dtype)
        b = torch.randn(batch_size * seqlen, num_heads_v, device=device, dtype=dtype)

        # SSM state must be float32
        # Shape: (batch + padding, num_heads_v, head_dim, head_dim)
        ssm_state = torch.randn(
            batch_size + 10, num_heads_v, head_dim, head_dim,
            device=device, dtype=torch.float32,
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

    # ------------------------------------------------------------------
    # Correctness test — compare HIP kernel against torch CPU reference
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("batch_size", [64])
    @pytest.mark.parametrize("seqlen", [1])
    @pytest.mark.parametrize("num_heads_qk", [4])
    @pytest.mark.parametrize("num_heads_v", [8])
    @pytest.mark.parametrize("head_dim", [128])
    def test_split_gdr_hip_correctness(
        self,
        batch_size,
        seqlen,
        num_heads_qk,
        num_heads_v,
        head_dim,
        device,
        dtype,
    ):
        """Test correctness of HIP fused_split_gdr_update against torch CPU reference."""
        torch.manual_seed(42)

        inputs = self.create_inputs(
            batch_size, seqlen, num_heads_qk, num_heads_v, head_dim, device, dtype,
        )

        key_dim = inputs["key_dim"]
        value_dim = inputs["value_dim"]

        softplus_beta = 1.0
        softplus_threshold = 20.0
        scale = head_dim ** -0.5

        # ---- Reference: pure PyTorch CPU ----
        ssm_state_ref = inputs["ssm_state"].clone()

        output_ref = split_gdr_reference(
            mixed_qkv=inputs["mixed_qkv"],
            A_log=inputs["A_log"],
            a=inputs["a"],
            dt_bias=inputs["dt_bias"],
            b=inputs["b"],
            initial_state_source=ssm_state_ref,
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

        # ---- HIP kernel under test ----
        hip_mod = _get_hip_module()

        # Convert state to swizzled layout for HIP kernel
        ssm_state_hip = inputs["ssm_state"].clone()
        ssm_state_swizzled = to_swizzled_layout(ssm_state_hip)

        output_hip = hip_mod.fused_split_gdr_update(
            mixed_qkv=inputs["mixed_qkv"],
            A_log=inputs["A_log"],
            a=inputs["a"],
            dt_bias=inputs["dt_bias"],
            b_gate=inputs["b"],
            initial_state_source=ssm_state_swizzled,  # swizzled layout
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

        # Convert state back from swizzled layout for comparison
        ssm_state_hip_final = from_swizzled_layout(ssm_state_swizzled)

        # ---- Compare ----
        output_diff = (output_ref - output_hip).abs().max().item()
        state_diff = (ssm_state_ref - ssm_state_hip_final).abs().max().item()

        print(f"\n{'='*70}")
        print(f"Split GDR HIP Correctness Test: batch={batch_size}, seqlen={seqlen}")
        print(f"  heads_qk={num_heads_qk}, heads_v={num_heads_v}, head_dim={head_dim}")
        print(f"{'='*70}")
        print(f"  Output max diff: {output_diff:.6f}")
        print(f"  State  max diff: {state_diff:.6f}")
        print(f"{'='*70}")

        # Tolerance: same as test_split_gdr_v3_correctness
        assert output_diff < 5e-3, f"Output diff too large: {output_diff}"
        assert state_diff < 5e-3, f"State diff too large: {state_diff}"

        print(f"  PASS — Split GDR HIP correctness test passed!")

    # ------------------------------------------------------------------
    # Performance test — HIP kernel vs Triton v3
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("batch_size", [64])
    @pytest.mark.parametrize("seqlen", [1])
    @pytest.mark.parametrize("num_heads_qk", [4])
    @pytest.mark.parametrize("num_heads_v", [8])
    @pytest.mark.parametrize("head_dim", [128])
    def test_split_gdr_hip_performance(
        self,
        batch_size,
        seqlen,
        num_heads_qk,
        num_heads_v,
        head_dim,
        device,
        dtype,
    ):
        """Benchmark HIP fused_split_gdr_update vs Triton fused_split_gdr_update_v3."""
        from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent import (
            fused_split_gdr_update_v3,
        )

        torch.manual_seed(42)

        inputs = self.create_inputs(
            batch_size, seqlen, num_heads_qk, num_heads_v, head_dim, device, dtype,
        )

        key_dim = inputs["key_dim"]
        value_dim = inputs["value_dim"]

        softplus_beta = 1.0
        softplus_threshold = 20.0
        scale = head_dim ** -0.5

        num_warmup = 10
        num_iters = 1000

        hip_mod = _get_hip_module()

        # ============================================================
        # Benchmark Triton v3
        # ============================================================
        for _ in range(num_warmup):
            ssm_state_tmp = inputs["ssm_state"].clone()
            _ = fused_split_gdr_update_v3(
                mixed_qkv=inputs["mixed_qkv"],
                A_log=inputs["A_log"],
                a=inputs["a"],
                dt_bias=inputs["dt_bias"],
                b=inputs["b"],
                initial_state_source=ssm_state_tmp,
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
            ssm_state_tmp = inputs["ssm_state"].clone()
            _ = fused_split_gdr_update_v3(
                mixed_qkv=inputs["mixed_qkv"],
                A_log=inputs["A_log"],
                a=inputs["a"],
                dt_bias=inputs["dt_bias"],
                b=inputs["b"],
                initial_state_source=ssm_state_tmp,
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

        triton_time_us = start_event.elapsed_time(end_event) / num_iters * 1000

        # ============================================================
        # Benchmark HIP kernel (with swizzled state layout)
        # ============================================================
        # Pre-convert state to swizzled layout (done once, not timed)
        ssm_state_swizzled_template = to_swizzled_layout(inputs["ssm_state"])

        for _ in range(num_warmup):
            ssm_state_tmp = ssm_state_swizzled_template.clone()
            _ = hip_mod.fused_split_gdr_update(
                mixed_qkv=inputs["mixed_qkv"],
                A_log=inputs["A_log"],
                a=inputs["a"],
                dt_bias=inputs["dt_bias"],
                b_gate=inputs["b"],
                initial_state_source=ssm_state_tmp,  # swizzled layout
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
            ssm_state_tmp = ssm_state_swizzled_template.clone()
            _ = hip_mod.fused_split_gdr_update(
                mixed_qkv=inputs["mixed_qkv"],
                A_log=inputs["A_log"],
                a=inputs["a"],
                dt_bias=inputs["dt_bias"],
                b_gate=inputs["b"],
                initial_state_source=ssm_state_tmp,  # swizzled layout
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

        hip_time_us = start_event.elapsed_time(end_event) / num_iters * 1000

        # ============================================================
        # Report
        # ============================================================
        speedup = triton_time_us / hip_time_us if hip_time_us > 0 else float('inf')

        print(f"\n{'='*70}")
        print(f"Split GDR Performance: batch={batch_size}, seqlen={seqlen}")
        print(f"  heads_qk={num_heads_qk}, heads_v={num_heads_v}, head_dim={head_dim}")
        print(f"  warmup={num_warmup}, iters={num_iters}")
        print(f"{'='*70}")
        print(f"  Triton v3:  {triton_time_us:8.2f} us")
        print(f"  HIP kernel: {hip_time_us:8.2f} us")
        print(f"  Speedup (Triton/HIP): {speedup:.3f}x")
        print(f"{'='*70}")


    # ------------------------------------------------------------------
    # Correctness test — ksplit2 kernel against torch CPU reference
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("batch_size", [64])
    @pytest.mark.parametrize("seqlen", [1])
    @pytest.mark.parametrize("num_heads_qk", [4])
    @pytest.mark.parametrize("num_heads_v", [8])
    @pytest.mark.parametrize("head_dim", [128])
    def test_split_gdr_ksplit2_correctness(
        self,
        batch_size,
        seqlen,
        num_heads_qk,
        num_heads_v,
        head_dim,
        device,
        dtype,
    ):
        """Test correctness of ksplit2 HIP kernel against torch CPU reference."""
        torch.manual_seed(42)

        inputs = self.create_inputs(
            batch_size, seqlen, num_heads_qk, num_heads_v, head_dim, device, dtype,
        )

        key_dim = inputs["key_dim"]
        value_dim = inputs["value_dim"]

        softplus_beta = 1.0
        softplus_threshold = 20.0
        scale = head_dim ** -0.5

        # ---- Reference: pure PyTorch CPU ----
        ssm_state_ref = inputs["ssm_state"].clone()
        output_ref = split_gdr_reference(
            mixed_qkv=inputs["mixed_qkv"],
            A_log=inputs["A_log"],
            a=inputs["a"],
            dt_bias=inputs["dt_bias"],
            b=inputs["b"],
            initial_state_source=ssm_state_ref,
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

        # ---- ksplit2 kernel under test ----
        hip_mod = _get_hip_module()
        ssm_state_hip = inputs["ssm_state"].clone()
        ssm_state_swizzled = to_swizzled_layout(ssm_state_hip)

        output_hip = hip_mod.fused_split_gdr_update_ksplit2(
            mixed_qkv=inputs["mixed_qkv"],
            A_log=inputs["A_log"],
            a=inputs["a"],
            dt_bias=inputs["dt_bias"],
            b_gate=inputs["b"],
            initial_state_source=ssm_state_swizzled,
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

        ssm_state_hip_final = from_swizzled_layout(ssm_state_swizzled)

        output_diff = (output_ref - output_hip).abs().max().item()
        state_diff = (ssm_state_ref - ssm_state_hip_final).abs().max().item()

        print(f"\n{'='*70}")
        print(f"Split GDR ksplit2 Correctness: batch={batch_size}, seqlen={seqlen}")
        print(f"  heads_qk={num_heads_qk}, heads_v={num_heads_v}, head_dim={head_dim}")
        print(f"{'='*70}")
        print(f"  Output max diff: {output_diff:.6f}")
        print(f"  State  max diff: {state_diff:.6f}")
        print(f"{'='*70}")

        assert output_diff < 5e-3, f"Output diff too large: {output_diff}"
        assert state_diff < 5e-3, f"State diff too large: {state_diff}"
        print(f"  PASS — ksplit2 correctness test passed!")

    # ------------------------------------------------------------------
    # Correctness test — ksplit4 kernel against torch CPU reference
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("batch_size", [64])
    @pytest.mark.parametrize("seqlen", [1])
    @pytest.mark.parametrize("num_heads_qk", [4])
    @pytest.mark.parametrize("num_heads_v", [8])
    @pytest.mark.parametrize("head_dim", [128])
    def test_split_gdr_ksplit4_correctness(
        self,
        batch_size,
        seqlen,
        num_heads_qk,
        num_heads_v,
        head_dim,
        device,
        dtype,
    ):
        """Test correctness of ksplit4 HIP kernel against torch CPU reference."""
        torch.manual_seed(42)

        inputs = self.create_inputs(
            batch_size, seqlen, num_heads_qk, num_heads_v, head_dim, device, dtype,
        )

        key_dim = inputs["key_dim"]
        value_dim = inputs["value_dim"]

        softplus_beta = 1.0
        softplus_threshold = 20.0
        scale = head_dim ** -0.5

        # ---- Reference: pure PyTorch CPU ----
        ssm_state_ref = inputs["ssm_state"].clone()
        output_ref = split_gdr_reference(
            mixed_qkv=inputs["mixed_qkv"],
            A_log=inputs["A_log"],
            a=inputs["a"],
            dt_bias=inputs["dt_bias"],
            b=inputs["b"],
            initial_state_source=ssm_state_ref,
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

        # ---- ksplit4 kernel under test ----
        hip_mod = _get_hip_module()
        ssm_state_hip = inputs["ssm_state"].clone()
        ssm_state_swizzled = to_swizzled_layout(ssm_state_hip)

        output_hip = hip_mod.fused_split_gdr_update_ksplit4(
            mixed_qkv=inputs["mixed_qkv"],
            A_log=inputs["A_log"],
            a=inputs["a"],
            dt_bias=inputs["dt_bias"],
            b_gate=inputs["b"],
            initial_state_source=ssm_state_swizzled,
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

        ssm_state_hip_final = from_swizzled_layout(ssm_state_swizzled)

        output_diff = (output_ref - output_hip).abs().max().item()
        state_diff = (ssm_state_ref - ssm_state_hip_final).abs().max().item()

        print(f"\n{'='*70}")
        print(f"Split GDR ksplit4 Correctness: batch={batch_size}, seqlen={seqlen}")
        print(f"  heads_qk={num_heads_qk}, heads_v={num_heads_v}, head_dim={head_dim}")
        print(f"{'='*70}")
        print(f"  Output max diff: {output_diff:.6f}")
        print(f"  State  max diff: {state_diff:.6f}")
        print(f"{'='*70}")

        assert output_diff < 5e-3, f"Output diff too large: {output_diff}"
        assert state_diff < 5e-3, f"State diff too large: {state_diff}"
        print(f"  PASS — ksplit4 correctness test passed!")

    # ------------------------------------------------------------------
    # Correctness test — ksplit4_db kernel against torch CPU reference
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("batch_size", [64])
    @pytest.mark.parametrize("seqlen", [1, 4])
    @pytest.mark.parametrize("num_heads_qk", [2, 4, 16])
    @pytest.mark.parametrize("num_heads_v", [4, 8, 32])
    @pytest.mark.parametrize("head_dim", [128])
    def test_split_gdr_ksplit4_db_correctness(
        self,
        batch_size,
        seqlen,
        num_heads_qk,
        num_heads_v,
        head_dim,
        device,
        dtype,
    ):
        """Test correctness of ksplit4_db (double-buffer) HIP kernel."""
        if num_heads_v < num_heads_qk:
            pytest.skip("num_heads_v must be >= num_heads_qk")
        torch.manual_seed(42)

        inputs = self.create_inputs(
            batch_size, seqlen, num_heads_qk, num_heads_v, head_dim, device, dtype,
        )

        key_dim = inputs["key_dim"]
        value_dim = inputs["value_dim"]

        softplus_beta = 1.0
        softplus_threshold = 20.0
        scale = head_dim ** -0.5

        ssm_state_ref = inputs["ssm_state"].clone()
        output_ref = split_gdr_reference(
            mixed_qkv=inputs["mixed_qkv"],
            A_log=inputs["A_log"],
            a=inputs["a"],
            dt_bias=inputs["dt_bias"],
            b=inputs["b"],
            initial_state_source=ssm_state_ref,
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

        hip_mod = _get_hip_module()
        ssm_state_hip = inputs["ssm_state"].clone()
        ssm_state_swizzled = to_swizzled_layout(ssm_state_hip)

        output_hip = hip_mod.fused_split_gdr_update_ksplit4_db(
            mixed_qkv=inputs["mixed_qkv"],
            A_log=inputs["A_log"],
            a=inputs["a"],
            dt_bias=inputs["dt_bias"],
            b_gate=inputs["b"],
            initial_state_source=ssm_state_swizzled,
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

        ssm_state_hip_final = from_swizzled_layout(ssm_state_swizzled)

        output_diff = (output_ref - output_hip).abs().max().item()
        state_diff = (ssm_state_ref - ssm_state_hip_final).abs().max().item()

        print(f"\n{'='*70}")
        print(f"Split GDR ksplit4_db Correctness: batch={batch_size}, seqlen={seqlen}")
        print(f"  heads_qk={num_heads_qk}, heads_v={num_heads_v}, head_dim={head_dim}")
        print(f"{'='*70}")
        print(f"  Output max diff: {output_diff:.6f}")
        print(f"  State  max diff: {state_diff:.6f}")
        print(f"{'='*70}")

        assert output_diff < 5e-3, f"Output diff too large: {output_diff}"
        assert state_diff < 5e-3, f"State diff too large: {state_diff}"
        print(f"  PASS — ksplit4_db correctness test passed!")

    # ------------------------------------------------------------------
    # Correctness test — vsplit kernel against torch CPU reference
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("batch_size", [64])
    @pytest.mark.parametrize("seqlen", [1])
    @pytest.mark.parametrize("num_heads_qk", [4])
    @pytest.mark.parametrize("num_heads_v", [8])
    @pytest.mark.parametrize("head_dim", [128])
    def test_split_gdr_vsplit_correctness(
        self,
        batch_size,
        seqlen,
        num_heads_qk,
        num_heads_v,
        head_dim,
        device,
        dtype,
    ):
        """Test correctness of vsplit HIP kernel against torch CPU reference."""
        torch.manual_seed(42)

        inputs = self.create_inputs(
            batch_size, seqlen, num_heads_qk, num_heads_v, head_dim, device, dtype,
        )

        key_dim = inputs["key_dim"]
        value_dim = inputs["value_dim"]

        softplus_beta = 1.0
        softplus_threshold = 20.0
        scale = head_dim ** -0.5

        # ---- Reference: pure PyTorch CPU ----
        ssm_state_ref = inputs["ssm_state"].clone()
        output_ref = split_gdr_reference(
            mixed_qkv=inputs["mixed_qkv"],
            A_log=inputs["A_log"],
            a=inputs["a"],
            dt_bias=inputs["dt_bias"],
            b=inputs["b"],
            initial_state_source=ssm_state_ref,
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

        # ---- vsplit kernel under test ----
        hip_mod = _get_hip_module()
        ssm_state_hip = to_vsplit_layout(inputs["ssm_state"].clone())

        output_hip = hip_mod.fused_split_gdr_update_vsplit(
            mixed_qkv=inputs["mixed_qkv"],
            A_log=inputs["A_log"],
            a=inputs["a"],
            dt_bias=inputs["dt_bias"],
            b_gate=inputs["b"],
            initial_state_source=ssm_state_hip,
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

        ssm_state_hip_final = from_vsplit_layout(ssm_state_hip)

        output_diff = (output_ref - output_hip).abs().max().item()
        state_diff = (ssm_state_ref - ssm_state_hip_final).abs().max().item()

        print(f"\n{'='*70}")
        print(f"Split GDR vsplit Correctness: batch={batch_size}, seqlen={seqlen}")
        print(f"  heads_qk={num_heads_qk}, heads_v={num_heads_v}, head_dim={head_dim}")
        print(f"{'='*70}")
        print(f"  Output max diff: {output_diff:.6f}")
        print(f"  State  max diff: {state_diff:.6f}")
        print(f"{'='*70}")

        assert output_diff < 5e-3, f"Output diff too large: {output_diff}"
        assert state_diff < 5e-3, f"State diff too large: {state_diff}"
        print(f"  PASS — vsplit correctness test passed!")

    # ------------------------------------------------------------------
    # Performance test — compare all kernels
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("batch_size", [64])
    @pytest.mark.parametrize("seqlen", [1])
    @pytest.mark.parametrize("num_heads_qk", [4])
    @pytest.mark.parametrize("num_heads_v", [8])
    @pytest.mark.parametrize("head_dim", [128])
    def test_split_gdr_all_kernels_performance(
        self,
        batch_size,
        seqlen,
        num_heads_qk,
        num_heads_v,
        head_dim,
        device,
        dtype,
    ):
        """Benchmark kernels: Triton v3, origin HIP, ksplit2/4, vsplit."""
        from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent import (
            fused_split_gdr_update_v3,
        )

        torch.manual_seed(42)

        inputs = self.create_inputs(
            batch_size, seqlen, num_heads_qk, num_heads_v, head_dim, device, dtype,
        )

        key_dim = inputs["key_dim"]
        value_dim = inputs["value_dim"]

        softplus_beta = 1.0
        softplus_threshold = 20.0
        scale = head_dim ** -0.5

        num_warmup = 10
        num_iters = 1000

        hip_mod = _get_hip_module()
        ssm_state_swizzled_template = to_swizzled_layout(inputs["ssm_state"])
        ssm_state_vsplit_template = to_vsplit_layout(inputs["ssm_state"])

        # Helper: benchmark a callable that takes a cloned state
        def _benchmark(run_fn, use_swizzled=False, use_vsplit=False):
            if use_swizzled:
                template = ssm_state_swizzled_template
            elif use_vsplit:
                template = ssm_state_vsplit_template
            else:
                template = inputs["ssm_state"]
            for _ in range(num_warmup):
                run_fn(template.clone())
            torch.cuda.synchronize()

            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_evt.record()
            for _ in range(num_iters):
                run_fn(template.clone())
            end_evt.record()
            torch.cuda.synchronize()
            return start_evt.elapsed_time(end_evt) / num_iters * 1000  # us

        common_args = dict(
            mixed_qkv=inputs["mixed_qkv"],
            A_log=inputs["A_log"],
            a=inputs["a"],
            dt_bias=inputs["dt_bias"],
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

        hip_common = dict(
            **{k: v for k, v in common_args.items() if k != "a"},
            a=inputs["a"],
            b_gate=inputs["b"],
        )

        # 1) Triton v3
        triton_us = _benchmark(
            lambda st: fused_split_gdr_update_v3(
                **common_args,
                b=inputs["b"],
                initial_state_source=st,
                initial_state_indices=inputs["ssm_state_indices"],
            ),
            use_swizzled=False,
        )

        # 2) Origin HIP
        origin_us = _benchmark(
            lambda st: hip_mod.fused_split_gdr_update(
                **hip_common,
                initial_state_source=st,
                initial_state_indices=inputs["ssm_state_indices"],
            ),
            use_swizzled=True,
        )

        # 3) ksplit2
        ksplit2_us = _benchmark(
            lambda st: hip_mod.fused_split_gdr_update_ksplit2(
                **hip_common,
                initial_state_source=st,
                initial_state_indices=inputs["ssm_state_indices"],
            ),
            use_swizzled=True,
        )

        # 4) ksplit4
        ksplit4_us = _benchmark(
            lambda st: hip_mod.fused_split_gdr_update_ksplit4(
                **hip_common,
                initial_state_source=st,
                initial_state_indices=inputs["ssm_state_indices"],
            ),
            use_swizzled=True,
        )

        # 5) ksplit4_db (double-buffer)
        ksplit4_db_us = _benchmark(
            lambda st: hip_mod.fused_split_gdr_update_ksplit4_db(
                **hip_common,
                initial_state_source=st,
                initial_state_indices=inputs["ssm_state_indices"],
            ),
            use_swizzled=True,
        )

        # 5b) ksplit4_opt (LDS-read-pipelined Phase 1/3)
        ksplit4_opt_us = _benchmark(
            lambda st: hip_mod.fused_split_gdr_update_ksplit4_opt(
                **hip_common,
                initial_state_source=st,
                initial_state_indices=inputs["ssm_state_indices"],
            ),
            use_swizzled=True,
        )

        # 6) ksplit8
        ksplit8_us = _benchmark(
            lambda st: hip_mod.fused_split_gdr_update_ksplit8(
                **hip_common,
                initial_state_source=st,
                initial_state_indices=inputs["ssm_state_indices"],
            ),
            use_swizzled=True,
        )

        # 7) ksplit4_hfuse
        ksplit4_hfuse_us = _benchmark(
            lambda st: hip_mod.fused_split_gdr_update_ksplit4_hfuse(
                **hip_common,
                initial_state_source=st,
                initial_state_indices=inputs["ssm_state_indices"],
            ),
            use_swizzled=True,
        )

        # 8) ksplit8_hfuse
        ksplit8_hfuse_us = _benchmark(
            lambda st: hip_mod.fused_split_gdr_update_ksplit8_hfuse(
                **hip_common,
                initial_state_source=st,
                initial_state_indices=inputs["ssm_state_indices"],
            ),
            use_swizzled=True,
        )

        # 9) vsplit (uses vsplit [N,HV,V/4,K,4] layout)
        vsplit_us = _benchmark(
            lambda st: hip_mod.fused_split_gdr_update_vsplit(
                **hip_common,
                initial_state_source=st,
                initial_state_indices=inputs["ssm_state_indices"],
            ),
            use_vsplit=True,
        )

        # ============================================================
        # Report
        # ============================================================
        baseline = triton_us
        print(f"\n{'='*70}")
        print(f"Split GDR ALL Kernels Performance Comparison")
        print(f"  batch={batch_size}, seqlen={seqlen}")
        print(f"  heads_qk={num_heads_qk}, heads_v={num_heads_v}, head_dim={head_dim}")
        print(f"  warmup={num_warmup}, iters={num_iters}")
        print(f"{'='*70}")
        print(f"  {'Kernel':<20s} {'Time (us)':>10s} {'vs Triton':>10s}")
        print(f"  {'-'*20} {'-'*10} {'-'*10}")
        print(f"  {'Triton v3':<20s} {triton_us:10.2f} {'1.000x':>10s}")
        print(f"  {'Origin HIP':<20s} {origin_us:10.2f} {baseline/origin_us:9.3f}x")
        print(f"  {'ksplit2':<20s} {ksplit2_us:10.2f} {baseline/ksplit2_us:9.3f}x")
        print(f"  {'ksplit4':<20s} {ksplit4_us:10.2f} {baseline/ksplit4_us:9.3f}x")
        print(f"  {'ksplit4_db':<20s} {ksplit4_db_us:10.2f} {baseline/ksplit4_db_us:9.3f}x")
        print(f"  {'ksplit4_opt':<20s} {ksplit4_opt_us:10.2f} {baseline/ksplit4_opt_us:9.3f}x")
        print(f"  {'ksplit8':<20s} {ksplit8_us:10.2f} {baseline/ksplit8_us:9.3f}x")
        print(f"  {'ksplit4_hfuse':<20s} {ksplit4_hfuse_us:10.2f} {baseline/ksplit4_hfuse_us:9.3f}x")
        print(f"  {'ksplit8_hfuse':<20s} {ksplit8_hfuse_us:10.2f} {baseline/ksplit8_hfuse_us:9.3f}x")
        print(f"  {'vsplit':<20s} {vsplit_us:10.2f} {baseline/vsplit_us:9.3f}x")
        print(f"{'='*70}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
