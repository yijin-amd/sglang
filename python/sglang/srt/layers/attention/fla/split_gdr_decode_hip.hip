/**
 * Fused Split GDR (Gating Delta Rule) Decode Kernel - HIP Implementation.
 *
 * This file implements the HIP version of fused_split_gdr_update_kernel_v3
 * from fused_sigmoid_gating_recurrent.py.
 *
 * The kernel fuses:
 *   1. Splitting mixed QKV tensor into Q, K, V
 *   2. Sigmoid gating delta rule state update
 *   3. Output computation
 *
 * Hardware target: AMD MI series GPU
 *   - 80 CUs
 *   - 4 SIMDs per CU
 *   - 64 threads per wavefront
 *   - Max 8 wavefronts per CU  (= 2 WF/SIMD)
 *   - 512 VGPRs per SIMD
 *
 * =====================================================================
 *  Swizzled State Layout: (N_states, HV, K/4, V, 4)
 * =====================================================================
 *
 *  This layout enables BOTH cross-thread coalescing AND per-thread float4:
 *
 *  Original (K, V):
 *    Thread 0 accesses: [k=0,v=0], [k=1,v=0], [k=2,v=0]...
 *    → stride = V floats = 512 bytes between consecutive k ❌
 *
 *  Swizzled (K/4, V, 4):
 *    Memory: [v=0,k=0:3][v=1,k=0:3][v=2,k=0:3]...[v=63,k=0:3]...
 *    Thread 0 loads float4 from [kg=0,v=0,0:3] → k=0,1,2,3 ✅
 *    Thread 1 loads float4 from [kg=0,v=1,0:3] → k=0,1,2,3 ✅
 *    All 64 threads access consecutive 1024 bytes ✅
 *
 *  Benefits:
 *    - Cross-thread coalescing: 64 threads × 16 bytes = 1024 bytes
 *    - Per-thread vectorization: float4 load = 4 floats in 1 instruction
 *    - 4x fewer memory load instructions
 *
 * =====================================================================
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>

#include <cmath>
#include <cstdint>

// ---------------------------------------------------------------------------
// Helper device functions
// ---------------------------------------------------------------------------

__device__ __forceinline__ float hip_softplus(
    float x, float beta, float inv_beta, float threshold)
{
    float beta_x = beta * x;
    return (beta_x <= threshold) ? inv_beta * logf(1.0f + expf(beta_x)) : x;
}

__device__ __forceinline__ float hip_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// ---------------------------------------------------------------------------
// Vec2 bf16 load helper - loads 2 bf16 values as one uint32_t
// ---------------------------------------------------------------------------

__device__ __forceinline__ void load_bf16x2(
    float* __restrict__ smem,
    const __hip_bfloat16* __restrict__ src,
    int base, int K, int BK)
{
    if (base + 1 < K) {
        uint32_t packed = *reinterpret_cast<const uint32_t*>(src + base);
        const __hip_bfloat16* v = reinterpret_cast<const __hip_bfloat16*>(&packed);
        smem[base]     = static_cast<float>(v[0]);
        smem[base + 1] = static_cast<float>(v[1]);
    } else if (base < K) {
        smem[base]     = static_cast<float>(src[base]);
        if (base + 1 < BK) smem[base + 1] = 0.0f;
    } else {
        if (base < BK)     smem[base]     = 0.0f;
        if (base + 1 < BK) smem[base + 1] = 0.0f;
    }
}

// Padded variant: stores at smem[padded_idx] to avoid LDS bank conflicts
// when multiple k_splits read from offsets that are multiples of 32 (= NUM_BANKS).
// Padding inserts 1 float gap every CHUNK elements, so k_split N starts at
// N*(CHUNK+1) instead of N*CHUNK, mapping to different banks.
template <int CHUNK>
__device__ __forceinline__ int lds_padded_idx(int idx)
{
    return idx + idx / CHUNK;
}

template <int CHUNK>
__device__ __forceinline__ void load_bf16x2_padded(
    float* __restrict__ smem,
    const __hip_bfloat16* __restrict__ src,
    int base, int K, int BK)
{
    const int p0 = lds_padded_idx<CHUNK>(base);
    const int p1 = lds_padded_idx<CHUNK>(base + 1);
    if (base + 1 < K) {
        uint32_t packed = *reinterpret_cast<const uint32_t*>(src + base);
        const __hip_bfloat16* v = reinterpret_cast<const __hip_bfloat16*>(&packed);
        smem[p0] = static_cast<float>(v[0]);
        smem[p1] = static_cast<float>(v[1]);
    } else if (base < K) {
        smem[p0] = static_cast<float>(src[base]);
        if (base + 1 < BK) smem[p1] = 0.0f;
    } else {
        if (base < BK)     smem[p0] = 0.0f;
        if (base + 1 < BK) smem[p1] = 0.0f;
    }
}

// ---------------------------------------------------------------------------
// Kernel: fused_split_gdr_update_kernel
// ---------------------------------------------------------------------------
//
// Template parameters:
//   BK  - tile size along K dimension (= head_dim, power of 2, must be divisible by 4)
//   BV  - block size / V-tile width (= 64, matches wavefront size)
//
// Grid:  (NK, cdiv(V, BV), B * HV)        NK always 1
// Block: BV threads = 1 wavefront
//
// State layout: SWIZZLED (N_states, HV, K/4, V, 4)
//   - Enables float4 vectorized loads with cross-thread coalescing
// ---------------------------------------------------------------------------

template <int BK, int BV, bool USE_INITIAL_STATE, bool USE_QK_L2NORM>
__global__ __launch_bounds__(BV)
void fused_split_gdr_update_kernel(
    // ----- Input: mixed QKV tensor (B, dim, T) -----
    const __hip_bfloat16* __restrict__ mixed_qkv,
    // ----- Gating inputs -----
    const float*          __restrict__ A_log,       // (HV,)
    const __hip_bfloat16* __restrict__ a,           // (B*T, HV)
    const __hip_bfloat16* __restrict__ dt_bias,     // (HV,)
    float                             softplus_beta,
    float                             softplus_threshold,
    const __hip_bfloat16* __restrict__ b_gate,      // (B*T, HV)
    // ----- Output -----
    __hip_bfloat16*       __restrict__ o,           // (B, T, HV, V)
    // ----- State: SWIZZLED (N_states, HV, K/4, V, 4) -----
    float*                __restrict__ h0_source,
    const int32_t*        __restrict__ h0_indices,  // (B,)
    // ----- Dimensions -----
    int T,
    int key_dim,
    int value_dim,
    // Strides for mixed_qkv
    int64_t stride_x_batch,
    int64_t stride_x_dim,
    int64_t stride_x_seq,
    // Strides for output
    int64_t stride_o_batch,
    int64_t stride_o_seq,
    int64_t stride_o_head,
    int64_t stride_o_dim,
    // Meta dimensions
    int B,
    int H,          // num_heads_qk
    int HV,         // num_heads_v
    int K,          // head_dim (key)
    int V_dim,      // head_dim (value)
    float scale
) {
    // Compile-time check: BK must be divisible by 4 for float4
    static_assert(BK % 4 == 0, "BK must be divisible by 4 for float4 vectorization");
    constexpr int BK4 = BK / 4;  // Number of float4 groups

    // ================================================================
    // 1. Thread / block indexing
    // ================================================================
    const int i_v  = blockIdx.y;        // V-tile index (each tile = BV columns)
    const int i_nh = blockIdx.z;        // flattened (batch, head_v)
    const int i_n  = i_nh / HV;         // batch index
    const int i_hv = i_nh % HV;         // value-head index

    const int GROUP_SIZE = HV / H;
    const int i_h  = i_hv / GROUP_SIZE; // corresponding QK head

    const int lane = threadIdx.x;       // [0, BV) = [0, 64)

    // Global V column this thread owns
    const int v_col = i_v * BV + lane;
    const bool valid_v = (v_col < V_dim);

    // ================================================================
    // 2. LDS for cooperative K / Q vector sharing
    // ================================================================
    __shared__ float smem[BK];          // BK floats for K or Q

    // Pre-load A_log and dt_bias BEFORE h0 state loads so that HBM latency
    // is hidden behind the global_load_dwordx4 for h_vec.
    const float a_log_val   = A_log[i_hv];
    const float dt_bias_val = static_cast<float>(dt_bias[i_hv]);

    // ================================================================
    // 3. Per-thread hidden-state column (BK registers, stored as float4)
    // ================================================================
    float4 h_vec[BK4];
    #pragma unroll
    for (int i = 0; i < BK4; i++) {
        h_vec[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    // ================================================================
    // 4. Load initial state from SWIZZLED layout: h0[idx, hv, kg, v_col, 0:4]
    //    Layout: (N_states, HV, K/4, V, 4)
    //    Address: base + idx*(HV*K4*V*4) + hv*(K4*V*4) + kg*(V*4) + v*4 + k_inner
    //
    //    For float4 load: base + kg*(V*4) + v_col*4
    //    All 64 threads loading kg=0: addresses are consecutive (coalesced!)
    // ================================================================
    if constexpr (USE_INITIAL_STATE) {
        const int32_t idx = h0_indices[i_n];
        if (idx >= 0 && valid_v) {
            const int K4 = K / 4;
            const float4* h0_base = reinterpret_cast<const float4*>(
                h0_source
                + static_cast<int64_t>(idx) * HV * K4 * V_dim * 4
                + static_cast<int64_t>(i_hv) * K4 * V_dim * 4
            );
            
            // float4 vectorized loads - each thread loads from consecutive address
            #pragma unroll
            for (int kg = 0; kg < BK4; kg++) {
                h_vec[kg] = h0_base[kg * V_dim + v_col];
            }
        }
    }

    // ================================================================
    // 5. Pre-compute loop-invariant gating values
    // ================================================================
    const float neg_exp_A_log     = -expf(a_log_val);
    const float inv_softplus_beta = 1.0f / softplus_beta;

    // ================================================================
    // 6. Dimension offsets inside the mixed_qkv dim axis
    // ================================================================
    const int q_dim_off = i_h  * K;                    // Q start in dim
    const int k_dim_off = key_dim + i_h * K;           // K start in dim
    const int v_dim_off = 2 * key_dim + i_hv * V_dim;  // V start in dim

    // ================================================================
    // 7. Base pointers (per-batch)
    // ================================================================
    const __hip_bfloat16* x_base =
        mixed_qkv + static_cast<int64_t>(i_n) * stride_x_batch;

    const __hip_bfloat16* p_a =
        a      + static_cast<int64_t>(i_n) * T * HV + i_hv;
    const __hip_bfloat16* p_b =
        b_gate + static_cast<int64_t>(i_n) * T * HV + i_hv;

    __hip_bfloat16* p_o =
        o + static_cast<int64_t>(i_n) * stride_o_batch
          + static_cast<int64_t>(i_hv) * stride_o_head;

    // Check if vec2 load is possible (decode: stride_x_dim == 1)
    const bool use_vec2 = (stride_x_dim == 1);

    // ================================================================
    // 8. Main timestep loop
    // ================================================================
    for (int t = 0; t < T; t++) {
        const __hip_bfloat16* x_t =
            x_base + static_cast<int64_t>(t) * stride_x_seq;

        // ------------------------------------------------------------
        // 8a. Cooperatively load K[BK] into LDS
        // ------------------------------------------------------------
        if (use_vec2) {
            const int base = 2 * lane;
            if (base < BK) {
                load_bf16x2(smem, x_t + k_dim_off, base, K, BK);
            }
        } else {
            for (int i = lane; i < BK; i += BV) {
                smem[i] = (i < K)
                    ? static_cast<float>(
                          x_t[static_cast<int64_t>(k_dim_off + i) * stride_x_dim])
                    : 0.0f;
            }
        }
        __syncthreads();

        // ------------------------------------------------------------
        // 8b. Load this thread's V element
        // ------------------------------------------------------------
        float v_local = 0.0f;
        if (valid_v) {
            v_local = static_cast<float>(
                x_t[static_cast<int64_t>(v_dim_off + v_col) * stride_x_dim]);
        }

        // ------------------------------------------------------------
        // 8c. Load gating scalars for this timestep
        // ------------------------------------------------------------
        const float a_t = static_cast<float>(p_a[static_cast<int64_t>(t) * HV]);
        const float b_t = static_cast<float>(p_b[static_cast<int64_t>(t) * HV]);

        // g = -exp(A_log) * softplus(a_t + dt_bias)
        const float sp  = hip_softplus(
            a_t + dt_bias_val, softplus_beta, inv_softplus_beta, softplus_threshold);
        const float g     = neg_exp_A_log * sp;
        const float exp_g = expf(g);

        // beta = sigmoid(b_t)
        const float beta = hip_sigmoid(b_t);

        // ------------------------------------------------------------
        // 8d. K-vector L2 norm (warp-level reduction via __shfl_xor)
        // ------------------------------------------------------------
        float k_inv_norm = 1.0f;
        if constexpr (USE_QK_L2NORM) {
            float k_sq = 0.0f;
            for (int i = lane; i < BK; i += BV) {
                k_sq += smem[i] * smem[i];
            }
            for (int off = 32; off >= 1; off >>= 1) {
                k_sq += __shfl_xor(k_sq, off, 64);
            }
            k_inv_norm = rsqrtf(k_sq + 1e-6f);
        }

        // ------------------------------------------------------------
        // 8e. Decay:  h *= exp(g)
        // ------------------------------------------------------------
        #pragma unroll 
        for (int kg = 0; kg < BK4; kg++) {
            h_vec[kg].x *= exp_g;
            h_vec[kg].y *= exp_g;
            h_vec[kg].z *= exp_g;
            h_vec[kg].w *= exp_g;
        }

        // ------------------------------------------------------------
        // 8f. Delta rule:  v -= dot(h, k_normed)
        // ------------------------------------------------------------
        {
            float dot_hk = 0.0f;
            #pragma unroll
            for (int kg = 0; kg < BK4; kg++) {
                const int k0 = kg * 4;
                dot_hk += h_vec[kg].x * smem[k0];
                dot_hk += h_vec[kg].y * smem[k0 + 1];
                dot_hk += h_vec[kg].z * smem[k0 + 2];
                dot_hk += h_vec[kg].w * smem[k0 + 3];
            }
            v_local -= dot_hk * k_inv_norm;
        }

        // ------------------------------------------------------------
        // 8g. Beta gating:  v *= beta
        // ------------------------------------------------------------
        v_local *= beta;

        // ------------------------------------------------------------
        // 8h. State update:  h[i] += k_normed[i] * v
        // ------------------------------------------------------------
        {
            const float kv = k_inv_norm * v_local;
            #pragma unroll
            for (int kg = 0; kg < BK4; kg++) {
                const int k0 = kg * 4;
                h_vec[kg].x += smem[k0]     * kv;
                h_vec[kg].y += smem[k0 + 1] * kv;
                h_vec[kg].z += smem[k0 + 2] * kv;
                h_vec[kg].w += smem[k0 + 3] * kv;
            }
        }

        // ------------------------------------------------------------
        // 8i. Load Q[BK] into LDS
        // ------------------------------------------------------------
        __syncthreads();
        if (use_vec2) {
            const int base = 2 * lane;
            if (base < BK) {
                load_bf16x2(smem, x_t + q_dim_off, base, K, BK);
            }
        } else {
            for (int i = lane; i < BK; i += BV) {
                smem[i] = (i < K)
                    ? static_cast<float>(
                          x_t[static_cast<int64_t>(q_dim_off + i) * stride_x_dim])
                    : 0.0f;
            }
        }
        __syncthreads();

        // ------------------------------------------------------------
        // 8j. Q-vector L2 norm
        // ------------------------------------------------------------
        float q_inv_norm = 1.0f;
        if constexpr (USE_QK_L2NORM) {
            float q_sq = 0.0f;
            for (int i = lane; i < BK; i += BV) {
                q_sq += smem[i] * smem[i];
            }
            for (int off = 32; off >= 1; off >>= 1) {
                q_sq += __shfl_xor(q_sq, off, 64);
            }
            q_inv_norm = rsqrtf(q_sq + 1e-6f);
        }

        // ------------------------------------------------------------
        // 8k. Output:  o = dot(h, q_normed) * scale
        // ------------------------------------------------------------
        {
            float o_local = 0.0f;
            #pragma unroll
            for (int kg = 0; kg < BK4; kg++) {
                const int k0 = kg * 4;
                o_local += h_vec[kg].x * smem[k0];
                o_local += h_vec[kg].y * smem[k0 + 1];
                o_local += h_vec[kg].z * smem[k0 + 2];
                o_local += h_vec[kg].w * smem[k0 + 3];
            }
            o_local *= q_inv_norm * scale;

            if (valid_v) {
                p_o[static_cast<int64_t>(t) * stride_o_seq
                    + static_cast<int64_t>(v_col) * stride_o_dim] =
                    static_cast<__hip_bfloat16>(o_local);
            }
        }
    } // end timestep loop

    // ================================================================
    // 9. Store final state to SWIZZLED layout
    // ================================================================
    if constexpr (USE_INITIAL_STATE) {
        const int32_t idx = h0_indices[i_n];
        if (idx >= 0 && valid_v) {
            const int K4 = K / 4;
            float4* h0_base = reinterpret_cast<float4*>(
                h0_source
                + static_cast<int64_t>(idx) * HV * K4 * V_dim * 4
                + static_cast<int64_t>(i_hv) * K4 * V_dim * 4
            );
            
            // float4 vectorized stores
            #pragma unroll
            for (int kg = 0; kg < BK4; kg++) {
                h0_base[kg * V_dim + v_col] = h_vec[kg];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel: fused_split_gdr_update_kernel_ksplit2
//
// K-split=2: each V column served by 2 adjacent lanes.
//   lane 2i:   h[0     : BK/2-1, v_col=i]
//   lane 2i+1: h[BK/2  : BK-1,   v_col=i]
// Reduction: single __shfl_xor(..., 1, 64).
//
// Grid:  (1, cdiv(V, BV/2), B * HV)
// Block: BV threads = 1 wavefront
// State layout: SWIZZLED (N_states, HV, K/4, V, 4)
// ---------------------------------------------------------------------------

template <int BK, int BV, bool USE_INITIAL_STATE, bool USE_QK_L2NORM>
__global__ 
__launch_bounds__(BV)
void fused_split_gdr_update_kernel_ksplit2(
    const __hip_bfloat16* __restrict__ mixed_qkv,
    const float*          __restrict__ A_log,
    const __hip_bfloat16* __restrict__ a,
    const __hip_bfloat16* __restrict__ dt_bias,
    float                             softplus_beta,
    float                             softplus_threshold,
    const __hip_bfloat16* __restrict__ b_gate,
    __hip_bfloat16*       __restrict__ o,
    float*                __restrict__ h0_source,
    const int32_t*        __restrict__ h0_indices,
    int T,
    int key_dim,
    int value_dim,
    int64_t stride_x_batch,
    int64_t stride_x_dim,
    int64_t stride_x_seq,
    int64_t stride_o_batch,
    int64_t stride_o_seq,
    int64_t stride_o_head,
    int64_t stride_o_dim,
    int B,
    int H,
    int HV,
    int K,
    int V_dim,
    float scale
) {
    static_assert(BK % 4 == 0, "BK must be divisible by 4");
    static_assert(BK >= 8, "BK must be >= 8 for K_SPLIT=2");

    constexpr int BK_HALF  = BK / 2;
    constexpr int BK4_HALF = BK_HALF / 4;
    constexpr int BV_OUT   = BV / 2;

    const int i_v  = blockIdx.y;
    const int i_nh = blockIdx.z;
    const int i_n  = i_nh / HV;
    const int i_hv = i_nh % HV;

    const int GROUP_SIZE = HV / H;
    const int i_h  = i_hv / GROUP_SIZE;

    const int lane = threadIdx.x;

    // Remap lanes so each half-wave handles one K half:
    // lanes [0..31]  -> k_split=0 (K front half)
    // lanes [32..63] -> k_split=1 (K back half)
    const int v_idx   = lane & (BV_OUT - 1);
    const int k_split = lane >> 5;
    const int v_col   = i_v * BV_OUT + v_idx;
    const bool valid_v = (v_col < V_dim);
    const int k_start = k_split * BK_HALF;

    __shared__ float smem[BK];

    const float a_log_val   = A_log[i_hv];
    const float dt_bias_val = static_cast<float>(dt_bias[i_hv]);

    float4 h_vec[BK4_HALF];
    #pragma unroll
    for (int i = 0; i < BK4_HALF; i++) {
        h_vec[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    // Load initial state from SWIZZLED layout
    if constexpr (USE_INITIAL_STATE) {
        const int32_t idx = h0_indices[i_n];
        if (idx >= 0 && valid_v) {
            const int K4 = K / 4;
            const float4* h0_base = reinterpret_cast<const float4*>(
                h0_source
                + static_cast<int64_t>(idx) * HV * K4 * V_dim * 4
                + static_cast<int64_t>(i_hv) * K4 * V_dim * 4
            );
            const int kg_start = k_start / 4;
            #pragma unroll
            for (int i = 0; i < BK4_HALF; i++) {
                h_vec[i] = h0_base[(kg_start + i) * V_dim + v_col];
            }
        }
    }

    const float neg_exp_A_log     = -expf(a_log_val);
    const float inv_softplus_beta = 1.0f / softplus_beta;

    const int q_dim_off = i_h  * K;
    const int k_dim_off = key_dim + i_h * K;
    const int v_dim_off = 2 * key_dim + i_hv * V_dim;

    const __hip_bfloat16* x_base =
        mixed_qkv + static_cast<int64_t>(i_n) * stride_x_batch;

    const __hip_bfloat16* p_a =
        a      + static_cast<int64_t>(i_n) * T * HV + i_hv;
    const __hip_bfloat16* p_b =
        b_gate + static_cast<int64_t>(i_n) * T * HV + i_hv;

    __hip_bfloat16* p_o =
        o + static_cast<int64_t>(i_n) * stride_o_batch
          + static_cast<int64_t>(i_hv) * stride_o_head;

    const bool use_vec2 = (stride_x_dim == 1);
    const int safe_v_col = valid_v ? v_col : 0;

    // Main timestep loop
    for (int t = 0; t < T; t++) {
        const __hip_bfloat16* x_t =
            x_base + static_cast<int64_t>(t) * stride_x_seq;

        // Issue V and gating loads unconditionally BEFORE K cooperative load.
        // Unconditional V load (safe_v_col) avoids exec masking, letting the
        // compiler defer s_waitcnt so these overlap with K's HBM latency.
        const __hip_bfloat16 v_raw =
            x_t[static_cast<int64_t>(v_dim_off + safe_v_col) * stride_x_dim];
        const __hip_bfloat16 a_raw = p_a[static_cast<int64_t>(t) * HV];
        const __hip_bfloat16 b_raw = p_b[static_cast<int64_t>(t) * HV];

        // Cooperatively load K[BK] into LDS (all 64 threads participate)
        if (use_vec2) {
            const int base = 2 * lane;
            if (base < BK) {
                load_bf16x2(smem, x_t + k_dim_off, base, K, BK);
            }
        } else {
            for (int i = lane; i < BK; i += BV) {
                smem[i] = (i < K)
                    ? static_cast<float>(
                          x_t[static_cast<int64_t>(k_dim_off + i) * stride_x_dim])
                    : 0.0f;
            }
        }
        __syncthreads();

        float v_local = valid_v ? static_cast<float>(v_raw) : 0.0f;
        const float a_t = static_cast<float>(a_raw);
        const float b_t = static_cast<float>(b_raw);

        const float sp  = hip_softplus(
            a_t + dt_bias_val, softplus_beta, inv_softplus_beta, softplus_threshold);
        const float g     = neg_exp_A_log * sp;
        const float exp_g = expf(g);
        const float beta  = hip_sigmoid(b_t);

        // Phase 1: Fused K-L2-norm + Delta dot (on UN-DECAYED h_vec)
        float k_inv_norm = 1.0f;
        {
            float dot_partial = 0.0f;
            float k_sq = 0.0f;
            #pragma unroll
            for (int i = 0; i < BK4_HALF; i++) {
                const int k0 = k_start + i * 4;
                const float s0 = smem[k0], s1 = smem[k0 + 1];
                const float s2 = smem[k0 + 2], s3 = smem[k0 + 3];
                dot_partial += h_vec[i].x * s0 + h_vec[i].y * s1
                             + h_vec[i].z * s2 + h_vec[i].w * s3;
                if constexpr (USE_QK_L2NORM) {
                    k_sq += s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3;
                }
            }
            dot_partial += __shfl_xor(dot_partial, 32, 64);
            if constexpr (USE_QK_L2NORM) {
                k_sq += __shfl_xor(k_sq, 32, 64);
                k_inv_norm = rsqrtf(k_sq + 1e-6f);
            }
            v_local -= dot_partial * exp_g * k_inv_norm;
        }

        v_local *= beta;

        // Phase 2: Fused Decay + State update
        {
            const float kv = k_inv_norm * v_local;
            #pragma unroll
            for (int i = 0; i < BK4_HALF; i++) {
                const int k0 = k_start + i * 4;
                h_vec[i].x = h_vec[i].x * exp_g + smem[k0]     * kv;
                h_vec[i].y = h_vec[i].y * exp_g + smem[k0 + 1] * kv;
                h_vec[i].z = h_vec[i].z * exp_g + smem[k0 + 2] * kv;
                h_vec[i].w = h_vec[i].w * exp_g + smem[k0 + 3] * kv;
            }
        }

        // Load Q into LDS
        __syncthreads();
        if (use_vec2) {
            const int base = 2 * lane;
            if (base < BK) {
                load_bf16x2(smem, x_t + q_dim_off, base, K, BK);
            }
        } else {
            for (int i = lane; i < BK; i += BV) {
                smem[i] = (i < K)
                    ? static_cast<float>(
                          x_t[static_cast<int64_t>(q_dim_off + i) * stride_x_dim])
                    : 0.0f;
            }
        }
        __syncthreads();

        // Phase 3: Fused Q-L2-norm + Output dot
        {
            float out_partial = 0.0f;
            float q_sq = 0.0f;
            #pragma unroll
            for (int i = 0; i < BK4_HALF; i++) {
                const int k0 = k_start + i * 4;
                const float s0 = smem[k0], s1 = smem[k0 + 1];
                const float s2 = smem[k0 + 2], s3 = smem[k0 + 3];
                out_partial += h_vec[i].x * s0 + h_vec[i].y * s1
                             + h_vec[i].z * s2 + h_vec[i].w * s3;
                if constexpr (USE_QK_L2NORM) {
                    q_sq += s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3;
                }
            }
            out_partial += __shfl_xor(out_partial, 32, 64);
            float q_inv_norm = 1.0f;
            if constexpr (USE_QK_L2NORM) {
                q_sq += __shfl_xor(q_sq, 32, 64);
                q_inv_norm = rsqrtf(q_sq + 1e-6f);
            }
            float o_local = out_partial * q_inv_norm * scale;

            if (k_split == 0 && valid_v) {
                p_o[static_cast<int64_t>(t) * stride_o_seq
                    + static_cast<int64_t>(v_col) * stride_o_dim] =
                    static_cast<__hip_bfloat16>(o_local);
            }
        }
    } // end timestep loop

    // Store final state to SWIZZLED layout
    if constexpr (USE_INITIAL_STATE) {
        const int32_t idx = h0_indices[i_n];
        if (idx >= 0 && valid_v) {
            const int K4 = K / 4;
            float4* h0_base = reinterpret_cast<float4*>(
                h0_source
                + static_cast<int64_t>(idx) * HV * K4 * V_dim * 4
                + static_cast<int64_t>(i_hv) * K4 * V_dim * 4
            );
            const int kg_start = k_start / 4;
            #pragma unroll
            for (int i = 0; i < BK4_HALF; i++) {
                h0_base[(kg_start + i) * V_dim + v_col] = h_vec[i];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel: fused_split_gdr_update_kernel_ksplit4
//
// K-split=4: each V column served by 4 adjacent lanes.
//   lane 4i+j: h[j*BK/4 : (j+1)*BK/4-1, v_col=i]   (j=0,1,2,3)
// Reduction: __shfl_xor(...,1) then __shfl_xor(...,2).
//
// Grid:  (1, cdiv(V, BV/4), B * HV)
// Block: BV threads = 1 wavefront
// State layout: SWIZZLED (N_states, HV, K/4, V, 4)
// ---------------------------------------------------------------------------

template <int BK, int BV, bool USE_INITIAL_STATE, bool USE_QK_L2NORM>
__global__ __launch_bounds__(BV)
void fused_split_gdr_update_kernel_ksplit4(
    const __hip_bfloat16* __restrict__ mixed_qkv,
    const float*          __restrict__ A_log,
    const __hip_bfloat16* __restrict__ a,
    const __hip_bfloat16* __restrict__ dt_bias,
    float                             softplus_beta,
    float                             softplus_threshold,
    const __hip_bfloat16* __restrict__ b_gate,
    __hip_bfloat16*       __restrict__ o,
    float*                __restrict__ h0_source,
    const int32_t*        __restrict__ h0_indices,
    int T,
    int key_dim,
    int value_dim,
    int64_t stride_x_batch,
    int64_t stride_x_dim,
    int64_t stride_x_seq,
    int64_t stride_o_batch,
    int64_t stride_o_seq,
    int64_t stride_o_head,
    int64_t stride_o_dim,
    int B,
    int H,
    int HV,
    int K,
    int V_dim,
    float scale
) {
    static_assert(BK % 4 == 0, "BK must be divisible by 4");
    static_assert(BK >= 16, "BK must be >= 16 for K_SPLIT=4");

    constexpr int BK_QTR  = BK / 4;
    constexpr int BK4_QTR = BK_QTR / 4;
    constexpr int BV_OUT  = BV / 4;
    constexpr int SMEM_PAD = (BK_QTR == 32) ? 1 : 0;
    constexpr int SMEM_STRIDE = BK_QTR + SMEM_PAD;
    constexpr int SMEM_SIZE = SMEM_STRIDE * 4;

    const int i_v  = blockIdx.y;
    const int i_nh = blockIdx.z;
    const int i_n  = i_nh / HV;
    const int i_hv = i_nh % HV;
    const int i_h  = i_hv / (HV / H);
    const int lane = threadIdx.x;

    const int v_idx   = lane & (BV_OUT - 1);
    const int k_split = lane >> 4;
    const int v_col   = i_v * BV_OUT + v_idx;
    const int k_start_logical = k_split * BK_QTR;
    const int k_start = k_split * SMEM_STRIDE;

    __shared__ float smem[SMEM_SIZE];

    // Pre-load A_log and dt_bias BEFORE h0 state loads so that HBM latency
    // (~400-600 cyc) is hidden behind the 8x global_load_dwordx4 for h_vec.
    const float a_log_val   = A_log[i_hv];
    const float dt_bias_val = static_cast<float>(dt_bias[i_hv]);

    float4 h_vec[BK4_QTR];
    #pragma unroll
    for (int i = 0; i < BK4_QTR; i++) {
        h_vec[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    if constexpr (USE_INITIAL_STATE) {
        const int32_t idx = h0_indices[i_n];
        if (idx >= 0) {
            const int K4 = K / 4;
            const float4* h0_base = reinterpret_cast<const float4*>(
                h0_source
                + static_cast<int64_t>(idx) * HV * K4 * V_dim * 4
                + static_cast<int64_t>(i_hv) * K4 * V_dim * 4
            );
            const int kg_start = k_start_logical / 4;
            #pragma unroll
            for (int i = 0; i < BK4_QTR; i++) {
                h_vec[i] = h0_base[(kg_start + i) * V_dim + v_col];
            }
        }
    }

    const float neg_exp_A_log     = -expf(a_log_val);
    const float inv_softplus_beta = 1.0f / softplus_beta;

    const int q_dim_off = i_h  * K;
    const int k_dim_off = key_dim + i_h * K;
    const int v_dim_off = 2 * key_dim + i_hv * V_dim;

    const __hip_bfloat16* x_base =
        mixed_qkv + static_cast<int64_t>(i_n) * stride_x_batch;
    const __hip_bfloat16* p_a =
        a      + static_cast<int64_t>(i_n) * T * HV + i_hv;
    const __hip_bfloat16* p_b =
        b_gate + static_cast<int64_t>(i_n) * T * HV + i_hv;
    __hip_bfloat16* p_o =
        o + static_cast<int64_t>(i_n) * stride_o_batch
          + static_cast<int64_t>(i_hv) * stride_o_head;

    const bool use_vec2 = (stride_x_dim == 1);

    for (int t = 0; t < T; t++) {
        const __hip_bfloat16* x_t =
            x_base + static_cast<int64_t>(t) * stride_x_seq;

        const __hip_bfloat16 v_raw =
            x_t[static_cast<int64_t>(v_dim_off + v_col) * stride_x_dim];
        const __hip_bfloat16 a_raw = p_a[static_cast<int64_t>(t) * HV];
        const __hip_bfloat16 b_raw = p_b[static_cast<int64_t>(t) * HV];

        if (use_vec2) {
            const int base = 2 * lane;
            if constexpr (SMEM_PAD) {
                load_bf16x2_padded_fast<BK_QTR>(smem, x_t + k_dim_off, base);
            } else {
                load_bf16x2(smem, x_t + k_dim_off, base, K, BK);
            }
        } else {
            for (int i = lane; i < BK; i += BV) {
                const float val = static_cast<float>(
                    x_t[static_cast<int64_t>(k_dim_off + i) * stride_x_dim]);
                if constexpr (SMEM_PAD) {
                    smem[i + i / BK_QTR] = val;
                } else {
                    smem[i] = val;
                }
            }
        }
        __syncthreads();

        float v_local = static_cast<float>(v_raw);
        const float a_t = static_cast<float>(a_raw);
        const float b_t = static_cast<float>(b_raw);

        const float sp  = hip_softplus(
            a_t + dt_bias_val, softplus_beta, inv_softplus_beta, softplus_threshold);
        const float g     = neg_exp_A_log * sp;
        const float exp_g = expf(g);
        const float beta  = hip_sigmoid(b_t);

        // Phase 1: Pipelined K-L2-norm + Delta dot
        float4 k_cache[BK4_QTR];
        float k_inv_norm = 1.0f;
        {
            float dot_partial = 0.0f;
            float k_sq = 0.0f;

            float pf0 = smem[k_start], pf1 = smem[k_start + 1];
            float pf2 = smem[k_start + 2], pf3 = smem[k_start + 3];

            #pragma unroll
            for (int i = 0; i < BK4_QTR; i++) {
                const float s0 = pf0, s1 = pf1, s2 = pf2, s3 = pf3;

                if (i + 1 < BK4_QTR) {
                    const int k0_next = k_start + (i + 1) * 4;
                    pf0 = smem[k0_next];     pf1 = smem[k0_next + 1];
                    pf2 = smem[k0_next + 2]; pf3 = smem[k0_next + 3];
                }

                k_cache[i] = make_float4(s0, s1, s2, s3);
                dot_partial += h_vec[i].x * s0 + h_vec[i].y * s1
                             + h_vec[i].z * s2 + h_vec[i].w * s3;
                if constexpr (USE_QK_L2NORM) {
                    k_sq += s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3;
                }
            }
            dot_partial += __shfl_xor(dot_partial, 16, 64);
            dot_partial += __shfl_xor(dot_partial, 32, 64);
            if constexpr (USE_QK_L2NORM) {
                k_sq += __shfl_xor(k_sq, 16, 64);
                k_sq += __shfl_xor(k_sq, 32, 64);
                k_inv_norm = rsqrtf(k_sq + 1e-6f);
            }
            v_local -= dot_partial * exp_g * k_inv_norm;
        }

        v_local *= beta;

        // Phase 2: Fused Decay + State update
        {
            const float kv = k_inv_norm * v_local;
            #pragma unroll
            for (int i = 0; i < BK4_QTR; i++) {
                h_vec[i].x = h_vec[i].x * exp_g + k_cache[i].x * kv;
                h_vec[i].y = h_vec[i].y * exp_g + k_cache[i].y * kv;
                h_vec[i].z = h_vec[i].z * exp_g + k_cache[i].z * kv;
                h_vec[i].w = h_vec[i].w * exp_g + k_cache[i].w * kv;
            }
        }

        // Load Q into smem
        if (use_vec2) {
            const int base = 2 * lane;
            if constexpr (SMEM_PAD) {
                load_bf16x2_padded_fast<BK_QTR>(smem, x_t + q_dim_off, base);
            } else {
                load_bf16x2(smem, x_t + q_dim_off, base, K, BK);
            }
        } else {
            for (int i = lane; i < BK; i += BV) {
                const float val = static_cast<float>(
                    x_t[static_cast<int64_t>(q_dim_off + i) * stride_x_dim]);
                if constexpr (SMEM_PAD) {
                    smem[i + i / BK_QTR] = val;
                } else {
                    smem[i] = val;
                }
            }
        }
        __syncthreads();

        // Phase 3: Pipelined Q-L2-norm + Output dot
        {
            float out_partial = 0.0f;
            float q_sq = 0.0f;

            float qf0 = smem[k_start], qf1 = smem[k_start + 1];
            float qf2 = smem[k_start + 2], qf3 = smem[k_start + 3];

            #pragma unroll
            for (int i = 0; i < BK4_QTR; i++) {
                const float s0 = qf0, s1 = qf1, s2 = qf2, s3 = qf3;

                if (i + 1 < BK4_QTR) {
                    const int k0_next = k_start + (i + 1) * 4;
                    qf0 = smem[k0_next];     qf1 = smem[k0_next + 1];
                    qf2 = smem[k0_next + 2]; qf3 = smem[k0_next + 3];
                }

                out_partial += h_vec[i].x * s0 + h_vec[i].y * s1
                             + h_vec[i].z * s2 + h_vec[i].w * s3;
                if constexpr (USE_QK_L2NORM) {
                    q_sq += s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3;
                }
            }
            out_partial += __shfl_xor(out_partial, 16, 64);
            out_partial += __shfl_xor(out_partial, 32, 64);
            float q_inv_norm = 1.0f;
            if constexpr (USE_QK_L2NORM) {
                q_sq += __shfl_xor(q_sq, 16, 64);
                q_sq += __shfl_xor(q_sq, 32, 64);
                q_inv_norm = rsqrtf(q_sq + 1e-6f);
            }
            float o_local = out_partial * q_inv_norm * scale;

            if (k_split == 0) {
                p_o[static_cast<int64_t>(t) * stride_o_seq
                    + static_cast<int64_t>(v_col) * stride_o_dim] =
                    static_cast<__hip_bfloat16>(o_local);
            }
        }
    }

    if constexpr (USE_INITIAL_STATE) {
        const int32_t idx = h0_indices[i_n];
        if (idx >= 0) {
            const int K4 = K / 4;
            float4* h0_base = reinterpret_cast<float4*>(
                h0_source
                + static_cast<int64_t>(idx) * HV * K4 * V_dim * 4
                + static_cast<int64_t>(i_hv) * K4 * V_dim * 4
            );
            const int kg_start = k_start_logical / 4;
            #pragma unroll
            for (int i = 0; i < BK4_QTR; i++) {
                h0_base[(kg_start + i) * V_dim + v_col] = h_vec[i];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel: fused_split_gdr_update_kernel_ksplit4_opt
//
// LDS-read-pipelined variant of ksplit4.
// Phase 1 and Phase 3 use software-pipelined LDS reads: the next iteration's
// ds_read is issued BEFORE computing on the current iteration's data, so that
// LDS latency (~20-40 cyc) is hidden behind FMA computation.
//
// Pattern per unrolled iteration:
//   1. Move prefetched data to working regs
//   2. Issue ds_read for next iteration (overlaps with step 3)
//   3. Compute FMA / dot / norm with current data
//
// Assumes K == BK and V_dim % (BV/4) == 0 (no boundary checks needed).
//
// Grid:  (1, cdiv(V, BV/4), B * HV)
// Block: BV threads = 1 wavefront
// State layout: SWIZZLED (N_states, HV, K/4, V, 4)
// ---------------------------------------------------------------------------

// Simplified vec2 bf16 load for K==BK case — no boundary fallback needed.
template <int CHUNK>
__device__ __forceinline__ void load_bf16x2_padded_fast(
    float* __restrict__ smem,
    const __hip_bfloat16* __restrict__ src,
    int base)
{
    const int p0 = lds_padded_idx<CHUNK>(base);
    const int p1 = lds_padded_idx<CHUNK>(base + 1);
    uint32_t packed = *reinterpret_cast<const uint32_t*>(src + base);
    const __hip_bfloat16* v = reinterpret_cast<const __hip_bfloat16*>(&packed);
    smem[p0] = static_cast<float>(v[0]);
    smem[p1] = static_cast<float>(v[1]);
}

template <int BK, int BV, bool USE_INITIAL_STATE, bool USE_QK_L2NORM>
__global__ __launch_bounds__(BV)
void fused_split_gdr_update_kernel_ksplit4_opt(
    const __hip_bfloat16* __restrict__ mixed_qkv,
    const float*          __restrict__ A_log,
    const __hip_bfloat16* __restrict__ a,
    const __hip_bfloat16* __restrict__ dt_bias,
    float                             softplus_beta,
    float                             softplus_threshold,
    const __hip_bfloat16* __restrict__ b_gate,
    __hip_bfloat16*       __restrict__ o,
    float*                __restrict__ h0_source,
    const int32_t*        __restrict__ h0_indices,
    int T,
    int key_dim,
    int value_dim,
    int64_t stride_x_batch,
    int64_t stride_x_dim,
    int64_t stride_x_seq,
    int64_t stride_o_batch,
    int64_t stride_o_seq,
    int64_t stride_o_head,
    int64_t stride_o_dim,
    int B,
    int H,
    int HV,
    int K,
    int V_dim,
    float scale
) {
    static_assert(BK % 4 == 0, "BK must be divisible by 4");
    static_assert(BK >= 16, "BK must be >= 16 for K_SPLIT=4");

    constexpr int BK_QTR  = BK / 4;
    constexpr int BK4_QTR = BK_QTR / 4;
    constexpr int BV_OUT  = BV / 4;
    constexpr int SMEM_PAD = (BK_QTR == 32) ? 1 : 0;
    constexpr int SMEM_STRIDE = BK_QTR + SMEM_PAD;
    constexpr int SMEM_SIZE = SMEM_STRIDE * 4;

    const int i_v  = blockIdx.y;
    const int i_nh = blockIdx.z;
    const int i_n  = i_nh / HV;
    const int i_hv = i_nh % HV;
    const int i_h  = i_hv / (HV / H);
    const int lane = threadIdx.x;

    const int v_idx   = lane & (BV_OUT - 1);
    const int k_split = lane >> 4;
    const int v_col   = i_v * BV_OUT + v_idx;
    const int k_start_logical = k_split * BK_QTR;
    const int k_start = k_split * SMEM_STRIDE;

    __shared__ float smem[SMEM_SIZE];

    const float a_log_val   = A_log[i_hv];
    const float dt_bias_val = static_cast<float>(dt_bias[i_hv]);

    const int q_dim_off = i_h  * K;
    const int k_dim_off = key_dim + i_h * K;
    const int v_dim_off = 2 * key_dim + i_hv * V_dim;

    const __hip_bfloat16* x_base =
        mixed_qkv + static_cast<int64_t>(i_n) * stride_x_batch;
    const __hip_bfloat16* p_a =
        a      + static_cast<int64_t>(i_n) * T * HV + i_hv;
    const __hip_bfloat16* p_b =
        b_gate + static_cast<int64_t>(i_n) * T * HV + i_hv;
    __hip_bfloat16* p_o =
        o + static_cast<int64_t>(i_n) * stride_o_batch
          + static_cast<int64_t>(i_hv) * stride_o_head;

    const bool use_vec2 = (stride_x_dim == 1);

    // Issue K cooperative load for t=0 BEFORE h0 state loads so that K's
    // HBM latency (~630 cyc) is hidden behind the 8x global_load_dwordx4.
    const __hip_bfloat16* x_t0 = x_base;
    if (use_vec2) {
        const int base = 2 * lane;
        if constexpr (SMEM_PAD) {
            load_bf16x2_padded_fast<BK_QTR>(smem, x_t0 + k_dim_off, base);
        } else {
            load_bf16x2(smem, x_t0 + k_dim_off, base, K, BK);
        }
    } else {
        for (int i = lane; i < BK; i += BV) {
            const float val = static_cast<float>(
                x_t0[static_cast<int64_t>(k_dim_off + i) * stride_x_dim]);
            if constexpr (SMEM_PAD) {
                smem[i + i / BK_QTR] = val;
            } else {
                smem[i] = val;
            }
        }
    }

    // h0 state loads overlap with K load's HBM latency above
    float4 h_vec[BK4_QTR];
    #pragma unroll
    for (int i = 0; i < BK4_QTR; i++) {
        h_vec[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    if constexpr (USE_INITIAL_STATE) {
        const int32_t idx = h0_indices[i_n];
        if (idx >= 0) {
            const int K4 = K / 4;
            const float4* h0_base = reinterpret_cast<const float4*>(
                h0_source
                + static_cast<int64_t>(idx) * HV * K4 * V_dim * 4
                + static_cast<int64_t>(i_hv) * K4 * V_dim * 4
            );
            const int kg_start = k_start_logical / 4;
            #pragma unroll
            for (int i = 0; i < BK4_QTR; i++) {
                h_vec[i] = h0_base[(kg_start + i) * V_dim + v_col];
            }
        }
    }

    const float neg_exp_A_log     = -expf(a_log_val);
    const float inv_softplus_beta = 1.0f / softplus_beta;

    // t=0: K already loaded to smem above; t>0: load K at loop top
    for (int t = 0; t < T; t++) {
        const __hip_bfloat16* x_t =
            x_base + static_cast<int64_t>(t) * stride_x_seq;

        const __hip_bfloat16 v_raw =
            x_t[static_cast<int64_t>(v_dim_off + v_col) * stride_x_dim];
        const __hip_bfloat16 a_raw = p_a[static_cast<int64_t>(t) * HV];
        const __hip_bfloat16 b_raw = p_b[static_cast<int64_t>(t) * HV];

        if (t > 0) {
            if (use_vec2) {
                const int base = 2 * lane;
                if constexpr (SMEM_PAD) {
                    load_bf16x2_padded_fast<BK_QTR>(smem, x_t + k_dim_off, base);
                } else {
                    load_bf16x2(smem, x_t + k_dim_off, base, K, BK);
                }
            } else {
                for (int i = lane; i < BK; i += BV) {
                    const float val = static_cast<float>(
                        x_t[static_cast<int64_t>(k_dim_off + i) * stride_x_dim]);
                    if constexpr (SMEM_PAD) {
                        smem[i + i / BK_QTR] = val;
                    } else {
                        smem[i] = val;
                    }
                }
            }
        }
        __syncthreads();

        float v_local = static_cast<float>(v_raw);
        const float a_t = static_cast<float>(a_raw);
        const float b_t = static_cast<float>(b_raw);

        const float sp  = hip_softplus(
            a_t + dt_bias_val, softplus_beta, inv_softplus_beta, softplus_threshold);
        const float g     = neg_exp_A_log * sp;
        const float exp_g = expf(g);
        const float beta  = hip_sigmoid(b_t);

        // Phase 1: Pipelined K-L2-norm + Delta dot
        //   Prefetch next iteration's LDS data while computing on current.
        float4 k_cache[BK4_QTR];
        float k_inv_norm = 1.0f;
        {
            float dot_partial = 0.0f;
            float k_sq = 0.0f;

            float pf0 = smem[k_start], pf1 = smem[k_start + 1];
            float pf2 = smem[k_start + 2], pf3 = smem[k_start + 3];

            #pragma unroll
            for (int i = 0; i < BK4_QTR; i++) {
                const float s0 = pf0, s1 = pf1, s2 = pf2, s3 = pf3;

                if (i + 1 < BK4_QTR) {
                    const int k0_next = k_start + (i + 1) * 4;
                    pf0 = smem[k0_next];     pf1 = smem[k0_next + 1];
                    pf2 = smem[k0_next + 2]; pf3 = smem[k0_next + 3];
                }

                k_cache[i] = make_float4(s0, s1, s2, s3);
                dot_partial += h_vec[i].x * s0 + h_vec[i].y * s1
                             + h_vec[i].z * s2 + h_vec[i].w * s3;
                if constexpr (USE_QK_L2NORM) {
                    k_sq += s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3;
                }
            }
            dot_partial += __shfl_xor(dot_partial, 16, 64);
            dot_partial += __shfl_xor(dot_partial, 32, 64);
            if constexpr (USE_QK_L2NORM) {
                k_sq += __shfl_xor(k_sq, 16, 64);
                k_sq += __shfl_xor(k_sq, 32, 64);
                k_inv_norm = rsqrtf(k_sq + 1e-6f);
            }
            v_local -= dot_partial * exp_g * k_inv_norm;
        }

        v_local *= beta;

        // Phase 2: Fused Decay + State update (reuse cached K, no LDS re-read)
        {
            const float kv = k_inv_norm * v_local;
            #pragma unroll
            for (int i = 0; i < BK4_QTR; i++) {
                h_vec[i].x = h_vec[i].x * exp_g + k_cache[i].x * kv;
                h_vec[i].y = h_vec[i].y * exp_g + k_cache[i].y * kv;
                h_vec[i].z = h_vec[i].z * exp_g + k_cache[i].z * kv;
                h_vec[i].w = h_vec[i].w * exp_g + k_cache[i].w * kv;
            }
        }

        // Load Q into smem — no barrier needed here because BV == wavefront
        // size (64 on gfx942): single-wavefront lockstep guarantees all K
        // smem reads in Phase 1 complete before any Q smem writes begin.
        if (use_vec2) {
            const int base = 2 * lane;
            if constexpr (SMEM_PAD) {
                load_bf16x2_padded_fast<BK_QTR>(smem, x_t + q_dim_off, base);
            } else {
                load_bf16x2(smem, x_t + q_dim_off, base, K, BK);
            }
        } else {
            for (int i = lane; i < BK; i += BV) {
                const float val = static_cast<float>(
                    x_t[static_cast<int64_t>(q_dim_off + i) * stride_x_dim]);
                if constexpr (SMEM_PAD) {
                    smem[i + i / BK_QTR] = val;
                } else {
                    smem[i] = val;
                }
            }
        }
        __syncthreads();

        // Phase 3: Pipelined Q-L2-norm + Output dot
        //   Same prefetch technique as Phase 1.
        {
            float out_partial = 0.0f;
            float q_sq = 0.0f;

            float qf0 = smem[k_start], qf1 = smem[k_start + 1];
            float qf2 = smem[k_start + 2], qf3 = smem[k_start + 3];

            #pragma unroll
            for (int i = 0; i < BK4_QTR; i++) {
                const float s0 = qf0, s1 = qf1, s2 = qf2, s3 = qf3;

                if (i + 1 < BK4_QTR) {
                    const int k0_next = k_start + (i + 1) * 4;
                    qf0 = smem[k0_next];     qf1 = smem[k0_next + 1];
                    qf2 = smem[k0_next + 2]; qf3 = smem[k0_next + 3];
                }

                out_partial += h_vec[i].x * s0 + h_vec[i].y * s1
                             + h_vec[i].z * s2 + h_vec[i].w * s3;
                if constexpr (USE_QK_L2NORM) {
                    q_sq += s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3;
                }
            }
            out_partial += __shfl_xor(out_partial, 16, 64);
            out_partial += __shfl_xor(out_partial, 32, 64);
            float q_inv_norm = 1.0f;
            if constexpr (USE_QK_L2NORM) {
                q_sq += __shfl_xor(q_sq, 16, 64);
                q_sq += __shfl_xor(q_sq, 32, 64);
                q_inv_norm = rsqrtf(q_sq + 1e-6f);
            }
            float o_local = out_partial * q_inv_norm * scale;

            if (k_split == 0) {
                p_o[static_cast<int64_t>(t) * stride_o_seq
                    + static_cast<int64_t>(v_col) * stride_o_dim] =
                    static_cast<__hip_bfloat16>(o_local);
            }
        }
    }

    if constexpr (USE_INITIAL_STATE) {
        const int32_t idx = h0_indices[i_n];
        if (idx >= 0) {
            const int K4 = K / 4;
            float4* h0_base = reinterpret_cast<float4*>(
                h0_source
                + static_cast<int64_t>(idx) * HV * K4 * V_dim * 4
                + static_cast<int64_t>(i_hv) * K4 * V_dim * 4
            );
            const int kg_start = k_start_logical / 4;
            #pragma unroll
            for (int i = 0; i < BK4_QTR; i++) {
                h0_base[(kg_start + i) * V_dim + v_col] = h_vec[i];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel: fused_split_gdr_update_kernel_ksplit4_db
//
// Double-buffer variant of ksplit4: uses separate smem_k and smem_q buffers
// so Q and K cooperative loads are issued in parallel at the start of each
// iteration.  Global loads are split from LDS writes to maximize overlap.
// This eliminates the separate Q-load stall (~502 cyc) at the cost of 2x smem.
//
// Grid:  (1, cdiv(V, BV/4), B * HV)
// Block: BV threads = 1 wavefront
// State layout: SWIZZLED (N_states, HV, K/4, V, 4)
// ---------------------------------------------------------------------------

template <int BK, int BV, bool USE_INITIAL_STATE, bool USE_QK_L2NORM>
__global__ __launch_bounds__(BV)
void fused_split_gdr_update_kernel_ksplit4_db(
    const __hip_bfloat16* __restrict__ mixed_qkv,
    const float*          __restrict__ A_log,
    const __hip_bfloat16* __restrict__ a,
    const __hip_bfloat16* __restrict__ dt_bias,
    float                             softplus_beta,
    float                             softplus_threshold,
    const __hip_bfloat16* __restrict__ b_gate,
    __hip_bfloat16*       __restrict__ o,
    float*                __restrict__ h0_source,
    const int32_t*        __restrict__ h0_indices,
    int T,
    int key_dim,
    int value_dim,
    int64_t stride_x_batch,
    int64_t stride_x_dim,
    int64_t stride_x_seq,
    int64_t stride_o_batch,
    int64_t stride_o_seq,
    int64_t stride_o_head,
    int64_t stride_o_dim,
    int B,
    int H,
    int HV,
    int K,
    int V_dim,
    float scale
) {
    static_assert(BK % 4 == 0, "BK must be divisible by 4");
    static_assert(BK >= 16, "BK must be >= 16 for K_SPLIT=4");

    constexpr int BK_QTR  = BK / 4;
    constexpr int BK4_QTR = BK_QTR / 4;
    constexpr int BV_OUT  = BV / 4;
    constexpr int SMEM_PAD = (BK_QTR == 32) ? 1 : 0;
    constexpr int SMEM_STRIDE = BK_QTR + SMEM_PAD;
    constexpr int SMEM_SIZE = SMEM_STRIDE * 4;

    const int i_v  = blockIdx.y;
    const int i_nh = blockIdx.z;
    const int i_n  = i_nh / HV;
    const int i_hv = i_nh % HV;
    const int i_h  = i_hv / (HV / H);
    const int lane = threadIdx.x;

    const int v_idx   = lane & (BV_OUT - 1);
    const int k_split = lane >> 4;
    const int v_col   = i_v * BV_OUT + v_idx;
    const int k_start_logical = k_split * BK_QTR;
    const int k_start = k_split * SMEM_STRIDE;

    __shared__ float smem_k[SMEM_SIZE];
    __shared__ float smem_q[SMEM_SIZE];

    const float a_log_val   = A_log[i_hv];
    const float dt_bias_val = static_cast<float>(dt_bias[i_hv]);

    float4 h_vec[BK4_QTR];
    #pragma unroll
    for (int i = 0; i < BK4_QTR; i++) {
        h_vec[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    if constexpr (USE_INITIAL_STATE) {
        const int32_t idx = h0_indices[i_n];
        if (idx >= 0) {
            const int K4 = K / 4;
            const float4* h0_base = reinterpret_cast<const float4*>(
                h0_source
                + static_cast<int64_t>(idx) * HV * K4 * V_dim * 4
                + static_cast<int64_t>(i_hv) * K4 * V_dim * 4
            );
            const int kg_start = k_start_logical / 4;
            #pragma unroll
            for (int i = 0; i < BK4_QTR; i++) {
                h_vec[i] = h0_base[(kg_start + i) * V_dim + v_col];
            }
        }
    }

    const float neg_exp_A_log     = -expf(a_log_val);
    const float inv_softplus_beta = 1.0f / softplus_beta;

    const int q_dim_off = i_h  * K;
    const int k_dim_off = key_dim + i_h * K;
    const int v_dim_off = 2 * key_dim + i_hv * V_dim;

    const __hip_bfloat16* x_base =
        mixed_qkv + static_cast<int64_t>(i_n) * stride_x_batch;
    const __hip_bfloat16* p_a =
        a      + static_cast<int64_t>(i_n) * T * HV + i_hv;
    const __hip_bfloat16* p_b =
        b_gate + static_cast<int64_t>(i_n) * T * HV + i_hv;
    __hip_bfloat16* p_o =
        o + static_cast<int64_t>(i_n) * stride_o_batch
          + static_cast<int64_t>(i_hv) * stride_o_head;

    const bool use_vec2 = (stride_x_dim == 1);

    for (int t = 0; t < T; t++) {
        const __hip_bfloat16* x_t =
            x_base + static_cast<int64_t>(t) * stride_x_seq;

        if (use_vec2) {
            const int base = 2 * lane;
            if constexpr (SMEM_PAD) {
                load_bf16x2_padded_fast<BK_QTR>(smem_k, x_t + k_dim_off, base);
                load_bf16x2_padded_fast<BK_QTR>(smem_q, x_t + q_dim_off, base);
            } else {
                load_bf16x2(smem_k, x_t + k_dim_off, base, K, BK);
                load_bf16x2(smem_q, x_t + q_dim_off, base, K, BK);
            }
        } else {
            for (int i = lane; i < BK; i += BV) {
                const float q_val = static_cast<float>(
                    x_t[static_cast<int64_t>(q_dim_off + i) * stride_x_dim]);
                const float k_val = static_cast<float>(
                    x_t[static_cast<int64_t>(k_dim_off + i) * stride_x_dim]);
                if constexpr (SMEM_PAD) {
                    smem_q[i + i / BK_QTR] = q_val;
                    smem_k[i + i / BK_QTR] = k_val;
                } else {
                    smem_q[i] = q_val;
                    smem_k[i] = k_val;
                }
            }
        }

        const __hip_bfloat16 v_raw =
            x_t[static_cast<int64_t>(v_dim_off + v_col) * stride_x_dim];
        const __hip_bfloat16 a_raw = p_a[static_cast<int64_t>(t) * HV];
        const __hip_bfloat16 b_raw = p_b[static_cast<int64_t>(t) * HV];

        __syncthreads();

        // Phase 1: Pipelined K-L2-norm + Delta dot
        float4 k_cache[BK4_QTR];
        float k_inv_norm = 1.0f;
        float dot_partial;
        {
            float dp = 0.0f;
            float k_sq = 0.0f;

            float pf0 = smem_k[k_start], pf1 = smem_k[k_start + 1];
            float pf2 = smem_k[k_start + 2], pf3 = smem_k[k_start + 3];

            #pragma unroll
            for (int i = 0; i < BK4_QTR; i++) {
                const float s0 = pf0, s1 = pf1, s2 = pf2, s3 = pf3;

                if (i + 1 < BK4_QTR) {
                    const int k0_next = k_start + (i + 1) * 4;
                    pf0 = smem_k[k0_next];     pf1 = smem_k[k0_next + 1];
                    pf2 = smem_k[k0_next + 2]; pf3 = smem_k[k0_next + 3];
                }

                k_cache[i] = make_float4(s0, s1, s2, s3);
                dp += h_vec[i].x * s0 + h_vec[i].y * s1
                    + h_vec[i].z * s2 + h_vec[i].w * s3;
                if constexpr (USE_QK_L2NORM) {
                    k_sq += s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3;
                }
            }
            dp += __shfl_xor(dp, 16, 64);
            dp += __shfl_xor(dp, 32, 64);
            if constexpr (USE_QK_L2NORM) {
                k_sq += __shfl_xor(k_sq, 16, 64);
                k_sq += __shfl_xor(k_sq, 32, 64);
                k_inv_norm = rsqrtf(k_sq + 1e-6f);
            }
            dot_partial = dp;
        }

        float v_local = static_cast<float>(v_raw);
        const float a_t = static_cast<float>(a_raw);
        const float b_t = static_cast<float>(b_raw);

        const float sp  = hip_softplus(
            a_t + dt_bias_val, softplus_beta, inv_softplus_beta, softplus_threshold);
        const float g     = neg_exp_A_log * sp;
        const float exp_g = expf(g);
        const float beta  = hip_sigmoid(b_t);

        v_local -= dot_partial * exp_g * k_inv_norm;

        v_local *= beta;

        // Phase 2: Fused Decay + State update (reuse cached K, no LDS re-read)
        {
            const float kv = k_inv_norm * v_local;
            #pragma unroll
            for (int i = 0; i < BK4_QTR; i++) {
                h_vec[i].x = h_vec[i].x * exp_g + k_cache[i].x * kv;
                h_vec[i].y = h_vec[i].y * exp_g + k_cache[i].y * kv;
                h_vec[i].z = h_vec[i].z * exp_g + k_cache[i].z * kv;
                h_vec[i].w = h_vec[i].w * exp_g + k_cache[i].w * kv;
            }
        }

        // Phase 3: Pipelined Q-L2-norm + Output dot (Q already in smem_q)
        {
            float out_partial = 0.0f;
            float q_sq = 0.0f;

            float qf0 = smem_q[k_start], qf1 = smem_q[k_start + 1];
            float qf2 = smem_q[k_start + 2], qf3 = smem_q[k_start + 3];

            #pragma unroll
            for (int i = 0; i < BK4_QTR; i++) {
                const float s0 = qf0, s1 = qf1, s2 = qf2, s3 = qf3;

                if (i + 1 < BK4_QTR) {
                    const int k0_next = k_start + (i + 1) * 4;
                    qf0 = smem_q[k0_next];     qf1 = smem_q[k0_next + 1];
                    qf2 = smem_q[k0_next + 2]; qf3 = smem_q[k0_next + 3];
                }

                out_partial += h_vec[i].x * s0 + h_vec[i].y * s1
                             + h_vec[i].z * s2 + h_vec[i].w * s3;
                if constexpr (USE_QK_L2NORM) {
                    q_sq += s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3;
                }
            }
            out_partial += __shfl_xor(out_partial, 16, 64);
            out_partial += __shfl_xor(out_partial, 32, 64);
            float q_inv_norm = 1.0f;
            if constexpr (USE_QK_L2NORM) {
                q_sq += __shfl_xor(q_sq, 16, 64);
                q_sq += __shfl_xor(q_sq, 32, 64);
                q_inv_norm = rsqrtf(q_sq + 1e-6f);
            }
            float o_local = out_partial * q_inv_norm * scale;

            if (k_split == 0) {
                p_o[static_cast<int64_t>(t) * stride_o_seq
                    + static_cast<int64_t>(v_col) * stride_o_dim] =
                    static_cast<__hip_bfloat16>(o_local);
            }
        }
    }

    if constexpr (USE_INITIAL_STATE) {
        const int32_t idx = h0_indices[i_n];
        if (idx >= 0) {
            const int K4 = K / 4;
            float4* h0_base = reinterpret_cast<float4*>(
                h0_source
                + static_cast<int64_t>(idx) * HV * K4 * V_dim * 4
                + static_cast<int64_t>(i_hv) * K4 * V_dim * 4
            );
            const int kg_start = k_start_logical / 4;
            #pragma unroll
            for (int i = 0; i < BK4_QTR; i++) {
                h0_base[(kg_start + i) * V_dim + v_col] = h_vec[i];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel: fused_split_gdr_update_kernel_ksplit8
//
// K-split=8: 8 threads cooperate on K reduction, each handling BK/8 K
// elements.  Thread layout is row-major (V first, then K):
//   v_idx   = lane & 7     (8 V threads)
//   k_split = lane >> 3    (8 K threads)
// Threads 0, 8, 16, 24, 32, 40, 48, 56 reduce the same V column.
//
// Each block computes BK x (BV/8) state elements.  Grid y-dimension is
// doubled vs ksplit4: cdiv(V, BV/8).
//
// Grid:  (1, cdiv(V, BV/8), B * HV)
// Block: BV threads = 1 wavefront
// State layout: SWIZZLED (N_states, HV, K/4, V, 4)
// ---------------------------------------------------------------------------

template <int BK, int BV, bool USE_INITIAL_STATE, bool USE_QK_L2NORM>
__global__ __launch_bounds__(BV)
void fused_split_gdr_update_kernel_ksplit8(
    const __hip_bfloat16* __restrict__ mixed_qkv,
    const float*          __restrict__ A_log,
    const __hip_bfloat16* __restrict__ a,
    const __hip_bfloat16* __restrict__ dt_bias,
    float                             softplus_beta,
    float                             softplus_threshold,
    const __hip_bfloat16* __restrict__ b_gate,
    __hip_bfloat16*       __restrict__ o,
    float*                __restrict__ h0_source,
    const int32_t*        __restrict__ h0_indices,
    int T,
    int key_dim,
    int value_dim,
    int64_t stride_x_batch,
    int64_t stride_x_dim,
    int64_t stride_x_seq,
    int64_t stride_o_batch,
    int64_t stride_o_seq,
    int64_t stride_o_head,
    int64_t stride_o_dim,
    int B,
    int H,
    int HV,
    int K,
    int V_dim,
    float scale
) {
    static_assert(BK % 32 == 0, "BK must be divisible by 32 for K_SPLIT=8");

    constexpr int BK_OCT  = BK / 8;
    constexpr int BK4_OCT = BK_OCT / 4;
    constexpr int BV_OUT  = BV / 8;
    constexpr int SMEM_PAD = (BK_OCT >= 8) ? 1 : 0;
    constexpr int SMEM_STRIDE = BK_OCT + SMEM_PAD;
    constexpr int SMEM_SIZE = SMEM_STRIDE * 8;

    const int i_v  = blockIdx.y;
    const int i_nh = blockIdx.z;
    const int i_n  = i_nh / HV;
    const int i_hv = i_nh % HV;
    const int i_h  = i_hv / (HV / H);
    const int lane = threadIdx.x;

    const int v_idx   = lane & (BV_OUT - 1);
    const int k_split = lane >> 3;
    const int v_col   = i_v * BV_OUT + v_idx;
    const bool valid_v = (v_col < V_dim);
    const int k_start_logical = k_split * BK_OCT;
    const int k_start = k_split * SMEM_STRIDE;

    __shared__ float smem[SMEM_SIZE];

    const float a_log_val   = A_log[i_hv];
    const float dt_bias_val = static_cast<float>(dt_bias[i_hv]);

    float4 h_vec[BK4_OCT];
    #pragma unroll
    for (int i = 0; i < BK4_OCT; i++) {
        h_vec[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    if constexpr (USE_INITIAL_STATE) {
        const int32_t idx = h0_indices[i_n];
        if (idx >= 0 && valid_v) {
            const int K4 = K / 4;
            const float4* h0_base = reinterpret_cast<const float4*>(
                h0_source
                + static_cast<int64_t>(idx) * HV * K4 * V_dim * 4
                + static_cast<int64_t>(i_hv) * K4 * V_dim * 4
            );
            const int kg_start = k_start_logical / 4;
            #pragma unroll
            for (int i = 0; i < BK4_OCT; i++) {
                h_vec[i] = h0_base[(kg_start + i) * V_dim + v_col];
            }
        }
    }

    const float neg_exp_A_log     = -expf(a_log_val);
    const float inv_softplus_beta = 1.0f / softplus_beta;

    const int q_dim_off = i_h  * K;
    const int k_dim_off = key_dim + i_h * K;
    const int v_dim_off = 2 * key_dim + i_hv * V_dim;

    const __hip_bfloat16* x_base =
        mixed_qkv + static_cast<int64_t>(i_n) * stride_x_batch;
    const __hip_bfloat16* p_a =
        a      + static_cast<int64_t>(i_n) * T * HV + i_hv;
    const __hip_bfloat16* p_b =
        b_gate + static_cast<int64_t>(i_n) * T * HV + i_hv;
    __hip_bfloat16* p_o =
        o + static_cast<int64_t>(i_n) * stride_o_batch
          + static_cast<int64_t>(i_hv) * stride_o_head;

    const bool use_vec2 = (stride_x_dim == 1);
    const int safe_v_col = valid_v ? v_col : 0;

    for (int t = 0; t < T; t++) {
        const __hip_bfloat16* x_t =
            x_base + static_cast<int64_t>(t) * stride_x_seq;

        // Issue V and gating loads unconditionally BEFORE K cooperative load.
        const __hip_bfloat16 v_raw =
            x_t[static_cast<int64_t>(v_dim_off + safe_v_col) * stride_x_dim];
        const __hip_bfloat16 a_raw = p_a[static_cast<int64_t>(t) * HV];
        const __hip_bfloat16 b_raw = p_b[static_cast<int64_t>(t) * HV];

        // Load K into smem
        if (use_vec2) {
            const int base = 2 * lane;
            if (base < BK) {
                if constexpr (SMEM_PAD) {
                    load_bf16x2_padded<BK_OCT>(smem, x_t + k_dim_off, base, K, BK);
                } else {
                    load_bf16x2(smem, x_t + k_dim_off, base, K, BK);
                }
            }
        } else {
            for (int i = lane; i < BK; i += BV) {
                const float val = (i < K)
                    ? static_cast<float>(
                          x_t[static_cast<int64_t>(k_dim_off + i) * stride_x_dim])
                    : 0.0f;
                if constexpr (SMEM_PAD) {
                    smem[i + i / BK_OCT] = val;
                } else {
                    smem[i] = val;
                }
            }
        }
        __syncthreads();

        float v_local = valid_v ? static_cast<float>(v_raw) : 0.0f;
        const float a_t = static_cast<float>(a_raw);
        const float b_t = static_cast<float>(b_raw);

        const float sp  = hip_softplus(
            a_t + dt_bias_val, softplus_beta, inv_softplus_beta, softplus_threshold);
        const float g     = neg_exp_A_log * sp;
        const float exp_g = expf(g);
        const float beta  = hip_sigmoid(b_t);

        // Phase 1: K-L2-norm + Delta dot, cache K in registers
        float4 k_cache[BK4_OCT];
        float k_inv_norm = 1.0f;
        {
            float dot_partial = 0.0f;
            float k_sq = 0.0f;
            #pragma unroll
            for (int i = 0; i < BK4_OCT; i++) {
                const int k0 = k_start + i * 4;
                const float s0 = smem[k0], s1 = smem[k0 + 1];
                const float s2 = smem[k0 + 2], s3 = smem[k0 + 3];
                k_cache[i] = make_float4(s0, s1, s2, s3);
                dot_partial += h_vec[i].x * s0 + h_vec[i].y * s1
                             + h_vec[i].z * s2 + h_vec[i].w * s3;
                if constexpr (USE_QK_L2NORM) {
                    k_sq += s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3;
                }
            }
            // 8-way K reduction across threads with stride 8
            dot_partial += __shfl_xor(dot_partial, 8, 64);
            dot_partial += __shfl_xor(dot_partial, 16, 64);
            dot_partial += __shfl_xor(dot_partial, 32, 64);
            if constexpr (USE_QK_L2NORM) {
                k_sq += __shfl_xor(k_sq, 8, 64);
                k_sq += __shfl_xor(k_sq, 16, 64);
                k_sq += __shfl_xor(k_sq, 32, 64);
                k_inv_norm = rsqrtf(k_sq + 1e-6f);
            }
            v_local -= dot_partial * exp_g * k_inv_norm;
        }

        v_local *= beta;

        // Phase 2: Decay + State update using k_cache
        {
            const float kv = k_inv_norm * v_local;
            #pragma unroll
            for (int i = 0; i < BK4_OCT; i++) {
                h_vec[i].x = h_vec[i].x * exp_g + k_cache[i].x * kv;
                h_vec[i].y = h_vec[i].y * exp_g + k_cache[i].y * kv;
                h_vec[i].z = h_vec[i].z * exp_g + k_cache[i].z * kv;
                h_vec[i].w = h_vec[i].w * exp_g + k_cache[i].w * kv;
            }
        }

        // Load Q into smem
        __syncthreads();
        if (use_vec2) {
            const int base = 2 * lane;
            if (base < BK) {
                if constexpr (SMEM_PAD) {
                    load_bf16x2_padded<BK_OCT>(smem, x_t + q_dim_off, base, K, BK);
                } else {
                    load_bf16x2(smem, x_t + q_dim_off, base, K, BK);
                }
            }
        } else {
            for (int i = lane; i < BK; i += BV) {
                const float val = (i < K)
                    ? static_cast<float>(
                          x_t[static_cast<int64_t>(q_dim_off + i) * stride_x_dim])
                    : 0.0f;
                if constexpr (SMEM_PAD) {
                    smem[i + i / BK_OCT] = val;
                } else {
                    smem[i] = val;
                }
            }
        }
        __syncthreads();

        // Phase 3: Q-L2-norm + Output dot, 8-way K reduction
        {
            float out_partial = 0.0f;
            float q_sq = 0.0f;
            #pragma unroll
            for (int i = 0; i < BK4_OCT; i++) {
                const int k0 = k_start + i * 4;
                const float s0 = smem[k0], s1 = smem[k0 + 1];
                const float s2 = smem[k0 + 2], s3 = smem[k0 + 3];
                out_partial += h_vec[i].x * s0 + h_vec[i].y * s1
                             + h_vec[i].z * s2 + h_vec[i].w * s3;
                if constexpr (USE_QK_L2NORM) {
                    q_sq += s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3;
                }
            }
            out_partial += __shfl_xor(out_partial, 8, 64);
            out_partial += __shfl_xor(out_partial, 16, 64);
            out_partial += __shfl_xor(out_partial, 32, 64);
            float q_inv_norm = 1.0f;
            if constexpr (USE_QK_L2NORM) {
                q_sq += __shfl_xor(q_sq, 8, 64);
                q_sq += __shfl_xor(q_sq, 16, 64);
                q_sq += __shfl_xor(q_sq, 32, 64);
                q_inv_norm = rsqrtf(q_sq + 1e-6f);
            }
            float o_local = out_partial * q_inv_norm * scale;

            if (k_split == 0 && valid_v) {
                p_o[static_cast<int64_t>(t) * stride_o_seq
                    + static_cast<int64_t>(v_col) * stride_o_dim] =
                    static_cast<__hip_bfloat16>(o_local);
            }
        }
    }

    if constexpr (USE_INITIAL_STATE) {
        const int32_t idx = h0_indices[i_n];
        if (idx >= 0 && valid_v) {
            const int K4 = K / 4;
            float4* h0_base = reinterpret_cast<float4*>(
                h0_source
                + static_cast<int64_t>(idx) * HV * K4 * V_dim * 4
                + static_cast<int64_t>(i_hv) * K4 * V_dim * 4
            );
            const int kg_start = k_start_logical / 4;
            #pragma unroll
            for (int i = 0; i < BK4_OCT; i++) {
                h0_base[(kg_start + i) * V_dim + v_col] = h_vec[i];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel: fused_split_gdr_update_kernel_ksplit4_hfuse
//
// Multi-head fused variant of ksplit4.  When HV > H (GQA), multiple V-heads
// share the same QK head.  This kernel loads K and Q once per timestep,
// caches K in registers (k_cache) and Q in smem, then iterates over
// HPB = HV/H v-heads reusing both.
//
// Grid:  (1, cdiv(V, BV/4), B * H)       ← note: H, not HV
// Block: BV threads = 1 wavefront
// State layout: SWIZZLED (N_states, HV, K/4, V, 4)
// ---------------------------------------------------------------------------

template <int BK, int BV, bool USE_INITIAL_STATE, bool USE_QK_L2NORM>
__global__ __launch_bounds__(BV)
void fused_split_gdr_update_kernel_ksplit4_hfuse(
    const __hip_bfloat16* __restrict__ mixed_qkv,
    const float*          __restrict__ A_log,
    const __hip_bfloat16* __restrict__ a,
    const __hip_bfloat16* __restrict__ dt_bias,
    float                             softplus_beta,
    float                             softplus_threshold,
    const __hip_bfloat16* __restrict__ b_gate,
    __hip_bfloat16*       __restrict__ o,
    float*                __restrict__ h0_source,
    const int32_t*        __restrict__ h0_indices,
    int T,
    int key_dim,
    int value_dim,
    int64_t stride_x_batch,
    int64_t stride_x_dim,
    int64_t stride_x_seq,
    int64_t stride_o_batch,
    int64_t stride_o_seq,
    int64_t stride_o_head,
    int64_t stride_o_dim,
    int B,
    int H,
    int HV,
    int K,
    int V_dim,
    float scale
) {
    static_assert(BK % 4 == 0, "BK must be divisible by 4");
    static_assert(BK >= 16, "BK must be >= 16 for K_SPLIT=4");

    constexpr int BK_QTR  = BK / 4;
    constexpr int BK4_QTR = BK_QTR / 4;
    constexpr int BV_OUT  = BV / 4;
    constexpr int SMEM_PAD = (BK_QTR == 32) ? 1 : 0;
    constexpr int SMEM_STRIDE = BK_QTR + SMEM_PAD;
    constexpr int SMEM_SIZE = SMEM_STRIDE * 4;

    const int i_v  = blockIdx.y;
    const int i_nh = blockIdx.z;
    const int i_n  = i_nh / H;
    const int i_h  = i_nh % H;
    const int HPB  = HV / H;
    const int lane = threadIdx.x;

    const int v_idx   = lane & (BV_OUT - 1);
    const int k_split = lane >> 4;
    const int v_col   = i_v * BV_OUT + v_idx;
    const bool valid_v = (v_col < V_dim);
    const int k_start_logical = k_split * BK_QTR;
    const int k_start = k_split * SMEM_STRIDE;

    __shared__ float smem[SMEM_SIZE];

    const int q_dim_off = i_h * K;
    const int k_dim_off = key_dim + i_h * K;
    const __hip_bfloat16* x_base =
        mixed_qkv + static_cast<int64_t>(i_n) * stride_x_batch;
    const bool use_vec2 = (stride_x_dim == 1);
    const float inv_softplus_beta = 1.0f / softplus_beta;

    const int K4 = K / 4;
    const int kg_start = k_start_logical / 4;
    const int32_t h0_idx = USE_INITIAL_STATE ? h0_indices[i_n] : -1;

    for (int t = 0; t < T; t++) {
        const __hip_bfloat16* x_t =
            x_base + static_cast<int64_t>(t) * stride_x_seq;

        // ---- Load K into smem (once per timestep) ----
        if (t > 0) __syncthreads();
        if (use_vec2) {
            const int base = 2 * lane;
            if (base < BK) {
                if constexpr (SMEM_PAD) {
                    load_bf16x2_padded<BK_QTR>(smem, x_t + k_dim_off, base, K, BK);
                } else {
                    load_bf16x2(smem, x_t + k_dim_off, base, K, BK);
                }
            }
        } else {
            for (int i = lane; i < BK; i += BV) {
                const float val = (i < K)
                    ? static_cast<float>(
                          x_t[static_cast<int64_t>(k_dim_off + i) * stride_x_dim])
                    : 0.0f;
                if constexpr (SMEM_PAD) {
                    smem[i + i / BK_QTR] = val;
                } else {
                    smem[i] = val;
                }
            }
        }
        __syncthreads();

        // ---- Cache K in registers, compute k_inv_norm (shared across heads) ----
        float4 k_cache[BK4_QTR];
        float k_inv_norm = 1.0f;
        {
            float k_sq = 0.0f;
            #pragma unroll
            for (int i = 0; i < BK4_QTR; i++) {
                const int k0 = k_start + i * 4;
                const float s0 = smem[k0], s1 = smem[k0 + 1];
                const float s2 = smem[k0 + 2], s3 = smem[k0 + 3];
                k_cache[i] = make_float4(s0, s1, s2, s3);
                if constexpr (USE_QK_L2NORM) {
                    k_sq += s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3;
                }
            }
            if constexpr (USE_QK_L2NORM) {
                k_sq += __shfl_xor(k_sq, 16, 64);
                k_sq += __shfl_xor(k_sq, 32, 64);
                k_inv_norm = rsqrtf(k_sq + 1e-6f);
            }
        }

        // ---- Load Q into smem (once per timestep, overwrites K) ----
        // No barrier needed: BV == wavefront size, single-wavefront lockstep
        // guarantees all K smem reads (k_cache) complete before Q writes.
        if (use_vec2) {
            const int base = 2 * lane;
            if (base < BK) {
                if constexpr (SMEM_PAD) {
                    load_bf16x2_padded<BK_QTR>(smem, x_t + q_dim_off, base, K, BK);
                } else {
                    load_bf16x2(smem, x_t + q_dim_off, base, K, BK);
                }
            }
        } else {
            for (int i = lane; i < BK; i += BV) {
                const float val = (i < K)
                    ? static_cast<float>(
                          x_t[static_cast<int64_t>(q_dim_off + i) * stride_x_dim])
                    : 0.0f;
                if constexpr (SMEM_PAD) {
                    smem[i + i / BK_QTR] = val;
                } else {
                    smem[i] = val;
                }
            }
        }
        __syncthreads();

        // ---- Process each v-head using k_cache (registers) and Q (smem) ----
        for (int hh = 0; hh < HPB; hh++) {
            const int i_hv = i_h * HPB + hh;

            const float a_log_val   = A_log[i_hv];
            const float dt_bias_val = static_cast<float>(dt_bias[i_hv]);
            const int v_dim_off     = 2 * key_dim + i_hv * V_dim;

            // Load state for this v-head
            float4 h_vec[BK4_QTR];
            #pragma unroll
            for (int i = 0; i < BK4_QTR; i++) {
                h_vec[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
            if constexpr (USE_INITIAL_STATE) {
                if (h0_idx >= 0 && valid_v) {
                    const float4* h0_base = reinterpret_cast<const float4*>(
                        h0_source
                        + static_cast<int64_t>(h0_idx) * HV * K4 * V_dim * 4
                        + static_cast<int64_t>(i_hv)   * K4 * V_dim * 4
                    );
                    #pragma unroll
                    for (int i = 0; i < BK4_QTR; i++) {
                        h_vec[i] = h0_base[(kg_start + i) * V_dim + v_col];
                    }
                }
            }

            const float neg_exp_A_log = -expf(a_log_val);

            // Per-head scalars
            const __hip_bfloat16* p_a = a      + static_cast<int64_t>(i_n) * T * HV + i_hv;
            const __hip_bfloat16* p_b = b_gate + static_cast<int64_t>(i_n) * T * HV + i_hv;

            // Issue V and gating loads early so they overlap with h_vec global loads.
            const int safe_v_col = valid_v ? v_col : 0;
            const __hip_bfloat16 v_raw =
                x_t[static_cast<int64_t>(v_dim_off + safe_v_col) * stride_x_dim];
            const __hip_bfloat16 a_raw = p_a[static_cast<int64_t>(t) * HV];
            const __hip_bfloat16 b_raw = p_b[static_cast<int64_t>(t) * HV];

            float v_local = valid_v ? static_cast<float>(v_raw) : 0.0f;
            const float a_t = static_cast<float>(a_raw);
            const float b_t = static_cast<float>(b_raw);

            const float sp  = hip_softplus(
                a_t + dt_bias_val, softplus_beta, inv_softplus_beta, softplus_threshold);
            const float g     = neg_exp_A_log * sp;
            const float exp_g = expf(g);
            const float beta  = hip_sigmoid(b_t);

            // Phase 1: dot(h_vec, k_cache)
            {
                float dot_partial = 0.0f;
                #pragma unroll
                for (int i = 0; i < BK4_QTR; i++) {
                    dot_partial += h_vec[i].x * k_cache[i].x
                                + h_vec[i].y * k_cache[i].y
                                + h_vec[i].z * k_cache[i].z
                                + h_vec[i].w * k_cache[i].w;
                }
                dot_partial += __shfl_xor(dot_partial, 16, 64);
                dot_partial += __shfl_xor(dot_partial, 32, 64);
                v_local -= dot_partial * exp_g * k_inv_norm;
            }

            v_local *= beta;

            // Phase 2: Decay + State update using k_cache
            {
                const float kv = k_inv_norm * v_local;
                #pragma unroll
                for (int i = 0; i < BK4_QTR; i++) {
                    h_vec[i].x = h_vec[i].x * exp_g + k_cache[i].x * kv;
                    h_vec[i].y = h_vec[i].y * exp_g + k_cache[i].y * kv;
                    h_vec[i].z = h_vec[i].z * exp_g + k_cache[i].z * kv;
                    h_vec[i].w = h_vec[i].w * exp_g + k_cache[i].w * kv;
                }
            }

            // Phase 3: dot(h_vec, Q from smem) → output
            {
                float out_partial = 0.0f;
                float q_sq = 0.0f;
                #pragma unroll
                for (int i = 0; i < BK4_QTR; i++) {
                    const int k0 = k_start + i * 4;
                    const float s0 = smem[k0], s1 = smem[k0 + 1];
                    const float s2 = smem[k0 + 2], s3 = smem[k0 + 3];
                    out_partial += h_vec[i].x * s0 + h_vec[i].y * s1
                                 + h_vec[i].z * s2 + h_vec[i].w * s3;
                    if constexpr (USE_QK_L2NORM) {
                        q_sq += s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3;
                    }
                }
                out_partial += __shfl_xor(out_partial, 16, 64);
                out_partial += __shfl_xor(out_partial, 32, 64);
                float q_inv_norm = 1.0f;
                if constexpr (USE_QK_L2NORM) {
                    q_sq += __shfl_xor(q_sq, 16, 64);
                    q_sq += __shfl_xor(q_sq, 32, 64);
                    q_inv_norm = rsqrtf(q_sq + 1e-6f);
                }
                float o_local = out_partial * q_inv_norm * scale;

                __hip_bfloat16* p_o = o + static_cast<int64_t>(i_n) * stride_o_batch
                                        + static_cast<int64_t>(i_hv) * stride_o_head;
                if (k_split == 0 && valid_v) {
                    p_o[static_cast<int64_t>(t) * stride_o_seq
                        + static_cast<int64_t>(v_col) * stride_o_dim] =
                        static_cast<__hip_bfloat16>(o_local);
                }
            }

            // Store updated state for this v-head
            if constexpr (USE_INITIAL_STATE) {
                if (h0_idx >= 0 && valid_v) {
                    float4* h0_base = reinterpret_cast<float4*>(
                        h0_source
                        + static_cast<int64_t>(h0_idx) * HV * K4 * V_dim * 4
                        + static_cast<int64_t>(i_hv)   * K4 * V_dim * 4
                    );
                    #pragma unroll
                    for (int i = 0; i < BK4_QTR; i++) {
                        h0_base[(kg_start + i) * V_dim + v_col] = h_vec[i];
                    }
                }
            }
        } // end head loop
    } // end timestep loop
}

// ---------------------------------------------------------------------------
// Kernel: fused_split_gdr_update_kernel_ksplit8_hfuse
//
// Multi-head fused variant of ksplit8.  When HV > H (GQA), multiple V-heads
// share the same QK head.  This kernel loads K and Q once per timestep,
// caches K in registers (k_cache) and Q in smem, then iterates over
// HPB = HV/H v-heads reusing both.
//
// Grid:  (1, cdiv(V, BV/8), B * H)       <- note: H, not HV
// Block: BV threads = 1 wavefront
// State layout: SWIZZLED (N_states, HV, K/4, V, 4)
// ---------------------------------------------------------------------------

template <int BK, int BV, bool USE_INITIAL_STATE, bool USE_QK_L2NORM>
__global__ __launch_bounds__(BV)
void fused_split_gdr_update_kernel_ksplit8_hfuse(
    const __hip_bfloat16* __restrict__ mixed_qkv,
    const float*          __restrict__ A_log,
    const __hip_bfloat16* __restrict__ a,
    const __hip_bfloat16* __restrict__ dt_bias,
    float                             softplus_beta,
    float                             softplus_threshold,
    const __hip_bfloat16* __restrict__ b_gate,
    __hip_bfloat16*       __restrict__ o,
    float*                __restrict__ h0_source,
    const int32_t*        __restrict__ h0_indices,
    int T,
    int key_dim,
    int value_dim,
    int64_t stride_x_batch,
    int64_t stride_x_dim,
    int64_t stride_x_seq,
    int64_t stride_o_batch,
    int64_t stride_o_seq,
    int64_t stride_o_head,
    int64_t stride_o_dim,
    int B,
    int H,
    int HV,
    int K,
    int V_dim,
    float scale
) {
    static_assert(BK % 32 == 0, "BK must be divisible by 32 for K_SPLIT=8");

    constexpr int BK_OCT  = BK / 8;
    constexpr int BK4_OCT = BK_OCT / 4;
    constexpr int BV_OUT  = BV / 8;
    constexpr int SMEM_PAD = (BK_OCT >= 8) ? 1 : 0;
    constexpr int SMEM_STRIDE = BK_OCT + SMEM_PAD;
    constexpr int SMEM_SIZE = SMEM_STRIDE * 8;

    const int i_v  = blockIdx.y;
    const int i_nh = blockIdx.z;
    const int i_n  = i_nh / H;
    const int i_h  = i_nh % H;
    const int HPB  = HV / H;
    const int lane = threadIdx.x;

    const int v_idx   = lane & (BV_OUT - 1);
    const int k_split = lane >> 3;
    const int v_col   = i_v * BV_OUT + v_idx;
    const bool valid_v = (v_col < V_dim);
    const int k_start_logical = k_split * BK_OCT;
    const int k_start = k_split * SMEM_STRIDE;

    __shared__ float smem[SMEM_SIZE];

    const int q_dim_off = i_h * K;
    const int k_dim_off = key_dim + i_h * K;
    const __hip_bfloat16* x_base =
        mixed_qkv + static_cast<int64_t>(i_n) * stride_x_batch;
    const bool use_vec2 = (stride_x_dim == 1);
    const float inv_softplus_beta = 1.0f / softplus_beta;

    const int K4 = K / 4;
    const int kg_start = k_start_logical / 4;
    const int32_t h0_idx = USE_INITIAL_STATE ? h0_indices[i_n] : -1;

    for (int t = 0; t < T; t++) {
        const __hip_bfloat16* x_t =
            x_base + static_cast<int64_t>(t) * stride_x_seq;

        // ---- Load K into smem (once per timestep) ----
        if (t > 0) __syncthreads();
        if (use_vec2) {
            const int base = 2 * lane;
            if (base < BK) {
                if constexpr (SMEM_PAD) {
                    load_bf16x2_padded<BK_OCT>(smem, x_t + k_dim_off, base, K, BK);
                } else {
                    load_bf16x2(smem, x_t + k_dim_off, base, K, BK);
                }
            }
        } else {
            for (int i = lane; i < BK; i += BV) {
                const float val = (i < K)
                    ? static_cast<float>(
                          x_t[static_cast<int64_t>(k_dim_off + i) * stride_x_dim])
                    : 0.0f;
                if constexpr (SMEM_PAD) {
                    smem[i + i / BK_OCT] = val;
                } else {
                    smem[i] = val;
                }
            }
        }
        __syncthreads();

        // ---- Cache K in registers, compute k_inv_norm (shared across heads) ----
        float4 k_cache[BK4_OCT];
        float k_inv_norm = 1.0f;
        {
            float k_sq = 0.0f;
            #pragma unroll
            for (int i = 0; i < BK4_OCT; i++) {
                const int k0 = k_start + i * 4;
                const float s0 = smem[k0], s1 = smem[k0 + 1];
                const float s2 = smem[k0 + 2], s3 = smem[k0 + 3];
                k_cache[i] = make_float4(s0, s1, s2, s3);
                if constexpr (USE_QK_L2NORM) {
                    k_sq += s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3;
                }
            }
            if constexpr (USE_QK_L2NORM) {
                k_sq += __shfl_xor(k_sq, 8, 64);
                k_sq += __shfl_xor(k_sq, 16, 64);
                k_sq += __shfl_xor(k_sq, 32, 64);
                k_inv_norm = rsqrtf(k_sq + 1e-6f);
            }
        }

        // ---- Load Q into smem (once per timestep, overwrites K) ----
        // No barrier needed: BV == wavefront size, single-wavefront lockstep
        // guarantees all K smem reads (k_cache) complete before Q writes.
        if (use_vec2) {
            const int base = 2 * lane;
            if (base < BK) {
                if constexpr (SMEM_PAD) {
                    load_bf16x2_padded<BK_OCT>(smem, x_t + q_dim_off, base, K, BK);
                } else {
                    load_bf16x2(smem, x_t + q_dim_off, base, K, BK);
                }
            }
        } else {
            for (int i = lane; i < BK; i += BV) {
                const float val = (i < K)
                    ? static_cast<float>(
                          x_t[static_cast<int64_t>(q_dim_off + i) * stride_x_dim])
                    : 0.0f;
                if constexpr (SMEM_PAD) {
                    smem[i + i / BK_OCT] = val;
                } else {
                    smem[i] = val;
                }
            }
        }
        __syncthreads();

        // ---- Process each v-head using k_cache (registers) and Q (smem) ----
        for (int hh = 0; hh < HPB; hh++) {
            const int i_hv = i_h * HPB + hh;

            const float a_log_val   = A_log[i_hv];
            const float dt_bias_val = static_cast<float>(dt_bias[i_hv]);
            const int v_dim_off     = 2 * key_dim + i_hv * V_dim;

            float4 h_vec[BK4_OCT];
            #pragma unroll
            for (int i = 0; i < BK4_OCT; i++) {
                h_vec[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
            if constexpr (USE_INITIAL_STATE) {
                if (h0_idx >= 0 && valid_v) {
                    const float4* h0_base = reinterpret_cast<const float4*>(
                        h0_source
                        + static_cast<int64_t>(h0_idx) * HV * K4 * V_dim * 4
                        + static_cast<int64_t>(i_hv)   * K4 * V_dim * 4
                    );
                    #pragma unroll
                    for (int i = 0; i < BK4_OCT; i++) {
                        h_vec[i] = h0_base[(kg_start + i) * V_dim + v_col];
                    }
                }
            }

            const float neg_exp_A_log = -expf(a_log_val);

            const __hip_bfloat16* p_a = a      + static_cast<int64_t>(i_n) * T * HV + i_hv;
            const __hip_bfloat16* p_b = b_gate + static_cast<int64_t>(i_n) * T * HV + i_hv;

            // Issue V and gating loads early so they overlap with h_vec global loads.
            const int safe_v_col = valid_v ? v_col : 0;
            const __hip_bfloat16 v_raw =
                x_t[static_cast<int64_t>(v_dim_off + safe_v_col) * stride_x_dim];
            const __hip_bfloat16 a_raw = p_a[static_cast<int64_t>(t) * HV];
            const __hip_bfloat16 b_raw = p_b[static_cast<int64_t>(t) * HV];

            float v_local = valid_v ? static_cast<float>(v_raw) : 0.0f;
            const float a_t = static_cast<float>(a_raw);
            const float b_t = static_cast<float>(b_raw);

            const float sp  = hip_softplus(
                a_t + dt_bias_val, softplus_beta, inv_softplus_beta, softplus_threshold);
            const float g     = neg_exp_A_log * sp;
            const float exp_g = expf(g);
            const float beta  = hip_sigmoid(b_t);

            // Phase 1: dot(h_vec, k_cache) with 8-way reduction
            {
                float dot_partial = 0.0f;
                #pragma unroll
                for (int i = 0; i < BK4_OCT; i++) {
                    dot_partial += h_vec[i].x * k_cache[i].x
                                + h_vec[i].y * k_cache[i].y
                                + h_vec[i].z * k_cache[i].z
                                + h_vec[i].w * k_cache[i].w;
                }
                dot_partial += __shfl_xor(dot_partial, 8, 64);
                dot_partial += __shfl_xor(dot_partial, 16, 64);
                dot_partial += __shfl_xor(dot_partial, 32, 64);
                v_local -= dot_partial * exp_g * k_inv_norm;
            }

            v_local *= beta;

            // Phase 2: Decay + State update using k_cache
            {
                const float kv = k_inv_norm * v_local;
                #pragma unroll
                for (int i = 0; i < BK4_OCT; i++) {
                    h_vec[i].x = h_vec[i].x * exp_g + k_cache[i].x * kv;
                    h_vec[i].y = h_vec[i].y * exp_g + k_cache[i].y * kv;
                    h_vec[i].z = h_vec[i].z * exp_g + k_cache[i].z * kv;
                    h_vec[i].w = h_vec[i].w * exp_g + k_cache[i].w * kv;
                }
            }

            // Phase 3: dot(h_vec, Q from smem) -> output, 8-way reduction
            {
                float out_partial = 0.0f;
                float q_sq = 0.0f;
                #pragma unroll
                for (int i = 0; i < BK4_OCT; i++) {
                    const int k0 = k_start + i * 4;
                    const float s0 = smem[k0], s1 = smem[k0 + 1];
                    const float s2 = smem[k0 + 2], s3 = smem[k0 + 3];
                    out_partial += h_vec[i].x * s0 + h_vec[i].y * s1
                                 + h_vec[i].z * s2 + h_vec[i].w * s3;
                    if constexpr (USE_QK_L2NORM) {
                        q_sq += s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3;
                    }
                }
                out_partial += __shfl_xor(out_partial, 8, 64);
                out_partial += __shfl_xor(out_partial, 16, 64);
                out_partial += __shfl_xor(out_partial, 32, 64);
                float q_inv_norm = 1.0f;
                if constexpr (USE_QK_L2NORM) {
                    q_sq += __shfl_xor(q_sq, 8, 64);
                    q_sq += __shfl_xor(q_sq, 16, 64);
                    q_sq += __shfl_xor(q_sq, 32, 64);
                    q_inv_norm = rsqrtf(q_sq + 1e-6f);
                }
                float o_local = out_partial * q_inv_norm * scale;

                __hip_bfloat16* p_o = o + static_cast<int64_t>(i_n) * stride_o_batch
                                        + static_cast<int64_t>(i_hv) * stride_o_head;
                if (k_split == 0 && valid_v) {
                    p_o[static_cast<int64_t>(t) * stride_o_seq
                        + static_cast<int64_t>(v_col) * stride_o_dim] =
                        static_cast<__hip_bfloat16>(o_local);
                }
            }

            // Store updated state for this v-head
            if constexpr (USE_INITIAL_STATE) {
                if (h0_idx >= 0 && valid_v) {
                    float4* h0_base = reinterpret_cast<float4*>(
                        h0_source
                        + static_cast<int64_t>(h0_idx) * HV * K4 * V_dim * 4
                        + static_cast<int64_t>(i_hv)   * K4 * V_dim * 4
                    );
                    #pragma unroll
                    for (int i = 0; i < BK4_OCT; i++) {
                        h0_base[(kg_start + i) * V_dim + v_col] = h_vec[i];
                    }
                }
            }
        } // end head loop
    } // end timestep loop
}

// ---------------------------------------------------------------------------
// Kernel: fused_split_gdr_update_kernel_vsplit
//
// V-split: each thread handles 4 contiguous V columns and BK/8 K elements.
//   Thread layout: 8 K-threads x 8 V-threads = 64 threads (1 wavefront)
//     lane 0-7   -> (k=0..7, same v_thread)
//     lane 8-15  -> (k=0..7, next v_thread), etc.
//     k_thread = lane & 7    (0..7)
//     v_thread = lane >> 3   (0..7)
//   Each thread owns h[k_chunk, v_group:v_group+4] = float4[BK/8]
//
// Use LDS staging for K/Q so each timestep loads once per block and
// reuses across the 8 V-threads sharing the same K chunk.
//
// Grid:  (1, cdiv(V, 32), B * HV)
// Block: 64 threads = 1 wavefront
// State layout: Standard (N_states, HV, K, V) = (N_states, HV, K, V/4, 4)
//   - float4 = 4 contiguous V values for one K position (zero-copy view)
// ---------------------------------------------------------------------------

template <int BK, int BV, bool USE_INITIAL_STATE, bool USE_QK_L2NORM>
__global__ __launch_bounds__(BV)
void fused_split_gdr_update_kernel_vsplit(
    const __hip_bfloat16* __restrict__ mixed_qkv,
    const float*          __restrict__ A_log,
    const __hip_bfloat16* __restrict__ a,
    const __hip_bfloat16* __restrict__ dt_bias,
    float                             softplus_beta,
    float                             softplus_threshold,
    const __hip_bfloat16* __restrict__ b_gate,
    __hip_bfloat16*       __restrict__ o,
    float*                __restrict__ h0_source,
    const int32_t*        __restrict__ h0_indices,
    int T,
    int key_dim,
    int value_dim,
    int64_t stride_x_batch,
    int64_t stride_x_dim,
    int64_t stride_x_seq,
    int64_t stride_o_batch,
    int64_t stride_o_seq,
    int64_t stride_o_head,
    int64_t stride_o_dim,
    int B,
    int H,
    int HV,
    int K,
    int V_dim,
    float scale
) {
    static_assert(BV == 64, "vsplit requires BV=64");
    static_assert(BK >= 8 && BK % 8 == 0, "BK must be >= 8 and divisible by 8");

    constexpr int K_THREADS    = 8;
    constexpr int V_THREADS    = BV / K_THREADS;    // 8
    constexpr int V_PER_THREAD = 4;
    constexpr int K_PER_THREAD = BK / K_THREADS;    // BK=128 -> 16
    constexpr int BV_OUT       = V_THREADS * V_PER_THREAD;  // 32

    const int i_v  = blockIdx.y;
    const int i_nh = blockIdx.z;
    const int i_n  = i_nh / HV;
    const int i_hv = i_nh % HV;

    const int GROUP_SIZE = HV / H;
    const int i_h  = i_hv / GROUP_SIZE;

    const int lane     = threadIdx.x;
    const int k_thread = lane & (K_THREADS - 1);
    const int v_thread = lane >> 3;

    // Interleaved-K mapping:
    // thread 0 -> k={0,8,16,...}, thread 1 -> k={1,9,17,...}, ...
    // This gives k0..k7 on lanes 0..7 for the same h_vec[i].
    const int v_group     = i_v * V_THREADS + v_thread;
    const int v_col_start = v_group * V_PER_THREAD;
    const bool valid_v    = (v_col_start + V_PER_THREAD - 1 < V_dim);

    __shared__ float smem[BK];

    const float a_log_val   = A_log[i_hv];
    const float dt_bias_val = static_cast<float>(dt_bias[i_hv]);

    // h_vec[i] = (v0, v1, v2, v3) for K position (k_thread + i * K_THREADS)
    float4 h_vec[K_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < K_PER_THREAD; i++) {
        h_vec[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    // Load initial state from vsplit layout [N, HV, V/4, K, 4].
    // float4 packs 4 contiguous V values for one K position.
    const int V4 = V_dim / 4;
    if constexpr (USE_INITIAL_STATE) {
        const int32_t idx = h0_indices[i_n];
        if (idx >= 0 && valid_v) {
            const float4* h0_base = reinterpret_cast<const float4*>(
                h0_source
                + static_cast<int64_t>(idx) * HV * K * V_dim
                + static_cast<int64_t>(i_hv) * K * V_dim
            );
            #pragma unroll
            for (int i = 0; i < K_PER_THREAD; i++) {
                const int k = k_thread + i * K_THREADS;
                if (k < K) {
                    h_vec[i] = h0_base[static_cast<int64_t>(v_group) * K + k];
                }
            }
        }
    }

    const float neg_exp_A_log     = -expf(a_log_val);
    const float inv_softplus_beta = 1.0f / softplus_beta;

    const int q_dim_off = i_h  * K;
    const int k_dim_off = key_dim + i_h * K;
    const int v_dim_off = 2 * key_dim + i_hv * V_dim;

    const __hip_bfloat16* x_base =
        mixed_qkv + static_cast<int64_t>(i_n) * stride_x_batch;

    const __hip_bfloat16* p_a =
        a      + static_cast<int64_t>(i_n) * T * HV + i_hv;
    const __hip_bfloat16* p_b =
        b_gate + static_cast<int64_t>(i_n) * T * HV + i_hv;

    __hip_bfloat16* p_o =
        o + static_cast<int64_t>(i_n) * stride_o_batch
          + static_cast<int64_t>(i_hv) * stride_o_head;

    const int safe_v_start = valid_v ? v_col_start : 0;

    for (int t = 0; t < T; t++) {
        const __hip_bfloat16* x_t =
            x_base + static_cast<int64_t>(t) * stride_x_seq;

        // Issue V and gating loads unconditionally to avoid exec masking,
        // allowing the compiler to overlap them with subsequent K LDS loads.
        const __hip_bfloat16 v0_raw =
            x_t[static_cast<int64_t>(v_dim_off + safe_v_start) * stride_x_dim];
        const __hip_bfloat16 v1_raw =
            x_t[static_cast<int64_t>(v_dim_off + safe_v_start + 1) * stride_x_dim];
        const __hip_bfloat16 v2_raw =
            x_t[static_cast<int64_t>(v_dim_off + safe_v_start + 2) * stride_x_dim];
        const __hip_bfloat16 v3_raw =
            x_t[static_cast<int64_t>(v_dim_off + safe_v_start + 3) * stride_x_dim];
        const __hip_bfloat16 a_raw = p_a[static_cast<int64_t>(t) * HV];
        const __hip_bfloat16 b_raw = p_b[static_cast<int64_t>(t) * HV];

        float v0 = valid_v ? static_cast<float>(v0_raw) : 0.0f;
        float v1 = valid_v ? static_cast<float>(v1_raw) : 0.0f;
        float v2 = valid_v ? static_cast<float>(v2_raw) : 0.0f;
        float v3 = valid_v ? static_cast<float>(v3_raw) : 0.0f;

        const float a_t = static_cast<float>(a_raw);
        const float b_t = static_cast<float>(b_raw);

        const float sp    = hip_softplus(
            a_t + dt_bias_val, softplus_beta, inv_softplus_beta, softplus_threshold);
        const float g     = neg_exp_A_log * sp;
        const float exp_g = expf(g);
        const float beta  = hip_sigmoid(b_t);

        // Phase 1: K-L2-norm + Delta dot (LDS staged K)
        float k_inv_norm = 1.0f;
        {
            for (int kk = lane; kk < BK; kk += BV) {
                smem[kk] = (kk < K)
                    ? static_cast<float>(
                          x_t[static_cast<int64_t>(k_dim_off + kk) * stride_x_dim])
                    : 0.0f;
            }
            __syncthreads();

            float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
            float k_sq = 0.0f;
            #pragma unroll
            for (int i = 0; i < K_PER_THREAD; i++) {
                const int k = k_thread + i * K_THREADS;
                const float kv = smem[k];
                d0 = fmaf(h_vec[i].x, kv, d0);
                d1 = fmaf(h_vec[i].y, kv, d1);
                d2 = fmaf(h_vec[i].z, kv, d2);
                d3 = fmaf(h_vec[i].w, kv, d3);
                if constexpr (USE_QK_L2NORM) {
                    k_sq = fmaf(kv, kv, k_sq);
                }
            }
            d0 += __shfl_xor(d0, 1, 64); d0 += __shfl_xor(d0, 2, 64); d0 += __shfl_xor(d0, 4, 64);
            d1 += __shfl_xor(d1, 1, 64); d1 += __shfl_xor(d1, 2, 64); d1 += __shfl_xor(d1, 4, 64);
            d2 += __shfl_xor(d2, 1, 64); d2 += __shfl_xor(d2, 2, 64); d2 += __shfl_xor(d2, 4, 64);
            d3 += __shfl_xor(d3, 1, 64); d3 += __shfl_xor(d3, 2, 64); d3 += __shfl_xor(d3, 4, 64);
            if constexpr (USE_QK_L2NORM) {
                k_sq += __shfl_xor(k_sq, 1, 64);
                k_sq += __shfl_xor(k_sq, 2, 64);
                k_sq += __shfl_xor(k_sq, 4, 64);
                k_inv_norm = rsqrtf(k_sq + 1e-6f);
            }
            const float factor = exp_g * k_inv_norm;
            v0 -= d0 * factor;
            v1 -= d1 * factor;
            v2 -= d2 * factor;
            v3 -= d3 * factor;
        }

        v0 *= beta; v1 *= beta; v2 *= beta; v3 *= beta;

        // Phase 2: Decay + State update (reuse staged K)
        {
            const float kv0 = k_inv_norm * v0, kv1 = k_inv_norm * v1;
            const float kv2 = k_inv_norm * v2, kv3 = k_inv_norm * v3;
            #pragma unroll
            for (int i = 0; i < K_PER_THREAD; i++) {
                const int k = k_thread + i * K_THREADS;
                const float xval = smem[k];
                h_vec[i].x = fmaf(h_vec[i].x, exp_g, xval * kv0);
                h_vec[i].y = fmaf(h_vec[i].y, exp_g, xval * kv1);
                h_vec[i].z = fmaf(h_vec[i].z, exp_g, xval * kv2);
                h_vec[i].w = fmaf(h_vec[i].w, exp_g, xval * kv3);
            }
        }

        // Phase 3: Q-L2-norm + Output dot (LDS staged Q)
        {
            for (int kk = lane; kk < BK; kk += BV) {
                smem[kk] = (kk < K)
                    ? static_cast<float>(
                          x_t[static_cast<int64_t>(q_dim_off + kk) * stride_x_dim])
                    : 0.0f;
            }
            __syncthreads();

            float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
            float q_sq = 0.0f;
            #pragma unroll
            for (int i = 0; i < K_PER_THREAD; i++) {
                const int k = k_thread + i * K_THREADS;
                const float qv = smem[k];
                o0 = fmaf(h_vec[i].x, qv, o0);
                o1 = fmaf(h_vec[i].y, qv, o1);
                o2 = fmaf(h_vec[i].z, qv, o2);
                o3 = fmaf(h_vec[i].w, qv, o3);
                if constexpr (USE_QK_L2NORM) {
                    q_sq = fmaf(qv, qv, q_sq);
                }
            }
            o0 += __shfl_xor(o0, 1, 64); o0 += __shfl_xor(o0, 2, 64); o0 += __shfl_xor(o0, 4, 64);
            o1 += __shfl_xor(o1, 1, 64); o1 += __shfl_xor(o1, 2, 64); o1 += __shfl_xor(o1, 4, 64);
            o2 += __shfl_xor(o2, 1, 64); o2 += __shfl_xor(o2, 2, 64); o2 += __shfl_xor(o2, 4, 64);
            o3 += __shfl_xor(o3, 1, 64); o3 += __shfl_xor(o3, 2, 64); o3 += __shfl_xor(o3, 4, 64);
            float q_inv_norm = 1.0f;
            if constexpr (USE_QK_L2NORM) {
                q_sq += __shfl_xor(q_sq, 1, 64);
                q_sq += __shfl_xor(q_sq, 2, 64);
                q_sq += __shfl_xor(q_sq, 4, 64);
                q_inv_norm = rsqrtf(q_sq + 1e-6f);
            }
            if (k_thread == 0 && valid_v) {
                const float s = q_inv_norm * scale;
                const int64_t o_off = static_cast<int64_t>(t) * stride_o_seq;
                p_o[o_off + static_cast<int64_t>(v_col_start)     * stride_o_dim] =
                    static_cast<__hip_bfloat16>(o0 * s);
                p_o[o_off + static_cast<int64_t>(v_col_start + 1) * stride_o_dim] =
                    static_cast<__hip_bfloat16>(o1 * s);
                p_o[o_off + static_cast<int64_t>(v_col_start + 2) * stride_o_dim] =
                    static_cast<__hip_bfloat16>(o2 * s);
                p_o[o_off + static_cast<int64_t>(v_col_start + 3) * stride_o_dim] =
                    static_cast<__hip_bfloat16>(o3 * s);
            }
            __syncthreads();
        }
    } // end timestep loop

    // Store final state to vsplit layout [N, HV, V/4, K, 4].
    if constexpr (USE_INITIAL_STATE) {
        const int32_t idx = h0_indices[i_n];
        if (idx >= 0 && valid_v) {
            float4* h0_base = reinterpret_cast<float4*>(
                h0_source
                + static_cast<int64_t>(idx) * HV * K * V_dim
                + static_cast<int64_t>(i_hv) * K * V_dim
            );
            #pragma unroll
            for (int i = 0; i < K_PER_THREAD; i++) {
                const int k = k_thread + i * K_THREADS;
                if (k < K) {
                    h0_base[static_cast<int64_t>(v_group) * K + k] = h_vec[i];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Host wrappers: fused_split_gdr_update_ksplit2 / ksplit4 / vsplit
// ---------------------------------------------------------------------------

// Common launch macro for ksplit wrappers
#define LAUNCH_KS(KNAME, BK_CT, USE_INIT, USE_L2)                             \
   hipLaunchKernelGGL(( KNAME<BK_CT, BV_VAL, USE_INIT, USE_L2>)                                    \
        , dim3(grid), dim3(block), 0, stream,                                           \
            reinterpret_cast<const __hip_bfloat16*>(mixed_qkv.data_ptr()),     \
            A_log.data_ptr<float>(),                                           \
            reinterpret_cast<const __hip_bfloat16*>(a.data_ptr()),             \
            reinterpret_cast<const __hip_bfloat16*>(dt_bias.data_ptr()),       \
            softplus_beta, softplus_threshold,                                 \
            reinterpret_cast<const __hip_bfloat16*>(b_gate.data_ptr()),        \
            reinterpret_cast<__hip_bfloat16*>(o.data_ptr()),                   \
            use_initial_state                                                  \
                ? initial_state_source.data_ptr<float>() : nullptr,            \
            initial_state_indices.data_ptr<int32_t>(),                         \
            T, key_dim, value_dim,                                             \
            stride_x_batch, stride_x_dim, stride_x_seq,                       \
            stride_o_batch, stride_o_seq, stride_o_head, stride_o_dim,         \
            B, H, HV, K, V, scale                                             \
        )

#define DISPATCH_KS_BOOL(KNAME, BK_CT)                                        \
    if (use_initial_state && use_qk_l2norm_in_kernel) {                        \
        LAUNCH_KS(KNAME, BK_CT, true, true);                                  \
    } else if (use_initial_state && !use_qk_l2norm_in_kernel) {                \
        LAUNCH_KS(KNAME, BK_CT, true, false);                                 \
    } else if (!use_initial_state && use_qk_l2norm_in_kernel) {                \
        LAUNCH_KS(KNAME, BK_CT, false, true);                                 \
    } else {                                                                   \
        LAUNCH_KS(KNAME, BK_CT, false, false);                                \
    }

#define DISPATCH_KS_BK(KNAME)                                                 \
    if (bk_runtime == 128)      { DISPATCH_KS_BOOL(KNAME, 128); }             \
    else if (bk_runtime == 64)  { DISPATCH_KS_BOOL(KNAME, 64);  }             \
    else if (bk_runtime == 256) { DISPATCH_KS_BOOL(KNAME, 256); }             \
    else if (bk_runtime == 32)  { DISPATCH_KS_BOOL(KNAME, 32);  }             \
    else { TORCH_CHECK(false, "Unsupported BK: ", bk_runtime); }

#define DEFINE_KSPLIT_WRAPPER(FUNC_NAME, KERNEL_NAME, KS_VAL)                  \
torch::Tensor FUNC_NAME(                                                       \
    torch::Tensor mixed_qkv, torch::Tensor A_log, torch::Tensor a,            \
    torch::Tensor dt_bias, torch::Tensor b_gate,                               \
    torch::Tensor initial_state_source, torch::Tensor initial_state_indices,   \
    int key_dim, int value_dim, int num_heads_qk, int num_heads_v,             \
    int head_dim, float softplus_beta, float softplus_threshold, float scale,  \
    bool use_qk_l2norm_in_kernel, c10::optional<torch::Tensor> output          \
) {                                                                            \
    TORCH_CHECK(mixed_qkv.dim() == 3, "mixed_qkv must be 3-D (B, dim, T)");   \
    const int B = mixed_qkv.size(0), dim = mixed_qkv.size(1),                 \
              T = mixed_qkv.size(2);                                           \
    const int H = num_heads_qk, HV = num_heads_v;                             \
    const int K = head_dim, V = head_dim;                                      \
    TORCH_CHECK(dim == 2 * key_dim + value_dim, "mixed_qkv dim mismatch");    \
    TORCH_CHECK(K % 4 == 0, "head_dim must be divisible by 4");                \
    int bk_runtime = 1;                                                        \
    while (bk_runtime < K) bk_runtime <<= 1;                                  \
    TORCH_CHECK((K + bk_runtime - 1) / bk_runtime == 1, "NK > 1 unsupported");\
    torch::Tensor o;                                                           \
    if (output.has_value()) {                                                  \
        o = output.value();                                                    \
        TORCH_CHECK(o.size(0)==B && o.size(1)==T && o.size(2)==HV && o.size(3)==V,\
                    "Output shape mismatch");                                  \
    } else { o = torch::empty({B, T, HV, V}, mixed_qkv.options()); }          \
    bool use_initial_state = initial_state_source.defined() &&                 \
                             initial_state_source.numel() > 0;                 \
    constexpr int BV_VAL = 64;                                                 \
    constexpr int BV_OUT = BV_VAL / KS_VAL;                                   \
    dim3 grid(1, (V + BV_OUT - 1) / BV_OUT, B * HV);                          \
    dim3 block(BV_VAL);                                                        \
    const int64_t stride_x_batch = mixed_qkv.stride(0);                        \
    const int64_t stride_x_dim   = mixed_qkv.stride(1);                        \
    const int64_t stride_x_seq   = mixed_qkv.stride(2);                        \
    const int64_t stride_o_batch = o.stride(0);                                \
    const int64_t stride_o_seq   = o.stride(1);                                \
    const int64_t stride_o_head  = o.stride(2);                                \
    const int64_t stride_o_dim   = o.stride(3);                                \
    auto stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();                            \
    DISPATCH_KS_BK(KERNEL_NAME);                                               \
    return o;                                                                  \
}

DEFINE_KSPLIT_WRAPPER(fused_split_gdr_update_ksplit2,
                      fused_split_gdr_update_kernel_ksplit2, 2)
DEFINE_KSPLIT_WRAPPER(fused_split_gdr_update_ksplit4,
                      fused_split_gdr_update_kernel_ksplit4, 4)
DEFINE_KSPLIT_WRAPPER(fused_split_gdr_update_ksplit4_opt,
                      fused_split_gdr_update_kernel_ksplit4_opt, 4)
DEFINE_KSPLIT_WRAPPER(fused_split_gdr_update_ksplit4_db,
                      fused_split_gdr_update_kernel_ksplit4_db, 4)
DEFINE_KSPLIT_WRAPPER(fused_split_gdr_update_ksplit8,
                      fused_split_gdr_update_kernel_ksplit8, 8)
DEFINE_KSPLIT_WRAPPER(fused_split_gdr_update_vsplit,
                      fused_split_gdr_update_kernel_vsplit, 2)
#undef DEFINE_KSPLIT_WRAPPER

// hfuse wrapper: grid Z = B * H (not B * HV)
torch::Tensor fused_split_gdr_update_ksplit4_hfuse(
    torch::Tensor mixed_qkv, torch::Tensor A_log, torch::Tensor a,
    torch::Tensor dt_bias, torch::Tensor b_gate,
    torch::Tensor initial_state_source, torch::Tensor initial_state_indices,
    int key_dim, int value_dim, int num_heads_qk, int num_heads_v,
    int head_dim, float softplus_beta, float softplus_threshold, float scale,
    bool use_qk_l2norm_in_kernel, c10::optional<torch::Tensor> output
) {
    TORCH_CHECK(mixed_qkv.dim() == 3, "mixed_qkv must be 3-D (B, dim, T)");
    const int B = mixed_qkv.size(0), dim = mixed_qkv.size(1),
              T = mixed_qkv.size(2);
    const int H = num_heads_qk, HV = num_heads_v;
    const int K = head_dim, V = head_dim;
    TORCH_CHECK(dim == 2 * key_dim + value_dim, "mixed_qkv dim mismatch");
    TORCH_CHECK(K % 4 == 0, "head_dim must be divisible by 4");
    TORCH_CHECK(HV % H == 0, "HV must be divisible by H for head fusion");
    int bk_runtime = 1;
    while (bk_runtime < K) bk_runtime <<= 1;
    TORCH_CHECK((K + bk_runtime - 1) / bk_runtime == 1, "NK > 1 unsupported");
    torch::Tensor o;
    if (output.has_value()) {
        o = output.value();
        TORCH_CHECK(o.size(0)==B && o.size(1)==T && o.size(2)==HV && o.size(3)==V,
                    "Output shape mismatch");
    } else { o = torch::empty({B, T, HV, V}, mixed_qkv.options()); }
    bool use_initial_state = initial_state_source.defined() &&
                             initial_state_source.numel() > 0;
    constexpr int BV_VAL = 64;
    constexpr int BV_OUT = BV_VAL / 4;
    dim3 grid(1, (V + BV_OUT - 1) / BV_OUT, B * H);
    dim3 block(BV_VAL);
    const int64_t stride_x_batch = mixed_qkv.stride(0);
    const int64_t stride_x_dim   = mixed_qkv.stride(1);
    const int64_t stride_x_seq   = mixed_qkv.stride(2);
    const int64_t stride_o_batch = o.stride(0);
    const int64_t stride_o_seq   = o.stride(1);
    const int64_t stride_o_head  = o.stride(2);
    const int64_t stride_o_dim   = o.stride(3);
    auto stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    DISPATCH_KS_BK(fused_split_gdr_update_kernel_ksplit4_hfuse);
    return o;
}

// hfuse wrapper for ksplit8: grid Z = B * H (not B * HV)
torch::Tensor fused_split_gdr_update_ksplit8_hfuse(
    torch::Tensor mixed_qkv, torch::Tensor A_log, torch::Tensor a,
    torch::Tensor dt_bias, torch::Tensor b_gate,
    torch::Tensor initial_state_source, torch::Tensor initial_state_indices,
    int key_dim, int value_dim, int num_heads_qk, int num_heads_v,
    int head_dim, float softplus_beta, float softplus_threshold, float scale,
    bool use_qk_l2norm_in_kernel, c10::optional<torch::Tensor> output
) {
    TORCH_CHECK(mixed_qkv.dim() == 3, "mixed_qkv must be 3-D (B, dim, T)");
    const int B = mixed_qkv.size(0), dim = mixed_qkv.size(1),
              T = mixed_qkv.size(2);
    const int H = num_heads_qk, HV = num_heads_v;
    const int K = head_dim, V = head_dim;
    TORCH_CHECK(dim == 2 * key_dim + value_dim, "mixed_qkv dim mismatch");
    TORCH_CHECK(K % 4 == 0, "head_dim must be divisible by 4");
    TORCH_CHECK(HV % H == 0, "HV must be divisible by H for head fusion");
    int bk_runtime = 1;
    while (bk_runtime < K) bk_runtime <<= 1;
    TORCH_CHECK((K + bk_runtime - 1) / bk_runtime == 1, "NK > 1 unsupported");
    torch::Tensor o;
    if (output.has_value()) {
        o = output.value();
        TORCH_CHECK(o.size(0)==B && o.size(1)==T && o.size(2)==HV && o.size(3)==V,
                    "Output shape mismatch");
    } else { o = torch::empty({B, T, HV, V}, mixed_qkv.options()); }
    bool use_initial_state = initial_state_source.defined() &&
                             initial_state_source.numel() > 0;
    constexpr int BV_VAL = 64;
    constexpr int BV_OUT = BV_VAL / 8;
    dim3 grid(1, (V + BV_OUT - 1) / BV_OUT, B * H);
    dim3 block(BV_VAL);
    const int64_t stride_x_batch = mixed_qkv.stride(0);
    const int64_t stride_x_dim   = mixed_qkv.stride(1);
    const int64_t stride_x_seq   = mixed_qkv.stride(2);
    const int64_t stride_o_batch = o.stride(0);
    const int64_t stride_o_seq   = o.stride(1);
    const int64_t stride_o_head  = o.stride(2);
    const int64_t stride_o_dim   = o.stride(3);
    auto stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    DISPATCH_KS_BK(fused_split_gdr_update_kernel_ksplit8_hfuse);
    return o;
}

#undef DISPATCH_KS_BK
#undef DISPATCH_KS_BOOL
#undef LAUNCH_KS

// ---------------------------------------------------------------------------
// Host wrapper: fused_split_gdr_update
// ---------------------------------------------------------------------------
//
// NOTE: State tensor must be in SWIZZLED layout (N_states, HV, K/4, V, 4)
//       Use state.reshape(N, HV, K//4, 4, V).permute(0,1,2,4,3).reshape(N, HV, K//4, V, 4)
//       to convert from standard (N, HV, K, V) layout.

torch::Tensor fused_split_gdr_update(
    torch::Tensor mixed_qkv,        // (B, dim, T), bfloat16
    torch::Tensor A_log,             // (HV,), float32
    torch::Tensor a,                 // (B*T, HV), bfloat16
    torch::Tensor dt_bias,           // (HV,), bfloat16
    torch::Tensor b_gate,            // (B*T, HV), bfloat16
    torch::Tensor initial_state_source,   // (N_states, HV, K/4, V, 4), float32 - SWIZZLED!
    torch::Tensor initial_state_indices,  // (B,), int32
    int key_dim,
    int value_dim,
    int num_heads_qk,
    int num_heads_v,
    int head_dim,
    float softplus_beta,
    float softplus_threshold,
    float scale,
    bool use_qk_l2norm_in_kernel,
    c10::optional<torch::Tensor> output    // optional pre-allocated output
) {
    // --- Dimension extraction ---
    TORCH_CHECK(mixed_qkv.dim() == 3, "mixed_qkv must be 3-D (B, dim, T)");
    const int B   = mixed_qkv.size(0);
    const int dim = mixed_qkv.size(1);
    const int T   = mixed_qkv.size(2);

    const int H  = num_heads_qk;
    const int HV = num_heads_v;
    const int K  = head_dim;
    const int V  = head_dim;

    TORCH_CHECK(dim == 2 * key_dim + value_dim,
                "mixed_qkv dim mismatch: expected ", 2 * key_dim + value_dim,
                " got ", dim);
    TORCH_CHECK(K % 4 == 0, "head_dim must be divisible by 4 for swizzled layout");

    // --- BK: next power of 2 >= K (compile-time for template) ---
    int bk_runtime = 1;
    while (bk_runtime < K) bk_runtime <<= 1;

    const int NK = (K + bk_runtime - 1) / bk_runtime;
    TORCH_CHECK(NK == 1, "NK > 1 is not supported yet");

    // --- Output allocation ---
    torch::Tensor o;
    if (output.has_value()) {
        o = output.value();
        TORCH_CHECK(o.size(0) == B && o.size(1) == T &&
                    o.size(2) == HV && o.size(3) == V,
                    "Output shape mismatch");
    } else {
        o = torch::empty({B, T, HV, V}, mixed_qkv.options());
    }

    // --- Determine whether initial state is used ---
    bool use_initial_state = initial_state_source.defined() &&
                             initial_state_source.numel() > 0;

    // --- Configuration: BV=64 (1 wavefront per block) ---
    constexpr int BV_VAL = 64;

    // --- Grid: (NK, cdiv(V, BV), B * HV) ---
    const int grid_x = NK;
    const int grid_y = (V + BV_VAL - 1) / BV_VAL;
    const int grid_z = B * HV;
    dim3 grid(grid_x, grid_y, grid_z);

    // Block = BV threads = 1 wavefront
    dim3 block(BV_VAL);

    // --- Strides ---
    const int64_t stride_x_batch = mixed_qkv.stride(0);
    const int64_t stride_x_dim   = mixed_qkv.stride(1);
    const int64_t stride_x_seq   = mixed_qkv.stride(2);
    const int64_t stride_o_batch = o.stride(0);
    const int64_t stride_o_seq   = o.stride(1);
    const int64_t stride_o_head  = o.stride(2);
    const int64_t stride_o_dim   = o.stride(3);

    auto stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();

    // --- Kernel dispatch ---
    #define LAUNCH_KERNEL(BK_CT, USE_INIT, USE_L2)                               \
       hipLaunchKernelGGL(( fused_split_gdr_update_kernel<BK_CT, BV_VAL, USE_INIT, USE_L2>)           \
            , dim3(grid), dim3(block), 0, stream,                                          \
                reinterpret_cast<const __hip_bfloat16*>(mixed_qkv.data_ptr()),   \
                A_log.data_ptr<float>(),                                         \
                reinterpret_cast<const __hip_bfloat16*>(a.data_ptr()),           \
                reinterpret_cast<const __hip_bfloat16*>(dt_bias.data_ptr()),     \
                softplus_beta,                                                   \
                softplus_threshold,                                              \
                reinterpret_cast<const __hip_bfloat16*>(b_gate.data_ptr()),      \
                reinterpret_cast<__hip_bfloat16*>(o.data_ptr()),                 \
                use_initial_state                                                \
                    ? initial_state_source.data_ptr<float>() : nullptr,          \
                initial_state_indices.data_ptr<int32_t>(),                       \
                T, key_dim, value_dim,                                           \
                stride_x_batch, stride_x_dim, stride_x_seq,                     \
                stride_o_batch, stride_o_seq, stride_o_head, stride_o_dim,       \
                B, H, HV, K, V, scale                                           \
            )

    #define DISPATCH_BK(BK_CT)                                                   \
        if (use_initial_state && use_qk_l2norm_in_kernel) {                      \
            LAUNCH_KERNEL(BK_CT, true, true);                                    \
        } else if (use_initial_state && !use_qk_l2norm_in_kernel) {              \
            LAUNCH_KERNEL(BK_CT, true, false);                                   \
        } else if (!use_initial_state && use_qk_l2norm_in_kernel) {              \
            LAUNCH_KERNEL(BK_CT, false, true);                                   \
        } else {                                                                 \
            LAUNCH_KERNEL(BK_CT, false, false);                                  \
        }

    if (bk_runtime == 128) {
        DISPATCH_BK(128);
    } else if (bk_runtime == 64) {
        DISPATCH_BK(64);
    } else if (bk_runtime == 256) {
        DISPATCH_BK(256);
    } else if (bk_runtime == 32) {
        DISPATCH_BK(32);
    } else {
        TORCH_CHECK(false, "Unsupported BK value: ", bk_runtime,
                    ". Supported: 32, 64, 128, 256");
    }

    #undef DISPATCH_BK
    #undef LAUNCH_KERNEL

    return o;
}

// ---------------------------------------------------------------------------
// PyBind11 / Torch extension binding
// ---------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "fused_split_gdr_update",
        &fused_split_gdr_update,
        "Fused Split GDR (Gating Delta Rule) decode update (HIP)\n"
        "NOTE: State must be in SWIZZLED layout (N_states, HV, K/4, V, 4)",
        py::arg("mixed_qkv"),
        py::arg("A_log"),
        py::arg("a"),
        py::arg("dt_bias"),
        py::arg("b_gate"),
        py::arg("initial_state_source"),
        py::arg("initial_state_indices"),
        py::arg("key_dim"),
        py::arg("value_dim"),
        py::arg("num_heads_qk"),
        py::arg("num_heads_v"),
        py::arg("head_dim"),
        py::arg("softplus_beta") = 1.0f,
        py::arg("softplus_threshold") = 20.0f,
        py::arg("scale") = -1.0f,
        py::arg("use_qk_l2norm_in_kernel") = true,
        py::arg("output") = c10::nullopt
    );
    m.def(
        "fused_split_gdr_update_ksplit2",
        &fused_split_gdr_update_ksplit2,
        "Fused Split GDR decode update with K-split=2 (HIP)\n"
        "K dimension split across 2 threads per V column.\n"
        "NOTE: State must be in SWIZZLED layout (N_states, HV, K/4, V, 4)",
        py::arg("mixed_qkv"),
        py::arg("A_log"),
        py::arg("a"),
        py::arg("dt_bias"),
        py::arg("b_gate"),
        py::arg("initial_state_source"),
        py::arg("initial_state_indices"),
        py::arg("key_dim"),
        py::arg("value_dim"),
        py::arg("num_heads_qk"),
        py::arg("num_heads_v"),
        py::arg("head_dim"),
        py::arg("softplus_beta") = 1.0f,
        py::arg("softplus_threshold") = 20.0f,
        py::arg("scale") = -1.0f,
        py::arg("use_qk_l2norm_in_kernel") = true,
        py::arg("output") = c10::nullopt
    );
    m.def(
        "fused_split_gdr_update_ksplit4",
        &fused_split_gdr_update_ksplit4,
        "Fused Split GDR decode update with K-split=4 (HIP)\n"
        "K dimension split across 4 threads per V column.\n"
        "NOTE: State must be in SWIZZLED layout (N_states, HV, K/4, V, 4)",
        py::arg("mixed_qkv"),
        py::arg("A_log"),
        py::arg("a"),
        py::arg("dt_bias"),
        py::arg("b_gate"),
        py::arg("initial_state_source"),
        py::arg("initial_state_indices"),
        py::arg("key_dim"),
        py::arg("value_dim"),
        py::arg("num_heads_qk"),
        py::arg("num_heads_v"),
        py::arg("head_dim"),
        py::arg("softplus_beta") = 1.0f,
        py::arg("softplus_threshold") = 20.0f,
        py::arg("scale") = -1.0f,
        py::arg("use_qk_l2norm_in_kernel") = true,
        py::arg("output") = c10::nullopt
    );
    m.def(
        "fused_split_gdr_update_ksplit4_db",
        &fused_split_gdr_update_ksplit4_db,
        "Fused Split GDR decode update with K-split=4, double-buffer smem (HIP)\n"
        "Q and K loaded into separate smem buffers in parallel.\n"
        "NOTE: State must be in SWIZZLED layout (N_states, HV, K/4, V, 4)",
        py::arg("mixed_qkv"),
        py::arg("A_log"),
        py::arg("a"),
        py::arg("dt_bias"),
        py::arg("b_gate"),
        py::arg("initial_state_source"),
        py::arg("initial_state_indices"),
        py::arg("key_dim"),
        py::arg("value_dim"),
        py::arg("num_heads_qk"),
        py::arg("num_heads_v"),
        py::arg("head_dim"),
        py::arg("softplus_beta") = 1.0f,
        py::arg("softplus_threshold") = 20.0f,
        py::arg("scale") = -1.0f,
        py::arg("use_qk_l2norm_in_kernel") = true,
        py::arg("output") = c10::nullopt
    );
    m.def(
        "fused_split_gdr_update_ksplit4_opt",
        &fused_split_gdr_update_ksplit4_opt,
        "Fused Split GDR decode update with K-split=4, LDS-read-pipelined (HIP)\n"
        "Phase 1/3 prefetch next iteration's LDS data while computing on current,\n"
        "hiding ds_read latency behind FMA computation.\n"
        "NOTE: State must be in SWIZZLED layout (N_states, HV, K/4, V, 4)",
        py::arg("mixed_qkv"),
        py::arg("A_log"),
        py::arg("a"),
        py::arg("dt_bias"),
        py::arg("b_gate"),
        py::arg("initial_state_source"),
        py::arg("initial_state_indices"),
        py::arg("key_dim"),
        py::arg("value_dim"),
        py::arg("num_heads_qk"),
        py::arg("num_heads_v"),
        py::arg("head_dim"),
        py::arg("softplus_beta") = 1.0f,
        py::arg("softplus_threshold") = 20.0f,
        py::arg("scale") = -1.0f,
        py::arg("use_qk_l2norm_in_kernel") = true,
        py::arg("output") = c10::nullopt
    );
    m.def(
        "fused_split_gdr_update_ksplit8",
        &fused_split_gdr_update_ksplit8,
        "Fused Split GDR decode update with K-split=8 (HIP)\n"
        "K dimension split across 8 threads per V column, row-major layout.\n"
        "NOTE: State must be in SWIZZLED layout (N_states, HV, K/4, V, 4)",
        py::arg("mixed_qkv"),
        py::arg("A_log"),
        py::arg("a"),
        py::arg("dt_bias"),
        py::arg("b_gate"),
        py::arg("initial_state_source"),
        py::arg("initial_state_indices"),
        py::arg("key_dim"),
        py::arg("value_dim"),
        py::arg("num_heads_qk"),
        py::arg("num_heads_v"),
        py::arg("head_dim"),
        py::arg("softplus_beta") = 1.0f,
        py::arg("softplus_threshold") = 20.0f,
        py::arg("scale") = -1.0f,
        py::arg("use_qk_l2norm_in_kernel") = true,
        py::arg("output") = c10::nullopt
    );
    m.def(
        "fused_split_gdr_update_ksplit4_hfuse",
        &fused_split_gdr_update_ksplit4_hfuse,
        "Fused Split GDR decode update with K-split=4 + multi-head fusion (HIP)\n"
        "Fuses HV/H v-heads sharing the same QK head into one block,\n"
        "loading K/Q only once per timestep.\n"
        "NOTE: State must be in SWIZZLED layout (N_states, HV, K/4, V, 4)",
        py::arg("mixed_qkv"),
        py::arg("A_log"),
        py::arg("a"),
        py::arg("dt_bias"),
        py::arg("b_gate"),
        py::arg("initial_state_source"),
        py::arg("initial_state_indices"),
        py::arg("key_dim"),
        py::arg("value_dim"),
        py::arg("num_heads_qk"),
        py::arg("num_heads_v"),
        py::arg("head_dim"),
        py::arg("softplus_beta") = 1.0f,
        py::arg("softplus_threshold") = 20.0f,
        py::arg("scale") = -1.0f,
        py::arg("use_qk_l2norm_in_kernel") = true,
        py::arg("output") = c10::nullopt
    );
    m.def(
        "fused_split_gdr_update_ksplit8_hfuse",
        &fused_split_gdr_update_ksplit8_hfuse,
        "Fused Split GDR decode update with K-split=8 + multi-head fusion (HIP)\n"
        "Fuses HV/H v-heads sharing the same QK head into one block,\n"
        "loading K/Q only once per timestep.  8-way K reduction.\n"
        "NOTE: State must be in SWIZZLED layout (N_states, HV, K/4, V, 4)",
        py::arg("mixed_qkv"),
        py::arg("A_log"),
        py::arg("a"),
        py::arg("dt_bias"),
        py::arg("b_gate"),
        py::arg("initial_state_source"),
        py::arg("initial_state_indices"),
        py::arg("key_dim"),
        py::arg("value_dim"),
        py::arg("num_heads_qk"),
        py::arg("num_heads_v"),
        py::arg("head_dim"),
        py::arg("softplus_beta") = 1.0f,
        py::arg("softplus_threshold") = 20.0f,
        py::arg("scale") = -1.0f,
        py::arg("use_qk_l2norm_in_kernel") = true,
        py::arg("output") = c10::nullopt
    );
    m.def(
        "fused_split_gdr_update_vsplit",
        &fused_split_gdr_update_vsplit,
        "Fused Split GDR decode update with V-split (HIP)\n"
        "8x8 thread layout: 8 K-threads x 8 V-threads, no LDS.\n"
        "NOTE: State must be in VSPLIT layout (N_states, HV, V/4, K, 4)",
        py::arg("mixed_qkv"),
        py::arg("A_log"),
        py::arg("a"),
        py::arg("dt_bias"),
        py::arg("b_gate"),
        py::arg("initial_state_source"),
        py::arg("initial_state_indices"),
        py::arg("key_dim"),
        py::arg("value_dim"),
        py::arg("num_heads_qk"),
        py::arg("num_heads_v"),
        py::arg("head_dim"),
        py::arg("softplus_beta") = 1.0f,
        py::arg("softplus_threshold") = 20.0f,
        py::arg("scale") = -1.0f,
        py::arg("use_qk_l2norm_in_kernel") = true,
        py::arg("output") = c10::nullopt
    );
}
