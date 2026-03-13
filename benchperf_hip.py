#!/usr/bin/env python3
"""Benchmark hip kernel across batch sizes and head configurations via rocprofv3.

Each kernel type (triton / flydsl / hip) runs in its own isolated subprocess
so that L2 cache state, memory pressure, and rocprofv3 tracing do not
interfere across kernel types.

Usage:
    python3 benchperf_hip.py --device 7
"""

import argparse
import csv
import os
import subprocess
import sys

SCENARIOS = [
    {"num_heads_qk": 4, "num_heads_v": 8, "head_dim": 128, "seqlen": 1},
    {"num_heads_qk": 2, "num_heads_v": 4, "head_dim": 128, "seqlen": 1},
    # {"num_heads_qk": 16, "num_heads_v": 32, "head_dim": 128, "seqlen": 1},
    # {"num_heads_qk": 16, "num_heads_v": 32, "head_dim": 128, "seqlen": 4},
]
BATCH_SIZES = [1, 16, 32, 64, 128, 256]
TRACE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trace_bench")
QUIET = True  # Set to False to show rocprofv3 subprocess stdout/stderr
FLYDSL_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "flydsl-examples"))

KERNEL_TYPES = ["triton", "flydsl", "hip"]


# =====================================================================
# Worker mode — invoked by rocprofv3 as a subprocess
# =====================================================================

def _worker(args):
    import torch

    device = "cuda"
    dtype = torch.bfloat16
    B, T = args.batch_size, args.seqlen
    H, HV, K, V = args.num_heads_qk, args.num_heads_v, args.head_dim, args.head_dim

    torch.manual_seed(42)
    q = torch.randn(B, T, H, K, device=device, dtype=dtype)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype)
    v = torch.randn(B, T, HV, V, device=device, dtype=dtype)
    A_log = torch.randn(HV, device=device, dtype=torch.float32)
    dt_bias = torch.randn(HV, device=device, dtype=dtype)
    a = torch.randn(B * T, HV, device=device, dtype=dtype)
    b_gate = torch.randn(B * T, HV, device=device, dtype=dtype)
    ssm_state = torch.randn(B, HV, K, V, device=device, dtype=torch.float32)
    ssm_state_indices = torch.arange(B, device=device, dtype=torch.int32)

    kt = args.kernel

    if kt == "triton":
        from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent import (
            fused_sigmoid_gating_delta_rule_update,
        )
        common = dict(
            A_log=A_log, a=a, dt_bias=dt_bias, softplus_beta=1.0,
            softplus_threshold=20.0, q=q, k=k, v=v, b=b_gate,
            scale=K ** -0.5, use_qk_l2norm_in_kernel=True,
        )
        def run(st):
            fused_sigmoid_gating_delta_rule_update(
                **common, initial_state_source=st,
                initial_state_indices=ssm_state_indices,
            )
        bench_state = ssm_state.clone()

    elif kt == "flydsl":
        sys.path.insert(0, FLYDSL_DIR)
        from gdn import Args as FlyDSLArgs, get_func as flydsl_get_func
        flydsl_args = FlyDSLArgs(
            dtype=dtype, b=B, sq=T, num_k_heads=H, num_v_heads=HV,
            head_k_dim=K, head_v_dim=V, use_qk_l2norm=True,
        )
        flydsl_exe = flydsl_get_func(flydsl_args)
        flydsl_a = a.reshape(B, T, HV)
        flydsl_b = b_gate.reshape(B, T, HV)
        flydsl_out = torch.zeros(B, T, HV, V, device=device, dtype=dtype)
        flydsl_scale = float(K ** -0.5)
        flydsl_state = ssm_state.permute(0, 1, 3, 2).contiguous()
        def run(st):
            flydsl_exe(q, k, v, flydsl_a, flydsl_b, dt_bias, A_log,
                       ssm_state_indices, st, flydsl_out, B, flydsl_scale)
        bench_state = flydsl_state.clone()

    elif kt == "hip":
        from torch.utils.cpp_extension import load
        key_dim = H * K
        value_dim = HV * V
        dim = 2 * key_dim + value_dim
        mixed_qkv = torch.randn(B, dim, T, device=device, dtype=dtype)
        hip_mod = load(
            name="split_gdr_hip",
            sources=[os.path.normpath(os.path.join(
                os.path.dirname(__file__),
                "python", "sglang", "srt", "layers", "attention", "fla",
                "split_gdr_decode_hip.hip",
            ))],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
        common = dict(
            mixed_qkv=mixed_qkv, A_log=A_log, a=a, b_gate=b_gate, dt_bias=dt_bias,
            key_dim=key_dim, value_dim=value_dim, num_heads_qk=H, num_heads_v=HV,
            head_dim=K, softplus_beta=1.0, softplus_threshold=20.0,
            scale=K ** -0.5, use_qk_l2norm_in_kernel=True,
        )
        ssm_swiz = ssm_state.reshape(B, HV, K // 4, 4, V) \
                             .permute(0, 1, 2, 4, 3).contiguous()
        def run(st):
            hip_mod.fused_split_gdr_update_ksplit4_opt(
                **common, initial_state_source=st,
                initial_state_indices=ssm_state_indices,
            )
        bench_state = ssm_swiz.clone()

    else:
        raise ValueError(f"Unknown kernel type: {kt}")

    # MI300X has 256 MB L2; writing a same-sized buffer evicts all prior data.
    l2_size = 256 * 1024 * 1024
    l2_flush = torch.empty(l2_size // 4, dtype=torch.float32, device=device)

    for _ in range(args.warmup):
        l2_flush.zero_()
        run(bench_state)
    torch.cuda.synchronize()
    for _ in range(args.iters):
        l2_flush.zero_()
        run(bench_state)
    torch.cuda.synchronize()


# =====================================================================
# Main mode — orchestrate rocprofv3 runs and collect results
# =====================================================================

_KERNEL_CSV_MATCH = {
    "triton": lambda name: "fused_sigmoid_gating_delta_rule_update_kernel" in name,
    "flydsl": lambda name: name == "fused_gdn_kernel",
    "hip":    lambda name: "ksplit4_opt<" in name,
}


def _run_one(batch_size, scenario_idx, scenario, gpu_id):
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = gpu_id
    quiet_opts = dict(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) if QUIET else {}

    result = {}
    for kt in KERNEL_TYPES:
        tag = f"s{scenario_idx}_b{batch_size}_{kt}"
        out_dir = os.path.join(TRACE_DIR, tag)
        os.makedirs(out_dir, exist_ok=True)

        cmd = [
            "rocprofv3", "--kernel-trace", "--output-format", "csv",
            "-d", out_dir, "-o", "k", "--stats", "--",
            sys.executable, __file__, "--worker",
            "--kernel", kt,
            "--batch_size", str(batch_size),
            "--seqlen", str(scenario["seqlen"]),
            "--num_heads_qk", str(scenario["num_heads_qk"]),
            "--num_heads_v", str(scenario["num_heads_v"]),
            "--head_dim", str(scenario["head_dim"]),
        ]

        try:
            subprocess.run(cmd, env=env, check=True, **quiet_opts)
        except subprocess.CalledProcessError:
            continue

        stats_file = os.path.join(out_dir, "k_kernel_stats.csv")
        if not os.path.exists(stats_file):
            continue

        matcher = _KERNEL_CSV_MATCH[kt]
        with open(stats_file) as f:
            for row in csv.DictReader(f):
                if matcher(row["Name"]):
                    result[kt] = float(row["AverageNs"])
                    break

    return result if result else None


def _main(gpu_id):
    os.makedirs(TRACE_DIR, exist_ok=True)
    results = {}

    for si, scenario in enumerate(SCENARIOS):
        label = f"H={scenario['num_heads_qk']},HV={scenario['num_heads_v']},T={scenario['seqlen']}"
        results[label] = {}
        for bs in BATCH_SIZES:
            short = f"H={scenario['num_heads_qk']},HV={scenario['num_heads_v']},T={scenario['seqlen']}"
            print(f"  Running scenario {si+1} ({short}), batch_size={bs} ...",
                  end="", flush=True)
            r = _run_one(bs, si, scenario, gpu_id)
            if r:
                results[label][bs] = r
                t_us = r.get('triton', 0) / 1000
                f_us = r.get('flydsl', 0) / 1000
                h_us = r.get('hip', 0) / 1000
                print(f"  triton={t_us:.1f}us  flydsl={f_us:.1f}us  hip={h_us:.1f}us")
            else:
                print("  FAILED")

    rows = []
    for scenario in SCENARIOS:
        H = scenario["num_heads_qk"]
        HV = scenario["num_heads_v"]
        T = scenario["seqlen"]
        K = scenario["head_dim"]
        tp = 16 // H
        label = f"H={H},HV={HV},T={T}"
        for bs in BATCH_SIZES:
            r = results.get(label, {}).get(bs)
            triton_us = r["triton"] / 1000 if r and "triton" in r else None
            flydsl_us = r["flydsl"] / 1000 if r and "flydsl" in r else None
            hip_us = r["hip"] / 1000 if r and "hip" in r else None
            sp_triton = triton_us / hip_us if triton_us and hip_us else None
            sp_flydsl = flydsl_us / hip_us if flydsl_us and hip_us else None
            rows.append({
                "tp": f"TP{tp}",
                "batch": bs,
                "num_heads_qk": H,
                "num_heads_v": HV,
                "head_dim": K,
                "seqlen": T,
                "triton_us": triton_us,
                "flydsl_us": flydsl_us,
                "hip_us": hip_us,
                "sp_triton": sp_triton,
                "sp_flydsl": sp_flydsl,
            })

    tw, fw, hw, sw, sf = 16, 16, 14, 22, 22
    hdr = (f"{'TP':>4s}  {'num_heads_qk':>12s}  {'num_heads_v':>11s}  {'head_dim':>8s}"
           f"  {'batch':>6s}  {'seqlen':>6s}  {'triton-perf (us)':>{tw}s}  {'flydsl-perf (us)':>{fw}s}"
           f"  {'hip-perf (us)':>{hw}s}  {'speedup(hip vs triton)':>{sw}s}  {'speedup(hip vs flydsl)':>{sf}s}")
    sep = "-" * len(hdr)

    print(f"\n{'='*len(hdr)}")
    print("Kernel Performance: triton vs flydsl vs hip (ksplit4)")
    print(f"{'='*len(hdr)}")
    print(hdr)
    print(sep)
    for row in rows:
        t_str = f"{row['triton_us']:>{tw}.2f}" if row["triton_us"] is not None else f"{'N/A':>{tw}s}"
        f_str = f"{row['flydsl_us']:>{fw}.2f}" if row["flydsl_us"] is not None else f"{'N/A':>{fw}s}"
        h_str = f"{row['hip_us']:>{hw}.2f}" if row["hip_us"] is not None else f"{'N/A':>{hw}s}"
        s_str = f"{row['sp_triton']:>{sw}.3f}" if row["sp_triton"] is not None else f"{'N/A':>{sw}s}"
        sf_str = f"{row['sp_flydsl']:>{sf}.3f}" if row["sp_flydsl"] is not None else f"{'N/A':>{sf}s}"
        print(f"{row['tp']:>4s}  {row['num_heads_qk']:>12d}  {row['num_heads_v']:>11d}  {row['head_dim']:>8d}"
              f"  {row['batch']:>6d}  {row['seqlen']:>6d}  {t_str}  {f_str}  {h_str}  {s_str}  {sf_str}")
    print(sep)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=7,
                        help="HIP device id to run benchmarks on")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--kernel", choices=KERNEL_TYPES, default="triton")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seqlen", type=int, default=1)
    parser.add_argument("--num_heads_qk", type=int, default=4)
    parser.add_argument("--num_heads_v", type=int, default=8)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=1000)
    args = parser.parse_args()

    if args.worker:
        _worker(args)
    else:
        _main(str(args.device))
