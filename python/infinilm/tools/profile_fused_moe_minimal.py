#!/usr/bin/env python3
# Copyright (c) 2025, InfiniCore
"""Minimal FusedMoE microbench for hcTracer (no vLLM, no serve).

Modes:
  launcher    — fused_moe_routed only for M in {1,16,256,2048}, TOP_K=16
  align_only  — moe_align_block_size x2 (stage1+stage2 block sizes), no Triton
  align_tax   — E_align (eager) vs C_align (capture path) @ M (default 2048)
  aoti_b16    — aoti_load_package(moe_B16/segment.pt2) with hidden [1,16,H]
  cpp_pad_m1  — C++ inductor_moe_ path: hidden [1,1,H] → pad to B16 (decode pad tax)
  host_split  — Phase 0 attribution: timed align/.item/opaque + capture-align @ M
  kernel_only — precompute align+silu once; time `_invoke_kernel` x2 only
  capture_replay — BeginCapture → one fused_moe_routed → EndCapture/Instantiate
                   → timed Launch×N (optional INFINI_MOE_STUB_* stage stubs)

Env (required; wrapper sets defaults):
  INFINI_MOE_CONFIGS, INFINI_MOE_TRITON_CACHE / TRITON_CACHE_DIR,
  INFINI_MOE_ALLOW_JIT=0 (default). Set to 1 only for raised-SMEM /
  side-cache experiments that must JIT missing cubins.

Refuses if Triton cache grows during JIT-off runs (G4 spirit).
When INFINI_MOE_ALLOW_JIT=1, cache growth is expected and not refused.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


H, E, N, TOP_K = 2048, 160, 512, 16
LAUNCHER_M_ALLOWED = (1, 16, 256, 2048)
DEFAULT_WARMUP = 5
DEFAULT_ITERS = 20


def _count_cache_entries(cache_dir: Path) -> int:
    if not cache_dir.is_dir():
        return 0
    return sum(1 for _ in cache_dir.rglob("*") if _.is_file())


def _cache_dir() -> Path:
    raw = (
        os.environ.get("INFINI_MOE_TRITON_CACHE", "").strip()
        or os.environ.get("TRITON_CACHE_DIR", "").strip()
    )
    if not raw:
        raise RuntimeError("INFINI_MOE_TRITON_CACHE / TRITON_CACHE_DIR unset")
    return Path(raw)


def _jit_allowed() -> bool:
    return os.environ.get("INFINI_MOE_ALLOW_JIT", "0").strip() in ("1", "true", "TRUE", "yes")


def _require_jit_off_or_allow() -> None:
    """Default: force JIT off. Opt-in ALLOW_JIT=1 for side-cache SMEM experiments."""
    if _jit_allowed():
        os.environ["INFINI_MOE_ALLOW_JIT"] = "1"
        print("[profile-moe] INFINI_MOE_ALLOW_JIT=1 (side-cache / raised-SMEM path)", flush=True)
        return
    os.environ["INFINI_MOE_ALLOW_JIT"] = "0"


def _refuse_vllm() -> None:
    if "vllm" in sys.modules:
        raise RuntimeError("vllm already imported; refuse (serve-free microbench)")


def _make_launcher_inputs(M: int, device, dtype):
    import torch

    x = torch.randn(M, H, device=device, dtype=dtype)
    topk_ids = torch.randint(0, E, (M, TOP_K), device=device, dtype=torch.int32)
    topk_w = torch.softmax(
        torch.randn(M, TOP_K, device=device, dtype=torch.float32), dim=-1
    ).to(dtype)
    w_gu = torch.randn(E, 2 * N, H, device=device, dtype=dtype).contiguous()
    w_d = torch.randn(E, H, N, device=device, dtype=dtype).contiguous()
    return x, topk_w, topk_ids, w_gu, w_d


def _run_timed(label: str, fn, *, warmup: int, iters: int, device) -> float:
    import torch

    print(f"=== WARMUP_BEGIN mode={label} warmup={warmup} ===", flush=True)
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
    print(f"=== WARMUP_END mode={label} ===", flush=True)

    print(f"=== TIMED_BEGIN mode={label} iters={iters} ===", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    ms_per_iter = elapsed_ms / max(iters, 1)
    print(
        f"=== TIMED_END mode={label} host_ms_total={elapsed_ms:.3f} "
        f"host_ms_per_iter={ms_per_iter:.3f} iters={iters} ===",
        flush=True,
    )
    print(f"[profile-moe] {label} host_ms/iter={ms_per_iter:.3f}", flush=True)
    return ms_per_iter


def _run_launcher(args) -> int:
    import torch
    from infinilm.kernels.fused_moe_runtime import fused_moe_routed, launcher_hash

    _refuse_vllm()
    os.environ["INFINI_MOE_PROFILE_PHASES"] = "1"
    device = torch.device("cuda", 0)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    M = int(args.M)
    if M not in LAUNCHER_M_ALLOWED:
        raise ValueError(
            f"launcher mode expects M in {LAUNCHER_M_ALLOWED}, got {M}"
        )

    print(f"[profile-moe] mode=launcher M={M} TOP_K={TOP_K} H={H} E={E} N={N}")
    print(f"[profile-moe] launcher_hash={launcher_hash()}")
    print(f"[profile-moe] configs={os.environ.get('INFINI_MOE_CONFIGS')}")
    print(f"[profile-moe] triton_cache={_cache_dir()}")

    x, topk_w, topk_ids, w_gu, w_d = _make_launcher_inputs(M, device, dtype)

    def once():
        return fused_moe_routed(x, topk_w, topk_ids, w_gu, w_d)

    ms = _run_timed(
        f"launcher_m{M}",
        once,
        warmup=args.warmup,
        iters=args.iters,
        device=device,
    )
    if args.out_dir:
        import json

        out = {
            "mode": "launcher",
            "M": M,
            "host_ms_per_iter": ms,
            "warmup": args.warmup,
            "iters": args.iters,
            "launcher_hash": launcher_hash(),
        }
        path = Path(args.out_dir) / f"launcher_m{M}_summary.json"
        path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
        print(f"[profile-moe] wrote {path}", flush=True)
    return 0


def _run_align_only(args) -> int:
    from infinilm.kernels.fused_moe_runtime import (
        get_moe_config_for_m,
        launcher_hash,
        moe_align_block_size,
    )

    _refuse_vllm()
    device = __import__("torch").device("cuda", 0)
    M = int(args.M)
    if M not in LAUNCHER_M_ALLOWED:
        raise ValueError(
            f"align_only mode expects M in {LAUNCHER_M_ALLOWED}, got {M}"
        )

    print(f"[profile-moe] mode=align_only M={M} TOP_K={TOP_K} H={H} E={E} N={N}")
    print(f"[profile-moe] launcher_hash={launcher_hash()}")

    import torch

    topk_ids = torch.randint(0, E, (M, TOP_K), device=device, dtype=torch.int32)
    cfg1 = get_moe_config_for_m(M, E=E, N=N, H=H, stage="stage1")
    cfg2 = get_moe_config_for_m(M, E=E, N=N, H=H, stage="stage2")

    def once():
        print("=== PHASE align1 ===", flush=True)
        moe_align_block_size(topk_ids, int(cfg1["BLOCK_SIZE_M"]), E)
        print("=== PHASE align2 ===", flush=True)
        moe_align_block_size(topk_ids, int(cfg2["BLOCK_SIZE_M"]), E)

    _run_timed(
        f"align_only_m{M}",
        once,
        warmup=args.warmup,
        iters=args.iters,
        device=device,
    )
    return 0


def _run_align_tax(args) -> int:
    """E_align (eager argsort) vs C_align (capture O(n²)/kernel path) @ M."""
    import json

    import torch

    from infinilm.kernels.fused_moe_runtime import (
        _moe_align_block_size_capture,
        get_moe_config_for_m,
        launcher_hash,
        moe_align_block_size,
    )

    _refuse_vllm()
    device = torch.device("cuda", 0)
    M = int(args.M)
    if M not in LAUNCHER_M_ALLOWED:
        raise ValueError(
            f"align_tax mode expects M in {LAUNCHER_M_ALLOWED}, got {M}"
        )

    print(f"[profile-moe] mode=align_tax M={M} TOP_K={TOP_K} E={E} N={N}")
    print(f"[profile-moe] launcher_hash={launcher_hash()}")
    print(
        "[profile-moe] C_align calls _moe_align_block_size_capture directly "
        "(same device algo as under hcStream capture; arena→torch.empty fallback)",
        flush=True,
    )

    topk_ids = torch.randint(0, E, (M, TOP_K), device=device, dtype=torch.int32)
    cfg1 = get_moe_config_for_m(M, E=E, N=N, H=H, stage="stage1")
    cfg2 = get_moe_config_for_m(M, E=E, N=N, H=H, stage="stage2")
    bs1 = int(cfg1["BLOCK_SIZE_M"])
    bs2 = int(cfg2["BLOCK_SIZE_M"])
    numel = M * TOP_K
    graph_eager_delta_ms = 22.0  # ~post_graph_run A0−A1 MoE layer (pw_graph_tax)

    def eager_once():
        moe_align_block_size(topk_ids, bs1, E)
        moe_align_block_size(topk_ids, bs2, E)

    def capture_once():
        # Force capture algorithm without TLS/CaptureArena (empty→torch.empty).
        _moe_align_block_size_capture(topk_ids, bs1, E)
        _moe_align_block_size_capture(topk_ids, bs2, E)

    e_ms = _run_timed(
        f"E_align_m{M}",
        eager_once,
        warmup=args.warmup,
        iters=args.iters,
        device=device,
    )
    c_ms = _run_timed(
        f"C_align_m{M}",
        capture_once,
        warmup=args.warmup,
        iters=args.iters,
        device=device,
    )
    ratio = (c_ms / e_ms) if e_ms > 0 else float("inf")
    explains = (c_ms / graph_eager_delta_ms) if graph_eager_delta_ms > 0 else None
    gate_ratio_ok = ratio >= 3.0
    gate_explain_ok = explains is not None and explains >= 0.5
    gate_pass = bool(gate_ratio_ok or gate_explain_ok)

    out = {
        "mode": "align_tax",
        "M": M,
        "TOP_K": TOP_K,
        "E": E,
        "numel": numel,
        "BLOCK_SIZE_M_stage1": bs1,
        "BLOCK_SIZE_M_stage2": bs2,
        "E_align_ms": e_ms,
        "C_align_ms": c_ms,
        "C_over_E": ratio,
        "graph_eager_delta_ms_ref": graph_eager_delta_ms,
        "C_align_frac_of_delta": explains,
        "gate": {
            "C_over_E_ge_3": gate_ratio_ok,
            "C_explains_ge_50pct_delta": gate_explain_ok,
            "pass": gate_pass,
        },
        "warmup": args.warmup,
        "iters": args.iters,
        "launcher_hash": launcher_hash(),
    }
    print("[profile-moe] ALIGN_TAX " + json.dumps(out, sort_keys=True), flush=True)
    if args.out_dir:
        path = Path(args.out_dir) / "ALIGN_TAX.json"
        path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
        print(f"[profile-moe] wrote {path}", flush=True)
    return 0 if gate_pass else 2


def _run_aoti_b16(args) -> int:
    import torch
    from torch._inductor import aoti_load_package

    from infinilm.compile.piecewise_moe_segment import make_moe_example_inputs
    from infinilm.kernels.fused_moe_runtime import launcher_hash

    _refuse_vllm()
    device = torch.device("cuda", 0)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    pkg = Path(args.segment_pt2).resolve()
    if not pkg.is_file():
        raise FileNotFoundError(f"moe_B16 segment.pt2 missing: {pkg}")

    print(f"[profile-moe] mode=aoti_b16 package={pkg}")
    print(f"[profile-moe] launcher_hash={launcher_hash()}")
    print(
        f"[profile-moe] shapes hidden=[1,16,{H}] (bucket pad; valid baked in pt2) "
        f"TOP_K={TOP_K} (segment routing)"
    )

    from infinilm.torch_llama.moe_ops import register_fused_moe_routed_op

    register_fused_moe_routed_op()

    device_index = device.index if device.index is not None else 0
    runner = aoti_load_package(str(pkg), device_index=device_index)
    inputs = make_moe_example_inputs(
        bucket=16,
        hidden_size=H,
        moe_intermediate_size=N,
        n_routed_experts=E,
        device=device,
        dtype=dtype,
    )

    def once():
        return runner(*inputs)

    _run_timed(
        "aoti_b16",
        once,
        warmup=args.warmup,
        iters=args.iters,
        device=device,
    )
    return 0


def _run_cpp_pad_m1(args) -> int:
    """Decode-like pad tax: seq=1 through C++ inductor_moe_ → eager MoE (or AOTI)."""
    import infinicore
    import torch
    from infinicore.lib import _infinicore as _ic
    from infinilm.compile.piecewise_moe_segment import make_moe_example_inputs
    from infinilm.compile.piecewise_segments import LAYER_AGNOSTIC_IDX, SEGMENT_MOE
    from infinilm.kernels.fused_moe_runtime import launcher_hash
    from infinilm.torch_llama.moe_ops import (
        configure_moe_block_routing,
        register_fused_moe_routed_op,
    )

    _refuse_vllm()
    register_fused_moe_routed_op()
    # Match MiniCPM5 routing (same as serve bootstrap / C++ eager decode).
    configure_moe_block_routing(
        top_k=TOP_K,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=3.66,
    )
    os.environ["INFINI_MOE_TOP_K"] = str(TOP_K)
    os.environ["INFINI_MOE_N_GROUP"] = "1"
    os.environ["INFINI_MOE_TOPK_GROUP"] = "1"
    os.environ["INFINI_MOE_ROUTED_SCALING"] = "3.66"
    os.environ["INFINI_MOE_NORM_TOPK"] = "1"

    # Decode valid_len=1 (resolver / env); package stays B16 (prefill fallback).
    os.environ["INFINI_PIECEWISE_VALID_LEN"] = "1"
    os.environ["INFINI_PIECEWISE_INDUCTOR_SEGMENT"] = "1"

    device = torch.device("cuda", 0)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    pkg = Path(args.segment_pt2).resolve()
    if not pkg.is_file():
        raise FileNotFoundError(f"moe_B16 segment.pt2 missing: {pkg}")

    print(f"[profile-moe] mode=cpp_pad_m1 package={pkg}")
    print(f"[profile-moe] launcher_hash={launcher_hash()}")
    print(
        f"[profile-moe] shapes hidden=[1,1,{H}] eager_decode valid_seq_len=1 TOP_K={TOP_K}",
        flush=True,
    )
    print(
        f"[profile-moe] INFINI_MOE_EAGER_DECODE_MAX="
        f"{os.environ.get('INFINI_MOE_EAGER_DECODE_MAX', '16 (default)')}",
        flush=True,
    )

    ic_dev = infinicore.device("cuda", 0)
    infinicore.set_device(ic_dev)

    _ic.register_piecewise_inductor_package(
        SEGMENT_MOE,
        LAYER_AGNOSTIC_IDX,
        16,
        str(pkg),
        0,
        True,
    )
    if hasattr(_ic, "set_piecewise_inductor_lookup_tp_rank"):
        _ic.set_piecewise_inductor_lookup_tp_rank(0)

    examples = make_moe_example_inputs(
        bucket=16,
        hidden_size=H,
        moe_intermediate_size=N,
        n_routed_experts=E,
        device=device,
        dtype=dtype,
    )
    _h, gate_w, bias, w_gu, w_d, shared_gu, shared_d = examples
    _ic.register_moe_external_weights(
        0,
        *[
            infinicore.from_torch(t.contiguous())._underlying
            for t in (gate_w, bias, w_gu, w_d, shared_gu, shared_d)
        ],
    )

    hidden_t = torch.randn(1, 1, H, device=device, dtype=dtype)
    out_t = torch.empty(1, 1, H, device=device, dtype=dtype)
    hidden = infinicore.from_torch(hidden_t)
    out = infinicore.from_torch(out_t)

    def once():
        _ic.inductor_moe_(hidden._underlying, out._underlying, 0, 16)

    _run_timed(
        "cpp_pad_m1",
        once,
        warmup=args.warmup,
        iters=args.iters,
        device=device,
    )
    return 0


def _run_host_split(args) -> int:
    """Phase 0: attribute align/.item/opaque + AOTI vs launcher vs cpp_pad @ M."""
    import json

    import torch
    from infinilm.kernels.fused_moe_runtime import (
        _HOST_SPLIT,
        _moe_align_block_size_capture,
        fused_moe_routed,
        get_moe_config_for_m,
        host_split_report,
        host_split_reset,
        launcher_hash,
        moe_align_block_size,
    )

    _refuse_vllm()
    os.environ["INFINI_MOE_HOST_SPLIT"] = "1"
    os.environ["INFINI_MOE_PROFILE_PHASES"] = "0"
    host_split_reset()

    device = torch.device("cuda", 0)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    M = int(args.M)
    if M not in LAUNCHER_M_ALLOWED:
        raise ValueError(
            f"host_split mode expects M in {LAUNCHER_M_ALLOWED}, got {M}"
        )
    warmup = max(int(args.warmup), 3)
    iters = max(int(args.iters), 10)

    print(f"[profile-moe] mode=host_split M={M} TOP_K={TOP_K} H={H} E={E} N={N}")
    print(f"[profile-moe] launcher_hash={launcher_hash()}")
    print(f"[profile-moe] configs={os.environ.get('INFINI_MOE_CONFIGS')}")
    print(f"[profile-moe] triton_cache={_cache_dir()}")

    x, topk_w, topk_ids, w_gu, w_d = _make_launcher_inputs(M, device, dtype)
    cfg1 = get_moe_config_for_m(M, E=E, N=N, H=H, stage="stage1")
    cfg2 = get_moe_config_for_m(M, E=E, N=N, H=H, stage="stage2")
    bs1 = int(cfg1["BLOCK_SIZE_M"])
    bs2 = int(cfg2["BLOCK_SIZE_M"])

    # --- align_only with .item split ---
    def align_once():
        moe_align_block_size(topk_ids, bs1, E)
        moe_align_block_size(topk_ids, bs2, E)

    print(f"=== WARMUP_BEGIN mode=align_only_m{M} warmup={warmup} ===", flush=True)
    with torch.no_grad():
        for _ in range(warmup):
            align_once()
        torch.cuda.synchronize()
    print(f"=== WARMUP_END mode=align_only_m{M} ===", flush=True)

    host_split_reset()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            align_once()
        torch.cuda.synchronize()
    align_ms = (time.perf_counter() - t0) * 1000.0 / iters
    align_split = host_split_report(f"align_only_m{M} x{iters}")
    align_per = {k: v / iters for k, v in align_split.items()}
    print(f"[profile-moe] align_only_m{M} host_ms/iter={align_ms:.3f}", flush=True)
    print(
        f"[moe-host-split] align_only_m{M} per_iter_ms "
        + " ".join(f"{k}={v:.3f}" for k, v in sorted(align_per.items())),
        flush=True,
    )

    # --- capture-align residual (GraphExec path; no TLS capture needed) ---
    def capture_align_once():
        _moe_align_block_size_capture(topk_ids, bs1, E)
        _moe_align_block_size_capture(topk_ids, bs2, E)

    print(f"=== WARMUP_BEGIN mode=align_capture_m{M} warmup={warmup} ===", flush=True)
    with torch.no_grad():
        for _ in range(warmup):
            capture_align_once()
        torch.cuda.synchronize()
    print(f"=== WARMUP_END mode=align_capture_m{M} ===", flush=True)

    host_split_reset()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            # Mirror fused_moe_runtime HOST_SPLIT key used under hcStream capture.
            _HOST_SPLIT.begin("align_capture")
            capture_align_once()
            _HOST_SPLIT.end("align_capture")
        torch.cuda.synchronize()
    align_capture_ms = (time.perf_counter() - t0) * 1000.0 / iters
    capture_split = host_split_report(f"align_capture_m{M} x{iters}")
    capture_per = {k: v / iters for k, v in capture_split.items()}
    print(
        f"[profile-moe] align_capture_m{M} host_ms/iter={align_capture_ms:.3f}",
        flush=True,
    )
    print(
        f"[moe-host-split] align_capture_m{M} per_iter_ms "
        + " ".join(f"{k}={v:.3f}" for k, v in sorted(capture_per.items())),
        flush=True,
    )

    # --- launcher opaque stage split ---
    def launch_once():
        return fused_moe_routed(x, topk_w, topk_ids, w_gu, w_d)

    print(f"=== WARMUP_BEGIN mode=launcher_m{M} warmup={warmup} ===", flush=True)
    with torch.no_grad():
        for _ in range(warmup):
            launch_once()
        torch.cuda.synchronize()
    print(f"=== WARMUP_END mode=launcher_m{M} ===", flush=True)

    host_split_reset()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            launch_once()
        torch.cuda.synchronize()
    launcher_ms = (time.perf_counter() - t0) * 1000.0 / iters
    launcher_split = host_split_report(f"launcher_m{M} x{iters}")
    launcher_per = {k: v / iters for k, v in launcher_split.items()}
    print(f"[profile-moe] launcher_m{M} host_ms/iter={launcher_ms:.3f}", flush=True)
    print(
        f"[moe-host-split] launcher_m{M} per_iter_ms "
        + " ".join(f"{k}={v:.3f}" for k, v in sorted(launcher_per.items())),
        flush=True,
    )

    # --- aoti_b16 / cpp_pad_m1 wall (decode-pad tax; M=1 only) ---
    aoti_ms = None
    cpp_ms = None
    if args.segment_pt2 and M == 1:
        # Reuse existing runners without host-split noise inside Triton.
        os.environ["INFINI_MOE_HOST_SPLIT"] = "0"
        host_split_reset()
        from torch._inductor import aoti_load_package

        from infinilm.compile.piecewise_moe_segment import make_moe_example_inputs
        from infinilm.torch_llama.moe_ops import register_fused_moe_routed_op

        register_fused_moe_routed_op()
        pkg = Path(args.segment_pt2).resolve()
        runner = aoti_load_package(
            str(pkg),
            device_index=device.index if device.index is not None else 0,
        )
        inputs = make_moe_example_inputs(
            bucket=16,
            hidden_size=H,
            moe_intermediate_size=N,
            n_routed_experts=E,
            device=device,
            dtype=dtype,
        )

        def aoti_once():
            return runner(*inputs)

        aoti_ms = _run_timed(
            "aoti_b16",
            aoti_once,
            warmup=warmup,
            iters=iters,
            device=device,
        )

        cpp_ms = None
        try:
            import infinicore
            from infinicore.lib import _infinicore as _ic
            from infinilm.compile.piecewise_segments import (
                LAYER_AGNOSTIC_IDX,
                SEGMENT_MOE,
            )

            os.environ["INFINI_PIECEWISE_VALID_LEN"] = "1"
            os.environ["INFINI_PIECEWISE_INDUCTOR_SEGMENT"] = "1"
            ic_dev = infinicore.device("cuda", 0)
            infinicore.set_device(ic_dev)
            _ic.register_piecewise_inductor_package(
                SEGMENT_MOE, LAYER_AGNOSTIC_IDX, 16, str(pkg), 0, True
            )
            if hasattr(_ic, "set_piecewise_inductor_lookup_tp_rank"):
                _ic.set_piecewise_inductor_lookup_tp_rank(0)
            _h, gate_w, bias, w_gu2, w_d2, shared_gu, shared_d = inputs
            _ic.register_moe_external_weights(
                0,
                *[
                    infinicore.from_torch(t.contiguous())._underlying
                    for t in (gate_w, bias, w_gu2, w_d2, shared_gu, shared_d)
                ],
            )
            hidden_t = torch.randn(1, 1, H, device=device, dtype=dtype)
            out_t = torch.empty(1, 1, H, device=device, dtype=dtype)
            hidden = infinicore.from_torch(hidden_t)
            out = infinicore.from_torch(out_t)

            def cpp_once():
                _ic.inductor_moe_(hidden._underlying, out._underlying, 0, 16)

            cpp_ms = _run_timed(
                "cpp_pad_m1",
                cpp_once,
                warmup=warmup,
                iters=iters,
                device=device,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[profile-moe] cpp_pad_m1 skipped: {exc}", flush=True)
    elif M != 1:
        print(
            f"[profile-moe] skip aoti_b16/cpp_pad_m1 (decode-pad tax; M={M})",
            flush=True,
        )

    # Derived AOTI / pad taxes (plan Phase 0).
    aoti_tax = (aoti_ms - launcher_ms) if aoti_ms is not None else None
    cpp_vs_launcher = (cpp_ms - launcher_ms) if cpp_ms is not None else None
    cpp_vs_aoti = (cpp_ms - aoti_ms) if (cpp_ms is not None and aoti_ms is not None) else None

    # Leaf keys for align (avoid double-counting nested walls).
    leaf_align = [
        "align_pre_item",
        "align_item_n_post",
        "align_item_total_pad",
        "align_pad_fill",
        "align_expert_ids",
    ]
    item_ms = (
        align_per.get("align_item_n_post", 0.0)
        + align_per.get("align_item_total_pad", 0.0)
        + align_per.get("align_pad_fill", 0.0)
    )
    # Launcher: prefer opaque walls; item keys are nested inside align walls.
    opaque_keys = [
        "opaque_alloc",
        "opaque_align1_wall",
        "opaque_kernel1",
        "opaque_silu",
        "opaque_align2_wall",
        "opaque_kernel2",
        "opaque_moe_sum",
    ]
    opaque_sum = sum(launcher_per.get(k, 0.0) for k in opaque_keys)
    kernel_ms = launcher_per.get("opaque_kernel1", 0.0) + launcher_per.get(
        "opaque_kernel2", 0.0
    )

    align_capture_key_ms = capture_per.get("align_capture", align_capture_ms)
    summary = {
        "M": M,
        "align_only_ms": align_ms,
        "align_only_m1_ms": align_ms if M == 1 else None,
        "align_capture_ms": align_capture_key_ms,
        "launcher_ms": launcher_ms,
        "launcher_m1_ms": launcher_ms if M == 1 else None,
        "aoti_b16_ms": aoti_ms,
        "cpp_pad_m1_ms": cpp_ms,
        "aoti_minus_launcher_ms": aoti_tax,
        "cpp_minus_launcher_ms": cpp_vs_launcher,
        "cpp_minus_aoti_ms": cpp_vs_aoti,
        "align_item_sync_ms": item_ms,
        "align_leaf_ms": {k: align_per.get(k, 0.0) for k in leaf_align},
        "launcher_opaque_ms": {k: launcher_per.get(k, 0.0) for k in opaque_keys},
        "launcher_opaque_sum_ms": opaque_sum,
        "launcher_kernel_stages_ms": kernel_ms,
        "opaque_kernel1_ms": launcher_per.get("opaque_kernel1", 0.0),
        "opaque_kernel2_ms": launcher_per.get("opaque_kernel2", 0.0),
        "opaque_silu_ms": launcher_per.get("opaque_silu", 0.0),
        "opaque_moe_sum_ms": launcher_per.get("opaque_moe_sum", 0.0),
        "align_capture_vs_opaque_kernels": (
            (align_capture_key_ms / kernel_ms) if kernel_ms > 0 else None
        ),
        "launcher_hash": launcher_hash(),
        "iters": iters,
        "warmup": warmup,
    }
    print("[moe-host-split] SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    if args.out_dir:
        out_path = Path(args.out_dir) / "host_split_summary.json"
        out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(f"[moe-host-split] wrote {out_path}", flush=True)
    return 0


def _run_kernel_only(args) -> int:
    """Time InfiniLM Triton GEMMs only (align+silu precomputed outside timed loop)."""
    import json

    import torch
    import triton.language as tl
    from infinilm.kernels.fused_moe_runtime import (
        _invoke_kernel,
        get_moe_config_for_m,
        launcher_hash,
        moe_align_block_size,
    )

    _refuse_vllm()
    os.environ["INFINI_MOE_PROFILE_PHASES"] = "0"

    device = torch.device("cuda", 0)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    M = int(args.M)
    if M not in LAUNCHER_M_ALLOWED:
        raise ValueError(f"kernel_only expects M in {LAUNCHER_M_ALLOWED}, got {M}")

    compute_type = tl.bfloat16 if dtype == torch.bfloat16 else tl.float16
    x, topk_w, topk_ids, w_gu, w_d = _make_launcher_inputs(M, device, dtype)
    cfg1 = get_moe_config_for_m(M, E=E, N=N, H=H, stage="stage1")
    cfg2 = get_moe_config_for_m(M, E=E, N=N, H=H, stage="stage2")

    sorted1, expert1, npost1 = moe_align_block_size(topk_ids, int(cfg1["BLOCK_SIZE_M"]), E)
    if int(cfg2["BLOCK_SIZE_M"]) == int(cfg1["BLOCK_SIZE_M"]):
        sorted2, expert2, npost2 = sorted1, expert1, npost1
    else:
        sorted2, expert2, npost2 = moe_align_block_size(
            topk_ids, int(cfg2["BLOCK_SIZE_M"]), E
        )

    cache1 = torch.empty(M, TOP_K, 2 * N, device=device, dtype=dtype)
    cache2 = torch.empty(M * TOP_K, N, device=device, dtype=dtype)
    cache3 = torch.empty(M, TOP_K, H, device=device, dtype=dtype)

    # One setup pass so cache2 is a valid stage2 input (not timed).
    with torch.no_grad():
        _invoke_kernel(
            x, w_gu, cache1, topk_w, sorted1, expert1, npost1, False, TOP_K, cfg1, compute_type
        )
        gate, up = cache1.view(-1, 2 * N).chunk(2, dim=-1)
        torch.mul(torch.nn.functional.silu(gate), up, out=cache2)

    def once():
        _invoke_kernel(
            x, w_gu, cache1, topk_w, sorted1, expert1, npost1, False, TOP_K, cfg1, compute_type
        )
        _invoke_kernel(
            cache2, w_d, cache3, topk_w, sorted2, expert2, npost2, True, 1, cfg2, compute_type
        )

    print(f"[profile-moe] mode=kernel_only M={M} TOP_K={TOP_K}")
    print(f"[profile-moe] launcher_hash={launcher_hash()}")
    print(
        f"[profile-moe] cfg1={cfg1.get('BLOCK_SIZE_M')}x{cfg1.get('BLOCK_SIZE_N')}x"
        f"{cfg1.get('BLOCK_SIZE_K')} s={cfg1.get('num_stages')} w={cfg1.get('num_warps')}"
    )
    host_ms = _run_timed(
        f"kernel_only_m{M}",
        once,
        warmup=args.warmup,
        iters=args.iters,
        device=device,
    )
    summary = {
        "M": M,
        "mode": "kernel_only",
        "host_ms_per_iter": host_ms,
        "iters": int(args.iters),
        "warmup": int(args.warmup),
        "cfg1": {
            "BLOCK_SIZE_M": cfg1.get("BLOCK_SIZE_M"),
            "BLOCK_SIZE_N": cfg1.get("BLOCK_SIZE_N"),
            "BLOCK_SIZE_K": cfg1.get("BLOCK_SIZE_K"),
            "num_stages": cfg1.get("num_stages"),
            "num_warps": cfg1.get("num_warps"),
        },
        "cfg2": {
            "BLOCK_SIZE_M": cfg2.get("BLOCK_SIZE_M"),
            "BLOCK_SIZE_N": cfg2.get("BLOCK_SIZE_N"),
            "BLOCK_SIZE_K": cfg2.get("BLOCK_SIZE_K"),
            "num_stages": cfg2.get("num_stages"),
            "num_warps": cfg2.get("num_warps"),
        },
        "launcher_hash": launcher_hash(),
    }
    if args.out_dir:
        out_path = Path(args.out_dir) / f"kernel_only_m{M}_summary.json"
        out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(f"[profile-moe] wrote {out_path}", flush=True)
    return 0


def _set_device_stream_capturing(on: bool) -> None:
    """Bridge INFINI_DEVICE_STREAM_CAPTURING for capture-path align (no InfiniCore TLS)."""
    import ctypes

    libc = ctypes.CDLL(None)
    if on:
        os.environ["INFINI_DEVICE_STREAM_CAPTURING"] = "1"
        libc.setenv(b"INFINI_DEVICE_STREAM_CAPTURING", b"1", 1)
    else:
        os.environ.pop("INFINI_DEVICE_STREAM_CAPTURING", None)
        try:
            libc.unsetenv(b"INFINI_DEVICE_STREAM_CAPTURING")
        except Exception:  # noqa: BLE001
            libc.setenv(b"INFINI_DEVICE_STREAM_CAPTURING", b"0", 1)


def _run_capture_replay(args) -> int:
    """Capture InductorMoe (fused_moe_routed body) via InfiniCore GraphExec; time replay.

    Product-faithful path: ``start_graph_recording`` → ``inductor_moe_`` →
    ``stop_graph_recording`` (BeginCapture/Instantiate inside) → ``graph.run()``×N.

    Record-time stubs via ``INFINI_MOE_STUB_{ALIGN,SILU,SUM,KERNEL}``.
    ``torch.cuda.CUDAGraph`` around raw ``fused_moe_routed`` hangs on MetaX;
    InfiniCore hcGraph is the supported capture path.
    """
    import json

    import torch
    from infinilm.kernels import fused_moe_runtime as fmr
    from infinilm.kernels.fused_moe_runtime import launcher_hash

    _refuse_vllm()
    os.environ["INFINI_MOE_PROFILE_PHASES"] = "0"
    _set_device_stream_capturing(False)
    fmr.clear_moe_stub_align_cache()

    M = int(args.M)
    if M not in LAUNCHER_M_ALLOWED:
        raise ValueError(f"capture_replay expects M in {LAUNCHER_M_ALLOWED}, got {M}")

    stubs = {
        "ALIGN": fmr._moe_stub_enabled("ALIGN"),
        "SILU": fmr._moe_stub_enabled("SILU"),
        "SUM": fmr._moe_stub_enabled("SUM"),
        "KERNEL": fmr._moe_stub_enabled("KERNEL"),
    }

    # Product MoE-in-graph env (match serve Decode / METAX UNSAFE).
    os.environ.setdefault("INFINI_CUDAGRAPH_POLICY", "full_and_piecewise")
    os.environ.setdefault("INFINI_MOE_INGRAPH", "both")
    os.environ.setdefault("INFINI_MOE_METAX_INGRAPH_UNSAFE", "1")
    os.environ.setdefault("INFINI_GRAPH_STRICT_REPLAY", "1")
    os.environ["INFINI_PIECEWISE_VALID_LEN"] = str(M)
    os.environ["INFINI_PIECEWISE_INDUCTOR_SEGMENT"] = "1"
    os.environ.pop("INFINI_MOE_FORCE_HOST_BREAK", None)
    os.environ.pop("INFINI_MOE_CAPTURE_SAFE", None)

    segment = (args.segment_pt2 or "").strip()
    if not segment:
        cache_root = os.environ.get(
            "INFINI_PIECEWISE_CACHE",
            str(_cache_dir().parent),
        )
        # Prefer moe_B{M}/segment.pt2 under deploy cache.
        for cand in (
            Path(cache_root) / "tp1" / "rank0" / f"moe_B{M}" / "segment.pt2",
            Path(os.environ.get("INFINI_MOE_CONFIGS", "")).resolve().parent
            / "tp1"
            / "rank0"
            / f"moe_B{M}"
            / "segment.pt2",
        ):
            if cand.is_file():
                segment = str(cand)
                break
    if not segment or not Path(segment).is_file():
        raise RuntimeError(
            f"capture_replay needs moe_B{M}/segment.pt2 (--segment-pt2 or deploy cache)"
        )

    import infinicore
    from infinicore.lib import _infinicore as _ic
    from infinilm.compile.piecewise_moe_segment import make_moe_example_inputs
    from infinilm.compile.piecewise_segments import LAYER_AGNOSTIC_IDX, SEGMENT_MOE
    from infinilm.torch_llama.moe_ops import register_fused_moe_routed_op

    register_fused_moe_routed_op()
    device_index = 0
    device = infinicore.device("cuda", device_index)
    infinicore.set_device(device)
    cuda_dev = f"cuda:{device_index}"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    layer_idx = 0
    bucket = int(M)

    _ic.register_piecewise_inductor_package(
        SEGMENT_MOE,
        LAYER_AGNOSTIC_IDX,
        bucket,
        str(Path(segment).resolve()),
        0,
        True,
    )
    if hasattr(_ic, "set_piecewise_inductor_lookup_tp_rank"):
        _ic.set_piecewise_inductor_lookup_tp_rank(0)

    examples = make_moe_example_inputs(
        bucket=bucket,
        hidden_size=H,
        moe_intermediate_size=N,
        n_routed_experts=E,
        device=torch.device(cuda_dev),
        dtype=dtype,
    )
    _hidden_ex, gate_w, bias, w_gu, w_d, shared_gu, shared_d = examples
    _ic.register_moe_external_weights(
        int(layer_idx),
        *[
            infinicore.from_torch(t.contiguous())._underlying
            for t in (gate_w, bias, w_gu, w_d, shared_gu, shared_d)
        ],
    )

    seq = bucket
    hidden_t = torch.randn(1, seq, H, device=cuda_dev, dtype=dtype)
    out_t = torch.empty(1, seq, H, device=cuda_dev, dtype=dtype)
    hidden = infinicore.from_torch(hidden_t)
    out = infinicore.from_torch(out_t)

    def sync():
        torch.cuda.synchronize(device_index)
        infinicore.sync_stream()

    def eager_once():
        _ic.inductor_moe_(
            hidden._underlying,
            out._underlying,
            int(layer_idx),
            int(bucket),
        )

    # Warm stub-align cache outside capture if needed (routed path uses topk from router).
    if stubs["ALIGN"]:
        # Prefill stub cache with a representative topk shape; router may differ
        # per call but stub skips align ops once precomputed for first ids.
        pass

    print(
        f"[profile-moe] mode=capture_replay M={M} TOP_K={TOP_K} stubs={stubs} "
        f"path=inductor_moe_graph",
        flush=True,
    )
    print(f"[profile-moe] launcher_hash={launcher_hash()}")
    print(f"[profile-moe] segment={segment}")
    print(f"[profile-moe] configs={os.environ.get('INFINI_MOE_CONFIGS')}")
    print(f"[profile-moe] triton_cache={_cache_dir()}")

    print("=== WARMUP_BEGIN mode=capture_replay_eager ===", flush=True)
    for _ in range(max(int(args.warmup), 2)):
        eager_once()
    sync()
    print("=== WARMUP_END mode=capture_replay_eager ===", flush=True)

    print("=== CAPTURE_BEGIN mode=capture_replay ===", flush=True)
    if hasattr(_ic, "set_inference_phase"):
        _ic.set_inference_phase("decode")
    try:
        infinicore.start_graph_recording(device)
        _ic.inductor_moe_(
            hidden._underlying,
            out._underlying,
            int(layer_idx),
            int(bucket),
        )
        graph = infinicore.stop_graph_recording()
    finally:
        if hasattr(_ic, "set_inference_phase"):
            try:
                _ic.set_inference_phase("unknown")
            except Exception:  # noqa: BLE001
                pass
    print("=== CAPTURE_END mode=capture_replay ===", flush=True)

    has_exec = bool(graph.has_device_exec())
    seg_count = int(graph.device_segment_count())
    log_msg = graph.device_graph_log() or ""
    print(
        f"[profile-moe] has_device_exec={int(has_exec)} device_segment_count={seg_count}",
        flush=True,
    )
    if log_msg:
        print(f"[profile-moe] device_graph_log={log_msg[:500]}", flush=True)
    if not has_exec:
        raise RuntimeError(
            "capture_replay: has_device_exec=false (MoE host-break or capture failed)"
        )

    def once():
        graph.run()

    host_ms = _run_timed(
        f"capture_replay_m{M}",
        once,
        warmup=args.warmup,
        iters=args.iters,
        device=torch.device(cuda_dev),
    )
    summary = {
        "M": M,
        "mode": "capture_replay",
        "capture_path": "inductor_moe_graph",
        "host_ms_per_iter": host_ms,
        "iters": int(args.iters),
        "warmup": int(args.warmup),
        "stubs": stubs,
        "launcher_hash": launcher_hash(),
        "fast_align": os.environ.get("INFINI_MOE_FAST_ALIGN", ""),
        "has_device_exec": has_exec,
        "device_segment_count": seg_count,
        "segment_pt2": segment,
        "last_replay_used_device": bool(graph.last_replay_used_device()),
        "replay_op_list_fallback": int(graph.replay_op_list_fallback()),
    }
    if args.out_dir:
        out_path = Path(args.out_dir) / f"capture_replay_m{M}_summary.json"
        out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(f"[profile-moe] wrote {out_path}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mode",
        required=True,
        choices=(
            "launcher",
            "align_only",
            "align_tax",
            "aoti_b16",
            "cpp_pad_m1",
            "host_split",
            "kernel_only",
            "capture_replay",
        ),
    )
    ap.add_argument(
        "--M",
        type=int,
        default=1,
        help=f"launcher/align_only token count {LAUNCHER_M_ALLOWED}",
    )
    ap.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16"))
    ap.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    ap.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    ap.add_argument(
        "--out-dir",
        default="",
        help="Optional dir for a small stdout copy / marker (profiling tree)",
    )
    ap.add_argument(
        "--segment-pt2",
        default="",
        help="Path to moe_B16/segment.pt2 (aoti_b16); default under deploy cache",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    try:
        _require_jit_off_or_allow()
        if not os.environ.get("INFINI_MOE_CONFIGS", "").strip():
            raise RuntimeError("INFINI_MOE_CONFIGS unset")
        cache = _cache_dir()
        before = _count_cache_entries(cache)
        jit_on = _jit_allowed()

        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required")
        torch.manual_seed(args.seed)

        if args.out_dir:
            Path(args.out_dir).mkdir(parents=True, exist_ok=True)

        if args.mode == "launcher":
            rc = _run_launcher(args)
        elif args.mode == "align_only":
            rc = _run_align_only(args)
        elif args.mode == "align_tax":
            rc = _run_align_tax(args)
        elif args.mode == "kernel_only":
            rc = _run_kernel_only(args)
        elif args.mode == "capture_replay":
            rc = _run_capture_replay(args)
        elif args.mode == "host_split":
            if not args.segment_pt2:
                # Allow host_split without AOTI if segment missing; warn.
                print(
                    "[profile-moe] WARN: --segment-pt2 unset; aoti/cpp_pad skipped",
                    flush=True,
                )
            rc = _run_host_split(args)
        elif args.mode == "cpp_pad_m1":
            if not args.segment_pt2:
                raise RuntimeError("--segment-pt2 required for cpp_pad_m1 (or set via wrapper)")
            rc = _run_cpp_pad_m1(args)
        else:
            if not args.segment_pt2:
                raise RuntimeError("--segment-pt2 required for aoti_b16 (or set via wrapper)")
            rc = _run_aoti_b16(args)

        after = _count_cache_entries(cache)
        print(
            f"[profile-moe] cache_files before={before} after={after} jit={int(jit_on)}",
            flush=True,
        )
        if (
            after > before
            and not jit_on
            and args.mode not in ("align_only", "align_tax", "host_split")
        ):
            raise RuntimeError(
                f"Triton cache grew during JIT-off profile ({before} → {after}); "
                "cubins incomplete for this shape/TOP_K"
            )
        return rc
    except Exception as exc:  # noqa: BLE001
        print(f"[profile-moe] FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
