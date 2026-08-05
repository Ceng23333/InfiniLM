#!/usr/bin/env python3
# Copyright (c) 2025, InfiniCore
"""vLLM Triton GEMM-only microbench (container attribution; not a product path).

Precomputes align + silu outside the timed loop; times two
``invoke_fused_moe_triton_kernel`` launches (stage1 + stage2) at stock default tiles.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict


H, E, N, TOP_K = 2048, 160, 512, 16
DEFAULT_WARMUP = 5
DEFAULT_ITERS = 20
ALLOWED_M = (1, 16, 2048)


def _default_cfg(M: int) -> Dict[str, Any]:
    from vllm.model_executor.layers.fused_moe.fused_moe import get_default_config

    return dict(get_default_config(M, E, N, H, TOP_K, "bfloat16"))


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
    print(f"[profile-vllm-kernel-only] {label} host_ms/iter={ms_per_iter:.3f}", flush=True)
    return ms_per_iter


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--M", type=int, required=True)
    ap.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16"))
    ap.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    ap.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()

    M = int(args.M)
    if M not in ALLOWED_M:
        raise ValueError(f"expected M in {set(ALLOWED_M)}, got {M}")

    try:
        import torch
        import triton.language as tl
        from vllm import _custom_ops as ops  # noqa: F401 — ensure custom ops loaded
        from vllm.model_executor.layers.fused_moe.fused_moe import (
            invoke_fused_moe_triton_kernel,
        )
        from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
            moe_align_block_size,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[profile-vllm-kernel-only] FAIL: {exc}", file=sys.stderr)
        return 1

    if not torch.cuda.is_available():
        print("[profile-vllm-kernel-only] FAIL: CUDA required", file=sys.stderr)
        return 1

    device = torch.device("cuda", 0)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    compute_type = tl.bfloat16 if dtype == torch.bfloat16 else tl.float16
    torch.manual_seed(args.seed)

    cfg = _default_cfg(M)
    print(f"[profile-vllm-kernel-only] M={M} cfg={cfg}", flush=True)

    # Match vLLM fused_experts: w1 [E, 2N, H], w2 [E, H, N]
    x = torch.randn(M, H, device=device, dtype=dtype)
    topk_ids = torch.randint(0, E, (M, TOP_K), device=device, dtype=torch.int32)
    topk_w = torch.softmax(
        torch.randn(M, TOP_K, device=device, dtype=torch.float32), dim=-1
    ).to(dtype)
    w_gu = torch.randn(E, 2 * N, H, device=device, dtype=dtype).contiguous()
    w_d = torch.randn(E, H, N, device=device, dtype=dtype).contiguous()

    sorted_ids, expert_ids, npost = moe_align_block_size(
        topk_ids, int(cfg["BLOCK_SIZE_M"]), E
    )

    # vLLM: cache1 [M,topk,2N], cache2 [M*topk,N], cache3 [M,topk,H]
    cache1 = torch.empty(M, TOP_K, 2 * N, device=device, dtype=dtype)
    cache2 = torch.empty(M * TOP_K, N, device=device, dtype=dtype)
    cache3 = torch.empty(M, TOP_K, H, device=device, dtype=dtype)

    inv_kwargs = dict(
        A_scale=None,
        B_scale=None,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=False,
        block_shape=None,
        B_bias=None,
    )

    def invoke(A, B, C, mul_w, top_k):
        invoke_fused_moe_triton_kernel(
            A,
            B,
            C,
            topk_weights=topk_w,
            sorted_token_ids=sorted_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=npost,
            mul_routed_weight=mul_w,
            top_k=top_k,
            config=cfg,
            compute_type=compute_type,
            **inv_kwargs,
        )

    with torch.no_grad():
        invoke(x, w_gu, cache1, False, TOP_K)
        torch.ops._C.silu_and_mul(cache2, cache1.view(-1, 2 * N))

    def once():
        invoke(x, w_gu, cache1, False, TOP_K)
        invoke(cache2, w_d, cache3, True, 1)

    host_ms = _run_timed(
        f"vllm_kernel_only_m{M}",
        once,
        warmup=args.warmup,
        iters=args.iters,
        device=device,
    )
    summary = {
        "M": M,
        "mode": "vllm_kernel_only",
        "host_ms_per_iter": host_ms,
        "iters": int(args.iters),
        "warmup": int(args.warmup),
        "cfg": cfg,
    }
    out_dir = (args.out_dir or "").strip()
    if out_dir:
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        out_path = path / f"vllm_kernel_only_m{M}_summary.json"
        out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(f"[profile-vllm-kernel-only] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
