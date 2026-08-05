#!/usr/bin/env python3
# Copyright (c) 2025, InfiniCore
"""vLLM fused_experts microbench for hcTracer baseline (recompile container only).

Not used on InfiniLM serve path. Compare host ms/iter vs InfiniLM launcher modes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

H, E, N, TOP_K = 2048, 160, 512, 16
DEFAULT_WARMUP = 5
DEFAULT_ITERS = 20
ALLOWED_M = (1, 16, 2048)


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
    print(f"[profile-vllm-moe] {label} host_ms/iter={ms_per_iter:.3f}", flush=True)
    return ms_per_iter


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--M", type=int, required=True, help="token count (1, 16, or 2048)")
    ap.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16"))
    ap.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    ap.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out-dir",
        default="",
        help="optional dir for vllm_m${M}_summary.json",
    )
    args = ap.parse_args()

    M = int(args.M)
    if M not in ALLOWED_M:
        raise ValueError(f"expected M in {set(ALLOWED_M)}, got {M}")

    try:
        import torch
        from vllm.model_executor.layers.fused_moe import fused_experts
    except Exception as exc:  # noqa: BLE001
        print(f"[profile-vllm-moe] FAIL: vLLM fused_experts unavailable: {exc}", file=sys.stderr)
        return 1

    if not torch.cuda.is_available():
        print("[profile-vllm-moe] FAIL: CUDA required", file=sys.stderr)
        return 1

    device = torch.device("cuda", 0)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    torch.manual_seed(args.seed)

    print(f"[profile-vllm-moe] mode=vllm M={M} TOP_K={TOP_K} H={H} E={E} N={N}")

    x = torch.randn(M, H, device=device, dtype=dtype)
    topk_ids = torch.randint(0, E, (M, TOP_K), device=device, dtype=torch.int32)
    topk_w = torch.softmax(
        torch.randn(M, TOP_K, device=device, dtype=torch.float32), dim=-1
    ).to(dtype)
    w_gate_up = torch.randn(E, 2 * N, H, device=device, dtype=dtype).contiguous()
    w_down = torch.randn(E, H, N, device=device, dtype=dtype).contiguous()

    def once():
        return fused_experts(
            x,
            w_gate_up,
            w_down,
            topk_w,
            topk_ids,
            inplace=False,
        )

    host_ms = _run_timed(
        f"vllm_m{M}",
        once,
        warmup=args.warmup,
        iters=args.iters,
        device=device,
    )

    out_dir = (args.out_dir or "").strip()
    if out_dir:
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        summary = {
            "M": M,
            "host_ms_per_iter": host_ms,
            "iters": int(args.iters),
            "warmup": int(args.warmup),
            "mode": "vllm",
            "H": H,
            "E": E,
            "N": N,
            "TOP_K": TOP_K,
        }
        out_path = path / f"vllm_m{M}_summary.json"
        out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(f"[profile-vllm-moe] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
