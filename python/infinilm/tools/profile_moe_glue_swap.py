#!/usr/bin/env python3
# Copyright (c) 2025, InfiniCore
"""Glue-swap microbench: InfiniLM Triton + optional vLLM align/silu/sum (container only).

Not a product path — imports vLLM custom ops for attribution. Cells:
  base, +align, +silu, +sum, +all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


H, E, N, TOP_K = 2048, 160, 512, 16
DEFAULT_WARMUP = 5
DEFAULT_ITERS = 20
CELLS = ("base", "align", "silu", "sum", "all")


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
    print(f"[glue-swap] {label} host_ms/iter={ms_per_iter:.3f}", flush=True)
    return ms_per_iter


def _make_inputs(M: int, device, dtype):
    import torch

    x = torch.randn(M, H, device=device, dtype=dtype)
    topk_ids = torch.randint(0, E, (M, TOP_K), device=device, dtype=torch.int32)
    topk_w = torch.softmax(
        torch.randn(M, TOP_K, device=device, dtype=torch.float32), dim=-1
    ).to(dtype)
    w_gu = torch.randn(E, 2 * N, H, device=device, dtype=dtype).contiguous()
    w_d = torch.randn(E, H, N, device=device, dtype=dtype).contiguous()
    return x, topk_w, topk_ids, w_gu, w_d


def _build_once(
    *,
    swap_align: bool,
    swap_silu: bool,
    swap_sum: bool,
    x,
    topk_w,
    topk_ids,
    w_gu,
    w_d,
    cfg1: Dict[str, Any],
    cfg2: Dict[str, Any],
    compute_type,
):
    """Return a no-arg callable that runs one fused MoE with selected glue swaps."""
    import torch
    import torch.nn.functional as F
    from infinilm.kernels.fused_moe_runtime import (
        _invoke_kernel,
        moe_align_block_size as ilm_align,
        moe_sum as ilm_sum,
    )
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size as vllm_align,
    )

    M = x.size(0)
    cache1 = torch.empty(M, TOP_K, 2 * N, device=x.device, dtype=x.dtype)
    cache2 = torch.empty(M * TOP_K, N, device=x.device, dtype=x.dtype)
    cache3 = torch.empty(M, TOP_K, H, device=x.device, dtype=x.dtype)
    out = torch.empty(M, H, device=x.device, dtype=x.dtype)

    def align(ids, bs):
        if swap_align:
            return vllm_align(ids, bs, E)
        return ilm_align(ids, bs, E)

    def silu_fn():
        if swap_silu:
            torch.ops._C.silu_and_mul(cache2, cache1.view(-1, 2 * N))
        else:
            gate, up = cache1.view(-1, 2 * N).chunk(2, dim=-1)
            torch.mul(F.silu(gate), up, out=cache2)

    def sum_fn():
        if swap_sum:
            ops.moe_sum(cache3.view(M, TOP_K, H), out)
        else:
            ilm_sum(cache3, out)

    def once():
        s1, e1, n1 = align(topk_ids, int(cfg1["BLOCK_SIZE_M"]))
        _invoke_kernel(
            x, w_gu, cache1, topk_w, s1, e1, n1, False, TOP_K, cfg1, compute_type
        )
        silu_fn()
        if int(cfg2["BLOCK_SIZE_M"]) == int(cfg1["BLOCK_SIZE_M"]):
            s2, e2, n2 = s1, e1, n1
        else:
            s2, e2, n2 = align(topk_ids, int(cfg2["BLOCK_SIZE_M"]))
        _invoke_kernel(
            cache2, w_d, cache3, topk_w, s2, e2, n2, True, 1, cfg2, compute_type
        )
        sum_fn()
        return out

    return once


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--M", type=int, default=2048)
    ap.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16"))
    ap.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    ap.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--cells",
        default="base,align,silu,sum,all",
        help="comma-separated subset of base,align,silu,sum,all",
    )
    args = ap.parse_args()

    if not os.environ.get("INFINI_MOE_CONFIGS", "").strip():
        print("[glue-swap] FAIL: INFINI_MOE_CONFIGS unset", file=sys.stderr)
        return 1

    # Import vLLM before InfiniLM runtime so custom ops register; we never call
    # fused_moe_routed (which asserts no vllm).
    try:
        import torch
        import triton.language as tl
        import vllm  # noqa: F401
        from vllm import _custom_ops as ops  # noqa: F401

        from infinilm.kernels.fused_moe_runtime import get_moe_config_for_m
    except Exception as exc:  # noqa: BLE001
        print(f"[glue-swap] FAIL: {exc}", file=sys.stderr)
        return 1

    if not torch.cuda.is_available():
        print("[glue-swap] FAIL: CUDA required", file=sys.stderr)
        return 1

    device = torch.device("cuda", 0)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    compute_type = tl.bfloat16 if dtype == torch.bfloat16 else tl.float16
    torch.manual_seed(args.seed)
    M = int(args.M)

    cfg1 = get_moe_config_for_m(M, E=E, N=N, H=H, stage="stage1")
    cfg2 = get_moe_config_for_m(M, E=E, N=N, H=H, stage="stage2")
    x, topk_w, topk_ids, w_gu, w_d = _make_inputs(M, device, dtype)

    cell_flags = {
        "base": (False, False, False),
        "align": (True, False, False),
        "silu": (False, True, False),
        "sum": (False, False, True),
        "all": (True, True, True),
    }
    wanted = [c.strip() for c in str(args.cells).split(",") if c.strip()]
    for c in wanted:
        if c not in cell_flags:
            raise ValueError(f"unknown cell {c}; expected one of {CELLS}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = {
        "M": M,
        "cfg1": {
            "BLOCK_SIZE_M": cfg1.get("BLOCK_SIZE_M"),
            "BLOCK_SIZE_N": cfg1.get("BLOCK_SIZE_N"),
            "BLOCK_SIZE_K": cfg1.get("BLOCK_SIZE_K"),
            "num_stages": cfg1.get("num_stages"),
            "num_warps": cfg1.get("num_warps"),
            "pipeline": cfg1.get("pipeline"),
        },
        "cells": {},
    }

    for cell in wanted:
        sa, ss, sm = cell_flags[cell]
        label = f"glue_{cell}_m{M}"
        print(
            f"[glue-swap] cell={cell} swap_align={sa} swap_silu={ss} swap_sum={sm}",
            flush=True,
        )
        once = _build_once(
            swap_align=sa,
            swap_silu=ss,
            swap_sum=sm,
            x=x,
            topk_w=topk_w,
            topk_ids=topk_ids,
            w_gu=w_gu,
            w_d=w_d,
            cfg1=cfg1,
            cfg2=cfg2,
            compute_type=compute_type,
        )
        host_ms = _run_timed(
            label, once, warmup=args.warmup, iters=args.iters, device=device
        )
        results["cells"][cell] = {
            "host_ms_per_iter": host_ms,
            "swap_align": sa,
            "swap_silu": ss,
            "swap_sum": sm,
        }

    base = (results["cells"].get("base") or {}).get("host_ms_per_iter")
    for cell, payload in results["cells"].items():
        if base and cell != "base":
            payload["delta_vs_base_ms"] = payload["host_ms_per_iter"] - base
            payload["ratio_vs_base"] = payload["host_ms_per_iter"] / base

    out_path = out_dir / f"glue_swap_m{M}_summary.json"
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(f"[glue-swap] wrote {out_path}", flush=True)
    print("[glue-swap] SUMMARY " + json.dumps(results["cells"], sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
