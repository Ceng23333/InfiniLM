#!/usr/bin/env python3
# Copyright (c) 2025, InfiniCore
"""Dump vLLM effective MoE Triton config (default / tuned file) @ given M.

No GPU required for the default / file-lookup path. Optional ``--probe-launch``
runs one ``fused_experts`` warmup to confirm launch and log any override.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Optional


H, E, N, TOP_K = 2048, 160, 512, 16
DEFAULT_M = 2048


def _cfg_fields(cfg: Dict[str, Any]) -> Dict[str, Any]:
    keys = (
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_K",
        "GROUP_SIZE_M",
        "num_warps",
        "num_stages",
        "SPLIT_K",
    )
    out: Dict[str, Any] = {}
    for k in keys:
        if k in cfg:
            out[k] = cfg[k]
    return out


def _detect_source(
    *,
    E: int,
    N: int,
    dtype: Optional[str],
    optimal: Dict[str, Any],
    default: Dict[str, Any],
) -> str:
    """Return ``file`` if a tuned JSON map was used, else ``default``."""
    try:
        from vllm.model_executor.layers.fused_moe.fused_moe import get_moe_configs
    except Exception:  # noqa: BLE001
        return "default" if optimal == default else "file"

    configs = get_moe_configs(E, N, dtype, 0, 0)
    if configs:
        return "file"
    # Same tiles as get_default_config → stock default path.
    return "default"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--M", type=int, default=DEFAULT_M)
    ap.add_argument("--E", type=int, default=E)
    ap.add_argument("--N", type=int, default=N)
    ap.add_argument("--H", type=int, default=H)
    ap.add_argument("--topk", type=int, default=TOP_K)
    ap.add_argument(
        "--dtype",
        default="bfloat16",
        help="config dtype string passed to vLLM (bf16/fp16 path uses this)",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="output JSON path (e.g. b0/vllm_default_cfg.json)",
    )
    ap.add_argument(
        "--probe-launch",
        action="store_true",
        help="optional fused_experts warmup on GPU to confirm launch",
    )
    args = ap.parse_args()

    try:
        from vllm.model_executor.layers.fused_moe.fused_moe import (
            get_default_config,
            try_get_optimal_moe_config,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[dump-vllm-moe-cfg] FAIL: vLLM unavailable: {exc}", file=sys.stderr)
        return 1

    M = int(args.M)
    dtype = str(args.dtype) if args.dtype else None
    # vLLM config lookup uses dtype=None for unquant bf16 in some paths;
    # dump both the explicit dtype and the None-optimal used at runtime.
    default_cfg = get_default_config(
        M, int(args.E), int(args.N), int(args.H), int(args.topk), dtype
    )
    w1_shape = (int(args.E), 2 * int(args.N), int(args.H))
    w2_shape = (int(args.E), int(args.H), int(args.N))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        optimal_cfg = try_get_optimal_moe_config(
            w1_shape, w2_shape, int(args.topk), dtype, M
        )

    source = _detect_source(
        E=int(args.E),
        N=int(args.N),
        dtype=dtype,
        optimal=optimal_cfg,
        default=default_cfg,
    )
    warn_msgs = [str(w.message) for w in caught]

    payload: Dict[str, Any] = {
        "M": M,
        "E": int(args.E),
        "N": int(args.N),
        "H": int(args.H),
        "topk": int(args.topk),
        "dtype": dtype,
        "source": source,
        "BLOCK_SIZE_M": optimal_cfg.get("BLOCK_SIZE_M"),
        "BLOCK_SIZE_N": optimal_cfg.get("BLOCK_SIZE_N"),
        "BLOCK_SIZE_K": optimal_cfg.get("BLOCK_SIZE_K"),
        "GROUP_SIZE_M": optimal_cfg.get("GROUP_SIZE_M"),
        "num_warps": optimal_cfg.get("num_warps"),
        "num_stages": optimal_cfg.get("num_stages"),
        "SPLIT_K": optimal_cfg.get("SPLIT_K"),
        "default_config": _cfg_fields(default_cfg),
        "optimal_config": _cfg_fields(optimal_cfg),
        "warnings": warn_msgs,
    }

    if args.probe_launch:
        try:
            import torch
            from vllm.model_executor.layers.fused_moe import fused_experts
        except Exception as exc:  # noqa: BLE001
            print(
                f"[dump-vllm-moe-cfg] FAIL probe: fused_experts unavailable: {exc}",
                file=sys.stderr,
            )
            return 1
        if not torch.cuda.is_available():
            print("[dump-vllm-moe-cfg] FAIL probe: CUDA required", file=sys.stderr)
            return 1
        device = torch.device("cuda", 0)
        torch_dtype = torch.bfloat16 if dtype in (None, "bfloat16") else torch.float16
        x = torch.randn(M, int(args.H), device=device, dtype=torch_dtype)
        topk_ids = torch.randint(
            0, int(args.E), (M, int(args.topk)), device=device, dtype=torch.int32
        )
        topk_w = torch.softmax(
            torch.randn(M, int(args.topk), device=device, dtype=torch.float32), dim=-1
        ).to(torch_dtype)
        w_gate_up = torch.randn(
            int(args.E), 2 * int(args.N), int(args.H), device=device, dtype=torch_dtype
        ).contiguous()
        w_down = torch.randn(
            int(args.E), int(args.H), int(args.N), device=device, dtype=torch_dtype
        ).contiguous()
        with torch.no_grad():
            fused_experts(x, w_gate_up, w_down, topk_w, topk_ids, inplace=False)
            torch.cuda.synchronize()
        payload["probe_launch"] = {"ok": True, "device": str(device)}
        print("[dump-vllm-moe-cfg] probe-launch OK", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    print(
        json.dumps(
            {
                "out": str(out_path),
                "source": source,
                "tiles": (
                    payload["BLOCK_SIZE_M"],
                    payload["BLOCK_SIZE_N"],
                    payload["BLOCK_SIZE_K"],
                ),
                "num_stages": payload["num_stages"],
                "num_warps": payload["num_warps"],
                "GROUP_SIZE_M": payload["GROUP_SIZE_M"],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
