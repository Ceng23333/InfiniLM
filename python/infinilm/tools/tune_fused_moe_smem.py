#!/usr/bin/env python3
# Copyright (c) 2025, InfiniCore
"""SMEM-legal InfiniLM fused_moe tile tuner (Mars ~64 KiB).

Searches ``pipeline=basic`` tiles that survive ``_sanitize_moe_config`` unchanged,
times them via ``fused_moe_routed`` @ M=2048, and writes winners into large-M
JSON keys (1024…4096) under ``INFINI_MOE_CONFIGS``. Does not rewrite M≤512.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


H, E, N, TOP_K = 2048, 160, 512, 16
DEFAULT_M = 2048
# Large buckets that currently share oversized 128³/cpasync tiles.
LARGE_M_KEYS = (1024, 1536, 2048, 3072, 4096)
SMEM_LIMIT_DEFAULT = 65536


def _smem_est(bm: int, bn: int, bk: int, stages: int) -> int:
    return (bm * bk + bk * bn) * 2 * max(stages, 1)


def _cfg_key(cfg: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        int(cfg["BLOCK_SIZE_M"]),
        int(cfg["BLOCK_SIZE_N"]),
        int(cfg["BLOCK_SIZE_K"]),
        int(cfg.get("GROUP_SIZE_M", 1)),
        int(cfg.get("num_warps", 4)),
        int(cfg.get("num_stages", 2)),
        str(cfg.get("pipeline", "basic") or "basic"),
    )


def _make_candidate(
    bm: int, bn: int, bk: int, stages: int, warps: int
) -> Dict[str, Any]:
    return {
        "BLOCK_SIZE_M": bm,
        "BLOCK_SIZE_N": bn,
        "BLOCK_SIZE_K": bk,
        "GROUP_SIZE_M": 1,
        "num_warps": warps,
        "num_stages": stages,
        "pipeline": "basic",
        "scenario": "",
        "ACCF32": False,
        "SPLIT_K": 1,
    }


def _iter_candidates(smem_limit: int) -> List[Dict[str, Any]]:
    from infinilm.kernels.fused_moe_runtime import _sanitize_moe_config

    out: List[Dict[str, Any]] = []
    seen = set()
    for bm, bn, bk, stages, warps in product(
        (32, 64), (32, 64), (32, 64), (2, 3), (4, 8)
    ):
        if _smem_est(bm, bn, bk, stages) > smem_limit:
            continue
        cand = _make_candidate(bm, bn, bk, stages, warps)
        san = _sanitize_moe_config(cand)
        # Exclude anything sanitize would still rewrite.
        if _cfg_key(san) != _cfg_key(cand):
            continue
        key = _cfg_key(cand)
        if key in seen:
            continue
        seen.add(key)
        out.append(cand)
    return out


def _stage_entry_from_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "BLOCK_SIZE_M": int(cfg["BLOCK_SIZE_M"]),
        "BLOCK_SIZE_N": int(cfg["BLOCK_SIZE_N"]),
        "BLOCK_SIZE_K": int(cfg["BLOCK_SIZE_K"]),
        "GROUP_SIZE_M": int(cfg.get("GROUP_SIZE_M", 1)),
        "num_warps": int(cfg.get("num_warps", 4)),
        "num_stages": int(cfg.get("num_stages", 2)),
        "pipeline": str(cfg.get("pipeline", "basic") or "basic"),
        "scenario": str(cfg.get("scenario", "") or ""),
        "ACCF32": bool(cfg.get("ACCF32", False)),
        "SPLIT_K": int(cfg.get("SPLIT_K", 1)),
    }


def _config_json_paths(configs_dir: Path) -> List[Path]:
    """Deploy-cache mirrors used by lookup (X203 / Mars_03 / nested H=2048)."""
    names = [
        f"H={H},E={E},N={N},device_name=X203.json",
        f"H={H},E={E},N={N},device_name=Mars_03.json",
    ]
    paths: List[Path] = []
    for name in names:
        for p in (configs_dir / name, configs_dir / f"H={H}" / name):
            if p not in paths:
                paths.append(p)
    return paths


def _load_primary_table(configs_dir: Path) -> Tuple[Dict[str, Any], Path]:
    for path in _config_json_paths(configs_dir):
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8")), path
    raise FileNotFoundError(
        f"No MoE config JSON under {configs_dir} matching H={H},E={E},N={N}"
    )


def _write_large_m_winners(
    configs_dir: Path,
    stage1: Dict[str, Any],
    stage2: Dict[str, Any],
    write_keys: Tuple[int, ...],
    *,
    dry_run: bool = False,
) -> List[str]:
    written: List[str] = []
    s1 = _stage_entry_from_cfg(stage1)
    s2 = _stage_entry_from_cfg(stage2)
    for path in _config_json_paths(configs_dir):
        if not path.is_file():
            continue
        table = json.loads(path.read_text(encoding="utf-8"))
        for m in write_keys:
            key = str(m)
            if key not in table:
                table[key] = {}
            entry = dict(table[key]) if isinstance(table[key], dict) else {}
            entry["stage1"] = deepcopy(s1)
            entry["stage2"] = deepcopy(s2)
            table[key] = entry
        if not dry_run:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(table, indent=2, sort_keys=False) + "\n",
                encoding="utf-8",
            )
        written.append(str(path))
    return written


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


def _time_routed(
    cfg1: Dict[str, Any],
    cfg2: Dict[str, Any],
    *,
    M: int,
    warmup: int,
    iters: int,
    device,
    dtype,
    inputs=None,
) -> float:
    import torch
    import infinilm.kernels.fused_moe_runtime as moe

    x, topk_w, topk_ids, w_gu, w_d = inputs or _make_inputs(M, device, dtype)

    def _override(M_tok: int, *, E: int, N: int, H: int, stage: str = "stage1"):
        del M_tok, E, N, H
        return dict(cfg1 if stage == "stage1" else cfg2)

    orig = moe.get_moe_config_for_m
    moe.get_moe_config_for_m = _override  # type: ignore[assignment]
    try:
        with torch.no_grad():
            for _ in range(warmup):
                moe.fused_moe_routed(x, topk_w, topk_ids, w_gu, w_d)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                moe.fused_moe_routed(x, topk_w, topk_ids, w_gu, w_d)
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return elapsed_ms / max(iters, 1)
    finally:
        moe.get_moe_config_for_m = orig


def _tune_stage(
    stage: str,
    candidates: List[Dict[str, Any]],
    fixed_other: Dict[str, Any],
    *,
    M: int,
    warmup: int,
    iters: int,
    device,
    dtype,
    inputs,
) -> Tuple[Dict[str, Any], float, List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    best_cfg: Optional[Dict[str, Any]] = None
    best_ms = float("inf")
    for i, cand in enumerate(candidates):
        cfg1 = cand if stage == "stage1" else fixed_other
        cfg2 = cand if stage == "stage2" else fixed_other
        try:
            ms = _time_routed(
                cfg1,
                cfg2,
                M=M,
                warmup=warmup,
                iters=iters,
                device=device,
                dtype=dtype,
                inputs=inputs,
            )
            err = None
        except Exception as exc:  # noqa: BLE001
            ms = None
            err = f"{type(exc).__name__}: {exc}"
        row = {
            "stage": stage,
            "idx": i,
            "cfg": _stage_entry_from_cfg(cand),
            "host_ms_per_iter": ms,
            "error": err,
        }
        rows.append(row)
        tag = (
            f"BM={cand['BLOCK_SIZE_M']} BN={cand['BLOCK_SIZE_N']} "
            f"BK={cand['BLOCK_SIZE_K']} stages={cand['num_stages']} "
            f"warps={cand['num_warps']}"
        )
        if ms is None:
            print(f"[tune-moe] {stage} [{i+1}/{len(candidates)}] {tag} FAIL {err}", flush=True)
            continue
        print(
            f"[tune-moe] {stage} [{i+1}/{len(candidates)}] {tag} "
            f"host_ms={ms:.3f}",
            flush=True,
        )
        if ms < best_ms:
            best_ms = ms
            best_cfg = dict(cand)
    if best_cfg is None:
        raise RuntimeError(f"no successful {stage} candidate")
    return best_cfg, best_ms, rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--M", type=int, default=DEFAULT_M)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16"))
    ap.add_argument(
        "--summary-out",
        default="",
        help="Path for TUNE_SUMMARY.json (default: <configs>/TUNE_SUMMARY.json)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Search + summarize without writing JSON winners",
    )
    ap.add_argument(
        "--write-keys",
        default=",".join(str(k) for k in LARGE_M_KEYS),
        help="Comma-separated M keys to overwrite (default large buckets only)",
    )
    args = ap.parse_args()

    os.environ["INFINI_MOE_ALLOW_JIT"] = "1"
    if not os.environ.get("INFINI_MOE_CONFIGS"):
        print("FATAL: set INFINI_MOE_CONFIGS", file=sys.stderr)
        return 2
    if not (
        os.environ.get("INFINI_MOE_TRITON_CACHE") or os.environ.get("TRITON_CACHE_DIR")
    ):
        print("FATAL: set INFINI_MOE_TRITON_CACHE or TRITON_CACHE_DIR", file=sys.stderr)
        return 2

    configs_dir = Path(os.environ["INFINI_MOE_CONFIGS"])
    cache = Path(
        os.environ.get("INFINI_MOE_TRITON_CACHE") or os.environ["TRITON_CACHE_DIR"]
    )
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(cache)

    smem_limit = int(os.environ.get("INFINI_MOE_SMEM_LIMIT", str(SMEM_LIMIT_DEFAULT)))
    write_keys = tuple(int(x) for x in args.write_keys.split(",") if x.strip())

    import torch
    from infinilm.kernels.fused_moe_runtime import (
        _flatten_stage,
        _sanitize_moe_config,
        get_moe_config_for_m,
        launcher_hash,
    )

    device = torch.device("cuda", 0)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    M = int(args.M)

    baseline_s1 = get_moe_config_for_m(M, E=E, N=N, H=H, stage="stage1")
    baseline_s2 = get_moe_config_for_m(M, E=E, N=N, H=H, stage="stage2")
    # Prove sanitize is currently clamping the seeded JSON for large M.
    raw_table, primary_path = _load_primary_table(configs_dir)
    raw_s1 = raw_s2 = None
    raw_entry = raw_table.get(str(M))
    if raw_entry is None:
        le_keys = sorted(
            int(k) for k in raw_table if str(k).isdigit() and int(k) <= M
        )
        if le_keys:
            raw_entry = raw_table.get(str(le_keys[-1]))
    if isinstance(raw_entry, dict):
        try:
            raw_s1 = _flatten_stage(raw_entry, "stage1")
            raw_s2 = _flatten_stage(raw_entry, "stage2")
        except Exception:  # noqa: BLE001
            raw_s1 = raw_s2 = None

    print(f"[tune-moe] launcher_hash={launcher_hash()}", flush=True)
    print(f"[tune-moe] configs={configs_dir} primary={primary_path}", flush=True)
    print(f"[tune-moe] cache={cache} smem_limit={smem_limit}", flush=True)
    print(f"[tune-moe] baseline stage1={baseline_s1}", flush=True)
    print(f"[tune-moe] baseline stage2={baseline_s2}", flush=True)

    candidates = _iter_candidates(smem_limit)
    print(f"[tune-moe] candidates={len(candidates)} (sanitize_noop filtered)", flush=True)
    if not candidates:
        print("FATAL: empty candidate grid", file=sys.stderr)
        return 1

    inputs = _make_inputs(M, device, dtype)
    base_ms = _time_routed(
        baseline_s1,
        baseline_s2,
        M=M,
        warmup=args.warmup,
        iters=args.iters,
        device=device,
        dtype=dtype,
        inputs=inputs,
    )
    print(f"[tune-moe] baseline host_ms/iter={base_ms:.3f}", flush=True)

    # Stage1 search with stage2 held at sanitized baseline; then stage2 with stage1 winner.
    win_s1, ms_s1, rows_s1 = _tune_stage(
        "stage1",
        candidates,
        baseline_s2,
        M=M,
        warmup=args.warmup,
        iters=args.iters,
        device=device,
        dtype=dtype,
        inputs=inputs,
    )
    win_s2, ms_s2, rows_s2 = _tune_stage(
        "stage2",
        candidates,
        win_s1,
        M=M,
        warmup=args.warmup,
        iters=args.iters,
        device=device,
        dtype=dtype,
        inputs=inputs,
    )
    final_ms = _time_routed(
        win_s1,
        win_s2,
        M=M,
        warmup=args.warmup,
        iters=args.iters,
        device=device,
        dtype=dtype,
        inputs=inputs,
    )
    print(f"[tune-moe] winner pair host_ms/iter={final_ms:.3f}", flush=True)

    # Prefer sanitized baseline when search does not improve host ms.
    if final_ms >= base_ms:
        print(
            f"[tune-moe] winner pair {final_ms:.3f}ms >= baseline {base_ms:.3f}ms; "
            "keeping baseline sanitized tiles",
            flush=True,
        )
        win_s1, win_s2 = dict(baseline_s1), dict(baseline_s2)
        final_ms = base_ms

    # Sanity: winners must be sanitize no-ops.
    san1 = _sanitize_moe_config(win_s1)
    san2 = _sanitize_moe_config(win_s2)
    sanitize_noop = _cfg_key(san1) == _cfg_key(win_s1) and _cfg_key(san2) == _cfg_key(
        win_s2
    )
    if not sanitize_noop:
        print("FATAL: winner fails sanitize_noop", file=sys.stderr)
        return 1

    # Write chosen tiles into large-M keys (baseline or true winner).
    # Callers restore on regression FAIL.
    written = _write_large_m_winners(
        configs_dir, win_s1, win_s2, write_keys, dry_run=args.dry_run
    )
    # Clear LRU so subsequent lookups see the new JSON.
    from infinilm.kernels.fused_moe_runtime import _load_config_table

    _load_config_table.cache_clear()

    # Post-write verify via real get_moe_config_for_m (unless dry-run).
    post_s1 = post_s2 = None
    post_noop = None
    if not args.dry_run:
        post_s1 = get_moe_config_for_m(M, E=E, N=N, H=H, stage="stage1")
        post_s2 = get_moe_config_for_m(M, E=E, N=N, H=H, stage="stage2")
        post_noop = (
            _cfg_key(post_s1) == _cfg_key(_sanitize_moe_config(post_s1))
            and _cfg_key(post_s2) == _cfg_key(_sanitize_moe_config(post_s2))
            and _cfg_key(post_s1) == _cfg_key(win_s1)
            and _cfg_key(post_s2) == _cfg_key(win_s2)
        )

    summary = {
        "M": M,
        "H": H,
        "E": E,
        "N": N,
        "TOP_K": TOP_K,
        "smem_limit": smem_limit,
        "launcher_hash": launcher_hash(),
        "primary_config": str(primary_path),
        "baseline_sanitized": {
            "stage1": _stage_entry_from_cfg(baseline_s1),
            "stage2": _stage_entry_from_cfg(baseline_s2),
            "host_ms_per_iter": base_ms,
        },
        "raw_seeded_before_sanitize": {
            "stage1": raw_s1,
            "stage2": raw_s2,
        },
        "n_candidates": len(candidates),
        "candidates": [_stage_entry_from_cfg(c) for c in candidates],
        "stage1_search": {
            "winner": _stage_entry_from_cfg(win_s1),
            "host_ms_per_iter": ms_s1,
            "rows": rows_s1,
        },
        "stage2_search": {
            "winner": _stage_entry_from_cfg(win_s2),
            "host_ms_per_iter": ms_s2,
            "rows": rows_s2,
        },
        "winner_pair": {
            "stage1": _stage_entry_from_cfg(win_s1),
            "stage2": _stage_entry_from_cfg(win_s2),
            "host_ms_per_iter": final_ms,
            "vs_baseline": (final_ms / base_ms) if base_ms > 0 else None,
        },
        "write_keys": list(write_keys),
        "written_paths": written,
        "dry_run": bool(args.dry_run),
        "sanitize_noop": bool(sanitize_noop),
        "post_write_sanitize_noop": post_noop,
        "post_write_cfg": {
            "stage1": _stage_entry_from_cfg(post_s1) if post_s1 else None,
            "stage2": _stage_entry_from_cfg(post_s2) if post_s2 else None,
        },
        "beats_baseline": final_ms < base_ms,
    }

    summary_path = Path(args.summary_out) if args.summary_out else (
        configs_dir / "TUNE_SUMMARY.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[tune-moe] wrote {summary_path}", flush=True)
    print(
        json.dumps(
            {
                "baseline_ms": base_ms,
                "winner_ms": final_ms,
                "vs_baseline": summary["winner_pair"]["vs_baseline"],
                "sanitize_noop": sanitize_noop,
                "beats_baseline": summary["beats_baseline"],
                "stage1": summary["winner_pair"]["stage1"],
                "stage2": summary["winner_pair"]["stage2"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
