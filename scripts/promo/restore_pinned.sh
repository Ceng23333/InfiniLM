#!/usr/bin/env bash
# Restore the campaign host to the pinned runtime after candidate proof.
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
default_itw_root="$(cd "${root}/.." && pwd)/InfiniTensorWorktree"
itw_root="${INFINI_TENSOR_WORKTREE:-${default_itw_root}}"
pin_lib="${RUNTIME_PIN_LIB:-${itw_root}/artifacts/runtime_pins/libinfinicore_runtime.so}"
[[ -f "${pin_lib}" ]] || { echo "[restore-pinned] missing pinned runtime: ${pin_lib}" >&2; exit 64; }

export CANDIDATE_WORKSPACE="${itw_root}/InfiniLM"
export RUNTIME_PIN_LIB="${pin_lib}"
export RUNTIME_PIN_SHA256="$(sha256sum "${pin_lib}" | awk '{print $1}')"
export NUM_BLOCKS="${NUM_BLOCKS:-2304}"
export BLOCK_SIZE="${BLOCK_SIZE:-256}"

launcher="${itw_root}/deployment/infinilm-prod/start.sh"
[[ -x "${launcher}" ]] || { echo "[restore-pinned] missing launcher: ${launcher}" >&2; exit 64; }
"${launcher}"

container_name="${CONTAINER_NAME:-qwen3-8b-mars-inf-tp4}"
actual_hash="$(docker exec "${container_name}" \
  sha256sum /workspace/InfiniLM/python/infinilm/lib/libinfinicore_runtime.so | awk '{print $1}')"
[[ "${actual_hash}" == "${RUNTIME_PIN_SHA256}" ]] || {
  echo "[restore-pinned] runtime hash mismatch: ${actual_hash} != ${RUNTIME_PIN_SHA256}" >&2
  exit 65
}

curl --noproxy '*' -fsS http://127.0.0.1:8000/health >/dev/null
echo "[restore-pinned] runtime=${RUNTIME_PIN_SHA256} health=ok"
