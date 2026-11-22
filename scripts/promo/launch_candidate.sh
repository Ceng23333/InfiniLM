#!/usr/bin/env bash
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
candidate_workspace="$(cd "${root}" && pwd)"
hash_file="${candidate_workspace}/artifacts/last_build_hashes.txt"
[[ -s "${hash_file}" ]] || {
  echo "[launch-candidate] missing build proof: ${hash_file}" >&2
  echo "[launch-candidate] run scripts/promo/build_candidate.sh first" >&2
  exit 64
}
itw_root="${INFINI_TENSOR_WORKTREE:-$(cd "${candidate_workspace}/.." && pwd)}"
launcher="${itw_root}/deployment/infinilm-prod/start.sh"

mkdir -p python/infinicore/lib python/infinilm/lib
runtime_hash="$(awk '$2 ~ /libinfinicore_runtime.so$/ { print $1; exit }' "${hash_file}")"
expected_runtime="$(sha256sum "${candidate_workspace}/build/linux/arm64/release/libinfinicore_runtime.so" | awk '{print $1}')"
[[ "${runtime_hash}" == "${expected_runtime}" ]] || {
  echo "[launch-candidate] runtime hash does not match build proof" >&2
  exit 64
}
cp "${candidate_workspace}/build/linux/arm64/release/libinfinicore_runtime.so" \
  python/infinicore/lib/libinfinicore_runtime.so
cp "${candidate_workspace}/build/linux/arm64/release/libinfinicore_runtime.so" \
  python/infinilm/lib/libinfinicore_runtime.so
cp "${candidate_workspace}/build/linux/arm64/release/_infinicore.cpython-310-aarch64-linux-gnu.so" \
  python/infinicore/lib/_infinicore.cpython-310-aarch64-linux-gnu.so
cp "${candidate_workspace}/build/linux/arm64/release/_infinilm.cpython-310-aarch64-linux-gnu.so" \
  python/infinilm/lib/_infinilm.cpython-310-aarch64-linux-gnu.so

[[ -f "${launcher}" ]] || {
  echo "[launch-candidate] missing launcher: ${launcher}" >&2
  exit 1
}

export CANDIDATE_WORKSPACE="${candidate_workspace}"
export MODEL_DIR="${MODEL_DIR:-/data-aisoft/mechdancer/models/Qwen3-8B}"
export NUM_BLOCKS="${NUM_BLOCKS:-2048}"
export ENABLE_GRAPH="${ENABLE_GRAPH:-1}"
export INFINI_MAX_NUM_BATCHED_TOKENS="${INFINI_MAX_NUM_BATCHED_TOKENS:-2048}"
export INFINI_V1_SCHEDULER="${INFINI_V1_SCHEDULER:-1}"
export RUNTIME_PIN_LIB="${candidate_workspace}/build/linux/arm64/release/libinfinicore_runtime.so"
export RUNTIME_PIN_SHA256="$(awk '$2 ~ /libinfinicore_runtime.so$/ { print $1; exit }' "${hash_file}")"

"${launcher}"
mkdir -p artifacts
curl --noproxy '*' -fsS http://127.0.0.1:8000/health > artifacts/candidate_health.json
echo "[launch-candidate] candidate source workspace: ${CANDIDATE_WORKSPACE}"
