#!/usr/bin/env bash
set -euo pipefail

require_source_edit() {
  local changed
  changed="$(
    {
      git diff --name-only HEAD -- csrc python/infinilm xmake.lua
      git ls-files --others --exclude-standard -- csrc python/infinilm xmake.lua
    } | sort -u
  )"
  if [[ -z "${changed}" ]]; then
    cat >&2 <<'EOF'
[build-candidate] no editable source change found.
Edit csrc/, python/infinilm/, or xmake.lua. Deployment-only knobs are rejected.
EOF
    exit 64
  fi
  printf '%s\n' "${changed}" > artifacts/source_changed_files.txt
}

root="$(git rev-parse --show-toplevel)"
cd "${root}"
mkdir -p artifacts
require_source_edit

itw_root="${INFINI_TENSOR_WORKTREE:-$(cd "${root}/../../.." && pwd)}"
campaign_python_root="${CAMPAIGN_PYTHON_ROOT:-${itw_root}/.campaign-python}"
python_include_dir="${campaign_python_root}/include/python3.10"
if [[ ! -f "${python_include_dir}/Python.h" ]]; then
  mkdir -p "${campaign_python_root}/include"
  docker cp qwen3-8b-mars-inf-tp4:/opt/conda/include/python3.10 "${python_include_dir}"
fi
if [[ ! -x "${root}/scripts/promo/python-container" ]]; then
  echo "[build-candidate] missing scripts/promo/python-container" >&2
  exit 1
fi

export XMAKE_ROOT=y
export HPCC_PATH="${HPCC_PATH:-/opt/hpcc}"
export PYBIND11_INCLUDE="${PYBIND11_INCLUDE:-$(python3 - <<'PY'
try:
    import pybind11
    print(pybind11.get_include())
except Exception:
    print("/opt/conda/include/python3.10")
PY
)}"
export PYTHON_INCLUDE_DIR="${PYTHON_INCLUDE_DIR:-${python_include_dir}}"
export PYTHON="${PYTHON:-${root}/scripts/promo/python-container}"

xmake f --root -P "${root}" -y -m release
xmake --root -P "${root}" -y -j32 infinicore_runtime
xmake --root -P "${root}" -y -j32 _infinicore
xmake --root -P "${root}" -y -j32 _infinilm
xmake install --root -P "${root}" -y infinicore_runtime

{
  sha256sum build/linux/arm64/release/libinfinicore_runtime.so
  sha256sum build/linux/arm64/release/_infinicore*.so
  sha256sum build/linux/arm64/release/_infinilm*.so
} > artifacts/artifact_hashes.txt
cp artifacts/artifact_hashes.txt artifacts/last_build_hashes.txt
printf '[build-candidate] rebuilt with source files:\n' >&2
cat artifacts/source_changed_files.txt >&2
printf '[build-candidate] artifact hashes:\n%s\n' "$(cat artifacts/last_build_hashes.txt)" >&2
