#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
for file in artifacts/source_changed_files.txt artifacts/last_build_hashes.txt; do
  [[ -s "${file}" ]] || { echo "[verify-rebuild] missing ${file}" >&2; exit 1; }
done

changed="$(
  {
    git diff --name-only HEAD -- csrc python/infinilm xmake.lua
    git ls-files --others --exclude-standard -- csrc python/infinilm xmake.lua
  } | sort -u
)"
[[ -n "${changed}" ]] || { echo "[verify-rebuild] source diff disappeared" >&2; exit 1; }

while IFS= read -r line; do
  [[ -n "${line}" ]] || continue
  set -- ${line}
  expected="${1}"
  artifact="${2}"
  [[ -f "${artifact}" ]] || { echo "[verify-rebuild] missing artifact ${artifact}" >&2; exit 1; }
  actual="$(sha256sum "${artifact}" | awk '{print $1}')"
  [[ "${actual}" == "${expected}" ]] || {
    echo "[verify-rebuild] hash mismatch: ${artifact}" >&2
    exit 1
  }
done < artifacts/last_build_hashes.txt

echo "[verify-rebuild] source diff and rebuilt artifacts verified"
