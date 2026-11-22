#!/usr/bin/env bash
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
cd "${root}"
mkdir -p artifacts
out="artifacts/smoke_response.json"
container=qwen3-8b-mars-inf-tp4

curl --noproxy '*' -fsS --max-time 30 http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3-8B","messages":[{"role":"user","content":"Reply with exactly OK."}],"max_tokens":32,"temperature":0,"stream":false}' \
  > "${out}"

python3 - "${out}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    payload = json.load(f)
assert payload.get("object") == "chat.completion"
choices = payload.get("choices") or []
assert choices
content = (choices[0].get("message") or {}).get("content")
assert content and content.strip()
print("[smoke-candidate] generation succeeded")
PY

curl --noproxy '*' -fsS http://127.0.0.1:8000/health > artifacts/smoke_health.json
if docker exec "${container}" bash -lc \
  'grep -E "RankWorker stopped|AssertionError|Traceback" /tmp/qwen3-8b-server.log 2>/dev/null | tail -5' \
  > artifacts/smoke_server_errors.txt; then
  [[ ! -s artifacts/smoke_server_errors.txt ]] || {
    echo "[smoke-candidate] server errors after generation" >&2
    cat artifacts/smoke_server_errors.txt >&2
    exit 65
  }
fi

echo "[smoke-candidate] health and real generation passed"
