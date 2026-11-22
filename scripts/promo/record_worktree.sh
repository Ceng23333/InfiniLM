#!/usr/bin/env bash
set -euo pipefail
python3 "$(git rev-parse --show-toplevel)/scripts/promo/record_worktree.py" --phase "${1:-edited}"
