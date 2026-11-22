#!/usr/bin/env python3
"""Run the parent Codex-aware AutoHelix monitor for this source worktree."""

import subprocess
import os
import sys
from pathlib import Path


def main() -> int:
    project = Path.cwd().resolve()
    if not (project / ".git").exists():
        project = Path(__file__).resolve().parents[2]

    parent = Path(__file__).resolve().parents[3]
    watcher = parent / "scripts/promo/watch_codex.py"
    if not watcher.is_file():
        print(f"parent watcher not found: {watcher}", file=sys.stderr)
        return 2

    watcher_python = os.environ.get(
        "AUTOHELIX_PYTHON",
        "/root/zenghua/workspace/deploy20261113/.venv-autohelix/bin/python",
    )
    return subprocess.call([watcher_python, str(watcher), str(project), *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
