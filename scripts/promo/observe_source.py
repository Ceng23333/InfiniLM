#!/usr/bin/env python3
"""Run the parent wall observer from an InfiniLM AutoHelix worktree."""

import os
import runpy
import sys
from pathlib import Path


def main() -> int:
    root = Path(os.environ.get("INFINI_TENSOR_WORKTREE", ""))
    if not root:
        candidate = Path(__file__).resolve().parents[3]
        if (candidate / "scripts/promo/observe_restore.py").is_file():
            root = candidate
    script = root / "scripts/promo/observe_restore.py"
    if not script.is_file():
        print(f"parent observer not found: {script}", file=sys.stderr)
        return 2
    os.environ["AUTOHELIX_ROOT"] = str(root)
    sys.argv = [str(script), *sys.argv[1:]]
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
