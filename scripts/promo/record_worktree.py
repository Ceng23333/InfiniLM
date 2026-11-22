#!/usr/bin/env python3
"""Write a durable provenance manifest for a campaign worktree."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List


SCHEMA = "campaign-worktree-version-1"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def git(args: List[str]) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=Path.cwd(),
        text=True,
    ).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_rows(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        parts = raw.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"invalid artifact hash row: {raw}")
        rows.append({"sha256": parts[0], "path": parts[1].strip()})
    return rows


def source_scope_hashes(root: Path, scope: List[str]) -> Dict[str, Any]:
    files: List[str] = []
    for item in scope:
        root_item = root / item
        if root_item.is_file():
            files.append(item)
        elif root_item.is_dir():
            files.extend(str(path.relative_to(root)) for path in root_item.rglob("*") if path.is_file())
    hashes = []
    total = hashlib.sha256()
    for relative in sorted(set(files)):
        path = root / relative
        digest = sha256(path)
        total.update(relative.encode("utf-8") + b"\0")
        total.update(bytes.fromhex(digest))
        hashes.append({"path": relative, "sha256": digest})
    return {"sha256": total.hexdigest(), "files": hashes}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["created", "edited", "built", "final", "failed"], required=True)
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/worktree_version.json"))
    args = parser.parse_args()

    root = Path.cwd().resolve()
    manifest_path = root / args.manifest
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    previous: Dict[str, Any] = {}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("schema") != SCHEMA:
            raise ValueError(f"unexpected worktree manifest schema: {previous.get('schema')}")

    spec_name = os.environ.get("CAMPAIGN_SPEC_PATH", "autohelix/source-edit.yaml")
    spec_path = root / spec_name
    spec_hash = os.environ.get("SPEC_SHA256")
    if spec_path.is_file():
        actual_spec_hash = sha256(spec_path)
        if spec_hash and spec_hash != actual_spec_hash:
            raise ValueError("spec hash does not match campaign state")
        spec_hash = actual_spec_hash
    elif not spec_hash:
        raise ValueError("campaign spec is missing and SPEC_SHA256 is not set")

    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root,
        text=True,
    ).splitlines()
    excluded_prefixes = ("artifacts/", ".campaign/", "autohelix/")
    changed_files = []
    for row in status:
        if len(row) < 4:
            continue
        path = row[3:].lstrip()
        if not path.startswith(excluded_prefixes):
            changed_files.append(path)

    artifact_path = root / "artifacts/artifact_hashes.txt"
    artifacts: Any = None
    artifact_hash = None
    if artifact_path.is_file():
        artifact_hash = sha256(artifact_path)
        artifacts = artifact_rows(artifact_path)

    manifest: Dict[str, Any] = {
        "schema": SCHEMA,
        "phase": args.phase,
        "iteration": int(os.environ["CAMPAIGN_ITERATION"]),
        "run_id": os.environ.get("CAMPAIGN_RUN_ID"),
        "spec_path": spec_name,
        "spec_sha256": spec_hash,
        "baseline_ref": os.environ.get("CAMPAIGN_BASELINE_REF"),
        "baseline_commit": os.environ.get("CAMPAIGN_BASELINE_COMMIT"),
        "parent_ref": os.environ.get("CAMPAIGN_PARENT_REF"),
        "git_commit": git(["rev-parse", "HEAD"]),
        "head_tree": git(["rev-parse", "HEAD^{tree}"]),
        "git_branch": git(["branch", "--show-current"]) or None,
        "git_describe": subprocess.check_output(
            ["git", "describe", "--always", "--dirty", "--tags"],
            cwd=root,
            text=True,
        ).strip(),
        "changed_files": sorted(set(changed_files)),
        "source_scope": source_scope_hashes(
            root,
            os.environ.get(
                "CAMPAIGN_EDITABLE_SCOPE",
                "csrc python/infinilm xmake.lua",
            ).split(),
        ),
        "artifact_hashes_sha256": artifact_hash,
        "artifacts": artifacts,
        "updated_utc": utc_now(),
    }
    manifest["created_utc"] = previous.get("created_utc", manifest["updated_utc"])
    manifest["phases"] = list(previous.get("phases", []))
    manifest["phases"].append({
        "phase": args.phase,
        "utc": manifest["updated_utc"],
        "head_tree": manifest["head_tree"],
    })

    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[record-worktree] {args.phase} commit={manifest['git_commit']} head_tree={manifest['head_tree']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
