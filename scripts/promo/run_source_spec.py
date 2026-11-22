#!/usr/bin/env python3
"""Execute the InfiniLM source-edit spec without the AutoHelix controller."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


METRIC_RE = re.compile(r"##autohelix\[([A-Za-z_][A-Za-z0-9_]*)=([-+0-9.eE]+)\]")
LEVER_RE = re.compile(r"##autohelix\[lever=([A-Za-z0-9][A-Za-z0-9._-]*)\]")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=check,
    )


def append_record(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def run_command(
    command: str,
    cwd: Path,
    timeout: int,
    log: Path,
    env: Optional[Dict[str, str]] = None,
) -> Tuple[bool, Optional[str]]:
    with log.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            shell=True,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            output.write(f"\n[spec-runner] timeout after {timeout}s\n")
            return False, f"timeout after {timeout}s"
    if process.returncode == 0:
        return True, None
    text = log.read_text(errors="replace")
    return False, text[-20000:]


def parse_metrics(path: Path) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for line in path.read_text(errors="replace").splitlines():
        for match in METRIC_RE.finditer(line):
            values[match.group(1)] = float(match.group(2))
    return values


def parse_lever(path: Path) -> Optional[str]:
    lever = None
    for line in path.read_text(errors="replace").splitlines():
        match = LEVER_RE.search(line)
        if match:
            lever = match.group(1)
    return lever


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def source_files(root: Path, scope: List[str]) -> List[str]:
    changed = set()
    for item in scope:
        diff = git(root, "diff", "--name-only", "HEAD", "--", item, check=False)
        changed.update(diff.stdout.splitlines())
        untracked = git(
            root, "ls-files", "--others", "--exclude-standard", "--", item, check=False
        )
        changed.update(untracked.stdout.splitlines())
    return sorted(changed)


def make_worktree(project: Path, path: Path, ref: str) -> None:
    if path.exists():
        raise RuntimeError(f"iteration worktree already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    result = git(project, "worktree", "add", "--detach", str(path), ref, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stdout)


def campaign_env(
    state: Dict[str, Any],
    iteration: int,
    spec_name: str,
    phase: str,
) -> Dict[str, str]:
    env = os.environ.copy()
    env.update({
        "CAMPAIGN_ITERATION": str(iteration),
        "CAMPAIGN_RUN_ID": str(state["run_id"]),
        "CAMPAIGN_SPEC_PATH": spec_name,
        "SPEC_SHA256": str(state["spec_sha256"]),
        "CAMPAIGN_BASELINE_REF": str(state["baseline_ref"]),
        "CAMPAIGN_BASELINE_COMMIT": str(state["baseline_commit"]),
        "CAMPAIGN_PARENT_REF": str(state["current_ref"]),
        "CAMPAIGN_PHASE": phase,
        "CAMPAIGN_EDITABLE_SCOPE": state["editable_scope"],
    })
    return env


def write_manifest(
    worktree: Path,
    state: Dict[str, Any],
    iteration: int,
    spec_name: str,
    phase: str,
) -> Dict[str, Any]:
    manifest = worktree / "artifacts/worktree_version.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "python3",
            "scripts/promo/record_worktree.py",
            "--phase",
            phase,
            "--manifest",
            str(manifest.relative_to(worktree)),
        ],
        cwd=worktree,
        env=campaign_env(state, iteration, spec_name, phase),
        check=True,
    )
    return json.loads(manifest.read_text(encoding="utf-8"))


def commit_candidate(
    project: Path,
    worktree: Path,
    scope: List[str],
    branch: str,
    title: str,
    state: Dict[str, Any],
) -> str:
    git(worktree, "add", "--", *scope)
    manifest_hash = sha256(worktree / "artifacts/worktree_version.json")
    git(
        worktree,
        "-c",
        "user.name=Codex",
        "-c",
        "user.email=codex@local",
        "commit",
        "-m",
        f"{title}\n\nWorktree-Version-Manifest: {manifest_hash}\n"
        f"Spec-SHA256: {state['spec_sha256']}\n"
        f"Campaign-Run-ID: {state['run_id']}",
    )
    commit = git(worktree, "rev-parse", "HEAD").stdout.strip()
    git(project, "branch", branch, commit)
    return commit


def remember_failed_lever(
    memory_path: Path,
    record: Dict[str, Any],
    failure_class: str,
) -> bool:
    rejected_commit = record.get("rejected_commit")
    if not rejected_commit:
        return False
    entry = {
        "schema": "campaign-failed-lever-1",
        "lever": record.get("lever") or f"unlabeled-{failure_class}",
        "failure_class": failure_class,
        "run_id": record.get("run_id"),
        "iteration": record.get("iteration"),
        "rejected_ref": record.get("rejected_branch"),
        "rejected_commit": rejected_commit,
        "baseline_commit": record.get("baseline_commit"),
        "source_files": record.get("source_files", []),
        "metrics": record.get("metrics"),
        "failure": (record.get("failure") or "")[-2000:],
        "finished_utc": record.get("finished_utc"),
    }
    for existing in load_jsonl(memory_path):
        if existing.get("rejected_commit") == rejected_commit:
            return False
    append_record(memory_path, entry)
    return True


def agent_prompt(spec: Dict[str, Any], iteration: int, baseline: Dict[str, Any]) -> str:
    scope = ", ".join(spec["scope"]["editable"])
    memory = spec.get("memory") or {}
    memory_path = Path(".campaign") / memory.get(
        "failed_levers_path",
        "failed_levers.jsonl",
    )
    failed_levers = (memory.get("known_failed_levers") or []) + load_jsonl(memory_path)
    return f"""Apply exactly one safe InfiniLM source mechanism for Qwen3-8B Mars serving.

This is spec-driven iteration {iteration}. Work only in the current Git worktree.
Editable paths: {scope}. Do not edit scripts/promo, autohelix, or deployment files.
The current baseline already contains the reproducible Qwen3-8B TP4 P64 Mars decode
graph collapse. Do not revert that mechanism. Inspect the matched InfiniProfiler and
promotion artifacts, then choose the next highest-evidence source mechanism. If evidence
is insufficient, leave the tree unchanged and explain why.

Baseline identity:
{json.dumps(baseline, indent=2, sort_keys=True)}

Known failed levers (do not repeat one of these mechanisms):
{json.dumps(failed_levers, indent=2, sort_keys=True)}

Finish by printing exactly one marker identifying the mechanism, for example:
##autohelix[lever=mars-batched-greedy-argmax]
Use only lowercase kebab-case, digits, dots, or hyphens. This marker must be the
last nonblank line of your final message.

Do not edit artifacts/worktree_version.json. The controller owns that manifest
and will run build, artifact verification, launch, real-generation smoke, and
unprofiled wall observation after you stop.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--baseline-ref", required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    project = Path.cwd().resolve()
    with args.spec.open(encoding="utf-8") as handle:
        spec = yaml.safe_load(handle)
    if spec.get("spec_version") != "source-edit-spec-2":
        raise SystemExit("spec must declare source-edit-spec-2")
    traceability = spec.get("traceability")
    if not traceability:
        raise SystemExit("spec must declare traceability")
    if traceability.get("worktree_manifest") != "artifacts/worktree_version.json":
        raise SystemExit("spec must require artifacts/worktree_version.json")
    if not traceability.get("require_rejected_commit"):
        raise SystemExit("spec must require rejected candidate commits")
    campaign = project / ".campaign"
    logs = campaign / "logs"
    worktrees = campaign / "worktrees"
    history = campaign / "history.jsonl"
    state_path = campaign / "state.json"
    memory_relative = Path((spec.get("memory") or {}).get(
        "failed_levers_path",
        "failed_levers.jsonl",
    ))
    if memory_relative.is_absolute() or ".." in memory_relative.parts:
        raise SystemExit("memory.failed_levers_path must be under .campaign")
    memory_path = campaign / memory_relative
    campaign.mkdir(parents=True, exist_ok=True)

    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        baseline_commit = git(project, "rev-parse", args.baseline_ref).stdout.strip()
        state = {
            "schema": "source-spec-campaign-1",
            "run_id": utc_now(),
            "spec": str(args.spec),
            "spec_sha256": sha256(args.spec),
            "baseline_ref": args.baseline_ref,
            "baseline_commit": baseline_commit,
            "baseline_summary": str(args.baseline_summary),
            "current_ref": args.baseline_ref,
            "editable_scope": " ".join(spec["scope"]["editable"]),
            "next_iteration": 1,
            "started_utc": utc_now(),
        }
        state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")

    if state.get("spec_sha256") != sha256(args.spec):
        raise SystemExit(
            "spec changed after campaign state was created; archive or reset .campaign"
        )
    if "run_id" not in state:
        state["run_id"] = utc_now()
    state.setdefault("editable_scope", " ".join(spec["scope"]["editable"]))

    max_iterations = args.max_iterations or int(spec.get("budget", {}).get("iterations", 1))
    with args.baseline_summary.open(encoding="utf-8") as handle:
        baseline = json.load(handle)
    stopped = False

    while state["next_iteration"] <= max_iterations and not stopped:
        iteration = int(state["next_iteration"])
        stamp = utc_now()
        worktree = worktrees / f"iter-{iteration}"
        iter_logs = logs / f"iter-{iteration}"
        iter_logs.mkdir(parents=True, exist_ok=True)
        record: Dict[str, Any] = {
            "schema": "source-spec-iteration-1",
            "iteration": iteration,
            "started_utc": stamp,
            "spec_sha256": state["spec_sha256"],
            "baseline_ref": state["baseline_ref"],
            "baseline_commit": state["baseline_commit"],
            "current_ref": state["current_ref"],
            "worktree": str(worktree),
            "run_id": state["run_id"],
        }
        try:
            make_worktree(project, worktree, state["current_ref"])
            record["worktree_version_created"] = write_manifest(
                worktree,
                state,
                iteration,
                args.spec.name,
                "created",
            )
            agent_log = iter_logs / "codex.log"
            with agent_log.open("w", encoding="utf-8") as output:
                process = subprocess.Popen(
                    [
                        "codex",
                        "exec",
                        "--dangerously-bypass-approvals-and-sandbox",
                        "--skip-git-repo-check",
                        "-C",
                        str(worktree),
                        "-m",
                        str(spec["agent"]["model"]),
                        agent_prompt(spec, iteration, baseline),
                    ],
                    cwd=worktree,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    process.wait(timeout=int(spec["agent"]["timeout_seconds"]))
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    raise RuntimeError(
                        f"agent timeout after {spec['agent']['timeout_seconds']}s"
                    )
            if process.returncode != 0:
                raise RuntimeError(f"codex exited {process.returncode}")

            lever = parse_lever(agent_log)
            record["lever"] = lever
            if (spec.get("memory") or {}).get("require_lever_marker") and not lever:
                raise RuntimeError("agent omitted the required lever marker")

            changed = source_files(worktree, spec["scope"]["editable"])
            record["source_files"] = changed
            if not changed:
                raise RuntimeError("agent made no editable source change")

            record["worktree_version_edited"] = write_manifest(
                worktree,
                state,
                iteration,
                args.spec.name,
                "edited",
            )

            forbidden = source_files(
                worktree, ["scripts/promo", "autohelix", "deployment"]
            )
            if forbidden:
                raise RuntimeError(f"agent edited forbidden paths: {forbidden}")

            for index, constraint in enumerate(spec["constraints"], 1):
                log = iter_logs / f"constraint-{index}.log"
                ok, failure = run_command(
                    constraint["command"],
                    worktree,
                    int(constraint["timeout"]),
                    log,
                    campaign_env(
                        state,
                        iteration,
                        args.spec.name,
                        f"constraint-{index}",
                    ),
                )
                record.setdefault("constraints", []).append(
                    {"command": constraint["command"], "log": str(log), "ok": ok}
                )
                if not ok:
                    raise RuntimeError(
                        f"constraint failed: {constraint['command']}: {failure}"
                    )
                record["worktree_version"] = write_manifest(
                    worktree,
                    state,
                    iteration,
                    args.spec.name,
                    "final" if index == len(spec["constraints"]) else "built",
                )

            metric_constraint = spec["metrics"][0]
            metric_log = iter_logs / "observe.log"
            ok, failure = run_command(
                metric_constraint["command"],
                worktree,
                int(metric_constraint.get("timeout", 3600)),
                metric_log,
                    campaign_env(state, iteration, args.spec.name, "observe"),
            )
            metrics = parse_metrics(metric_log) if ok else {}
            record["metrics"] = metrics
            if not ok:
                raise RuntimeError(f"metrics failed: {failure}")

            suite_root = Path(
                os.environ.get(
                    "INFINI_TENSOR_WORKTREE",
                    str(project.parent.parent.parent),
                )
            )
            compare_path = suite_root / "artifacts/promo/suite/compare.json"
            compare = json.loads(compare_path.read_text(encoding="utf-8"))
            record["compare"] = compare
            primary = metrics.get("primary_wall_ratio")
            success = metrics.get("success_rate", 0.0)
            ttft = metrics.get("ttft_p95_ratio", 999.0)
            gate_limit = 1.0
            for gate in spec.get("acceptance", {}).get("metric_gates", []):
                if gate.get("metric") == "primary_wall_ratio":
                    percent = float(gate.get("max_regression_pct", 0))
                    gate_limit = 1.0 - percent / 100.0
            accepted = bool(
                primary is not None
                and primary >= gate_limit
                and success >= 0.99
                and ttft <= 1.25
                and compare.get("status") == "PASS"
            )
            record["accepted"] = accepted
            record["surpasses_baseline"] = bool(primary is not None and primary > 1.0)
            record["worktree_version_final"] = write_manifest(
                worktree,
                state,
                iteration,
                args.spec.name,
                "final",
            )
            artifacts = record["worktree_version_final"].get("artifacts") or []
            record["artifact_hashes"] = artifacts

            if accepted:
                branch = f"campaign/accepted/{stamp}-iter-{iteration}"
                commit = commit_candidate(
                    project,
                    worktree,
                    spec["scope"]["editable"],
                    branch,
                    f"Spec iteration {iteration}: accepted source lever",
                    state,
                )
                record["commit"] = commit
                record["accepted_branch"] = branch
                if record["surpasses_baseline"]:
                    state["current_ref"] = branch
            else:
                diff_path = iter_logs / "rejected.diff"
                diff = git(
                    worktree,
                    "diff",
                    "HEAD",
                    "--",
                    *spec["scope"]["editable"],
                    check=False,
                )
                diff_path.write_text(diff.stdout)
                record["rejected_diff"] = str(diff_path)
                if traceability.get("require_rejected_commit"):
                    rejected_branch = f"campaign/rejected/{stamp}-iter-{iteration}"
                    rejected_commit = commit_candidate(
                        project,
                        worktree,
                        spec["scope"]["editable"],
                        rejected_branch,
                        f"Spec iteration {iteration}: rejected source lever",
                        state,
                    )
                    record["rejected_branch"] = rejected_branch
                    record["rejected_commit"] = rejected_commit
                record["finished_utc"] = utc_now()
                remember_failed_lever(memory_path, record, "metrics_no_go")
        except Exception as exc:
            record["accepted"] = False
            record["failure"] = str(exc)
            if worktree.exists():
                try:
                    changed = source_files(worktree, spec["scope"]["editable"])
                    if changed:
                        record["worktree_version_failed"] = write_manifest(
                            worktree,
                            state,
                            iteration,
                            args.spec.name,
                            "failed",
                        )
                        rejected_branch = f"campaign/rejected/{stamp}-iter-{iteration}"
                        rejected_commit = commit_candidate(
                            project,
                            worktree,
                            spec["scope"]["editable"],
                            rejected_branch,
                            f"Spec iteration {iteration}: failed source lever",
                            state,
                        )
                        record["rejected_branch"] = rejected_branch
                        record["rejected_commit"] = rejected_commit
                        state["last_rejected_ref"] = rejected_branch
                        record["finished_utc"] = utc_now()
                        failure_class = (
                            "missing_lever_marker" if not record.get("lever") else "constraint_failed"
                        )
                        remember_failed_lever(memory_path, record, failure_class)
                except Exception as provenance_exc:
                    record["provenance_failure"] = str(provenance_exc)
            stopped = True
        finally:
            record["finished_utc"] = utc_now()
            append_record(history, record)
            state["next_iteration"] = iteration + 1
            state["last_utc"] = record["finished_utc"]
            state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")

    return 0 if not stopped else 1


if __name__ == "__main__":
    raise SystemExit(main())
