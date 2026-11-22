# AutoHelix source-edit worktree layout

Run AutoHelix from the InfiniLM checkout, not the parent deployment checkout:

```bash
cd /root/zenghua/workspace/deploy20261113/InfiniTensorWorktree/InfiniLM
autohelix run -c autohelix/source-edit.yaml -n 4
```

With this layout, each iteration gets a complete InfiniLM source tree:

```text
InfiniLM/                          # AutoHelix project + editable source
  autohelix/source-edit.yaml       # tracked campaign contract
  csrc/                            # editable C++ runtime and operators
  python/infinilm/                 # editable Python service/model code
  scripts/promo/                   # tracked build, launch, and observer wrappers
  artifacts/                       # ignored build and measurement evidence
  .autohelix/worktrees/iter-N/     # complete iteration worktree
```

`build_candidate.sh` rejects a diff with no `csrc/`, `python/infinilm/`, or
`xmake.lua` change. It rebuilds `infinicore_runtime`, `_infinicore`, and
`_infinilm`, then records hashes in `artifacts/last_build_hashes.txt`.
`verify_source_rebuild.sh` verifies those hashes before the candidate can be
measured. `launch_candidate.sh` passes its own Git worktree as
`CANDIDATE_WORKSPACE`; the parent launcher bind-mounts that directory over
`/workspace/InfiniLM`, so the restarted service loads the edited worktree rather
than the parent source checkout.
