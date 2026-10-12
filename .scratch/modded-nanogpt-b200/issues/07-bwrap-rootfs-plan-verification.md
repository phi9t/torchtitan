# Issue 07: Bwrap Rootfs Plan Verification

Type: task
Status: complete
Blocked by: -

## Intent

Implement the first executable slice of the bwrap rootfs runtime spec: before a
payload runs, `scripts/rootfs/enter_rootfs.sh` can emit the concrete bwrap plan
it would execute, and a host-testable verifier can validate that plan against
the rootfs boundary invariants.

This issue stops at bwrap plan verification. It does not implement the full
uv/mise package sync, managed rootfs store, optimized-kernel verifier, full
preflight integration, or any B200 training launch.

## Authority Boundary

- No full B200 launch is authorized.
- Do not run `torchrun`, GPU preflight, package installation, or data
  preparation as part of this issue.
- Host-side tests may exercise plan emission without invoking `bwrap`.
- Rootfs entrypoint behavior must remain compatible with current wrappers.

## Acceptance Criteria

- `scripts/rootfs/enter_rootfs.sh` supports an emit-plan mode that resolves the
  same rootfs, mounts, cwd, env, network mode, and inner argv used for execution
  but exits before invoking `bwrap`.
- The emitted plan is structured JSON and can be written to
  `TORCHTITAN_ROOTFS_PLAN_OUTPUT` or stdout.
- The plan redacts no-needed secrets and does not contain launch authorization
  values.
- A Python verifier validates the plan's required rootfs invariants:
  rootfs path, repo mount, cwd, network mode, required mounts, required env, and
  inner argv.
- Focused tests cover success and fail-closed cases without requiring Docker,
  bwrap execution, CUDA, or a live GPU.
- Static verification passes for the B200 harness scope.

## Non-Goals

- Read-only rootfs conversion.
- uv or mise environment sync.
- Full runtime materialization schema implementation.
- Full preflight or launch-readiness integration.
- Cleaning generated results or rootfs trees.

## Verification

Run:

```bash
python3 -m pytest -q tests/unit_tests/test_rootfs_bwrap_plan.py
python3 experiments/modded_nanogpt_b200/verify_static.py
```

Completed evidence:

```text
python3 -m pytest -q tests/unit_tests/test_rootfs_bwrap_plan.py tests/unit_tests/test_execution_rootfs_selection_shell.py tests/unit_tests/test_execution_rootfs_identity.py tests/unit_tests/test_modded_nanogpt_b200_cli_guard.py
34 passed in 5.13s

bash -n scripts/rootfs/enter_rootfs.sh scripts/rootfs/verify_runtime_env.py

python3 experiments/modded_nanogpt_b200/verify_static.py
Static verification passed for 42 file(s).
```
