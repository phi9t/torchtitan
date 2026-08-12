# External Harness Installed-Preflight Smoke

Run ID: `20260812T105500Z-external-harness-preflight-smoke`

This run upgrades the external agentic-harness lane from dry-run artifact
ingestion to a real package-install and import preflight inside the TorchTitan
bwrap rootfs. It is infrastructure evidence only: no Terminal-Bench, Harbor, or
tau2 benchmark task was executed, and no model or adapter capability score is
claimed.

## Command

```bash
RUN_ID=20260812T105500Z-external-harness-preflight-smoke \
RESULTS_ROOT=experiments/scaffold_to_policy/results/external_harness_preflight_smoke \
experiments/scaffold_to_policy/run_external_harness_preflight_smoke.sh
```

The script re-entered through `scripts/rootfs/enter_rootfs.sh`, created an
isolated virtualenv under the ignored results tree, installed pinned harness
packages, wrote raw preflight artifacts, ingested them, and built the shared
report input.

## Summary

| Harness family | Packages | Import/version check | CLI check | Task execution |
| --- | --- | --- | --- | --- |
| `harbor_terminal` | `harbor==0.21.0`, `terminal-bench==0.2.18` | passed | no `harbor`, `terminal-bench`, or `tb` executable found | not run |
| `tau2` | `tau2==2.3.3` | passed | no `tau2` executable found | not run |

Report-input checks all passed:

- `ingested_present=true`
- `all_rootfs_selected=true`
- `all_modes_labeled=true`
- `all_pins_present=true`
- `installed_preflight_imports_available=true`
- `installed_preflight_versions_match=true`

## Pinned Evaluation Objects

Harbor/Terminal-Bench:

- Harbor repo: `https://github.com/harbor-framework/harbor.git`
- Harbor revision: `b7e2f71b4563618af3a42279740f5f412dcf7046`
- Harbor package: `harbor==0.21.0`
- Terminal-Bench repo: `https://github.com/harbor-framework/terminal-bench-2-1.git`
- Terminal-Bench revision: `7131e4375048a0e408a8fb404b5f499d726b695b`
- Terminal-Bench package: `terminal-bench==0.2.18`

tau2:

- tau2-bench repo: `https://github.com/sierra-research/tau2-bench.git`
- tau2-bench revision: `668d3bcd135c02aa3438f987ef45735b7c163ee3`
- tau2 package: `tau2==2.3.3`

## Environment Evidence

Both raw preflight artifacts report:

- `rootfs.in_rootfs=true`
- Python: `experiments/scaffold_to_policy/results/external_harness_preflight_smoke/.venv-harness-preflight/bin/python`
- Python version: `3.12.3`
- platform: `Linux-5.15.152.bsk.9-amd64-x86_64-with-glibc2.39`
- `git`: `git version 2.43.0`
- `docker`: unavailable inside the rootfs environment
- `bwrap`: unavailable as an inner-rootfs command

The outer rootfs launch still uses the repo-local bwrap path. The missing
inner `bwrap` binary matters only if a later nested harness run tries to invoke
bwrap from inside this already-entered rootfs.

## Example Artifact Snippets

Harbor/Terminal-Bench package evidence:

```json
{
  "name": "terminal-bench-2-1",
  "package_module": "terminal_bench",
  "package_name": "terminal-bench",
  "package_version": "0.2.18",
  "installed": true,
  "installed_version": "0.2.18"
}
```

tau2 package evidence:

```json
{
  "name": "tau2-bench",
  "package_module": "tau2",
  "package_name": "tau2",
  "package_version": "2.3.3",
  "installed": true,
  "installed_version": "2.3.3"
}
```

The ingested trajectories deliberately stop at preflight:

```json
[
  {
    "actor": "torchtitan",
    "event": "create_isolated_rootfs_virtualenv"
  },
  {
    "actor": "external_harness",
    "event": "import_and_version_preflight"
  }
]
```

## Infra Notes

The earlier `/tmp` virtualenv attempt failed because each bwrap entry gets a
fresh `/tmp`. The working path keeps the harness virtualenv under:

```text
experiments/scaffold_to_policy/results/external_harness_preflight_smoke/.venv-harness-preflight/
```

That directory is generated runtime state and remains outside git.

Pip emitted dependency warnings because the isolated harness virtualenv installs
only the external harness packages, not the full TorchTitan development
requirements. This is acceptable for this preflight because the script imports
only the scaffold-to-policy CLI and harness packages, but real benchmark task
execution should either install TorchTitan requirements into that virtualenv or
move the external harness packages into the standard rootfs Python environment.

## Remaining Work

This clears the package-install/import/version gate for the external harness
lane. The next gate is real upstream-harness task execution:

- identify the package-supported entrypoint for `terminal-bench==0.2.18` when
  no `terminal-bench` or `tb` executable is exposed;
- decide whether Harbor should be invoked as a Python API or via a repo checkout
  rather than the pip package alone;
- identify the package-supported tau2 task runner when no `tau2` executable is
  exposed;
- add a one-task or minimal-subset task-execution smoke that preserves each
  benchmark's upstream scorer and records final-state or executable-test
  artifacts;
- keep the result labeled as model + scaffold + harness + tools + environment,
  not a model-only score.
