# External Harness Installed-Preflight Smoke

Run IDs:

- initial preflight: `20260812T105500Z-external-harness-preflight-smoke`
- corrected preflight: `20260812T112500Z-external-harness-preflight-corrected`

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

The script re-entered through `scripts/rootfs/enter_rootfs.sh`, created
isolated virtualenvs under the ignored results tree, installed pinned harness
packages, wrote raw preflight artifacts, ingested them, and built the shared
report input. The corrected run uses separate virtualenvs for
Harbor/Terminal-Bench and tau2-bench because the pinned packages have
incompatible dependency constraints in one environment.

## Summary

| Harness family | Packages | Import/version check | CLI check | Task execution |
| --- | --- | --- | --- | --- |
| `harbor_terminal` | `harbor==0.21.0`, `terminal-bench==0.2.18` | passed | `harbor`, `terminal-bench`, and `tb` found | not run |
| `tau2` | Sierra tau2-bench git revision, package `tau2==1.0.1` | passed | `tau2` found | attempted mock runner; infra errors |

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
- tau2 package from the pinned repo: `tau2==1.0.1`

The initial run incorrectly installed PyPI `tau2==2.3.3`, which is a magnetic
relaxation package from a different project. The corrected run installs
Sierra's tau2-bench from the pinned git revision and verifies package version
`1.0.1`.

## Environment Evidence

The corrected raw preflight artifacts report:

- `rootfs.in_rootfs=true`
- Harbor/Terminal-Bench Python:
  `experiments/scaffold_to_policy/results/external_harness_preflight_corrected/.venv-harbor-terminal-preflight/bin/python`
- tau2 Python:
  `experiments/scaffold_to_policy/results/external_harness_preflight_corrected/.venv-tau2-preflight/bin/python`
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
  "package_version": "1.0.1",
  "installed": true,
  "installed_version": "1.0.1"
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
fresh `/tmp`. The working path keeps harness virtualenvs under:

```text
experiments/scaffold_to_policy/results/external_harness_preflight_corrected/.venv-harbor-terminal-preflight/
experiments/scaffold_to_policy/results/external_harness_preflight_corrected/.venv-tau2-preflight/
```

That directory is generated runtime state and remains outside git.

Pip emitted dependency warnings because the isolated harness virtualenvs install
only the external harness packages, not the full TorchTitan development
requirements. This is acceptable for this preflight because the script imports
only the scaffold-to-policy CLI and harness packages.

## Task-Execution Probe

After the corrected preflight, a minimal tau2 mock-domain runner probe was
attempted through the tau2-bench CLI with `TAU2_DATA_DIR` pointed at the pinned
repo checkout under the ignored results tree. This reached tau2's task loader,
runner, retry loop, and results writer, but both no-external-LLM pairings ended
as infrastructure errors:

- `llm_agent_gt` with `dummy_user` failed with `Dummy user can only be used with solo agent`;
- `llm_agent_solo` with `dummy_user` failed with
  `DummyUser.__init__() got an unexpected keyword argument 'tools'`.

The saved tau2 result files are generated artifacts under:

```text
experiments/scaffold_to_policy/results/external_harness_preflight_corrected/src/tau2-bench/data/simulations/
```

They contain `termination_reason: "infrastructure_error"` and `Evaluated: 0`,
so they are not successful tau2 benchmark evaluations. They are useful blocker
evidence for the next harness-integration step.

## Remaining Work

This clears the package-install/import/version gate for the external harness
lane. The next gate is real upstream-harness task execution:

- choose a Terminal-Bench task and execution environment; the CLI is available
  but expects containerized task execution;
- decide whether Harbor should run Terminal-Bench through `harbor run`,
  `harbor exec`, or a job config that preserves Harbor's own scorer;
- choose a tau2-bench agent/user pairing that can run at least one mock-domain
  task without external API keys, or wire a local model provider explicitly;
- add a one-task or minimal-subset task-execution smoke that preserves each
  benchmark's upstream scorer and records final-state or executable-test
  artifacts;
- keep the result labeled as model + scaffold + harness + tools + environment,
  not a model-only score.
