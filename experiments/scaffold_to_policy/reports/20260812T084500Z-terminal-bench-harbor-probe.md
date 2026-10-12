# Terminal-Bench / Harbor Probe

Run ID: `20260812TTEST-terminal-harbor`

This probe checks the next external coding/agentic benchmark boundary. It
installs Harbor and Terminal-Bench in an isolated rootfs virtualenv, clones the
pinned Terminal-Bench 2.1 task repo, verifies Terminal-Bench result-model
ingestion, and then attempts to run one Harbor-native task through Harbor's own
CLI.

It is not a Terminal-Bench or Harbor model result. The execution probe is
blocker evidence.

## Command

```bash
RUN_ID=20260812TTEST-terminal-harbor \
RESULTS_ROOT=experiments/scaffold_to_policy/results/terminal_bench_oracle_probe_harbor_test \
TASK_ID=headless-terminal \
TIMEOUT_SECONDS=120 \
experiments/scaffold_to_policy/run_terminal_bench_oracle_probe.sh
```

The entrypoint re-executes through:

```bash
scripts/rootfs/enter_rootfs.sh -- experiments/scaffold_to_policy/run_terminal_bench_oracle_probe.sh
```

## Pins

| Component | Pin |
| --- | --- |
| Harbor package | `harbor==0.21.0` |
| Terminal-Bench package | `terminal-bench==0.2.18` |
| Harbor repo revision | `b7e2f71b4563618af3a42279740f5f412dcf7046` |
| Terminal-Bench 2.1 task repo revision | `7131e4375048a0e408a8fb404b5f499d726b695b` |
| Task | `headless-terminal` |

## Result

Two separate artifacts were ingested:

| Artifact | Mode | Result |
| --- | --- | --- |
| `terminal_bench_result_smoke` | `task_score_smoke` | passed result-model ingestion with score `1.0` |
| `terminal_bench_execution_probe` | `task_execution_probe` | failed before task execution because Docker is unavailable in rootfs |

The corrected Harbor execution command reached the real environment blocker:

```text
Docker is not installed or not on PATH. Please install Docker and try again.
```

Report-input checks:

| Check | Result |
| --- | --- |
| `ingested_present` | true |
| `all_rootfs_selected` | true |
| `all_modes_labeled` | true |
| `all_pins_present` | true |
| `task_score_smokes_succeeded` | true |
| `task_execution_probes_succeeded` | false |

## Important Finding

The pinned Terminal-Bench 2.1 task repo uses Harbor's newer task layout:

```text
task.toml
instruction.md
environment/Dockerfile
tests/test.sh
solution/solve.sh
```

The installed `terminal-bench==0.2.18` CLI expects the older
`task.yaml`/`docker-compose.yaml` layout, so direct `tb runs create` is not the
right execution path for this pinned task corpus. The corrected probe uses:

```bash
harbor run \
  --path <terminal-bench-2-1>/tasks/headless-terminal \
  --jobs-dir <results>/runs \
  --job-name <run_id> \
  --env docker \
  --agent oracle \
  --n-concurrent 1 \
  --n-attempts 1 \
  --max-retries 0 \
  --yes \
  --no-delete
```

## Interpretation

TorchTitan's rootfs-managed package install, task-repo pinning, result-model
ingestion, and Harbor CLI launch boundary now work. Full Terminal-Bench/Harbor
task execution remains blocked until Docker or an equivalent Harbor environment
backend is available inside the bwrap rootfs. Once that is fixed, the next gate
is a one-task Harbor oracle run that writes real Harbor trial artifacts before
any model or adapter is evaluated.
