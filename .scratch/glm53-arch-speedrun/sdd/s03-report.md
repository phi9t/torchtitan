# S03 report: ablation queue and opportunistic scheduler

Date: 2026-10-14

## Outcome

Phase A is complete. I built the ablation queue, documented the 4c fractional-factorial design and alias structure,
added the host-only opportunistic scheduler, and added a rootfs analysis helper for completed slices.

Phase C was not run. Phase B did not have green gates: `.scratch/glm53-arch-speedrun/sdd/s01-report.md` was still
missing, and `.scratch/glm53-arch-speedrun/sdd/s02-report.md` reported the overfit/throughput gate as blocked because
the S01 harness, runner, baseline model, config registry, and data metadata were not present. Per the S03 brief, the
matrix must not run on a red or incomplete harness.

## Files

- `experiments/glm53_arch_speedrun/queue/generate_cells.py`: deterministic queue generator.
- `experiments/glm53_arch_speedrun/queue/cells.jsonl`: 93 runnable cells for slices 4a, 4b, and 4c.
- `experiments/glm53_arch_speedrun/queue/DESIGN.md`: human-readable queue and 4c design declaration.
- `experiments/glm53_arch_speedrun/queue/alias_structure.json`: machine-readable effect coding and alias declaration.
- `experiments/glm53_arch_speedrun/queue/deferred_4d_templates.jsonl`: deferred dose-sweep templates.
- `experiments/glm53_arch_speedrun/queue/deferred_c1_templates.jsonl`: deferred combined-winner templates.
- `experiments/glm53_arch_speedrun/queue/state/.gitignore`: keeps mutable per-cell state files out of source control.
- `experiments/glm53_arch_speedrun/scheduler.sh`: host scheduler.
- `experiments/glm53_arch_speedrun/analyze_results.py`: rootfs slice analysis helper.
- `docs/research/2026-10-14-glm-5.3-flash-arch-speedrun-ablations.md`: updated with the S03 queue declaration.

## Queue declaration

Runnable cells:

| Slice | Runs | Notes |
| --- | ---: | --- |
| 4a | 30 | B0 x3 and each add-one arm x2 at seq 4096, plus the required seq-1024 replicate. |
| 4b | 15 | G x3 and each leave-one-out arm x2 at seq 4096. |
| 4c | 48 | 24-row fraction x2 seeds at seq 4096. |

Deferred cells:

| Slice | Templates | Freeze rule |
| --- | ---: | --- |
| 4d | 36 | Choose each element's better add-one vs leave-one-out context after 4a/4b. |
| C1 | 3 | Include winners after 4a-4d where mean val CE improves by more than `2 * std(B0 seed CE)`. |

4c design:

- Generator seed: `5303`.
- Candidate set: full `2^4 x 3` over F1 mHC, F2 positions, F3 mixer, F4 sparse, and F5 MTP.
- Balance: exactly eight rows per sparse level and 12/12 balance on every two-level factor.
- Registered model: intercept, all main effects, all binary-binary two-way interactions, and all binary-by-sparse
  two-way interactions.
- Rank: `21/21`, so the registered main and two-way terms have no exact aliases. Higher-order effects are pooled into
  residual error.

## Scheduler behavior

`experiments/glm53_arch_speedrun/scheduler.sh` runs only on the host. It:

- waits for S01 and S02 reports with explicit green gate language;
- stops or waits rather than launching if reports are missing, blocked, or red;
- reads `nvidia-smi` free memory and launches only on GPUs with at least 30 GiB free;
- launches at most six cells globally and at most two cells per GPU;
- records per-cell JSON state under `queue/state/` with status, exit code, run ID, attempt ID, PID, devices, and paths;
- retries once only when logs classify the failure as infrastructure (`NCCL`, `torchrun`, launch, rendezvous, timeout);
- does not retry numeric failures (`NaN`, `Inf`, non-finite loss, or explicit numeric failure class);
- publishes completed attempt bundles with `experiments/glm53_flash_sft_ladder/publish.sh` to
  `torchtitan/runs/glm53-arch-speedrun/<run_id>/`;
- runs `experiments/glm53_arch_speedrun/analyze_results.py` inside `TORCHTITAN_ROOTFS_ENV=glm53_train` after a slice is
  complete.

## Verification

Commands run:

```bash
TORCHTITAN_ROOTFS_ENV=glm53_train scripts/rootfs/enter_rootfs.sh -- bash -lc 'unset CC CXX; cd /workspace/torchtitan && python experiments/glm53_arch_speedrun/queue/generate_cells.py'
```

Result: PASS. Summary: 93 runnable cells, 36 deferred 4d templates, 3 deferred C1 templates, 4c rank `21/21`.

```bash
bash -n experiments/glm53_arch_speedrun/scheduler.sh
TORCHTITAN_ROOTFS_ENV=glm53_train scripts/rootfs/enter_rootfs.sh -- bash -lc 'unset CC CXX; cd /workspace/torchtitan && python -m py_compile experiments/glm53_arch_speedrun/analyze_results.py experiments/glm53_arch_speedrun/queue/generate_cells.py'
```

Result: PASS.

```bash
jq -s 'length as $n | {lines:$n, unique_ids:(map(.cell_id)|unique|length), runnable:(map(select(.runnable != false))|length), slices:(group_by(.slice)|map({slice:.[0].slice,count:length}))}' experiments/glm53_arch_speedrun/queue/cells.jsonl
```

Result: PASS. `lines=93`, `unique_ids=93`, slice counts `4a=30`, `4b=15`, `4c=48`.

```bash
experiments/glm53_arch_speedrun/scheduler.sh --once --dry-run --no-wait
```

Result: PASS. The scheduler initialized pending state and refused to launch because
`.scratch/glm53-arch-speedrun/sdd/s01-report.md` was missing.

I also ran a temporary fake-green two-cell scheduler test with `GLM53_ARCH_RUNNER_CMD` and `GLM53_ARCH_PUBLISH_SCRIPT`
overrides. Result: PASS. The first cell launched, wrote exit code 0, transitioned to `done`, called the fake publisher,
and the second pending cell launched. No real HDFS publish or real training job was used in that test.

## Phase B gate status

At report time:

- S01 report: missing.
- S02 report: present but not green; it says the overfit/throughput gate is `BLOCKED: missing S01 harness`.
- S01/S02 worker processes were still alive.

Because the harness gates were not green, I did not run any 4a, 4b, 4c, 4d, or C1 matrix cells.
