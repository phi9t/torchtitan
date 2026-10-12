# Issue 09: Two-GPU Trial Retro and Next Iteration

Type: task
Status: blocked
Blocked by: trusted user request containing `launch-full-b200`

## Intent

Record the first two-GPU trial attempt retro and prepare the next safe
iteration.

The attempted run directory
`experiments/modded_nanogpt_b200/results/lane_b_two_gpu_trial_20260818T041445Z_attempt_001/`
contains only `active_jobs_prelaunch.json`. It is prelaunch evidence, not a
failed training run. No `attempt.json`, `preflight_report.json`, `run.log`,
`summary.json`, or training telemetry exists for that directory.

## Findings

- GPUs `0,1` were idle when selected for the trial.
- GPU `7` had another user's Python process, so it was not selected.
- `check_active_jobs.sh` reported `active_job_count=0`.
- The non-skip full launch did not start because the trusted user request did
  not contain `launch-full-b200`; wrapper or CLI authorization flags are
  lower-level launch markers, not primary authority.
- The harness still needed code alignment with the new policy before another
  run: preflight, runner, and summarizer had active 8-GPU assumptions.

## Completed Preparation

- `preflight.py` full mode now expects exactly two B200 GPUs.
- `run_speedrun.py` now passes `--expected-gpus 2` to preflight and launches
  `torchrun --nproc_per_node=2`.
- `summarize.py` launch-readiness and baseline GPU evidence gates now require
  the active two-GPU B200 allocation instead of eight GPUs.
- Focused rootfs tests cover the updated preflight, runner, and summarizer
  behavior.

## Next Iteration Gates

Before spending GPU time again:

- use a fresh `run_id` and result directory; do not reuse the prelaunch-only
  directory from this issue;
- select exactly two idle B200 GPUs, normally the lowest-index idle pair;
- set `CUDA_VISIBLE_DEVICES=<two selected host GPU indexes>` on the
  rootfs-aware wrapper invocation;
- rerun `check_active_jobs.sh --active-jobs-output <result_dir>/active_jobs_prelaunch.json`;
- rerun the focused rootfs tests and static verifier if the harness changed;
- require the trusted user request to contain `launch-full-b200` before setting
  `MODDED_NANOGPT_2GPU_FULL_LAUNCH_AUTHORIZATION=launch-full-b200` on the
  no-argument wrapper or passing `--launch-authorization=launch-full-b200` to
  lower-level launch tooling.

## Verification

Initial retro verification:

```bash
python3 -m pytest -q \
  tests/unit_tests/test_modded_nanogpt_b200_summarize.py \
  tests/unit_tests/test_modded_nanogpt_b200_preflight.py \
  tests/unit_tests/test_modded_nanogpt_b200_run_speedrun.py
```

Result:

```text
90 passed in 1.30s
```

Later master-plan Task 9 runtime-verification hardening superseded the earlier
outstanding verification item and completed the static check inside the rootfs.
That prerequisite hardening does not resolve this issue; this issue remains
blocked until an authorized non-skip two-GPU trial can run:

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 -m pytest -q tests/unit_tests/test_modded_nanogpt_b200_schemas.py tests/unit_tests/test_modded_nanogpt_b200_runtime_verifier.py tests/unit_tests/test_modded_nanogpt_b200_optimized_kernel_certifier.py tests/unit_tests/test_modded_nanogpt_b200_preflight.py tests/unit_tests/test_modded_nanogpt_b200_run_speedrun.py tests/unit_tests/test_modded_nanogpt_b200_parse_log.py tests/unit_tests/test_modded_nanogpt_b200_summarize.py && python3 -m py_compile experiments/modded_nanogpt_b200/runtime/verify_runtime.py experiments/modded_nanogpt_b200/run_speedrun.py && python3 experiments/modded_nanogpt_b200/verify_static.py'
```

Result:

```text
152 passed in 7.79s
Static verification passed for 31 file(s).
```

## Comments

2026-08-18: Retro classified the prior two-GPU directory as prelaunch-only.
Root cause for no training artifacts was missing explicit launch authorization,
not a training crash. A separate readiness issue was fixed in the summarizer:
two-GPU trial summaries would have been rejected by stale 8-GPU evidence gates.

2026-08-18: Next retry was blocked at GPU selection. The NanoGPT active-job scan
was clean, but every visible B200 had another user's `VLLM::Worker_TP*_EP*`
process attached and about 96 GiB allocated. Evidence is preserved under
`experiments/modded_nanogpt_b200/results/lane_b_two_gpu_blocked_busy_20260818T055607Z_attempt_001/`.
Do not launch onto those GPUs. Retry only after two B200 GPUs are idle and the
trusted request contains `launch-full-b200`; only then may the wrapper or
lower-level runner receive the launch-authorization argument for a non-skip full
run.

2026-08-19: Master-plan Task 9 non-launch prerequisite refresh completed for
the current two-GPU Lane B FA2/Triton gate. This prerequisite refresh does not
resolve issue `09`; the first non-skip two-GPU trial remains blocked until the
trusted request contains `launch-full-b200`. The first runtime-refresh skip-run artifact
(`lane_b_full_skiprun_runtime_refresh_20260819T082409Z`) exposed a runner
classification propagation bug: `attempt.json` was correctly conservative with
`claim_eligible=false`, but post-preflight `launch_readiness.json` and
`summary.json` failed to adopt `claim_eligible=true` from the matching
successful preflight report, so the run index demoted the artifact with
`classification.claim_eligible must be true`. A focused regression now proves
initial metadata remains conservative while post-preflight readiness and
summary classification become claim-eligible only after a matching successful
preflight. The fixed fresh skip-run artifact is
`experiments/modded_nanogpt_b200/results/lane_b_full_skiprun_runtime_refresh_20260819T082841Z/summary.json`;
it records 2x B200 evidence, `ready_to_launch=true`, `training_launched=false`,
`skip_run=true`, `blocked_by=[]`, launch-eligible optimized-kernel evidence, and
matching command-env/runtime-verification digests. Refreshed `run_index.json`
reports `total_attempts=58`, `baseline_stats.count=0`,
`len(launch_prerequisite_attempts)=2`, and `len(launch_ready_attempts)=0`.
Verification: `test_modded_nanogpt_b200_run_speedrun.py` reported
`47 passed in 1.19s`; combined runner/summarizer plus compile/static reported
`90 passed in 1.26s` and `Static verification passed for 16 file(s).`

2026-08-19: Final non-launch foundation audit reconciled this ticket's
verification block with the later runtime-verifier evidence above. No new
training, preflight, CUDA, NCCL, data-prep, or summarizer run was performed for
that reconciliation.

2026-08-19: A later non-launch refresh attempt after embedding
`training_launch_allowed` in launch-readiness sidecars did not become a new
prerequisite row. The skip-run artifact
`experiments/modded_nanogpt_b200/results/lane_b_full_skiprun_runtime_allowed_refresh_20260819T000000Z/`
stalled in preflight before writing `preflight_report.json`; the owned process
was terminated before training, and `operator_stopped_preflight_stall.json`
marks the partial evidence as not a baseline and not launch readiness. The
runner now has a focused regression for preflight no-output timeout evidence and
process-group cleanup.

2026-08-19: The follow-up timeout-hardened non-launch refresh completed as a
structured preflight failure, not as a launch-prerequisite row:
`experiments/modded_nanogpt_b200/results/lane_b_full_skiprun_timeout_hardened_refresh_20260819T102351Z/`.
It records `exit_code.phase=preflight`, `exit_code=124`,
`blocker.message="preflight produced no output for 120 seconds"`,
`ready_to_launch=false`, `training_launched=false`, `skip_run=true`, and
embedded `runtime_verification.training_launch_allowed=true`. No training
process was launched. The current two-GPU retry remains sequentially blocked:
refresh prerequisite evidence must first get past preflight, and any non-skip
full launch still requires the trusted user request to contain
`launch-full-b200`.

2026-08-19: Follow-up hardening added a preflight progress sidecar for the next
retry. `run_speedrun.py` now passes `--progress
<result_dir>/preflight_progress.json` to preflight, `preflight.py` atomically
records the active check and completed checks, and no-output timeout blockers
use that sidecar to name the active check when available. Focused rootfs tests
cover both progress writing and timeout blocker enrichment.

2026-08-19: The fresh progress-localized non-launch refresh is
`experiments/modded_nanogpt_b200/results/lane_b_full_skiprun_progress_refresh_20260819T103735Z/`.
It stopped before training with `exit_code.phase=preflight`, `exit_code=124`,
`blocker.phase=nccl_all_reduce`, and
`blocker.message="preflight produced no output for 120 seconds while running nccl_all_reduce"`.
`preflight_progress.json` proves the active check was `nccl_all_reduce` and
the completed checks reached `torch_primitives`, including rootfs marker, pinned
Torch/CUDA/Triton versions, 2x B200 inventory, and torch primitive smoke. This
is not a launch-prerequisite row. The runner outer preflight no-output timeout
is now 240 seconds so preflight's own 180-second subprocess timeout can write a
structured report before the outer backstop fires.

2026-08-19: The next sequential non-launch refresh is
`experiments/modded_nanogpt_b200/results/lane_b_full_skiprun_nccl_timeout_refresh_20260819T104618Z/`.
It also stopped before training, but the failure is now owned by preflight
rather than the runner backstop: `exit_code.phase=preflight`, `exit_code=21`,
`blocker.phase=nccl_all_reduce`, and
`blocker.message="NCCL all-reduce smoke timed out after 90s"`.
`preflight_report.json` exists and records the failed check with
`expected_gpus=2`, `timeout_seconds=90`, and c10d/TCPStore output. The earlier
checks through `torch_primitives` passed. The captured error repeatedly says
`The IPv6 network addresses of (localhost, 33215) cannot be retrieved (gai
error: -3 - Temporary failure in name resolution)` and then `The client socket
has timed out after 60000ms while trying to connect to (localhost, 33215)`.
This points the next repair at NCCL preflight rendezvous setup, likely replacing
`torchrun --standalone` hostname rendezvous with an explicit IPv4 loopback
rendezvous. The artifact remains non-launch evidence and does not satisfy a
two-GPU trial, baseline, or launch-prerequisite row.

2026-08-19: The IPv4 rendezvous repair was verified by the next sequential
non-launch refresh:
`experiments/modded_nanogpt_b200/results/lane_b_full_skiprun_ipv4_nccl_refresh_20260819T105321Z/summary.json`.
It records `preflight_ok=true`, `preflight_returncode=0`,
`full_mode_gates.nccl_checked=true`, `data_manifest_checked=true`,
`verified_sha=true`, `optimized_kernel_certified=true`,
`ready_to_launch=true`, `blocked_by=[]`, `training_launched=false`, and
`skip_run=true`. The refreshed rootfs-generated run index reports
`total_attempts=62`, `baseline_stats.count=0`,
`len(launch_prerequisite_attempts)=3`, and `len(launch_ready_attempts)=0`.
This refresh restored intermediate prerequisite evidence and remains non-launch
evidence only; it is superseded for current handoff purposes by the strict
runtime-env refresh below. The next two-GPU full attempt still requires the
trusted user request to contain `launch-full-b200`.

2026-08-19: Tracker status reconciled with the current evidence. The two-GPU
trial preparation and prerequisite refresh are complete, but the ticket remains
blocked on explicit trusted launch authorization for the first non-skip full
attempt.

2026-08-19: A stricter runtime-env prerequisite refresh supersedes the
IPv4-only skip-run row as current handoff evidence:
`experiments/modded_nanogpt_b200/results/lane_b_full_skiprun_runtime_env_refresh_20260819T131652Z/summary.json`.
It did not launch training and remains prerequisite evidence only. It records
`ready_to_launch=true`, `training_launched=false`, `skip_run=true`,
`blocked_by=[]`, rootfs-critical command-environment fields covered by the
runtime-verification digest, and
`runtime_verification.training_launch_allowed=true`. A fresh rebuilt run index
reports `total_attempts=63`, `baseline_stats.count=0`,
`len(launch_prerequisite_attempts)=1`, and `len(launch_ready_attempts)=0`
under the stricter current runtime-evidence predicate.

The ticket remains blocked rather than open because the next required step is a
non-skip full training attempt and the trusted request still lacks
`launch-full-b200`. Do not run another equivalent skip-run refresh or readiness
audit only to reconfirm this blocked state.
