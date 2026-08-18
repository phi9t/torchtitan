# Issue 09: Two-GPU Trial Retro and Next Iteration

Type: task
Status: open
Blocked by: explicit user authorization for non-skip launch

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
- The non-skip full launch did not start because the trusted request did not
  include `--launch-authorization=launch-full-b200`.
- The harness still needed code alignment with the new policy before another
  run: preflight, runner, and summarizer had active 8-GPU assumptions.

## Completed Preparation

- `preflight.py` full mode now expects exactly two B200 GPUs.
- `run_speedrun.py` now passes `--expected-gpus 2` to preflight and launches
  `torchrun --nproc_per_node=2`.
- `summarize.py` launch-readiness and baseline GPU evidence gates now require
  the active two-GPU B200 allocation instead of eight GPUs.
- Focused host-side tests cover the updated preflight, runner, and summarizer
  behavior.

## Next Iteration Gates

Before spending GPU time again:

- use a fresh `run_id` and result directory; do not reuse the prelaunch-only
  directory from this issue;
- select exactly two idle B200 GPUs, normally the lowest-index idle pair;
- set `CUDA_VISIBLE_DEVICES=<two selected host GPU indexes>` on the
  rootfs-aware wrapper invocation;
- rerun `check_active_jobs.sh --active-jobs-output <result_dir>/active_jobs_prelaunch.json`;
- rerun the focused host-side tests and static verifier if the harness changed;
- require the explicit trusted authorization token
  `--launch-authorization=launch-full-b200` for any non-skip full launch.

## Verification

Completed in this retro:

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

Still required before handoff:

```bash
python3 experiments/modded_nanogpt_b200/verify_static.py
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
trusted request includes `--launch-authorization=launch-full-b200` for a
non-skip full run.
