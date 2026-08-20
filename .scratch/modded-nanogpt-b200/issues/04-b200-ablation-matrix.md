# B200 ablation matrix

Type: task
Status: blocked
Blocked by: Lane A or Lane B non-skip baseline; active Lane B path requires
  trusted user request containing `launch-full-b200`

## Requirement

Add the B200 optimization matrix only after the faithful reproduction or minimal
compatibility baseline is available. The current RSI foundation path treats the
two-GPU Lane B FA2/Triton full attempt as the active minimal compatibility
baseline candidate, but that attempt has not launched yet because the trusted
request does not contain `launch-full-b200`.

Use `.scratch/modded-nanogpt-b200/spec.md` as the canonical spec.

## Scope

Allowed:

- compare B200 runtime stack candidates;
- compare Flash Attention 3 versus a documented fallback if FA3 is unstable;
- compare FP8 enabled versus `DISABLE_FP8=1` as a diagnostic arm;
- compare NCCL environment candidates if collective timing dominates;
- summarize median and best timings across replicated runs.
- run every arm through rootfs-aware harness wrappers.

Excluded:

- no train/validation token-stream changes for competition-comparable claims;
- no hidden ML-affecting changes;
- no banned upstream compile-policy changes for competition-comparable claims;
- no broad TorchTitan-native comparison in v1.
- no host-side Python, `torchrun`, parser, or summarizer execution for real GPU
  attempts.

## Verification Evidence

- Every arm labels itself as `competition-comparable`, `B200 systems-only`, or
  `B200 ML variant`.
- Every arm records the common schema fields, including `arm`,
  `environment_class`, `claim_label`, and `evidence_tier`.
- Every arm records source commit, patch diff if any, data manifest, hardware,
  environment, final validation loss, upstream train time, wall-clock breakdown,
  and memory evidence.
- The baseline artifact from Lane A or ticket 06 is named before any ablation
  launch.
- Claims below 3.28 validation loss are replicated enough for their stated
  evidence tier.

## Current State

Issue `04` remains intentionally blocked. The current strict prerequisite
artifact is
`experiments/modded_nanogpt_b200/results/lane_b_full_skiprun_runtime_env_refresh_20260819T131652Z/summary.json`;
it is a full-mode Lane B skip-run gate with `ready_to_launch=true`,
`training_launched=false`, `skip_run=true`, `blocked_by=[]`,
`runtime_verification.training_launch_allowed=true`, and the required
`launch_authorization_required_token=launch-full-b200`. The refreshed
`run_index.json` records `baseline_stats.count=0`, four skip-run
launch-prerequisite rows, and zero non-skip launch-ready rows.

Do not run ablation arms, matrix execution, or additional readiness refreshes
just to reconfirm this blocked state. The next material input is a successful
non-skip Lane A/B baseline; for the active two-GPU Lane B path, that requires
the trusted user request itself to contain `launch-full-b200`.
