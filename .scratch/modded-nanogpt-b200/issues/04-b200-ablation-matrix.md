# B200 ablation matrix

Type: task
Status: ready-for-agent
Blocked by: 03

## Requirement

Add the B200 optimization matrix only after the faithful reproduction or minimal
compatibility baseline is available.

## Scope

Allowed:

- compare B200 runtime stack candidates;
- compare Flash Attention 3 versus a documented fallback if FA3 is unstable;
- compare FP8 enabled versus `DISABLE_FP8=1` as a diagnostic arm;
- compare NCCL environment candidates if collective timing dominates;
- summarize median and best timings across replicated runs.

Excluded:

- no train/validation token-stream changes for competition-comparable claims;
- no hidden ML-affecting changes;
- no banned upstream compile-policy changes for competition-comparable claims;
- no broad TorchTitan-native comparison in v1.

## Verification Evidence

- Every arm labels itself as `competition-comparable`, `B200 systems-only`, or
  `B200 ML variant`.
- Every arm records source commit, patch diff if any, data manifest, hardware,
  environment, final validation loss, upstream train time, wall-clock breakdown,
  and memory evidence.
- Claims below 3.28 validation loss are replicated enough for their stated
  evidence tier.
