# Reproduction wrapper and log parser

Type: task
Status: ready-for-agent
Blocked by: 01, 02

## Requirement

Add the faithful upstream reproduction wrapper and parser for the Modded NanoGPT
B200 benchmark.

## Scope

Allowed:

- run upstream `run.sh` or equivalent `torchrun` through an experiment-local
  wrapper;
- capture complete logs and environment metadata;
- parse final validation loss, upstream `train_time`, `step_avg`, and CUDA
  memory lines;
- write a compact typed report.

Excluded:

- no upstream source edits for the Lane A reproduction path;
- no optimization ablations;
- no claim that shell wall-clock equals upstream train time;
- no long GPU run unless explicitly authorized for this ticket.

## Verification Evidence

- Parser test or fixture covering representative upstream log lines.
- Smoke wrapper run if GPU budget is authorized; otherwise exact blocker and
  dry-run evidence.
- Report clearly distinguishes smoke, upstream reproduction, compatibility
  patchset, and optimization ablation classifications.
