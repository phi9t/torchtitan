# Agentic Execution Trace Review

Date: 2026-08-11

## Scope

This review covers the agentic execution trace for the Countdown
scaffold-to-policy run: runtime repair, smoke, reduced pilot, full pilot, LoRA
export, adapter evaluation, and reporting.

## What Worked Well

The rootfs constraint was eventually enforced end to end. Dependency repair,
asset download, validation, vLLM generation, TorchTitan training, LoRA export,
adapter evaluation, and summary checks all ran through
`scripts/rootfs/enter_rootfs.sh -- ...`. This matched the user's hermeticity
requirement and avoided mixing host Python state with the experiment runtime.

The smoke -> reduced -> full progression caught real issues before the final
evidence pass. Smoke exposed runtime and checkpoint-load plumbing. Reduced mode
validated calibration, data coverage, and training. The first full attempt then
surfaced an output-scoping bug before adapter evaluation, and the corrected full
run produced fresh mode-scoped checkpoints.

The agent did not pass TorchTitan internal checkpoints directly to vLLM. It
stopped at the missing adapter-export boundary, inspected the checkpoint format,
implemented an experiment-local PEFT/vLLM exporter, and added unit tests for the
fragile mapping logic. That was the right risk boundary for scientific evidence.

Failures were handled by root cause rather than by retrying the same command.
The trace shows targeted fixes for:

- Missing rootfs dependencies and Qwen3 assets.
- Host-to-rootfs launcher argument handling in `run_common.sh`.
- Full-mode training outputs colliding with reduced-mode checkpoint folders.
- vLLM LoRA rank defaulting to 16 while the trained adapters used rank 32.
- FlashInfer JIT/toolkit incompatibility avoided by the experiment's vLLM
  defaults rather than by changing unrelated CUDA state.
- `safetensors` shared-storage rejection for copied Q/K/V LoRA A tensors.

The final result was checked structurally, not just by file existence. Every
base and adapter summary was validated for expected split size and exactly 32
rollouts per problem before the results were recorded.

## What Needs To Improve

Hermeticity should be a hard preflight, not an emergent discipline. Early
commands and checks had to be steered back into rootfs execution. Future
Countdown runners should include a single `run_preflight.sh` that verifies
`TORCHTITAN_IN_ROOTFS=1`, import availability, asset presence, GPU visibility,
vLLM backend defaults, and LoRA rank compatibility before any expensive stage.

The rootfs self-reexec path should have had a regression test. The missing `--`
in `run_common.sh` was small but blocked host-launched scripts. Add a shell test
or lightweight bats-style check that invokes each experiment script from the
host with a harmless command path and confirms the rootfs wrapper receives the
script arguments correctly.

Mode-scoped outputs should have been designed before the full run. Reusing
reduced checkpoint directories in full mode wasted GPU time and made artifact
provenance harder to reason about. Training, export, and eval paths should all
include `{mode}/{arm}` in the default path contract, with preflight refusing to
overwrite a checkpoint directory from a different mode unless explicitly asked.

Adapter export should be part of the experiment contract, not an afterthought.
The plan correctly warned not to use internal checkpoints as `LORA_ADAPTER`, but
the scaffold lacked a validated export step. The exporter is now present; the
next cleanup should wire it into `run_full_pilot.sh` or a documented
`run_export_adapters.sh`, then run adapter eval as a first-class stage.

Long-running GPU loops need better progress and resumability. The 12-arm
adapter eval loop was correct, but it was a monolithic shell loop. A small
manifest-driven runner should skip completed valid summaries, log start/end
times per split/arm, and fail with a concise status table. That would make
resuming after interruption safer and reduce manual process polling.

The trace-review evidence extraction was too noisy. Searching session artifacts
without tight file filters pulled in unrelated generated data. Future trace
reviews should use a small extractor that reads JSONL session events and emits
only user messages, agent updates, tool calls, tool exits, and patch summaries.

## Process Changes For The Next Run

1. Add an experiment preflight that must pass before smoke, reduced, full, export,
   or eval.
2. Promote LoRA export and adapter eval into scripted stages with resumable
   manifest semantics.
3. Make output paths mode-scoped by construction and refuse ambiguous reuse.
4. Add focused tests for host-to-rootfs re-exec and vLLM LoRA config defaults.
5. Record per-stage command, start time, end time, return code, and artifact
   paths in a machine-readable run manifest.

## Follow-Up Implemented

The next runner pass encoded the most important process fixes directly into the
Countdown experiment surface:

- `run_preflight.sh` now verifies rootfs activation, required imports, Qwen3
  assets, GPU availability, and vLLM FlashInfer-safe defaults before expensive
  work.
- `run_export_adapters.sh` exports reduced/full mode-scoped TorchTitan LoRA
  checkpoints to PEFT/vLLM adapter directories and skips already valid exports
  unless `FORCE=1`.
- `run_eval_adapters.sh` evaluates every exported adapter for the requested
  split/arm matrix, skips completed summaries by default, and writes a
  machine-readable matrix decision.
- `run_full_pilot.sh` now includes runtime preflight plus adapter export and
  adapter evaluation for non-smoke modes.
- `validate-eval-matrix` checks every expected summary for split size, rollout
  count, pass@k curve coverage, and bucket totals.
- `run_full_pilot.sh` writes a JSONL stage manifest with command argv, start and
  end times, duration, and return code for each canonical stage.

The full-run artifacts passed the new checks:

- `experiments/countdown_search_distill/results/runtime_preflight.json`
- `experiments/countdown_search_distill/results/eval/adapter_matrix_full.json`

## Bottom Line

The execution succeeded and produced real scaffold-to-policy evidence, but it
required several mid-run repairs that should become preflight checks and
first-class runner stages. The main improvement is to move the learned runtime
contracts from the agent's working memory into the experiment's executable
surface.
