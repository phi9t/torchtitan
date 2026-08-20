# NanoGPT performance probe ladder

Type: task
Status: complete
Blocked by: -
Blocks: 06, 09, 04

## Requirement

Implement a diagnostic-only performance probe ladder for the latest
B200-compatible no-output failure. The probes must explain whether the launched
attempt timed out because of a true hang, silent compile or kernel warmup, data
or runtime setup latency, or observability overhead before first training
output.

Use `.scratch/modded-nanogpt-b200/spec.md` as the canonical spec. This ticket is
not a Lane C ablation and cannot produce baseline or reproduction evidence.

## Current Evidence

- Latest launched B200-compatible attempt:
  `experiments/modded_nanogpt_b200/results/lane_b_full_20260818T014833Z_attempt_001/`
- Blocker: `training produced no output for 600 seconds`
- Blocker phase: `stall`
- Current baseline count: `0`
- Current policy: active non-diagnostic attempts use exactly two visible B200
  GPUs and still require the trusted user request to contain
  `launch-full-b200`.
- Latest completed probe ladder:
  - static wrapper/preflight:
    `experiments/modded_nanogpt_b200/results/issue14_diag_static_wrapper_preflight_20260819T073828Z_attempt_001/`
  - import/construction:
    `experiments/modded_nanogpt_b200/results/issue14_diag_import_construction_20260819T073828Z_attempt_001/`
  - one-GPU microstep, selected Triton tuple:
    `experiments/modded_nanogpt_b200/results/issue14_diag_microstep_1gpu_triton_20260819T073926Z_attempt_001/`
  - two-GPU microstep, selected Triton tuple:
    `experiments/modded_nanogpt_b200/results/issue14_diag_microstep_2gpu_triton_20260819T073926Z_attempt_001/`
  - watchdog heartbeat:
    `experiments/modded_nanogpt_b200/results/issue14_diag_watchdog_heartbeat_20260819T073828Z_attempt_001/`
  - observability overhead:
    `experiments/modded_nanogpt_b200/results/issue14_diag_observability_overhead_20260819T073828Z_attempt_001/`
- Diagnostic fallback blocker:
  - one-GPU torch MLP:
    `experiments/modded_nanogpt_b200/results/issue14_diag_microstep_1gpu_20260819T073828Z_attempt_001/`
  - two-GPU torch MLP:
    `experiments/modded_nanogpt_b200/results/issue14_diag_microstep_2gpu_20260819T073828Z_attempt_001/`
  - both record `CUDA error: no kernel image is available for execution on the
    device` and remain diagnostic-only evidence, not selected-tuple blockers.

## Scope

Allowed:

- add rootfs-aware wrappers for performance probes;
- add Python probe entrypoints under `experiments/modded_nanogpt_b200/`;
- add schema validation or parser/summarizer support for probe artifacts;
- run static, non-GPU, and GPU diagnostic probes through rootfs-aware wrappers;
- use `--allow-previous-stall` only for a diagnostic arm that revisits the named
  no-output blocker;
- capture monotonic phase timings, first CUDA operation time, compile/warmup
  timings when observable, max GPU memory, artifact sizes, and blocker phases;
- compare Tier 0 observability against the selected richer profile on the same
  diagnostic workload.

Excluded:

- no non-skip claim-bearing full baseline launch unless the trusted user
  request contains `launch-full-b200`;
- no Lane C optimization matrix;
- no broad runtime-stack search;
- no train/validation token-stream changes;
- no source, model, optimizer, schedule, validation-count, or compile-policy
  changes unless the diagnostic arm name declares that exact variable and the
  result remains `claim_eligible=false`;
- no host-side Python, CUDA, NCCL, parser, summarizer, package, or training
  execution.

## Probe Ladder

Implement the ladder in this order. Later probes depend on earlier gates being
fresh or explicitly recorded as diagnostic-only.

1. **Static wrapper/preflight timing**
   - rootfs entry;
   - source verification;
   - manifest validation;
   - SHA verification;
   - NCCL check;
   - active-job scan;
   - environment capture.

2. **Import and construction timing**
   - torch import;
   - CUDA context creation;
   - distributed initialization;
   - model import;
   - model construction;
   - selected attention and MLP backend imports.

3. **One-GPU microstep timing**
   - first forward;
   - first backward;
   - optimizer step;
   - first compile trigger if present;
   - steady microstep median after warmup if reached.

4. **Two-GPU microstep timing**
   - same phase timings as the one-GPU probe;
   - declared two-rank NCCL world size;
   - exactly two visible B200 GPUs.

5. **Watchdog heartbeat calibration**
   - structured heartbeat before compile;
   - structured heartbeat during warmup;
   - time to first train-like output;
   - clear classification of timeout versus forward progress.

6. **Observability overhead control**
   - same diagnostic workload under Tier 0;
   - same diagnostic workload under the selected richer profile;
   - artifact bytes, phase timing deltas, and missing evidence listed
     separately.

## Artifact Contract

Every probe writes an immutable result directory under the ignored
`experiments/modded_nanogpt_b200/results/` tree. At minimum, each result
contains:

- `probe.json`: schema-versioned probe metadata and phase timings;
- `attempt.json`: common classification fields with `claim_eligible=false`;
- `command.argv.json`: replayable rootfs-aware wrapper command with a SHA256
  digest;
- `command.env.json`: redacted environment metadata with a SHA256 digest;
- `summary.json`: parser/summarizer output or an explicit missing-summary
  blocker.

`probe.json` must include:

- `schema_version`;
- `probe_name`;
- `probe_kind`;
- `run_id`;
- `attempt_id`;
- `mode=diagnostic`;
- `claim_eligible=false`;
- source commit and source variant;
- data manifest pointer;
- attention backend;
- MLP backend;
- visible GPU IDs;
- declared world size;
- monotonic `start_ns` and `end_ns` per phase;
- first CUDA operation time when CUDA runs;
- first compile time when observable;
- first training-output time when reached;
- steady microstep median when reached;
- max GPU memory when CUDA runs;
- result artifact bytes;
- blocker phase and message when blocked.

## Acceptance Criteria

- [x] A clean-context implementer can run static probe validation without GPUs.
- [x] GPU probes refuse to run outside the rootfs.
- [x] Probe artifacts are classified as diagnostic and never counted in
  baseline statistics.
- [x] The two-GPU microstep probe uses exactly two visible B200 GPUs and a
  matching two-rank NCCL world size.
- [x] The watchdog calibration can distinguish no progress from long silent
  compile/warmup progress using structured heartbeats.
- [x] The observability overhead control reports timing deltas and artifact
  bytes separately from training metrics.
- [x] The final report recommends one of:
  - keep the 600-second watchdog;
  - raise the watchdog with measured compile/warmup evidence;
  - repair a specific backend, data, runtime, or observability issue;
  - stop because the selected tuple cannot pass the two-GPU microstep
    prerequisite.

Final recommendation: proceed only with the selected FA2/Triton tuple for the
next authorized two-GPU full attempt. The selected tuple passed the one-GPU
synthetic CUDA microstep, Triton backend smoke, two-rank NCCL smoke, and
two-GPU microstep prerequisite. Preserve the torch MLP fallback failure as a
diagnostic blocker and do not use it for the next full launch.

## Verification Evidence

Run inside rootfs:

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 experiments/modded_nanogpt_b200/verify_static.py'
```

Run focused unit tests inside rootfs after implementation:

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 -m pytest -q tests/unit_tests/test_modded_nanogpt_b200_performance_probe.py tests/unit_tests/test_modded_nanogpt_b200_summarize.py tests/unit_tests/test_modded_nanogpt_b200_run_experiment_matrix.py'
```

GPU diagnostic probes require fresh active-job evidence and must be launched
sequentially. Do not run them merely because this ticket exists; run them only
when the user has authorized the diagnostic GPU budget or when the current
trusted task explicitly asks for those probes.
