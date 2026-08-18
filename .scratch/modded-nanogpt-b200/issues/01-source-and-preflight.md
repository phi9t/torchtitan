# Source checkout and B200 preflight

Type: task
Status: resolved
Blocked by: -

## Requirement

Create the first runnable slice of the Modded NanoGPT B200 benchmark harness:
pin or verify the upstream source and fail loudly before any long run if the
host/runtime cannot support the benchmark.

Use `.scratch/modded-nanogpt-b200/spec.md` as the canonical spec.

## Scope

Allowed:

- add gitignored experiment-local `sources/`, `data/`, and `results/` paths;
- add durable tracked `.gitignore` rules for generated experiment paths;
- add a source-fetch or source-verify script;
- add a preflight script;
- add rootfs-aware shell wrappers for source/preflight work;
- add minimal README usage for the preflight;
- run cheap rootfs preflight checks.

Excluded:

- no full training run;
- no FineWeb bulk download;
- no upstream source edits;
- no TorchTitan core code edits;
- no optimization ablations;
- no commit, push, or pull request unless explicitly authorized.
- no host-side Python for setup, preflight, CUDA probes, NCCL probes, parsing,
  or summary generation of real GPU work.

## Behavior

The preflight must verify or record:

- common schema fields: `schema_version`, `lane`, `mode`, `arm`,
  `claim_label`, `evidence_tier`, `run_id`, `attempt_id`, and
  `environment_class`;
- mode policy: full jobs require NCCL, 8x B200, full data manifest,
  `--verify-sha`, and no known-stall overrides;
- upstream source path and commit, expected
  `ecbb586296d3dac36fd206211f25d63bad4a6b35`;
- 8 visible CUDA devices;
- every visible device matches the expected B200 device class unless overridden;
- PyTorch, CUDA, Triton, and NCCL version evidence where available;
- BF16 allocation smoke;
- FP8 tensor allocation and `torch._scaled_mm` smoke;
- Flash Attention 3 import and tiny varlen smoke, or a clear unsupported status
  for an explicit fallback arm;
- 8-rank NCCL all-reduce smoke, unless the script is in a documented dry-run
  mode;
- repo-local cache and compiler settings.
- `TORCHTITAN_IN_ROOTFS=1` before Python or CUDA code runs.
- rootfs sentinel evidence: `/workspace/torchtitan`, rootfs entrypoint path,
  `sys.executable`, and `sys.prefix`.
- `--skip-nccl` is diagnostic-only and fails closed unless explicitly
  acknowledged.
- subprocess probes use timeouts and preserve JSON failure reports on runtime
  exceptions, malformed manifests, wrong source commits, missing Torch/Triton,
  Git failures, FA failures, and NCCL failures.

## Verification Evidence

- Run the preflight on the current host or document the exact blocker.
- Show that host-launched wrappers re-enter the bwrap rootfs.
- Show that source verification fails on a wrong commit, or cover it with a
  focused unit/helper test.
  - `python -m pytest tests/unit_tests/test_modded_nanogpt_b200_fetch_upstream.py -q`
    passed for clean-source `source.json`, dirty-source refusal, and missing
    source clone from a local remote.
- `experiments/modded_nanogpt_b200/fetch_upstream.sh` re-enters
  `scripts/rootfs/enter_rootfs.sh` before invoking source helper Python.
- Show that direct host invocation of `preflight.py` fails closed.
- Show that `--skip-nccl` cannot pass outside `mode=diagnostic` plus explicit
  override.
- Show malformed and incomplete data manifests fail with a preserved report.
- MLP backend gates preserve structured detail in preflight reports: PyTorch
  fallback records the local-smoke output shape and full-mode policy block,
  while Triton now runs a bounded local forward/backward smoke and records
  `local_smoke.output_shape=[2, 16, 768]` when passing. Historical Triton
  failures still preserve the sm100 `linear_relu_square_kernel` compile blocker
  and `TritonNvidiaGPUOptimizeTMemLayoutsPass`.
- `experiments/modded_nanogpt_b200/diagnose_mlp_backend.sh` re-enters
  `scripts/rootfs/enter_rootfs.sh` before running the bounded MLP backend
  diagnostic. It records rootfs and environment details, sets the B200 CE
  compile capability for Triton probes, and writes an ignored JSON report.
  Unit coverage verifies direct-script import, rootfs failure reporting,
  subprocess failure reporting, and `triton_backward_shape_mismatch`
  classification, and now verifies that the standalone diagnostic delegates to
  the shared preflight Triton smoke helper instead of carrying duplicate probe
  code. The current bounded Triton diagnostic passes at
  `experiments/modded_nanogpt_b200/results/mlp_diag_triton_shared_20260816T005734Z/mlp_diagnostic.json`.
- Inspect the diff to confirm no generated source, data, logs, or caches are
  tracked.
- `git check-ignore -v` now also confirms `.scratch/trae-pytest-tmp/` is
  ignored, matching the repo-local pytest `--basetemp` used to avoid `/tmp`
  space failures during focused harness verification.
- Phase 1 process hygiene now uses the rootfs-aware structured scanner
  `experiments/modded_nanogpt_b200/check_active_jobs.sh` instead of raw
  `pgrep`. The scanner writes schema-v1 JSON, filters shell/search false
  positives where the command line merely contains
  `torchrun|train_gpt.py|cached_fineweb10B.py`, matches real target commands by
  process-token basename instead of arbitrary substrings, and returns exit `21`
  if a real training or data-prep process is active. Focused coverage now
  ignores substring lookalikes such as `nottrain_gpt.py`,
  `precached_fineweb10B.py`, and `mytorchrun_helper` while preserving real
  `torchrun` and `cached_fineweb10B.py` matches. Fresh rootfs evidence at
  `experiments/modded_nanogpt_b200/results/phase1_active_jobs_launch_guard_20260816T030725Z/active_jobs.json`
  records `ok=true`, `active_job_count=0`, `ignored_match_count=0`, and
  `active_jobs=[]`. A focused runner regression now proves the same scanner is
  enforced after preflight and before telemetry/training for authorized full
  launches, with
  `blocker.phase=active_jobs` and `exit_code.phase=active_jobs` when another
  matching job is active.
- Python CLIs now fail closed when invoked directly on the host. The shared
  `experiments/modded_nanogpt_b200/cli_guard.py` path is wired through
  `run_speedrun.py`, `parse_log.py`, `summarize.py`, `prepare_data.py`,
  `fetch_upstream.py`, `preflight.py`, and `diagnose_mlp_backend.py` from their
  `__main__` entrypoints. Direct host execution exits `21` before work if
  `TORCHTITAN_IN_ROOTFS=1` is missing, while imported helper APIs and unit tests
  remain usable.
- Shell wrappers now source `experiments/modded_nanogpt_b200/rootfs_guard.sh`
  after the rootfs re-entry branch. This blocks forged host-side
  `TORCHTITAN_IN_ROOTFS=1` markers by requiring `cwd=/workspace/torchtitan` and
  the rootfs sentinel before any wrapper-owned work such as package
  installation, preflight, parsing, summarization, or diagnostics can run.
