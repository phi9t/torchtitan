# Modded NanoGPT B200 Speedrun Harness

Status: ready-for-implementation

## Intent

Build a repo-local harness for running
`kellerjordan/modded-nanogpt` on the local B200 host while preserving the
upstream speedrun claim boundaries. The active launch policy is a small-scale
two-GPU trial, not an 8-GPU reproduction campaign. The harness must make three things
separate and auditable:

- the preflight checklist verifier;
- the speedrun execution path for upstream or B200-compatible sources;
- the result harness that prepares source/data, launches jobs, parses logs, and
  classifies claims.

The canonical implementation home is `experiments/modded_nanogpt_b200/`.
Generated source snapshots, data, caches, logs, and result bundles stay under
ignored generated paths. Reusable TorchTitan library code is out of scope unless
a later ticket separately justifies moving logic into `torchtitan/experiments/`
or core.

## Primary Sources

Authority order:

1. This file is the canonical implementation spec and ticket source of truth.
2. `experiments/modded_nanogpt_b200/preflight_checklist.md` is the executable
   launch checklist and mode policy for operators.
3. `experiments/modded_nanogpt_b200/upstream_reference.md` is the pinned
   upstream fact source.
4. `experiments/modded_nanogpt_b200/benchmark_design.md` is historical design
   context and is superseded wherever it conflicts with this spec.

Current implementation files:

- `experiments/modded_nanogpt_b200/preflight.py`
- `experiments/modded_nanogpt_b200/run_preflight.sh`
- `experiments/modded_nanogpt_b200/setup_flash_attention.sh`
- `experiments/modded_nanogpt_b200/parse_log.py`
- `experiments/modded_nanogpt_b200/parse_log.sh`
- `experiments/modded_nanogpt_b200/summarize.py`
- `experiments/modded_nanogpt_b200/summarize.sh`
- `experiments/modded_nanogpt_b200/verify_static.py`
- `experiments/modded_nanogpt_b200/run_speedrun.py`
- `experiments/modded_nanogpt_b200/run_speedrun.sh`
- `experiments/modded_nanogpt_b200/check_active_jobs.sh`
- `experiments/modded_nanogpt_b200/fetch_upstream.py`
- `experiments/modded_nanogpt_b200/fetch_upstream.sh`
- `experiments/modded_nanogpt_b200/prepare_data.py`
- `experiments/modded_nanogpt_b200/prepare_data.sh`
- `AGENTS.md`, which is a symlink to `.claude/CLAUDE.md`

Pinned upstream:

- repository: `https://github.com/kellerjordan/modded-nanogpt`
- commit: `ecbb586296d3dac36fd206211f25d63bad4a6b35`

## Classification

Route: gated path.

Reason:

- external source repository;
- real B200 runtime with explicit GPU allocation evidence;
- rootfs execution boundary;
- benchmark and validation-loss claims;
- source, data, distributed-runtime, timing, and hardware evidence;
- multi-ticket implementation with blocker edges.

Surface: repo-local research program. This effort must not change TorchTitan
core trainer semantics. TorchTitan core changes require a new design and a new
ticket.

## Authority Boundary

This spec authorizes bounded repo-local harness implementation and focused
verification. It does not authorize:

- committing, pushing, or opening a pull request;
- merging any branch;
- running a full long GPU job unless the user explicitly asks for that run;
- tracking generated source, data, logs, caches, or result bundles;
- modifying upstream source for Lane A;
- changing TorchTitan core training behavior;
- claiming a successful reproduction without final validation evidence.

Full source/data preparation and short preflight/diagnostic GPU probes are
allowed when they run through the bwrap rootfs and preserve generated artifacts
under ignored paths.

Blocked-state policy: the current harness state is blocked on explicit
`launch-full-b200` authorization. Until that authorization is supplied or a
material input changes, do not re-run equivalent clean-context audits,
summarizer refreshes, preflights, GPU probes, or launch-readiness checks merely
to reconfirm the same blocked state. Future agents should cite the latest
recorded evidence and stop at the authorization boundary.

## Rootfs Contract

Every real execution step must run under `scripts/rootfs/enter_rootfs.sh`.

Host shell may do only orchestration:

- create ignored result directories;
- set environment variables for wrappers;
- call repo-local shell wrappers;
- inspect git status and generated logs.

The following must happen only after `TORCHTITAN_IN_ROOTFS=1` is present:

- `python`;
- `pip install`;
- `torchrun`;
- HuggingFace or dataset download code;
- CUDA, BF16, FP8, FlashAttention, Triton, or NCCL probes;
- training and validation;
- log parsing;
- summary generation.

Every new executable script in this experiment must use this pattern:

```bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SCRIPT_REL="experiments/modded_nanogpt_b200/<script>.sh"

if [[ "${TORCHTITAN_IN_ROOTFS:-0}" != "1" ]]; then
  exec "${REPO_ROOT}/scripts/rootfs/enter_rootfs.sh" -- "${SCRIPT_REL}" "$@"
fi

source "${SCRIPT_DIR}/rootfs_guard.sh"
require_modded_nanogpt_rootfs "${SCRIPT_REL}"
```

The current approved wrappers are:

- `experiments/modded_nanogpt_b200/check_active_jobs.sh`
- `experiments/modded_nanogpt_b200/diagnose_mlp_backend.sh`
- `experiments/modded_nanogpt_b200/fetch_upstream.sh`
- `experiments/modded_nanogpt_b200/parse_log.sh`
- `experiments/modded_nanogpt_b200/prepare_data.sh`
- `experiments/modded_nanogpt_b200/run_preflight.sh`
- `experiments/modded_nanogpt_b200/run_speedrun.sh`
- `experiments/modded_nanogpt_b200/setup_flash_attention.sh`
- `experiments/modded_nanogpt_b200/summarize.sh`

Future wrappers must follow the same shape before they run Python or CUDA work.

Wrappers and verifiers must prove the rootfs boundary rather than trusting only
the marker variable. Shell wrappers source `rootfs_guard.sh` so a forged host
`TORCHTITAN_IN_ROOTFS=1` marker fails before wrapper-owned work such as package
installation, data preparation, parsing, summarization, or diagnostics. Python
entrypoints call `cli_guard.py` from `__main__` so direct host
`python experiments/modded_nanogpt_b200/<tool>.py ...` also fails closed before
real work. At minimum, rootfs-executed Python records:

- `TORCHTITAN_IN_ROOTFS=1`;
- `Path.cwd() == /workspace/torchtitan`;
- `/workspace/torchtitan/scripts/rootfs/enter_rootfs.sh` exists;
- `sys.executable` and `sys.prefix`.

## Lane Model and Claim Boundary

Every attempt uses this common classification schema in all run-level artifacts
that can represent an attempt: `preflight_report.json`, `attempt.json`, and
`summary.json`.

Attempt metadata must not preserve operator-supplied authorization values. For
full-launch authorization, `attempt.json` records only
`command.launch_authorization_present`; the explicit required token belongs in
`launch_readiness.json` and `launch_readiness.md` so operators can see the gate
without echoing a supplied token back into immutable attempt metadata.

```json
{
  "schema_version": 1,
  "classification": {
    "lane": "A|B|C",
    "mode": "full|smoke|diagnostic",
    "arm": "A0|B0|B1|...",
    "claim_label": "B200 upstream reproduction|B200 compatibility patchset|B200 systems-only|B200 ML variant|smoke|diagnostic",
    "evidence_tier": "full-single-attempt|smoke|diagnostic",
    "run_id": "...",
    "attempt_id": "...",
    "environment_class": "torchtitan-rootfs-b200",
    "claim_eligible": false
  }
}
```

`claim_eligible` means only that the attempt is structurally allowed to support
a claim. Post-run validity still requires final validation, valid source state,
valid data, and successful full-job preflight.

Mode policy:

- `full`: currently means a two-GPU small-scale B200 trial. It requires exactly
  two visible B200 GPUs, NCCL over the declared two-rank world size, full 900M
  FineWeb manifest, SHA verification, no known-stall overrides, sequential
  launch hygiene, and lane-specific source validity. Historical 8-GPU artifact
  requirements remain historical evidence only unless the user separately
  authorizes an 8-GPU reproduction campaign.
- `smoke`: may use a smoke data manifest, but must label the artifact `smoke`
  and cannot support a validation-loss or timing claim.
- `diagnostic`: may revisit known blockers, skip NCCL only with
  `MODDED_NANOGPT_ALLOW_SKIP_NCCL=1`, and must never enter successful baseline
  statistics.

### Lane A: Faithful Upstream B200 Reproduction

Allowed:

- clone or check out the pinned upstream source under an ignored source path;
- set rootfs/cache/compiler variables outside upstream;
- run wrapper-side preflight, logging, timing, and metadata capture;
- install missing non-Torch dependencies inside rootfs.

Disallowed:

- any upstream source edit;
- any train or validation data-stream change;
- any validation-token-count change;
- model, optimizer, schedule, precision-semantics, compile-policy, or kernel
  change;
- fallback attention backend while calling the run Lane A.

Success:

- source remains clean before and after the run;
- training reaches final validation;
- final `val_loss <= 3.28`;
- summary reports upstream log `train_time` separately from shell wall-clock;
- result label is `B200 upstream reproduction`, not an official upstream record
  unless upstream record rules are separately satisfied.

### Lane B: Minimal B200 Compatibility Patchset

Allowed:

- B200 environment adapter work;
- source-local B200 compatibility patches in a copied/generated source tree;
- kernel-compatibility patches justified by concrete Lane A failures;
- wrapper-side metadata and preflight additions.

Required:

- source descends from the pinned upstream commit;
- every source difference is saved as `variant_patch.diff`;
- every patch is classified as `environment`, `hardware-detection`,
  `kernel-compat`, `timing-harness`, or `ML-affecting`;
- every patch records the first Lane A blocker it addresses.

Disallowed for a source-equivalent claim:

- model, optimizer, schedule, train data, validation data, validation count, or
  banned compile-policy changes.

Success:

- full run reaches final validation;
- final `val_loss <= 3.28`;
- result label is `B200 compatibility patchset` or `B200 systems-only`.
- if any ML-affecting patch exists, result label is `B200 ML variant`.

### Lane C: B200 Optimization Ablations

Allowed only after a baseline exists:

- runtime stack comparisons;
- FA3 versus FA2 fallback comparisons;
- FP8 enabled versus `DISABLE_FP8=1` diagnostic comparisons;
- NCCL environment comparisons;
- compile-cache cold versus warm comparisons.

Required:

- one variable changes per arm;
- baseline arm is named before launch;
- every arm records source, patch diff, data manifest, environment, hardware,
  final validation loss, upstream `train_time`, shell wall-clock, and memory;
- every arm is classified as `competition-comparable`, `B200 systems-only`, or
  `B200 ML variant`.

## Artifact Layout

Tracked source and docs:

```text
experiments/modded_nanogpt_b200/
  benchmark_design.md
  upstream_reference.md
  preflight_checklist.md
  cli_guard.py
  rootfs_guard.sh
  preflight.py
  run_preflight.sh
  setup_flash_attention.sh
  fetch_upstream.py
  fetch_upstream.sh
  prepare_data.py
  prepare_data.sh
  run_speedrun.py
  run_speedrun.sh
  parse_log.py
  parse_log.sh
  summarize.py
  summarize.sh
  verify_static.py
  check_active_jobs.sh
```

Generated and ignored:

```text
experiments/modded_nanogpt_b200/sources/
experiments/modded_nanogpt_b200/data/
experiments/modded_nanogpt_b200/results/
.cache/huggingface/
```

Every run attempt has one immutable result directory:

```text
results/<run-id>/
  attempt.json
  preflight_report.json
  hardware.json
  environment.json
  source.json
  data_manifest.json            # copy or pointer record
  variant_patch.diff            # Lane B/C only
  command.env
  command.argv                  # replayable run_speedrun.sh invocation
  run.log
  summary.json
```

`attempt.json["command"]["argv"]` mirrors the replayable wrapper invocation in
`command.argv`. `attempt.json["command"]["training_argv"]` records the inner
`torchrun` command that will run from the selected source directory after
preflight and launch authorization gates pass.
`command.env` and `attempt.json["environment"]` persist only redacted
environment metadata. The runner still passes the real environment to preflight
and training subprocesses, but artifacted metadata replaces key, token, secret,
password, credential, and authorization-like values with `<REDACTED>`.
`data_manifest.json` is a result-local pointer record for the manifest supplied
to the attempt, so the bundle remains self-describing without copying or
mutating a reused generated manifest. Parser summaries resolve this pointer to
report the underlying manifest fields while preserving the result-local pointer
path.
After preflight runs, `environment.json` records a schema-v1
`preflight_environment` sidecar copied from `preflight_report.json["environment"]`,
and `hardware.json` records a schema-v1 `preflight_gpus` sidecar copied from
`preflight_report.json["gpus"]`. Parser summaries preserve those sidecars under
`environment_sidecar` and `hardware_sidecar`, prefer them for the top-level
environment/GPU fields, and render their kinds in `analysis.md`. Older bundles
without sidecars fall back to the preflight report fields.

Do not overwrite a prior attempt directory. Reuse requires a new run ID or a new
attempt ID.

## Preflight Checklist Verifier Spec

Current files:

- `preflight.py`
- `run_preflight.sh`
- `setup_flash_attention.sh`
- `preflight_checklist.md`

### CLI

`run_preflight.sh` is the normal public entrypoint. It re-enters rootfs and then
execs `preflight.py`.

Required arguments:

- `--mode {full,smoke,diagnostic}`
- `--lane {A,B}`
- `--source PATH`
- `--data-manifest PATH`
- `--attention-backend {fa3,fa2,flex}`
- `--mlp-backend {triton,torch}`
- `--report PATH`

Optional arguments:

- `--run-id STR`
- `--attempt-id STR`
- `--arm STR`
- `--claim-label STR`
- `--evidence-tier STR`
- `--environment-class STR`, default `torchtitan-rootfs-b200`
- `--expected-gpus INT`, default `8`
- `--expected-name STR`, default `B200`
- `--verify-sha`
- `--allow-previous-stall`, diagnostic only
- `--skip-nccl`, blocked by `run_preflight.sh` unless
  `MODDED_NANOGPT_ALLOW_SKIP_NCCL=1`

Exit codes:

- `0`: all requested gates passed
- `21`: preflight gate failed or diagnostic override required
- other nonzero: tool/runtime error that should be captured as a blocker

### Report JSON

The report must be written even on gate failure when possible:

```json
{
  "schema_version": 1,
  "ok": false,
  "classification": {
    "lane": "A",
    "mode": "full",
    "arm": "A0",
    "claim_label": "B200 upstream reproduction",
    "evidence_tier": "full-single-attempt",
    "run_id": "...",
    "attempt_id": "...",
    "environment_class": "torchtitan-rootfs-b200",
    "claim_eligible": false
  },
  "checks": [
    {
      "name": "rootfs",
      "ok": true
    }
  ],
  "failures": [
    {
      "name": "attention_backend",
      "error": "..."
    }
  ],
  "environment": {
    "python": "...",
    "torch": "...",
    "cuda_runtime": "...",
    "triton": "..."
  },
  "gpus": [
    {
      "index": 0,
      "name": "NVIDIA B200",
      "compute_capability": [10, 0]
    }
  ]
}
```

Required check names:

- `mode_policy`
- `rootfs`
- `torch_import`
- `gpu_inventory`
- `torch_primitives`
- `nccl_all_reduce`
- `source_policy`
- `attention_backend`
- `mlp_backend`
- `data_manifest`

Every check must be one of:

- `ok: true`
- `ok: false` plus `error`
- `ok: "skipped"` only when a diagnostic override was explicit

### Required Gates

Rootfs:

- `TORCHTITAN_IN_ROOTFS=1`;
- current directory is `/workspace/torchtitan`;
- rootfs workspace sentinel exists;
- rootfs Python details are recorded;
- host-side Python is never used for this verifier.

GPU inventory:

- exactly two visible CUDA devices for active full trial jobs;
- every device name contains `B200`;
- every visible device reports compute capability at least `(10, 0)`.

Torch primitives:

- BF16 allocation and reduction;
- FP8 tensor creation;
- `torch._scaled_mm` existence;
- one-block `_scaled_mm` smoke returning expected shape.

NCCL:

- declared-world-size `torchrun --standalone` smoke; the active two-GPU trial
  requires two ranks;
- all ranks initialize NCCL;
- all-reduce returns expected sum;
- full-job preflight cannot skip this gate.

Source:

- source exists;
- `train_gpt.py` exists;
- `git rev-parse HEAD` equals the pinned upstream commit;
- Lane A `git status --short` is empty;
- Lane B source contains required B200 compatibility hooks;
- Lane B B200 custom-kernel variant requires
  `MODDED_NANOGPT_CE_COMPUTE_CAPABILITY=100`.

Attention:

- Lane A requires FA3.
- Lane B may use FA2 only after `setup_flash_attention.sh` succeeds.
- `flex` is blocked for full jobs because prior full-width validation attempted
  a 256 GiB block-mask allocation.

MLP:

- `triton` must pass local forward/backward smoke before B200 full jobs.
- `torch` must pass local forward/backward smoke.
- `torch` remains diagnostic-only until full-job warmup no longer OOMs or
  stalls.

Data:

- manifest exists;
- manifest schema version is `1`;
- manifest dataset is `fineweb10B`;
- full-run manifest token budget is `900M`;
- full-run manifest source commit is the pinned upstream commit;
- full-run manifest preparation command is recorded;
- manifest has exactly 10 `.bin` shards for the 900M-token path;
- total bytes are `2000010240`;
- manifest `total_bytes` equals the sum of shard sizes;
- every shard exists and size matches;
- every shard has a SHA256 entry;
- full-job preflight requires `--verify-sha`.

### Verifier Tests

Minimum tests or command checks:

- host invocation of `run_preflight.sh` reports `rootfs: ok`;
- direct host invocation of `preflight.py` fails with exit `21`;
- `run_preflight.sh --skip-nccl` fails with exit `21` unless
  `MODDED_NANOGPT_ALLOW_SKIP_NCCL=1`;
- wrong source commit or dirty Lane A source fails;
- FA3 B200 missing-kernel failure is captured in JSON rather than killing the
  parent process;
- FA2 smoke passes after `setup_flash_attention.sh`;
- data manifest size mismatch fails.

## FlashAttention Setup Spec

`setup_flash_attention.sh` is the public entrypoint for Lane B FA2 setup.

Required behavior:

- re-enter rootfs if launched from host;
- set `CC=/usr/bin/gcc`, `CXX=/usr/bin/g++`;
- set `CUDA_HOME=/opt/cuda-synth`, `CUDA_PATH=/opt/cuda-synth`;
- default `FLASH_ATTN_VERSION=2.8.3.post1`;
- default `CUDA_WHEEL_VERSION=13.2.86`;
- default `TORCH_CUDA_ARCH_LIST=10.0`;
- install only CUDA build wheels needed for compilation:
  `nvidia-cuda-nvcc`, `nvidia-cuda-crt`, `nvidia-cuda-cccl`, `nvidia-nvvm`;
- maintain `/opt/cuda-synth/lib/libcudart.so -> libcudart.so.13`;
- source-build FlashAttention by default;
- support `FLASH_ATTN_SOURCE_BUILD=0` only as a fast repeat smoke after a clean
  source build passed;
- run the FA2 varlen/window B200 smoke before returning success.

Success evidence:

- wheel build completed;
- final smoke prints device `NVIDIA B200`, capability `(10, 0)`, output shape
  `(256, 6, 128)`, output dtype `torch.bfloat16`.

## Source Preparation Spec

Add `fetch_upstream.sh`.

Required behavior:

- re-enter rootfs before any Python or network-adjacent helper code, if any;
- create `experiments/modded_nanogpt_b200/sources/`;
- clone `https://github.com/kellerjordan/modded-nanogpt` into
  `sources/modded-nanogpt` when missing;
- check out `ecbb586296d3dac36fd206211f25d63bad4a6b35`;
- verify `git rev-parse HEAD`;
- write `source.json` into a caller-supplied result directory or print it to
  stdout for the harness to capture;
- refuse to overwrite a dirty existing source unless an explicit generated
  variant path is supplied.

Lane B variant preparation:

- copy or worktree from the pinned source into a distinct generated source path;
- apply compatibility patches there only;
- write `variant_patch.diff` before any run;
- classify every changed file and patch reason.

## Data Preparation Spec

Add `prepare_data.sh`.

Required behavior:

- re-enter rootfs before invoking Python;
- run from the pinned upstream source directory;
- support full path: `python data/cached_fineweb10B.py 9`;
- support smoke path: `python data/cached_fineweb10B.py 1`, clearly labeled
  `smoke`;
- do not modify upstream dataset code for competition-comparable claims;
- write a manifest with:
  - schema version;
  - run ID;
  - source path;
  - source commit;
  - data command;
  - environment snapshot;
  - fresh versus reused status;
  - shard paths;
  - shard byte sizes;
  - shard SHA256 checksums;
  - total bytes;
  - number of shards.

Manifest schema:

```json
{
  "schema_version": 1,
  "dataset": "fineweb10B",
  "token_budget": "900M",
  "source": {
    "path": "...",
    "commit": "ecbb586296d3dac36fd206211f25d63bad4a6b35"
  },
  "command": ["python", "data/cached_fineweb10B.py", "9"],
  "freshness": "fresh|reused",
  "total_bytes": 2000010240,
  "files": [
    {
      "path": ".../data/fineweb10B/...",
      "bytes": 123,
      "sha256": "..."
    }
  ]
}
```

Full-run manifest pass criteria:

- exactly 10 `.bin` files;
- total bytes `2000010240`;
- all files exist;
- all checksums are present.

## Speedrun Runner Spec

Add `run_speedrun.sh`.

Purpose:

- launch Lane A or Lane B speedrun attempts under rootfs;
- keep upstream source unchanged for Lane A;
- capture complete operational evidence;
- tee stdout/stderr to `run.log`;
- write an attempt manifest before launch and a summary after launch.

CLI:

```bash
experiments/modded_nanogpt_b200/run_speedrun.sh \
  --lane A|B \
  --mode smoke|full|diagnostic \
  --source PATH \
  --data-manifest PATH \
  --result-dir PATH \
  [--attention-backend fa3|fa2] \
  [--mlp-backend triton|torch] \
  [--allow-previous-stall] \
  [--verify-sha] \
  [--skip-run]
```

Required phases:

1. Re-enter rootfs.
2. Validate `result-dir` is under ignored `results/`.
3. Capture `attempt.json` before mutation or launch.
4. Run `setup_flash_attention.sh` automatically for Lane B FA2 unless an
   explicit `--skip-flash-setup` diagnostic option is added later.
5. Run `run_preflight.sh` with NCCL for full jobs.
6. Copy or point to the data manifest.
7. Capture source status and Lane B patch diff.
8. Set run environment:
   - `HF_HOME=$PWD/.cache/huggingface`;
   - `HF_HUB_CACHE=$HF_HOME/hub`;
   - `CC=/usr/bin/gcc`;
   - `CXX=/usr/bin/g++`;
   - `DATA_PATH=<upstream-compatible data root>`;
   - result-local `TORCHINDUCTOR_CACHE_DIR`;
   - result-local `TRITON_CACHE_DIR`.
9. Launch from the selected source directory:
   - Lane A: `torchrun --standalone --nproc_per_node=8 train_gpt.py` or
     upstream-equivalent `./run.sh` if it preserves the same behavior;
   - Lane B: same command in the generated variant source directory.
10. Preserve exit status.
11. Run parser even on failure so `summary.json` records the blocker.
12. Verify Lane A source remains clean after run.

Full jobs:

- must not pass `--skip-nccl`;
- must not use `--allow-previous-stall`;
- must use full data manifest;
- must pass `--verify-sha`;
- must fail if preflight exits nonzero.

Diagnostic jobs:

- may use `--allow-previous-stall`;
- may use `MODDED_NANOGPT_ALLOW_SKIP_NCCL=1` only with explicit diagnostic
  classification;
- must never be summarized as a successful reproduction.

## Log Parser Spec

Implemented in `parse_log.py`; run real artifact parsing through
`parse_log.sh` so host-launched parsing re-enters the rootfs.

Input:

- `--log PATH`
- `--result-dir PATH`
- `--source PATH`
- `--data-manifest PATH`
- `--preflight-report PATH`
- optional `--wall-clock-json PATH`

Output:

- `summary.json`
- optionally stdout summary for humans

Required fields:

```json
{
  "schema_version": 1,
  "ok": false,
  "classification": {
    "lane": "B",
    "mode": "diagnostic",
    "arm": "B0",
    "claim_label": "diagnostic",
    "evidence_tier": "diagnostic",
    "run_id": "...",
    "attempt_id": "...",
    "environment_class": "torchtitan-rootfs-b200",
    "claim_eligible": false
  },
  "upstream_commit": "ecbb586296d3dac36fd206211f25d63bad4a6b35",
  "source": {
    "path": "...",
    "dirty": true,
    "status": "..."
  },
  "data_manifest": "...",
  "environment": {
    "python": "...",
    "torch": "...",
    "cuda_runtime": "...",
    "triton": "..."
  },
  "hardware": {
    "num_gpus": 8,
    "gpus": []
  },
  "metrics": {
    "val_loss": null,
    "train_time": null,
    "step_avg": null,
    "peak_allocated_memory": null,
    "peak_reserved_memory": null
  },
  "wall_clock": {
    "setup_seconds": null,
    "preflight_seconds": null,
    "download_seconds": null,
    "warmup_seconds": null,
    "train_shell_seconds": null,
    "validation_seconds": null,
    "total_seconds": null
  },
  "validity": {
    "reached_final_validation": false,
    "val_loss_threshold_met": false,
    "source_clean_required_and_met": false,
    "data_manifest_verified": false,
    "full_job_preflight_passed": false,
    "claim_valid": false,
    "included_in_baseline_stats": false
  },
  "blocker": {
    "phase": "attention_backend",
    "message": "..."
  }
}
```

Parser requirements:

- Extract upstream `train_time` from the log, not shell elapsed time.
- Extract final validation loss only when final validation was reached.
- Extract `step_avg` from upstream log text only when present.
- Extract peak allocated and reserved memory when present.
- Preserve nulls for missing final metrics.
- Do not infer success from partial logs.
- Mark Lane A invalid if the source is dirty after the run.
- Mark successful B200 reproduction only when final validation exists,
  `val_loss <= 3.28`, and all full-baseline evidence gates pass.
- Include `claim_validation` in `summary.json` and render it in `analysis.md`
  so operators can see the successful-reproduction gates directly: full mode,
  Lane A, preflight, NCCL, SHA verification, source cleanliness, final
  validation, validation-loss threshold, upstream `train_time`, separate shell
  wall-clock, and the first claim blocker.
- Include derived telemetry signs in `summary.json` and `analysis.md` for
  thermal-or-clock throttling and CPU-or-RSS bottleneck suspicion. These signs
  are diagnostic evidence only; they do not replace the raw telemetry ranges or
  change baseline eligibility.
- Full-baseline evidence gates are fail-closed. `ok=true` and
  `included_in_baseline_stats=true` require
  `launch_readiness.training_launched=true`, not `--skip-run`, no
  `--allow-previous-stall`, launch preflight success, rootfs sentinel evidence
  (`TORCHTITAN_IN_ROOTFS=1`, `cwd=/workspace/torchtitan`, sentinel present),
  NCCL preflight evidence, SHA-verified full 900M manifest evidence in both
  preflight and manifest summaries, `launch_readiness.data_manifest` matching
  either the result-local manifest pointer or its resolved manifest target when
  present, the pinned data source commit, and exactly the declared B200 GPU
  allocation. The active trial policy requires two visible B200 GPUs.

Parser tests:

- fixture with successful final validation;
- fixture with final validation metrics but missing full-attempt evidence,
  proving baseline inclusion fails closed;
- fixture with preflight failure;
- fixture with compile/warmup failure before final validation;
- fixture with partial log that contains intermediate validation but no final
  result;
- fixture with Lane A dirty source status.

## Harness Summary Spec

Implemented in `summarize.py`; run real result aggregation through
`summarize.sh` so host-launched summary generation re-enters the rootfs.

Purpose:

- aggregate result directories without changing run evidence;
- produce compact run indexes and comparison tables;
- keep claim labels conservative.

Required behavior:

- read only `summary.json` and manifest files from result directories;
- never parse mixed historical logs if `summary.json` exists;
- group by lane, arm, source commit, data manifest, and environment class;
- report median and best train time only for runs that reached final
  validation;
- keep shell wall-clock separate;
- list failed/diagnostic attempts with blocker phase and message;
- preserve compact per-attempt `claim_validation` and `telemetry.signs` fields
  so final reporting can audit claim blockers and telemetry bottleneck signs
  from `run_index.json` without reopening every result directory;
- preserve `launch_readiness_exclusion` on failed/diagnostic attempt records
  that carry launch-readiness evidence but are not surfaced under
  `launch_ready_attempts`, so operators can see the first missing gate without
  reopening every summary directory;
- include a top-level `stale_or_demoted_artifacts` run-index section for
  historical or malformed attempt artifacts that cannot be treated as launch
  prerequisites or baselines under the current schema. The list is bounded by
  `MAX_STALE_OR_DEMOTED_ARTIFACTS = 20`. Each record must include the
  `summary` path, `reason`, `message`, `operator_action`, lane/mode/claim-label
  context when available, and `launch_readiness_exclusion` when the artifact
  was demoted by a launch-readiness gate. Legacy summaries may be reported with
  `reason=legacy_summary_schema`; malformed summaries must be preserved in this
  section instead of aborting the index.
- include top-level `launch_readiness_exclusion_stats` with the total exclusion
  count and per-phase objects keyed by exclusion phase. Each per-phase object
  must contain `count` and `example_summaries`, where `example_summaries` lists
  up to three `summary.json` paths for that phase, so operators can inspect
  launch blockers without scanning every failed attempt row;
- never include diagnostic attempts in successful baseline statistics;
- fail closed if a stale or malformed summary sets
  `included_in_baseline_stats=true` without also being Lane A or B,
  `mode=full`, `claim_eligible=true`, and `ok=true`.
- preserve launch-readiness evidence from sibling `launch_readiness.json` when
  present, and otherwise fall back to embedded `summary.json["launch_readiness"]`
  for archived or copied summary bundles.
- expose only Lane A or B full-mode, claim-eligible, ready-but-not-launched
  prerequisites under `launch_ready_attempts`; diagnostic and smoke attempts
  remain failed/diagnostic rows even if a malformed readiness sidecar says
  `ready_to_launch=true`.
- when a readiness sidecar records `lane` or `mode`, require those fields to
  match the normalized summary classification before surfacing the attempt
  under `launch_ready_attempts`.
- require summary-level parsed manifest evidence before surfacing an attempt
  under `launch_ready_attempts`; a readiness sidecar alone is not enough. The
  `data_manifest_summary` must carry the full 900M shape and pinned source
  commit, and `launch_readiness.data_manifest` must match either the
  result-local manifest pointer or its resolved target when present.
- require summary-level GPU evidence before surfacing an attempt under
  `launch_ready_attempts`; under the active trial policy the attempt summary
  must carry exactly two GPU entries whose names identify B200 devices.
- require summary-level rootfs sentinel evidence before surfacing an attempt
  under `launch_ready_attempts`; the attempt summary must show
  `TORCHTITAN_IN_ROOTFS=1`, `cwd=/workspace/torchtitan`, and the workspace
  sentinel present.
- require summary-level source provenance before surfacing an attempt under
  `launch_ready_attempts`; the parsed source commit must equal the pinned
  upstream commit `ecbb586296d3dac36fd206211f25d63bad4a6b35`.
- require summary-level Lane B patch provenance before surfacing a Lane B
  attempt under `launch_ready_attempts`; the parsed variant patch
  classification must include a diff artifact, a non-empty changed-file list,
  patch classes other than `unclassified`, and first Lane A blockers for every
  changed file.
- require summary-level active-job scan evidence before surfacing an attempt
  under `launch_ready_attempts`; `active_jobs.ok` must be true,
  `active_jobs.active_job_count` must be zero, and `active_jobs.scan_error`
  must be absent or false.
- full-mode runner attempts must capture active-job evidence after successful
  preflight and before returning for `--skip-run` or missing full-launch
  authorization. Active jobs or active-job scan errors must use the existing
  `active_jobs` blocker before a skip-run or launch-authority success path can
  be reported.

## Static Harness Verifier Spec

Implemented in `verify_static.py`.

Purpose:

- close the static-verification gap where `git diff --check` can miss
  untracked harness files;
- provide a focused, non-launch verification command for the Modded NanoGPT
  B200 harness and tracker files;
- check tracked modified files, staged files, and untracked files from the
  scoped paths only:
  `.scratch/modded-nanogpt-b200`,
  `experiments/modded_nanogpt_b200`, and
  `tests/unit_tests/test_modded_nanogpt_b200_*.py`.

Required behavior:

- inspect only `.py`, `.sh`, and `.md` candidate files;
- exclude generated or ignored trees, including `.scratch/trae-pytest-tmp`,
  experiment `data/`, `results/`, and `sources/`, and `__pycache__`;
- fail on trailing whitespace;
- fail on a missing final newline;
- fail on Python compile errors for `.py` files;
- fail on shell syntax errors for `.sh` files via `bash -n`;
- support `--list-files` for candidate-file inspection;
- support `--json` for machine-readable candidate and check results;
- avoid launch, GPU, training, preflight, data-prep, download, pip, CUDA,
  NCCL, `torchrun`, full-launch, and result-artifact mutation.

Minimum tests or command checks:

- untracked scoped files are included and checked;
- staged changes are included and checked even when the working tree version
  is otherwise not enough to expose the file through a plain diff check;
- generated and ignored paths are excluded;
- Python compile failures and shell syntax failures are reported;
- `--list-files` and `--json` report the same candidate set.

## Current Known Runtime Facts

Observed on 2026-08-15:

- rootfs Python: `3.12.3`;
- Torch: `2.13.0+cu132`;
- CUDA runtime: `13.2`;
- Triton: `3.7.1`;
- hardware: 8x `NVIDIA B200`, compute capability `(10, 0)`;
- upstream FA3 failed on B200 with `no kernel image is available for execution
  on the device`;
- FA2 `flash-attn==2.8.3.post1` source-builds cleanly in rootfs with CUDA
  build wheels `13.2.86` and passes the B200 varlen/window smoke;
- FlexAttention full-width fallback attempted a 256 GiB block-mask allocation;
- Triton MLP initially failed for sm100 in
  `TritonNvidiaGPUOptimizeTMemLayoutsPass`, then failed a bounded diagnostic in
  backward because the 3D MLP tensors were not flattened before `post.T @
  grad_output`; the current bounded Triton diagnostic passes with
  `output_shape=[2, 16, 768]`;
- PyTorch MLP fallback passed local forward/backward smoke but has not passed a
  full-job warmup.

These facts are blockers or implementation inputs, not success evidence for a
full reproduction.

## Acceptance Criteria

The v1 harness is complete when:

- repo-local rootfs wrappers exist for source prep, data prep, preflight,
  speedrun launch, parsing, and summarization;
- host-side Python is not used for real GPU experiment work;
- the upstream source commit is pinned and verified;
- Lane A preflight fails loudly on the current FA3 B200 blocker and preserves a
  JSON report;
- Lane B FA2 setup source-builds FlashAttention and passes the B200 smoke;
- full-job preflight includes NCCL over the declared launch world size;
- data preparation writes a checksum manifest matching the expected 900M-token
  path;
- the runner can launch smoke and full modes without editing upstream source
  for Lane A;
- Lane B variant runs preserve `variant_patch.diff`;
- parser writes `summary.json` for success and failure logs;
- summaries distinguish upstream `train_time` from shell wall-clock;
- result classifications are conservative and match the lane model;
- generated artifacts remain untracked.

## Child Tickets

Implementation tickets:

- `issues/01-source-and-preflight.md`
- `issues/02-data-and-manifest.md`
- `issues/03-reproduction-wrapper-and-parser.md`
- `issues/04-b200-ablation-matrix.md`

Recommended additional ticket split:

- `issues/05-harness-summary-and-run-index.md`
- `issues/06-lane-b-compatibility-baseline.md`

Ticket `01` is partly implemented in the current working tree through
`preflight.py`, `run_preflight.sh`, `setup_flash_attention.sh`, and
`preflight_checklist.md`. Remaining tickets should treat those files as current
source, not as final immutable design.
