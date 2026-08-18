# Modded NanoGPT B200 Preflight Checklist

Status: active preflight specification
Date: 2026-08-15
Surface: repo-local research program
Owner path: `experiments/modded_nanogpt_b200/`

This checklist is the launch gate for any
`kellerjordan/modded-nanogpt` B200 run from this TorchTitan checkout. The
active launch policy is a two-GPU small-scale trial, run sequentially. It is
written for two audiences:

- the operator launching the job;
- the reviewer deciding whether the resulting run can support a Lane A, Lane B,
  or Lane C claim.

The normal executable gate is
`experiments/modded_nanogpt_b200/run_preflight.sh`, which re-enters the bwrap
rootfs before invoking `experiments/modded_nanogpt_b200/preflight.py`. This
document is stricter than the script: it also defines launch hygiene, claim
boundaries, required artifacts, and no-go conditions that must be recorded in
the run summary.

Authority order:

1. `.scratch/modded-nanogpt-b200/spec.md` is the canonical implementation spec.
2. This checklist is the executable launch contract for preflight and job
   approval.
3. `experiments/modded_nanogpt_b200/upstream_reference.md` is the pinned
   upstream fact source.
4. `experiments/modded_nanogpt_b200/benchmark_design.md` is historical design
   context and is superseded where it conflicts with the files above.

## Non-Negotiable Rules

- All Python setup, dependency installation, data preparation, preflight,
  training, log parsing, and summarization must run through the bwrap rootfs.
  Prefer repo-local wrappers such as
  `experiments/modded_nanogpt_b200/setup_flash_attention.sh` and
  `experiments/modded_nanogpt_b200/run_preflight.sh`; they re-enter
  `scripts/rootfs/enter_rootfs.sh` automatically.
- Do not run upstream `pip install -r requirements.txt`; it can replace the
  B200-capable Torch stack.
- Keep upstream source unchanged for Lane A. Any source edit makes the result
  Lane B or Lane C, never a faithful upstream reproduction.
- Keep generated source snapshots, data, logs, caches, reports, and result
  bundles out of git.
- A preflight failure exits `21`. A full job must not launch after exit `21`
  unless the run is explicitly diagnostic and the summary says so.
- `--skip-nccl` is blocked by `run_preflight.sh` unless
  `MODDED_NANOGPT_ALLOW_SKIP_NCCL=1` is set. That override is diagnostic-only
  and must not appear in a full-job approval record.
- Do not use `--allow-previous-stall` for a full job. It is only for diagnostic
  launches that intentionally revisit a known blocked configuration.
- Do not call any run successful unless the training loop reaches final
  validation. If final validation is absent, `train_time`, `val_loss`, and
  `step_avg` must be null or omitted.
- Report upstream timed training from the upstream log's `train_time`.
  Preserve shell wall-clock separately as operational evidence.
- Active non-diagnostic launches must expose exactly two B200 GPUs to the
  rootfs and use a matching two-rank distributed world. Do not start a second
  NanoGPT attempt until the previous attempt has finished and its artifacts have
  been summarized.

## Rootfs Execution Boundary

The host shell may do only orchestration that does not import Python packages or
touch CUDA runtime state:

- create ignored result directories;
- set environment variables for the wrapper;
- call repo-local shell wrappers;
- inspect git status and generated logs.

Everything else must happen after `TORCHTITAN_IN_ROOTFS=1` is present:

- `python` and `torchrun`;
- `pip install`;
- HuggingFace or dataset download code;
- FlashAttention, Triton, CUDA, BF16, FP8, or NCCL probes;
- training and validation;
- log parsing and summary generation.

The approved wrappers for this experiment are:

- `experiments/modded_nanogpt_b200/check_active_jobs.sh`
- `experiments/modded_nanogpt_b200/diagnose_mlp_backend.sh`
- `experiments/modded_nanogpt_b200/fetch_upstream.sh`
- `experiments/modded_nanogpt_b200/parse_log.sh`
- `experiments/modded_nanogpt_b200/prepare_data.sh`
- `experiments/modded_nanogpt_b200/run_speedrun.sh`
- `experiments/modded_nanogpt_b200/summarize.sh`
- `experiments/modded_nanogpt_b200/setup_flash_attention.sh`
- `experiments/modded_nanogpt_b200/run_preflight.sh`

If a new executable step is added, give it the same rootfs re-entry shape before
using it in a real run. Shell wrappers must source `rootfs_guard.sh` after the
re-entry branch so a forged `TORCHTITAN_IN_ROOTFS=1` marker cannot skip the
rootfs workspace and sentinel checks. Python entrypoints must also call the
shared `cli_guard.py` path from `if __name__ == "__main__"` so direct host-side
`python experiments/modded_nanogpt_b200/<tool>.py ...` fails closed before any
work runs.

## Claim Classification Gate

Classify the run before preflight. The classification determines which source,
kernel, and patch rules apply.

Every attempt must record this common classification schema in
`preflight_report.json`, `attempt.json`, and `summary.json`:

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

`claim_eligible` may be true only for `mode=full`; final success still requires
the post-run validity checks.

### Mode Policy Matrix

| Gate | Full | Smoke | Diagnostic |
| --- | --- | --- | --- |
| NCCL | Required; no `--skip-nccl` | Required unless a smoke fixture has no distributed claim | May skip only with `MODDED_NANOGPT_ALLOW_SKIP_NCCL=1` and recorded classification |
| Data manifest | Full 900M FineWeb manifest | Smoke manifest allowed and labeled | Full or smoke manifest allowed, but no claim eligibility |
| SHA verification | Required before a claim-bearing run | Optional unless preserving a reusable manifest | Optional, record whether checked |
| GPU inventory | Exactly 2x B200 for the active trial policy | May be reduced only for a declared local fixture | Records actual visible devices; no claim eligibility if reduced |
| Known-stall overrides | Forbidden | Forbidden unless classified diagnostic instead | Allowed only to revisit a named blocker |
| Source cleanliness | Lane-specific required | Lane-specific required | Lane-specific recorded; no claim eligibility if violated |

The executable verifier enforces the full-job column. Runner and parser tickets
must preserve the same policy for `attempt.json` and `summary.json`.

### Lane A: Faithful Upstream B200 Reproduction

Pass criteria:

- Source checkout is exactly
  `ecbb586296d3dac36fd206211f25d63bad4a6b35`.
- `git status --short` in the upstream checkout is empty.
- Attention backend is upstream FA3.
- No changes are made to model math, optimizer math, schedule, data streams,
  validation count, compile policy, or kernel choices.
- Environment-only adaptation is allowed: rootfs entry, compiler variables,
  cache variables, and wrapper-side logging.

No-go conditions:

- Upstream source is dirty.
- FA3 smoke fails.
- A fallback attention backend is used.
- Any B200 compatibility patch is applied.

Result label:

- If final validation succeeds with `val_loss <= 3.28`, report
  `B200 upstream reproduction`.
- Do not call it an official upstream record unless upstream record rules are
  separately satisfied.

### Lane B: Minimal B200 Compatibility Patchset

Pass criteria:

- Source checkout descends from
  `ecbb586296d3dac36fd206211f25d63bad4a6b35`.
- Every source difference is preserved as an artifacted patch diff.
- Every patch is classified as `environment`, `hardware-detection`,
  `kernel-compat`, `timing-harness`, or `ML-affecting`.
- No patch changes the train token stream, validation token stream, validation
  count, model, optimizer, or schedule unless the run is reclassified as a
  `B200 ML variant`.
- The first Lane A blocker that each patch addresses is recorded.

No-go conditions:

- A patch has no concrete B200 incompatibility justification.
- A patch silently changes ML semantics while the result is presented as
  source-equivalent or competition-comparable.
- The selected full-job backend is listed as a live no-go in the current
  checklist, or the full launch lacks the explicit authorization token and
  full-mode gates required by this checklist. Historical backend blockers do
  not block the current Lane B full dry gate unless they remain listed as live
  no-go conditions.

Result label:

- If final validation succeeds with `val_loss <= 3.28`, report
  `B200 compatibility patchset` or `B200 systems-only`, depending on the
  patch classes.
- If any ML-affecting patch is present, report `B200 ML variant`.

### Lane C: B200 Optimization Ablation

Pass criteria:

- The baseline arm is declared before launch: `A0` if Lane A runs, otherwise
  nearest source-equivalent Lane B.
- The arm being launched is named (`B1`, `B2`, `B3`, `B4`, or another explicit
  ablation arm).
- The single changed variable is documented before launch.
- The source, data manifest, rootfs class, GPU allocation, and reporting parser
  are matched to the baseline unless the ablation intentionally changes one of
  them.

No-go conditions:

- Multiple uncontrolled variables change in one ablation arm.
- A timing claim compares against a different data stream, validation count,
  model, optimizer, schedule, or source patchset without saying so.

Result label:

- Use `competition-comparable`, `B200 systems-only`, or `B200 ML variant` in
  the run summary.

## Required Artifacts

Create a result directory before preflight:

```bash
RUN_ID="lane_<lane>_<timestamp>"
RESULT_DIR="experiments/modded_nanogpt_b200/results/${RUN_ID}"
mkdir -p "${RESULT_DIR}"
```

Preserve these artifacts for every attempted run:

- `preflight_report.json`
- complete run log if training is launched
- source path and upstream commit
- source `git status --short`
- source patch diff for Lane B or Lane C
- data manifest path
- hardware and environment inventory
- final parsed summary if training reaches or attempts validation
- explicit classification: Lane A, Lane B, Lane C, smoke, or diagnostic

Generated artifacts must remain under ignored paths:

- `experiments/modded_nanogpt_b200/sources/`
- `experiments/modded_nanogpt_b200/data/`
- `experiments/modded_nanogpt_b200/results/`
- repo-local caches such as `.cache/huggingface`

## Phase 0: Host and Process Hygiene

Run from the TorchTitan checkout root.

Pass criteria:

- `pwd` equals the checkout root derived by `git rev-parse --show-toplevel`.
- No unrelated generated artifacts are staged for commit.
- No existing `torchrun`, `train_gpt.py`, or stale diagnostic run is consuming
  GPUs.
- GPU inventory is captured by `run_preflight.sh` inside rootfs, not by a host
  Python or CUDA probe.
- The result directory exists and is under `experiments/modded_nanogpt_b200/results/`.

Suggested checks:

```bash
test "$(pwd)" = "$(git rev-parse --show-toplevel)"
git status --short --ignored=matching experiments/modded_nanogpt_b200 .gitignore
experiments/modded_nanogpt_b200/check_active_jobs.sh \
  --active-jobs-output experiments/modded_nanogpt_b200/results/<run_id>/active_jobs.json
```

No-go conditions:

- Another training process is active on the target GPUs.
- The result path is outside `experiments/modded_nanogpt_b200/results/`.
- Generated source, data, logs, or caches are staged for git.

## Phase 1: Rootfs and Toolchain Gate

All commands below must run inside the TorchTitan rootfs. Use repo-local
wrappers that re-enter the rootfs themselves. Use raw
`scripts/rootfs/enter_rootfs.sh -- ...` only for diagnostic probes that do not
have a wrapper yet.

Pass criteria:

- `TORCHTITAN_IN_ROOTFS=1`.
- `pwd` inside the wrapper is `/workspace/torchtitan`.
- `/workspace/torchtitan/scripts/rootfs/enter_rootfs.sh` exists.
- `sys.executable`, `sys.prefix`, and the rootfs marker are recorded in the
  report.
- Python is the rootfs Python, not host Python.
- Torch is the rootfs B200-capable stack.
- CUDA runtime reported by PyTorch is `13.2`.
- Triton imports.
- `CC=/usr/bin/gcc` and `CXX=/usr/bin/g++` are available.
- `CUDA_HOME=/opt/cuda-synth` and `CUDA_PATH=/opt/cuda-synth`.
- `/opt/cuda-synth/bin/nvcc` exists.
- `/opt/cuda-synth/lib/libcudart.so.13` exists.
- `/opt/cuda-synth/lib/libcudart.so` exists before building extensions.

Suggested check:

```bash
scripts/rootfs/enter_rootfs.sh -- /bin/bash -lc '
  set -euo pipefail
  test "${TORCHTITAN_IN_ROOTFS:-0}" = 1
  python - <<'"'"'PY'"'"'
import sys
import torch
import triton

print("python", sys.version.replace("\n", " "))
print("torch", torch.__version__)
print("cuda_runtime", torch.version.cuda)
print("triton", triton.__version__)
print("cuda_available", torch.cuda.is_available())
PY
  test -x /usr/bin/gcc
  test -x /usr/bin/g++
  test -x /opt/cuda-synth/bin/nvcc
  test -f /opt/cuda-synth/lib/libcudart.so.13
  test -e /opt/cuda-synth/lib/libcudart.so
'
```

No-go conditions:

- `TORCHTITAN_IN_ROOTFS` is missing.
- The verifier is not running from `/workspace/torchtitan`.
- The rootfs workspace sentinel is missing.
- Torch has been replaced by an upstream requirement install.
- CUDA build tools are missing from `/opt/cuda-synth`.
- `libcudart.so` is absent and FlashAttention has not been rebuilt after
  fixing it.

## Phase 2: Dependency Install and Import Gate

Do not install upstream `requirements.txt`. The upstream file currently includes
`torch==2.10`, which can replace the rootfs B200-capable Torch stack.

Allowed direct runtime dependencies:

- always allowed: `numpy`, `tqdm`, `huggingface-hub`, `datasets`, `tiktoken`,
  `typing-extensions`, and `setuptools`;
- Lane A or FA3-only: `kernels`;
- Lane B FA2: use `experiments/modded_nanogpt_b200/setup_flash_attention.sh`
  for FlashAttention and its pinned CUDA build wheels.

Safe install pattern inside rootfs:

```bash
scripts/rootfs/enter_rootfs.sh -- /bin/bash -lc '
  set -euo pipefail
  python -m pip install --break-system-packages --no-deps \
    numpy tqdm huggingface-hub datasets tiktoken typing-extensions setuptools
'
```

For Lane A or any FA3 diagnostic, add `kernels` with the same `--no-deps`
pattern. If a package reports a missing transitive import, add only the named
non-Torch package after confirming the pip plan will not install or downgrade
`torch`, `triton`, CUDA runtime packages, or FlashAttention.

Import check:

```bash
scripts/rootfs/enter_rootfs.sh -- /bin/bash -lc '
  set -euo pipefail
  python - <<'"'"'PY'"'"'
import importlib.util
import torch
import triton

required = [
    "numpy",
    "tqdm",
    "huggingface_hub",
    "datasets",
    "tiktoken",
    "typing_extensions",
    "setuptools",
]
missing = [name for name in required if importlib.util.find_spec(name) is None]
print("torch", torch.__version__)
print("cuda_runtime", torch.version.cuda)
print("triton", triton.__version__)
if missing:
    raise SystemExit("missing imports: " + ", ".join(missing))
PY
'
```

Pass criteria:

- Imports above succeed inside rootfs.
- Torch, CUDA runtime, and Triton versions remain the rootfs pinned versions.
- Any package install uses the allowlist and `--no-deps`, or records an explicit
  non-Torch transitive dependency decision.
- Lane B FA2 dependency setup is performed by `setup_flash_attention.sh`.

No-go conditions:

- `pip install -r requirements.txt` is used in the upstream source tree.
- A pip operation installs, downgrades, or replaces Torch, Triton, CUDA runtime
  packages, or FlashAttention outside the approved FA2 setup script.
- An import check is run on the host instead of inside rootfs.

## Phase 3: FlashAttention Gate

Lane A uses upstream FA3. Lane B currently uses the repo-local FA2 fallback
because upstream FA3 has not produced a working B200 kernel image in this
environment.

### Lane A FA3 Gate

Pass criteria:

- `kernels` imports.
- `get_kernel("kernels-community/flash-attn3", version=1)` succeeds.
- The FA3 varlen/window smoke in `preflight.py` passes on B200.

No-go conditions:

- FA3 fails with `no kernel image is available for execution on the device`.
- FA3 imports but fails the B200 varlen/window smoke.
- Any fallback attention backend is used while calling the run Lane A.

### Lane B FA2 Gate

First run the source-build proof:

```bash
experiments/modded_nanogpt_b200/setup_flash_attention.sh
```

Pass criteria:

- The script re-enters rootfs if launched from the host.
- CUDA build wheels are pinned to `13.2.86`:
  `nvidia-cuda-nvcc`, `nvidia-cuda-crt`, `nvidia-cuda-cccl`, and `nvidia-nvvm`.
- `flash-attn==2.8.3.post1` builds from source by default.
- `TORCH_CUDA_ARCH_LIST=10.0` is used by default.
- The final smoke prints `flash-attn ok`.
- The smoke output reports:
  - `torch: 2.13.0+cu132`
  - `cuda_runtime: 13.2`
  - `device: NVIDIA B200`
  - `capability: (10, 0)`
  - `output_shape: (256, 6, 128)`
  - `output_dtype: torch.bfloat16`

For repeat validation after the source-build proof has already passed:

```bash
FLASH_ATTN_SOURCE_BUILD=0 experiments/modded_nanogpt_b200/setup_flash_attention.sh
```

No-go conditions:

- The script runs on the host instead of inside rootfs.
- The build uses CUDA 13.3 wheels against the CUDA 13.2 Torch/runtime stack.
- The build fails with `/usr/bin/ld: cannot find -lcudart`.
- The extension imports but the B200 varlen/window smoke fails.

## Phase 4: Source Policy Gate

Source must be a generated checkout under
`experiments/modded_nanogpt_b200/sources/`, unless the run summary explicitly
records a different generated source path.

Pass criteria for all lanes:

- `git rev-parse HEAD` equals
  `ecbb586296d3dac36fd206211f25d63bad4a6b35`.
- `train_gpt.py` exists.
- `triton_kernels.py` exists when the selected source tree uses or patches the
  upstream custom kernels.
- The source path is under an ignored generated directory.

Lane A pass criteria:

- `git status --short` is empty.
- The source has not been patched.

Lane B and Lane C pass criteria:

- `git status --short` may be dirty only by intended B200 patch files.
- A patch diff is written under the result directory before launch.
- The patch diff is reviewed for ML-affecting changes.
- B200 CE custom-kernel variants set
  `MODDED_NANOGPT_CE_COMPUTE_CAPABILITY=100` before importing
  `triton_kernels.py`.

Suggested checks:

```bash
SOURCE="experiments/modded_nanogpt_b200/sources/<source-dir>"
git -C "${SOURCE}" rev-parse HEAD
git -C "${SOURCE}" status --short
git -C "${SOURCE}" diff --binary HEAD -- >"${RESULT_DIR}/variant_patch.diff"
git -C "${SOURCE}" ls-files --others --exclude-standard -z >"${RESULT_DIR}/untracked_files.zlist"
while IFS= read -r -d '' path; do
  mkdir -p "${RESULT_DIR}/untracked_files/$(dirname "${path}")"
  cp -a "${SOURCE}/${path}" "${RESULT_DIR}/untracked_files/${path}"
done <"${RESULT_DIR}/untracked_files.zlist"
```

The supported `run_speedrun.sh` path writes `variant_patch.diff` and
`variant_patch_classification.json` under the result directory before launch.
Treat those runner-generated artifacts as authoritative for Lane B/C launch
readiness. If collecting a manual diagnostic diff, capture tracked, staged,
added, and deleted changes with `git diff --binary HEAD --`; also preserve the
untracked file list and file contents so untracked source provenance is not
lost.

No-go conditions:

- Lane A source is dirty.
- Source commit is not the pinned upstream commit.
- Variant patch diff is missing for Lane B or Lane C.
- A variant source is used without reclassifying the run away from Lane A.

## Phase 5: Environment Variable Gate

For the supported full launch path, do not manually override `DATA_PATH`.
`run_speedrun.sh` reads the manifest shard paths, derives the upstream-compatible
root, writes it to `command.env`, and verifies source-visible train and
validation globs before `torchrun`.

Common required variables:

```bash
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export TORCHINDUCTOR_CACHE_DIR="${RESULT_DIR}/torchinductor_cache"
export TRITON_CACHE_DIR="${RESULT_DIR}/triton_cache"
```

`train_gpt.py` appends `data/fineweb10B/fineweb_{train,val}_*.bin` to
`DATA_PATH`. Therefore the runner-derived `DATA_PATH` must point to the
directory that contains `data/`, not to the `data/` directory itself. Verify the
recorded value after the runner writes `command.env`:

```bash
DATA_PATH="$(awk -F= '$1 == "DATA_PATH" {print substr($0, index($0, "=") + 1)}' "${RESULT_DIR}/command.env")"
test -n "${DATA_PATH}"
test -f "${DATA_PATH}/data/fineweb10B/fineweb_val_000000.bin"
test -n "$(find "${DATA_PATH}/data/fineweb10B" -maxdepth 1 -name 'fineweb_train_*.bin' -print -quit)"
test -n "$(find "${DATA_PATH}/data/fineweb10B" -maxdepth 1 -name 'fineweb_val_*.bin' -print -quit)"
```

Manual `DATA_PATH` exports are diagnostic/reference only. If a wrapper is not
used, set `DATA_PATH` to the upstream-compatible root that contains `data/` and
record why the supported runner path was bypassed.

Lane A required variables:

```bash
export MODDED_NANOGPT_ATTN_BACKEND=fa3
unset MODDED_NANOGPT_CE_COMPUTE_CAPABILITY
```

Lane B FA2 variant variables:

```bash
export MODDED_NANOGPT_ATTN_BACKEND=fa2
export MODDED_NANOGPT_CE_COMPUTE_CAPABILITY=100
```

MLP variables:

```bash
export MODDED_NANOGPT_MLP_BACKEND=<triton-or-torch>
```

Pass criteria:

- Cache directories are result-local or repo-local.
- `command.env` records the runner-derived `DATA_PATH`.
- The recorded `DATA_PATH` points to the intended upstream-compatible root that
  contains `data/fineweb10B/`.
- Both upstream train and validation globs match before launch:
  `${DATA_PATH}/data/fineweb10B/fineweb_train_*.bin` and
  `${DATA_PATH}/data/fineweb10B/fineweb_val_*.bin`.
- Lane B sets `MODDED_NANOGPT_CE_COMPUTE_CAPABILITY=100`.
- The attention backend matches the lane classification.

No-go conditions:

- `DATA_PATH` points at a directory named `data`, causing
  `data/data/fineweb10B/fineweb_val_*.bin` in the training process.
- `${SOURCE}/data/fineweb10B/fineweb_val_000000.bin` exists but the launcher
  would search `${SOURCE}/data/data/fineweb10B/fineweb_val_*.bin`.
- `MODDED_NANOGPT_ATTN_BACKEND=flex` for a full job.
- `TORCHDYNAMO_DISABLE=1` with `MODDED_NANOGPT_MLP_BACKEND=torch` for a full
  job.
- Result-local compile caches are reused between ablation arms without saying
  whether this is a warm-cache run.

## Phase 6: GPU Primitive Gate

The executable preflight checks the actual device path.

Pass criteria:

- Exactly 8 CUDA devices are visible.
- Every device name contains `B200`.
- Every visible device reports compute capability at least `(10, 0)`.
- `torch.cuda.is_available()` is true.
- BF16 allocation and reduction pass.
- FP8 tensor creation passes.
- `torch._scaled_mm` exists.
- A one-block FP8 `torch._scaled_mm` smoke returns the expected shape.

No-go conditions:

- Fewer or more than 8 CUDA devices are visible for a full 8-GPU run.
- Any visible device is not a B200.
- Any visible device reports capability below `(10, 0)`.
- BF16, FP8, or `_scaled_mm` fails.

## Phase 7: Distributed Communication Gate

Pass criteria:

- `torchrun --standalone --nproc_per_node=8` starts 8 ranks.
- NCCL initializes on all ranks.
- A one-value all-reduce returns the expected sum on every rank.
- The preflight is run without `--skip-nccl` before any full job.

No-go conditions:

- `--skip-nccl` is used for a full job.
- Any rank fails to initialize NCCL.
- The all-reduce result is wrong.
- A stale rank or prior process owns the rendezvous or GPU.

## Phase 8: Backend-Specific Compute Gate

Attention:

- Lane A requires FA3 smoke success.
- Lane B FA2 requires `setup_flash_attention.sh` success and FA2 smoke success.
- FlexAttention is blocked for full jobs because prior full-width validation
  attempted a 256 GiB block-mask allocation.

CE custom kernel:

- Lane A inherits upstream behavior.
- Lane B B200 variants must set
  `MODDED_NANOGPT_CE_COMPUTE_CAPABILITY=100`.
- If the CE kernel remains hardcoded to `compute_capability="90"` in a B200
  variant, do not launch the full job.

MLP:

- `MODDED_NANOGPT_MLP_BACKEND=triton` is the current Lane B full dry-gate
  backend. The current full-mode skip-run gate passed the local Triton MLP
  smoke with output shape `[2, 16, 768]`.
- Historical Triton MLP failures for sm100 must remain in the run history, but
  they are no longer the current local smoke/preflight blocker.
- `MODDED_NANOGPT_MLP_BACKEND=torch` passed a local forward/backward smoke in
  an older gate, but the current recommended Lane B full dry gate uses Triton
  MLP rather than PyTorch MLP.
- `MODDED_NANOGPT_MLP_BACKEND=torch` remains diagnostic-only unless a future
  operator intentionally revisits the previous eager OOM and graph-break
  compile-stall path.
- `--allow-previous-stall` may be used only to classify a diagnostic launch,
  never to approve a full run.

No-go conditions:

- Any backend is known to fail full-width validation and the launch is not
  explicitly diagnostic.
- A local smoke or skip-run gate is used to claim baseline success or full-job
  completion before a non-skip full launch reaches final validation.

## Phase 9: Data Manifest Gate

Create or refresh manifests with the repo-local rootfs wrapper. Do not rely on
historical result paths as the only full-run data source.

Fresh full 900M path:

```bash
RUN_ID="full_manifest_<timestamp>"
SOURCE="experiments/modded_nanogpt_b200/sources/modded-nanogpt"
DATA_DIR="${SOURCE}/data/fineweb10B"
RESULT_DIR="experiments/modded_nanogpt_b200/results/${RUN_ID}"
MANIFEST="${RESULT_DIR}/data_manifest.json"
mkdir -p "${RESULT_DIR}"

experiments/modded_nanogpt_b200/prepare_data.sh \
  --source "${SOURCE}" \
  --data-dir "${DATA_DIR}" \
  --output "${MANIFEST}" \
  --token-budget 900M \
  --freshness fresh
```

Reuse an existing full shard set after checking it is complete:

```bash
RUN_ID="full_manifest_reuse_<timestamp>"
SOURCE="experiments/modded_nanogpt_b200/sources/modded-nanogpt"
DATA_DIR="${SOURCE}/data/fineweb10B"
RESULT_DIR="experiments/modded_nanogpt_b200/results/${RUN_ID}"
MANIFEST="${RESULT_DIR}/data_manifest.json"
mkdir -p "${RESULT_DIR}"

experiments/modded_nanogpt_b200/prepare_data.sh \
  --source "${SOURCE}" \
  --data-dir "${DATA_DIR}" \
  --output "${MANIFEST}" \
  --token-budget 900M \
  --freshness reused \
  --skip-upstream-command
```

Smoke data uses the same interface with `--token-budget smoke`; it must never
support a full-run claim. `prepare_data.py` records the upstream command as
`python data/cached_fineweb10B.py 9` for full manifests and computes SHA256 for
every shard. Full preflight must still pass `--verify-sha` so the current shard
contents are rechecked against the manifest.

Pass criteria:

- The upstream data command was run from the pinned source checkout:
  `python data/cached_fineweb10B.py 9`.
- Manifest `schema_version` is `1`.
- Manifest `dataset` is `fineweb10B`.
- Full-run manifest `token_budget` is `900M`.
- Full-run manifest source commit is
  `ecbb586296d3dac36fd206211f25d63bad4a6b35`.
- The manifest records the upstream data command and source commit.
- The manifest records exact shard paths, byte sizes, and SHA256 checksums.
- The manifest records the exact command and environment used for data
  preparation.
- The manifest records whether data was freshly downloaded or reused.
- There are exactly 10 `.bin` shards.
- Total size is `2000010240` bytes.
- Manifest `total_bytes` equals the sum of shard sizes.
- Every listed shard exists and its current size matches the manifest.
- The launcher derives an upstream-compatible `DATA_PATH` from the manifest:
  if manifest shards live under `<root>/data/fineweb10B/`, `DATA_PATH` is
  `<root>`.
- Before `torchrun`, the launcher verifies that the source working directory
  will see train and validation shards through
  `${DATA_PATH}/data/fineweb10B/fineweb_{train,val}_*.bin`.
- Full preflight must pass `--verify-sha`; smoke and diagnostic runs must record
  whether SHA verification was skipped.

No-go conditions:

- The manifest is missing.
- Any shard is missing.
- Total byte count differs from `2000010240`.
- The manifest points to `<root>/data/fineweb10B/...` but the launch
  environment sets `DATA_PATH=<root>/data`.
- A smoke dataset from `cached_fineweb10B.py 1` is used for a full-run claim.
- The data token stream differs from upstream while the result is labeled
  competition-comparable.

## Phase 10: Executable Preflight Commands

Lane A command shape:

```bash
RUN_ID="lane_a_full_<timestamp>"
RESULT_DIR="experiments/modded_nanogpt_b200/results/${RUN_ID}"
SOURCE="experiments/modded_nanogpt_b200/sources/modded-nanogpt"
DATA_MANIFEST="experiments/modded_nanogpt_b200/results/<manifest-run-id>/data_manifest.json"
mkdir -p "${RESULT_DIR}"

MODDED_NANOGPT_ATTN_BACKEND=fa3 \
experiments/modded_nanogpt_b200/run_preflight.sh \
  --mode full \
  --lane A \
  --run-id "${RUN_ID}" \
  --attempt-id "${RUN_ID}_attempt_001" \
  --arm A0 \
  --source "${SOURCE}" \
  --data-manifest "${DATA_MANIFEST}" \
  --attention-backend fa3 \
  --mlp-backend triton \
  --verify-sha \
  --report "${RESULT_DIR}/preflight_report.json"
```

Lane B non-launch full dry-gate command shape for the current FA2/Triton-MLP
state:

```bash
RUN_ID="lane_b_full_skiprun_<timestamp>"
RESULT_DIR="experiments/modded_nanogpt_b200/results/${RUN_ID}"
SOURCE="experiments/modded_nanogpt_b200/sources/modded-nanogpt-b200-sdpa"
DATA_MANIFEST="experiments/modded_nanogpt_b200/results/<manifest-run-id>/data_manifest.json"
mkdir -p "${RESULT_DIR}"

MODDED_NANOGPT_CE_COMPUTE_CAPABILITY=100 \
MODDED_NANOGPT_ATTN_BACKEND=fa2 \
MODDED_NANOGPT_MLP_BACKEND=triton \
experiments/modded_nanogpt_b200/run_speedrun.sh \
  --mode full \
  --lane B \
  --run-id "${RUN_ID}" \
  --attempt-id "${RUN_ID}_attempt_001" \
  --arm B0 \
  --source "${SOURCE}" \
  --data-manifest "${DATA_MANIFEST}" \
  --attention-backend fa2 \
  --mlp-backend triton \
  --verify-sha \
  --skip-run \
  --result-dir "${RESULT_DIR}"
```

Important: the command above is a non-launch full dry gate. It must not include
`--launch-authorization=launch-full-b200`, and it is not a baseline because
`--skip-run` preserves `training_launched=false`.

One prior fresh Lane B full dry gate was:

- `experiments/modded_nanogpt_b200/results/lane_b_full_skiprun_refresh_20260816T111056Z`
- manifest:
  `experiments/modded_nanogpt_b200/results/full_manifest_refresh_20260816T111021Z/data_manifest.json`
- manifest facts: `verified_sha=true`, `token_budget=900M`, `num_files=10`,
  `total_bytes=2000010240`
- run-index facts: `total_attempts=24`, `launch_prerequisite_attempts=1`,
  `launch_ready_attempts=0`, `baseline_stats.count=0`

`launch_ready_attempts=0` is intentional for this state: the only current
prerequisite row is a `--skip-run` artifact. A non-skip full launch still needs
explicit operator authorization by adding
`--launch-authorization=launch-full-b200` to a full `run_speedrun.sh` command.
Agents must not infer that authorization from a passing dry gate.

A full-job preflight must include NCCL; do not pass `--skip-nccl`. If a wrapper
smoke must skip NCCL, set `MODDED_NANOGPT_ALLOW_SKIP_NCCL=1` and record the run
as diagnostic.

Before adding launch authorization, the result directory must show all of these
operator-visible prerequisite gates as passed:

- rootfs sentinel and `/workspace/torchtitan` working directory;
- Torch import with CUDA runtime, Triton, and FlashAttention versions recorded;
- exactly two visible B200 GPUs;
- BF16, FP8, and `_scaled_mm` primitive smoke;
- two-rank NCCL all-reduce for the active full trial mode;
- Lane B FA2 setup plus FA2 varlen/window smoke;
- selected MLP backend smoke;
- SHA-verified 900M FineWeb manifest with 10 shards and total bytes
  `2000010240`;
- source provenance at commit `ecbb586296d3dac36fd206211f25d63bad4a6b35`;
- Lane B variant patch diff and patch classification with no unclassified
  full-launch changes;
- result-local compile caches;
- source-visible train and validation data globs through `DATA_PATH`;
- active-job scan showing no concurrent `torchrun`, `train_gpt.py`, or
  `cached_fineweb10B.py` work;
- explicit `--launch-authorization=launch-full-b200` for non-skip full launch.

## Phase 11: Launch Approval Record

Before launching training, write or capture a short launch approval note in the
result directory. It must include:

- run ID;
- lane and claim classification;
- source path;
- source commit;
- source cleanliness or patch artifact path;
- data manifest path;
- attention backend;
- MLP backend;
- whether NCCL was checked;
- whether SHA256 was checked;
- whether compile caches are cold or reused;
- expected no-go blockers that remain;
- exact training command to be launched.

No full job is approved if any item is unknown.

Full launch authority is separate from preflight readiness. A non-skip
`run_speedrun.sh --mode full` command must stop before `torchrun` unless the
operator explicitly supplies `--launch-authorization=launch-full-b200`. Do not
add that token to a command unless the user has authorized the full B200 launch
in the current session.

## Phase 12: Post-Run Validity Checks

Immediately after any run attempt:

- Preserve the complete console log.
- Run `git diff --exit-code` or capture `git status --short` in the source
  checkout.
- If Lane A source is dirty after the run, classify the run invalid until the
  cause is understood.
- Extract summary fields from the log:
  - upstream commit;
  - GPU inventory;
  - Python, PyTorch, CUDA, Triton versions;
  - final `val_loss`;
  - final `train_time`;
  - final `step_avg`;
  - peak allocated memory;
  - peak reserved memory;
  - data manifest path;
  - source diff status;
  - total shell wall-clock.
- If final validation is missing, leave `val_loss`, `train_time`, and
  `step_avg` null or absent.
- Preserve shell wall-clock separately from upstream `train_time`.

## Current Known Blockers

- Lane A FA3: `kernels-community/flash-attn3` failed on B200 with
  `no kernel image is available for execution on the device`.
- FlexAttention fallback: full-width validation attempted a 256 GiB allocation
  while creating the block mask.
- FA2 attention: `experiments/modded_nanogpt_b200/setup_flash_attention.sh`
  cleanly source-builds `flash-attn==2.8.3.post1` against the rootfs CUDA 13.2
  stack and passes a B200 varlen/window smoke. This is the current workable
  FlashAttention implementation for Lane B.
- Triton MLP: the sm100 `linear_relu_square_kernel` compile failure in
  `TritonNvidiaGPUOptimizeTMemLayoutsPass` is historical. The current Lane B
  full dry gate uses Triton MLP and passes the local smoke/preflight gate.
- PyTorch MLP fallback: local forward/backward smoke passed in an older gate,
  but eager full warmup OOMed and graph-break compile stalled before training.
  It remains a historical diagnostic path, not the current recommended Lane B
  full dry-gate backend.
- Full Lane B baseline: no completed full baseline exists yet. The next
  non-skip full Lane B launch is blocked by missing explicit full-launch
  authorization. The fresh dry gate is a launch prerequisite only; it did not
  run training.
- Lane C: blocked until a Lane A or Lane B baseline exists.

Until explicit full-launch authorization is granted, a non-skip full run should
stop before training rather than spend GPU time. Diagnostic launches must say
exactly which historical blocker or new hypothesis they are revisiting.
