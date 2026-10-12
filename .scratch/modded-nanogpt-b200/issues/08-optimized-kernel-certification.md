# Issue 08: Optimized Kernel Certification Matrix

Type: task
Status: complete
Blocked by: -

## Intent

Promote optimized-kernel checking from scattered preflight smokes into one
schema-governed certification matrix for modded NanoGPT B200 attempts.

The matrix proves the selected launch tuple is buildable, supported, and smoke
verified while preserving hard evidence for unselected or currently unsupported
FA3, FA4, Triton, SDPA, torch fallback, and source-local kernel paths. The
certification decision is not an all-green matrix: known-unsupported unselected
paths must be visible, but they must not block a supported selected tuple.

Use `.scratch/modded-nanogpt-b200/spec.md`,
`.scratch/modded-nanogpt-b200/rootfs_runtime_env_spec.md`, and
`.scratch/modded-nanogpt-b200/master_plan.md` as the canonical specs.

## Authority Boundary

- No full B200 launch is authorized by this issue.
- Do not install packages, run CUDA probes, run NCCL, run `torchrun`, or mutate
  generated result trees from the host shell.
- Real package, CUDA, FlashAttention, Triton, NCCL, TorchInductor, and
  source-kernel verification must run only through rootfs-aware wrappers after
  `TORCHTITAN_IN_ROOTFS=1`.
- Host-side unit tests may mock package and CUDA-heavy smokes.
- Generated compiler caches, build logs, result artifacts, and kernel reports
  stay out of git.

## Requirements

Implement a first-class `optimized_kernel_report.json` written under the
attempt runtime artifact directory and governed by
`optimized_kernel_report.schema.json`.

Each backend or prerequisite row records:

- `name`, `kind`, `role`, `requested`, `selected_for_launch`,
  `required_for_selected_launch`, `diagnostic_only`;
- `support_status`, `build_status`, `smoke_status`, and `launch_eligible`;
- package/import versions, source path and digest, setup-wrapper provenance,
  selected backend env values, cache directories, GPU arch evidence, stdout
  tail, artifact paths, `failure_class`, and structured blockers.

The current matrix must include at least:

- `attention.fa2`: build or verify `flash-attn==2.8.3.post1` through
  `setup_flash_attention.sh`, preserve CUDA synth package and `sm_100` or
  compute-100 evidence where inspectable, and run the causal BF16 varlen/window
  smoke.
- `attention.fa3`: probe the selected `kernels` /
  `kernels-community/flash-attn3` path, preserving exact missing import,
  missing artifact, missing kernel image, or runtime failure evidence.
- `attention.fa4`: probe or classify as a named unsupported backend when no
  source, profile, package, or provider selects it. Do not silently omit it.
- `attention.flex`: keep diagnostic-only unless a later ticket promotes it.
- `attention.torch_sdpa`: record PyTorch SDPA availability and a causal BF16
  smoke when a source/profile selects it; otherwise record unselected status.
- `mlp.triton`: reuse the source-local
  `FusedLinearReLUSquareFunction.apply` forward/backward smoke and preserve
  compiler-pass failure detail such as
  `TritonNvidiaGPUOptimizeTMemLayoutsPass`.
- `mlp.torch`: verify the local fallback smoke but keep full-mode launch policy
  separate from smoke success.
- `source_triton_kernels` and `source_dc_triton_kernels`: import from the
  selected source tree, record file digests, and smoke the selected dynamic
  correction path when the selected train graph uses it.
- supporting prerequisites: `torch._scaled_mm` FP8, TorchInductor compile
  cache, CUDA NVCC/NVVM or synth package bridge, Triton tensor descriptor,
  NCCL all-reduce evidence, and forbidden component absence such as
  `flashinfer`.

## Acceptance Criteria

- `preflight.py` or the new runtime verifier writes
  `optimized_kernel_report.json` for every full-mode launch gate.
- Launch readiness fails closed when the selected attention backend, selected
  MLP backend, required source-local kernel, or required core prerequisite is
  not `launch_eligible=true`.
- Launch readiness does not fail solely because an unselected FA3, FA4, flex,
  torch fallback, or diagnostic Triton variant is `unsupported` or `unknown`,
  provided the report contains structured evidence.
- The report distinguishes selected launch failures from unselected diagnostic
  failures in summaries and blockers.
- The report digest is included in `runtime_verification.json`,
  `launch_readiness.json`, and attempt summary artifacts when those files
  exist.
- Unit tests cover selected-pass, selected-fail, unselected-unsupported, and
  forbidden-present cases without requiring a live GPU.
- Rootfs wrapper verification covers at least one real selected tuple before a
  later full launch is requested.

## Non-Goals

- No full training launch or retry campaign.
- No broad package manager redesign beyond the report inputs needed here.
- No promotion of FA4, flex, torch SDPA, PyTorch MLP fallback, or FlashInfer to
  full-mode eligibility unless a separate ticket adds a provider-specific
  verifier and launch policy.
- No deletion or cleanup of existing generated results or caches.

## Verification

Expected host-side checks:

```bash
python3 -m pytest -q tests/unit_tests/test_modded_nanogpt_b200_preflight.py tests/unit_tests/test_modded_nanogpt_b200_components.py
python3 experiments/modded_nanogpt_b200/verify_static.py
```

Expected rootfs-only checks for real package or CUDA evidence:

```bash
experiments/modded_nanogpt_b200/certify_optimized_kernels.sh \
  --source experiments/modded_nanogpt_b200/sources/modded-nanogpt-b200-sdpa \
  --attention-backend fa2 \
  --mlp-backend triton \
  --output experiments/modded_nanogpt_b200/results/<run_id>/runtime/optimized_kernel_report.json
```

The rootfs wrapper name is provisional until implementation, but the rootfs
boundary is not provisional.

## Comments

2026-08-18: Design direction approved as a certified matrix gate. Supported
selected paths must pass build and smoke checks; unsupported unselected FA3,
FA4, Triton, and fallback paths must produce structured evidence without
blocking a known-good selected launch tuple.

2026-08-19: Implemented `optimized_kernel_report.schema.json`,
`optimized_kernel_certifier.py`, and the rootfs-aware
`certify_optimized_kernels.sh` wrapper. `run_speedrun.py` now invokes the
certifier for full-mode launch gates after preflight and active-job safety
checks, before skip-run readiness or a real training launch, and includes the
report digest in `launch_readiness.json`.

Selected FA2/Triton certification passed through the rootfs without launching
training:

```bash
experiments/modded_nanogpt_b200/certify_optimized_kernels.sh \
  --source experiments/modded_nanogpt_b200/sources/modded-nanogpt-b200-sdpa \
  --attention-backend fa2 \
  --mlp-backend triton \
  --output experiments/modded_nanogpt_b200/results/issue08_kernel_cert_fa2_triton_20260819T083000Z/runtime/optimized_kernel_report.json \
  --run-id issue08_kernel_cert_fa2_triton_20260819T083000Z \
  --attempt-id issue08_kernel_cert_fa2_triton_20260819T083000Z_attempt_001 \
  --expected-gpus 2
```

The report records `launch_eligible=true`, `blockers=[]`, digest
`a789b17eef84c62d386f4f395d32ec0c0924bc261fcd450c07668d17d3882969`, and
selected-row success for FA2 attention, Triton MLP, source-local Triton/DC
kernels, FP8 `torch._scaled_mm`, TorchInductor cache, Triton tensor descriptor,
and two-rank NCCL. Unselected FA3, FA4, flex, torch SDPA, and torch MLP
fallback rows are preserved as diagnostic/nonblocking evidence. `flashinfer`
is installed in the reusable rootfs through `vllm`, but the selected
modded-nanogpt source does not import or select it, so the row is recorded as
`support_status=installed_unselected` and does not block the selected tuple.

Verification completed inside the rootfs:

```bash
python3 experiments/modded_nanogpt_b200/verify_static.py
python3 -m pytest -q \
  tests/unit_tests/test_modded_nanogpt_b200_optimized_kernel_certifier.py \
  tests/unit_tests/test_modded_nanogpt_b200_cli_guard.py \
  tests/unit_tests/test_modded_nanogpt_b200_run_speedrun.py \
  tests/unit_tests/test_modded_nanogpt_b200_performance_probe.py \
  tests/unit_tests/test_modded_nanogpt_b200_summarize.py \
  tests/unit_tests/test_modded_nanogpt_b200_run_experiment_matrix.py \
  tests/unit_tests/test_modded_nanogpt_b200_diagnose_mlp_backend.py
```

The final verification output was `Static verification passed for 13 file(s).`
and `124 passed in 7.91s`.
