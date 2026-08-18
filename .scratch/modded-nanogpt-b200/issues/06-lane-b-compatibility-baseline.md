# Lane B compatibility baseline

Type: task
Status: blocked
Blocked by: explicit user authorization for full Lane B launch

## Requirement

Define and execute the first Lane B compatibility baseline only after the
preflight verifier, data manifest, speedrun runner, and parser are in place.
The goal is to establish the smallest B200-compatible path that can plausibly
replace the failed Lane A FA3 path without changing upstream train or
validation token streams.

Use `.scratch/modded-nanogpt-b200/spec.md` as the canonical spec.

## Scope

Allowed:

- use a generated variant source tree under
  `experiments/modded_nanogpt_b200/sources/`;
- use repo-local FA2 from `setup_flash_attention.sh`;
- preserve a `variant_patch.diff`;
- run diagnostic and smoke attempts through `run_speedrun.sh`;
- run a full attempt only when explicitly authorized by the user.

Excluded:

- no Lane A claim;
- no hidden ML-affecting patches;
- no train or validation data-stream change;
- no validation-token-count change;
- no generated source, data, logs, or result bundles tracked in git;
- no full run that bypasses preflight or skips NCCL.

## Behavior

The Lane B baseline must:

- start from upstream commit `ecbb586296d3dac36fd206211f25d63bad4a6b35`;
- classify every patch as `environment`, `hardware-detection`,
  `kernel-compat`, `timing-harness`, or `ML-affecting`;
- identify the concrete Lane A blocker each patch addresses;
- set `MODDED_NANOGPT_CE_COMPUTE_CAPABILITY=100` for B200 custom-kernel
  variants;
- use `MODDED_NANOGPT_ATTN_BACKEND=fa2` only after the FA2 setup script passes;
- block `flex` attention for full jobs;
- require Triton MLP and PyTorch MLP to pass local forward/backward smoke before
  any full job, and keep PyTorch MLP blocked for full jobs until its known
  full-job blockers have verified fixes;
- record whether an attempt is smoke, diagnostic, or full.
- produce an explicit baseline artifact path before ticket 04 can launch any
  ablation arm.

## Verification Evidence

- FA2 setup script source-builds or fast-smoke-verifies FlashAttention inside
  rootfs.
- Full-job preflight includes NCCL unless the attempt is explicitly diagnostic.
- Full-job preflight includes SHA verification and a full 900M data manifest.
  - Latest full-mode gate used
    `experiments/modded_nanogpt_b200/results/full_manifest_refresh_20260816T111021Z/data_manifest.json`,
    a schema-v1 `900M` manifest with top-level `verified_sha=true`, 10 shards,
    and `total_bytes=2000010240`.
  - Clean-context manifest SHA hardening fixed the evidence contract before any
    full launch: `prepare_data.build_manifest` now writes top-level
    `verified_sha: true` when shard SHA entries are computed; full-mode
    preflight with `verify_sha=True` now rejects manifests whose top-level
    `verified_sha` is missing or not true while still recomputing hashes for
    accepted manifests; and summarize launch-prerequisite gating now requires
    both parsed manifest SHA evidence and `preflight_data_manifest.verified_sha`
    evidence. Root cause: launch-prerequisite indexing could rely on sidecar or
    preflight SHA while parsed manifest SHA was null because older manifests had
    per-shard `sha256` entries but no top-level `verified_sha`.
    TDD evidence recorded RED `3 failed, 45 passed`, GREEN
    `48 passed in 0.15s`, clean-context verification `48 passed in 0.08s`,
    `py_compile` passing for `prepare_data.py`, `preflight.py`, and
    `summarize.py`, static `rg` confirmation of `verified_sha`,
    `manifest_verified_sha`, and `require_sha` references, and
    `git diff --check` passing. A review found no blocking correctness bugs;
    a test-only follow-up added recomputation and false-preflight-SHA gating
    coverage and recorded `47 passed in 0.15s` plus `git diff --check`
    passing. No launch, GPU, training, data-prep, download, pip, CUDA, NCCL,
    `torchrun`, or artifact-mutating command was run, so this evidence does not
    create a Lane B baseline artifact or refresh launch readiness.
- Full-mode operator dry runs can stop after preflight with
  `launch_readiness.json` and `launch_readiness.md`; those artifacts identify
  whether SHA/NCCL/manifest gates are launch-ready while preserving
  `training_launched=false` and excluding the attempt from baseline stats.
  - `experiments/modded_nanogpt_b200/results/lane_b_full_gate_20260816T001547Z/`
    is a full-mode `--skip-run` gate attempt. It records rootfs, Torch/CUDA,
    FlashAttention, 8x B200 inventory, Torch primitives, NCCL all-reduce,
    source policy, full data-manifest SHA verification, and FA2 attention
    passing. It stops before launch at `mlp_backend`; `launch_readiness.json`
    records `ready_to_launch=false`, `training_launched=false`,
    `nccl_checked=true`, `data_manifest_checked=true`, `verified_sha=true`, and
    `blocked_by=[mlp_backend]`.
  - The latest `preflight_report.json` preserves the successful local PyTorch
    MLP smoke separately from the full-mode policy block:
    `local_smoke.output_shape=[2, 16, 768]` and
    `full_mode_policy=blocked_without_allow_previous_stall`.
  - The refreshed `summary.json` preserves the same detail under
    `preflight_mlp_backend`, and `analysis.md` renders the MLP backend,
    local-smoke backend/output shape, and full-mode policy blocker for human
    review without requiring direct JSON inspection.
  - Historical Triton MLP full-mode no-go paths preserve structured preflight
    detail: `backend=triton`, `blocked_kernel=linear_relu_square_kernel`,
    `blocked_arch=sm100`, `failure_class=triton_compile`, and
    `compiler_pass=TritonNvidiaGPUOptimizeTMemLayoutsPass`.
  - `experiments/modded_nanogpt_b200/results/lane_b_full_gate_triton_20260816T005014Z/`
    is the current paired full-mode `--skip-run` gate for `mlp_backend=triton`.
    It records the same rootfs, Torch/CUDA, FlashAttention, 8x B200, Torch
    primitive, NCCL, source policy, full manifest, and SHA gates as the PyTorch
    gate, and now also passes the Triton MLP local smoke with
    `local_smoke.output_shape=[2, 16, 768]`. It stops only because `--skip-run`
    was requested; `launch_readiness.json` records `ready_to_launch=true`,
    `training_launched=false`, `launch_authority_required=true`, and
    `blocked_by=[]`.
  - `experiments/modded_nanogpt_b200/results/lane_b_full_authority_guard_20260816T010926Z/`
    reran the same full-mode Triton preflight without `--skip-run` and without
    the explicit full-launch authorization token. It records
    `ready_to_launch=true`, `training_launched=false`, exit code `21`,
    `blocker.phase=launch_authority`,
    `launch_authorization_required_token=launch-full-b200`, and proves the
    runner stops before `torchrun` unless
    `--launch-authorization=launch-full-b200` is present.
  - `experiments/modded_nanogpt_b200/results/lane_b_full_patch_class_guard_20260816T012032Z/`
    reran the same full-mode Triton authority guard after adding structured
    Lane B patch classification. It records the same launch-ready/no-training
    state plus `variant_patch_classification.json`, which classifies
    `train_gpt.py` as `hardware-detection` for the first Lane A blocker
    `FA3 no kernel image on B200`, and `triton_kernels.py` as `kernel-compat`
    for the first Lane A blocker
    `Triton sm100 custom kernel compile/runtime blocker`.
  - `experiments/modded_nanogpt_b200/results/lane_b_full_prelaunch_patch_guard_20260816T012647Z/`
    reran the same authority guard after moving Lane B `variant_patch.diff` and
    `variant_patch_classification.json` capture before preflight and before any
    possible launch. It records the same classifications, embeds them in
    `summary.json`, renders them in `analysis.md`, and still exits `21` at
    `launch_authority` before `torchrun`.
  - Full Lane B attempts now have a preflight-side source-classification guard:
    any `variant_patch_classification.json` entry with
    `patch_class=unclassified` stops the run before preflight or training with
    exit code `21`, `blocker.phase=variant_patch_classification`, and
    `launch_readiness.ready_to_launch=false`, even when the explicit
    full-launch token is present. This is covered by
    `test_full_lane_b_rejects_unclassified_variant_patch_before_preflight`.
  - A clean-context runner hardening slice fixed the Lane B FA2 setup path
    before launch work: for Lane B `attention_backend=fa2`, the runner now
    invokes `setup_flash_attention.sh` before preflight, and setup failure
    writes blocker and exit evidence without running preflight or training.
    The same slice fixed manifest-derived `DATA_PATH` for both absolute and
    repo-relative shard paths by deriving the data root from shard roots and
    resolving repo-relative paths according to repo/rootfs cwd semantics. It
    also made the initial full-mode `attempt.json` classification fail closed
    with `claim_eligible: False` until preflight evidence can justify changing
    it. Unit/static evidence only: RED `4 failed, 24 passed`, then RED
    `1 failed, 27 passed` for the repo-relative `DATA_PATH` regression, then
    GREEN `29 passed in 0.70s` for the focused runner tests; independent
    verification recorded `29 passed in 0.61s`, `py_compile` passing,
    `git diff --check` passing, and static `rg` confirmation of
    `_manifest_shard_paths`, `_data_root_from_manifest`,
    `setup_flash_attention`, `DATA_PATH`, and `claim_eligible: False`.
    Independent review found no blocking findings remain; residual risk is
    that the tests use fake command runners/environment construction rather
    than a live rootfs launch. No launch, GPU, training, data-prep, download,
    pip, CUDA, or `torchrun` command was run, so this does not create a Lane B
    baseline artifact.
  - Clean-context early active-job evidence capture hardening fixed the
    remaining ready-row timing gap without running a real launch. Full-mode
    attempts now capture parsed `active_jobs`
    evidence after successful preflight and before either `--skip-run` or
    missing launch authorization returns. Active jobs or active-job scan errors
    block through the existing `active_jobs` blocker before skip-run or
    launch-authority success can be reported. Independent verifier evidence:
    `tests/unit_tests/test_modded_nanogpt_b200_run_speedrun.py -q` reported
    `31 passed in 1.08s`,
    `tests/unit_tests/test_modded_nanogpt_b200_summarize.py -q` reported
    `43 passed in 0.10s`,
    `python -m py_compile experiments/modded_nanogpt_b200/run_speedrun.py experiments/modded_nanogpt_b200/summarize.py`
    passed,
    `git diff --check -- experiments/modded_nanogpt_b200/run_speedrun.py tests/unit_tests/test_modded_nanogpt_b200_run_speedrun.py`
    passed, and `python experiments/modded_nanogpt_b200/verify_static.py`
    reported `Static verification passed for 39 file(s).` The overall
    objective remains incomplete: no full Lane A/B baseline exists and full
    launch still requires explicit user authorization with `launch-full-b200`.
  - The latest non-launch Lane B full-mode refresh is
    `experiments/modded_nanogpt_b200/results/lane_b_full_skiprun_refresh_20260816T111056Z`.
    Its summary records `preflight_ok=true`,
    `launch_readiness.ready_to_launch=true`,
    `launch_readiness.training_launched=false`,
    `launch_readiness.skip_run=true`, `blocked_by=[]`,
    `full_mode_gates.verified_sha=true`,
    `full_mode_gates.nccl_checked=true`, and
    `full_mode_gates.data_manifest_checked=true`. Its active-job scan is clear:
    `active_jobs.ok=true`, `active_job_count=0`, and `active_jobs=[]`. The
    blocker is `phase=not_launched` with a message that skip-run was requested
    and training was not launched. `attempt.json["command"]["argv"]` and
    `command.argv` include `--skip-run` and do not include
    `--launch-authorization=launch-full-b200`.
  - The current audit snapshot of
    `experiments/modded_nanogpt_b200/results/run_index.json` records
    `total_attempts=24`, `baseline_stats.count=0`,
    `len(launch_prerequisite_attempts)=1`, and
    `len(launch_ready_attempts)=0`. The single prerequisite row points to
    `experiments/modded_nanogpt_b200/results/lane_b_full_skiprun_refresh_20260816T111056Z/summary.json`.
    No full Lane A or Lane B baseline exists, no non-skip launch-ready row
    exists, no training launch occurred, and a full launch still requires
    explicit user authorization.
  - Blocked-state policy: until `launch-full-b200` authorization is supplied or
    a material input changes, do not re-run equivalent audits, summarizer
    refreshes, preflights, GPU probes, or launch-readiness checks merely to
    reconfirm this same blocked state. Cite the latest recorded evidence above
    and stop at the authorization boundary.
  - A clean-context subagent later refreshed the ignored real run index through
    the approved rootfs-aware summarizer:
    `experiments/modded_nanogpt_b200/summarize.sh --results-root experiments/modded_nanogpt_b200/results --output experiments/modded_nanogpt_b200/results/run_index.json`,
    which exited `0`. It did not run launch, GPU, training, preflight,
    data-prep, download, pip, CUDA, NCCL, `torchrun`, or full-launch commands.
    Historical refreshed facts, now superseded by the
    `lane_b_full_skiprun_refresh_20260816T111056Z` index refresh:
    `total_attempts=23`, `baseline_count=0`, `launch_prerequisite_count=0`,
    `launch_ready_count=0`, `launch_readiness_exclusion_stats.count=20`,
    `by_phase.data_manifest_summary.verified_sha.count=11`,
    `by_phase.preflight_data_manifest.verified_sha.count=2`, and
    `by_phase.launch_readiness.count=7`; both
    `launch_prerequisite_attempts=[]` and `launch_ready_attempts=[]`.
    Representative SHA-gated exclusion paths include
    `experiments/modded_nanogpt_b200/results/lane_b_full_authority_guard_20260816T010335Z/summary.json`,
    `experiments/modded_nanogpt_b200/results/lane_b_full_authority_guard_20260816T010926Z/summary.json`,
    `experiments/modded_nanogpt_b200/results/lane_b_full_gate_20260816T000411Z/summary.json`,
    `experiments/modded_nanogpt_b200/results/lane_b_full_gate_20260816T000617Z/summary.json`,
    `experiments/modded_nanogpt_b200/results/lane_b_full_gate_20260816T001037Z/summary.json`,
    and
    `experiments/modded_nanogpt_b200/results/lane_b_full_rootfs_guard_20260816T031154Z/summary.json`.
    The requested SHA-exclusion `jq` query failed on a null phase during
    operator inspection, so a null-safe equivalent was used. This refresh
    confirms the stricter SHA evidence gates demote older generated artifacts;
    it does not establish a Lane B baseline or runtime launch readiness.
  - Clean-context parser pointer hardening changed only `parse_log.py` and its
    focused parser tests. `parse_log.py` now resolves non-absolute
    `data_manifest_pointer.path` values relative to the pointer file directory.
    For pointer manifests, it propagates explicit top-level `verified_sha` from
    the resolved target only when the pointer lacks top-level `verified_sha`;
    it does not infer `verified_sha` from shard `sha256` entries. RED failed
    because relative result-local pointer targets were not resolved and target
    manifest fields were not loaded. GREEN evidence: parse_log plus summarize
    tests passed with `71 passed in 0.14s`, clean-context verification recorded
    `71 passed in 0.13s`, `py_compile` passed for `parse_log.py` and
    `summarize.py`, and `git diff --check` passed for scoped files, with the
    static verifier covering untracked scoped content. Clean-context review
    found no blocking issues, confirmed summarize fail-closed gates remain
    intact, and confirmed `verified_sha` is not inferred from shard hashes.
    After the parser fix, the ignored real index was refreshed through the
    approved rootfs-aware summarizer. Historical refreshed facts, now
    superseded by the `lane_b_full_skiprun_refresh_20260816T111056Z` index
    refresh: `total_attempts=23`, `baseline_count=0`,
    `launch_prerequisite_count=0`, `launch_ready_count=0`, `stale_count=20`,
    and `launch_readiness_exclusion_stats.count=20`. Exclusion phases were
    `data_manifest_summary.verified_sha=11`, `launch_readiness=7`, and
    `preflight_data_manifest.verified_sha=2`, plus 3 diagnostic failed-attempt
    rows with no `launch_readiness_exclusion` phase. Existing resolved
    manifests still lacked top-level `verified_sha`, so those demotions did
    not clear during that historical refresh. The later
    `lane_b_full_skiprun_refresh_20260816T111056Z` index refresh now surfaces
    one skip-run launch-prerequisite row while keeping
    `launch_ready_attempts` empty. This does not establish a Lane B baseline or
    runtime launch readiness; no launch, GPU, training, preflight, data-prep,
    download, pip, CUDA, NCCL, `torchrun`, or full-launch command was run.
- Bounded Triton MLP diagnostic evidence:
  `experiments/modded_nanogpt_b200/results/mlp_diag_triton_shared_20260816T005734Z/mlp_diagnostic.json`
  ran through `diagnose_mlp_backend.sh` inside rootfs. It records Torch
  `2.13.0+cu132`, Triton `3.7.1`, CUDA runtime `13.2`, rootfs marker `1`,
  `ok=true`, and `detail.output_shape=[2, 16, 768]`. The diagnostic and
  full-mode preflight now share the same Triton MLP smoke helper, with unit
  coverage preventing future drift.
  The previous diagnostic at
  `experiments/modded_nanogpt_b200/results/mlp_diag_triton_20260816T004154Z/mlp_diagnostic.json`
  identified the fixed bug as `triton_backward_shape_mismatch` at
  `FusedLinearReLUSquareFunction.backward`, caused by computing `post.T @
  grad_output` before flattening 3D MLP tensors.
- `summary.json` records null final metrics for attempts that do not reach final
  validation.
- Any full attempt that reaches final validation reports final `val_loss`,
  upstream `train_time`, `step_avg`, peak memory, shell wall-clock, source diff
  status, and data manifest.
