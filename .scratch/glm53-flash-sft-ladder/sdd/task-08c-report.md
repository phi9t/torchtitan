# T08c report: GLM-5.3-Flash bf16 drift diagnosis

Status: COMPLETE. Claim label: `deterministic_comparison`. No commit was made.

Surface: repo-local GLM experiment diagnostics under
`torchtitan/experiments/glm53_flash_bf16_diag/`. The T08b-owned files under
`torchtitan/experiments/glm53_flash/` were read but not edited.

## Result

TT-bf16 is not the outlier. HF-bf16 and TT-bf16 are essentially equally far from
the HF-fp32 truth on the real-weight layers 20-27 Countdown slice. HF-bf16 is
slightly worse on the aggregate maxima, but the difference is not meaningful
enough to identify a TT-only bug.

Conclusion: the layers 20-27 bf16 gap is reference/noise from the bf16 path, not
a TorchTitan-specific correctness bug. The TT sub-block bisection and monkeypatch
fix branch was skipped by the ticket's decision rule.

Patch path: not applicable. No
`.scratch/glm53-flash-sft-ladder/evidence/t08c-bf16-fix.patch` was created.

## Evidence

Accepted evidence artifacts:

- `.scratch/glm53-flash-sft-ladder/evidence/task-08c/bf16-truth-compare.json`
- `.scratch/glm53-flash-sft-ladder/evidence/task-08c/bf16-truth-compare.stdout.json`
- `.scratch/glm53-flash-sft-ladder/evidence/task-08c/stream-smoke-layer20.json`
- `.scratch/glm53-flash-sft-ladder/evidence/task-08c/LARGE_FILES.md`

Large tensor dumps were moved out of the repo scratch tree and are pointed to by
`LARGE_FILES.md`:

- `/data05/philip.yang/glm53-ladder-cache/evidence-large/task-08c/fp32-truth-tensors.pt`
- `/data05/philip.yang/glm53-ladder-cache/evidence-large/task-08c/stream-smoke-layer20-truth.pt`

Command shape:

```bash
TORCHTITAN_ROOTFS_ENV=glm53_ref \
TORCHTITAN_ROOTFS_FORMAL_CACHE_HOST=/data05/philip.yang/glm53-ladder-cache \
scripts/rootfs/enter_rootfs.sh -- bash -lc \
  'unset CC CXX; cd /workspace/torchtitan && python -m torchtitan.experiments.glm53_flash_bf16_diag.bf16_truth_diag stream-compare \
    --checkpoint /project/formal-cache/model-weights/zai-org/GLM-5.3-Flash-BF16 \
    --data-file /project/formal-cache/datasets/countdown-sft/v1/train/formatting.jsonl \
    --layer-start 20 --num-layers 8 --device cpu --no-long-sequence \
    --output .scratch/glm53-flash-sft-ladder/evidence/task-08c/bf16-truth-compare.json \
    --truth-tensors .scratch/glm53-flash-sft-ladder/evidence/task-08c/fp32-truth-tensors.pt'
```

The accepted run streams one real layer at a time and carries hidden states
forward between layers. This preserved the fp32 identity gate while keeping
accepted-run peak RSS below the 200 GiB guard: recorded max RSS was
`117.60253143310547` GiB, and final RSS was `1.9832305908203125` GiB.

A discarded first monolithic 8-layer CPU attempt exceeded the host guard before
producing evidence: observed RSS was about `361398632` KiB, roughly `344.7` GiB.
It was killed and replaced with the streamed one-layer-at-a-time run above.

## Summary Table

Truth is HF-fp32. The fp32 TT sanity check is included to prove the streamed
diagnostic still reproduces the existing deterministic comparison.

| Candidate vs truth | Min top-1 | Max KL truth->candidate | Max final abs | Max layer abs | Max layer mean abs |
| --- | ---: | ---: | ---: | ---: | ---: |
| TT-fp32 | 1.0 | 0.0001911335129989311 | 0.7716797590255737 | 0.058231860399246216 | 0.00014791548892389983 |
| TT-bf16 | 0.02631578966975212 | 1.3880523443222046 | 27.459651947021484 | 44.55101776123047 | 0.12575286626815796 |
| HF-bf16 | 0.02631578966975212 | 1.3974062204360962 | 27.584596633911133 | 45.30101776123047 | 0.12854929268360138 |

Per-sample final-logit results:

| Sample | TT-bf16 top-1 | TT-bf16 KL | TT-bf16 max abs | HF-bf16 top-1 | HF-bf16 KL | HF-bf16 max abs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `cd-14-16-49-t47` | 0.06578947603702545 | 1.335719347000122 | 27.459651947021484 | 0.05263157933950424 | 1.3353179693222046 | 27.459651947021484 |
| `cd-24-8-22-t54` | 0.06578947603702545 | 1.355976939201355 | 27.459651947021484 | 0.06578947603702545 | 1.3540655374526978 | 27.459651947021484 |
| `cd-17-95-3-t75` | 0.09210526198148727 | 1.375407338142395 | 27.334596633911133 | 0.09210526198148727 | 1.388088583946228 | 27.584596633911133 |
| `cd-24-49-15-t58` | 0.02631578966975212 | 1.3880523443222046 | 27.459638595581055 | 0.02631578966975212 | 1.3974062204360962 | 27.459638595581055 |

## Per-Layer Hidden Drift

Layer stats are max over the four Countdown prompts.

| Layer | TT kind | TT-bf16 max abs | TT-bf16 max mean abs | HF kind | HF-bf16 max abs | HF-bf16 max mean abs |
| ---: | --- | ---: | ---: | --- | ---: | ---: |
| 20 | `linear_attention` | 0.3923139274120331 | 0.011373242363333702 | `linear_attention` | 0.3893842399120331 | 0.011401507072150707 |
| 21 | `linear_attention` | 0.6317768096923828 | 0.021336454898118973 | `linear_attention` | 0.5770893096923828 | 0.021328549832105637 |
| 22 | `linear_attention` | 0.650048017501831 | 0.03706388175487518 | `linear_attention` | 0.6995453834533691 | 0.03718133270740509 |
| 23 | `deepseek_sparse_attention` | 1.112874150276184 | 0.06351237744092941 | `indexed_attention` | 1.112874150276184 | 0.06364411115646362 |
| 24 | `linear_attention` | 1.4090150594711304 | 0.071302630007267 | `linear_attention` | 1.4246399402618408 | 0.07166606932878494 |
| 25 | `linear_attention` | 44.55101776123047 | 0.11162739992141724 | `linear_attention` | 45.30101776123047 | 0.11324950307607651 |
| 26 | `linear_attention` | 25.74811553955078 | 0.1227523684501648 | `linear_attention` | 26.998111724853516 | 0.12502069771289825 |
| 27 | `deepseek_sparse_attention` | 25.86159896850586 | 0.12575286626815796 | `indexed_attention` | 27.111597061157227 | 0.12854929268360138 |

The shared layer-25 jump is the branch decision: both implementations drift from
fp32 truth in the same region and with the same scale. TT-bf16 is not uniquely
far from truth.

## Bisection Result

Bisection was not run because the prerequisite was not met. The ticket asked to
bisect only if TT-bf16 is the outlier. It is not: TT-bf16 is marginally closer
than HF-bf16 on aggregate max KL, max final abs, max layer abs, and max layer
mean abs.

No responsible TT sub-block or exact TT op was named, because the evidence does
not support a TT-specific culprit. The categories listed in the brief
(`mHC`, attention/KDA/sparse MLA, router, experts, shared expert, norms,
Sinkhorn, recurrence state dtype) remain possible bf16-noise contributors in the
shared/reference sense, not isolated TT defects.

## Verification

Passed syntax check:

```bash
TORCHTITAN_ROOTFS_ENV=glm53_ref \
TORCHTITAN_ROOTFS_FORMAL_CACHE_HOST=/data05/philip.yang/glm53-ladder-cache \
scripts/rootfs/enter_rootfs.sh -- bash -lc \
  'unset CC CXX; cd /workspace/torchtitan && python -m py_compile torchtitan/experiments/glm53_flash_bf16_diag/bf16_truth_diag.py'
```

Passed deterministic stream smoke:

```text
layers=20..20, device=cpu, mode=stream_one_layer_at_a_time
TT-fp32 vs HF-fp32 truth: min top-1 1.0, max final abs 3.814697265625e-06
TT-bf16 vs HF-fp32 truth: min top-1 0.34210526943206787, max KL 0.2543221712112427
HF-bf16 vs HF-fp32 truth: min top-1 0.3552631437778473, max KL 0.2557189464569092
accepted-run max RSS 112.34434509277344 GiB
```

Passed scoped hooks on the new diagnostic files and this report:

```text
ufmt
flake8
pydoclint
codespell
trailing-whitespace
end-of-file-fixer
check-ast
check-merge-conflict
```

The first offline pre-commit attempt could not fetch hook environments from
GitHub. The passing run used the documented `TORCHTITAN_ROOTFS_NETWORK=networked`
rootfs mode and was scoped to:

```text
torchtitan/experiments/glm53_flash_bf16_diag/__init__.py
torchtitan/experiments/glm53_flash_bf16_diag/bf16_truth_diag.py
.scratch/glm53-flash-sft-ladder/sdd/task-08c-report.md
```

Full `pre-commit run --all-files` was not run; the repo environment notes say
the pyrefly hook currently fails repo-wide and may auto-edit unrelated files.
