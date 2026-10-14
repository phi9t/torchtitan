# T17 report: GLM MoE grouped expert fast path

## Result

T17 is complete. I added a native GLM routed-expert grouped implementation behind
`GLM53_FLASH_EXPERTS_IMPL in {loop, grouped}` with default `loop`, so existing
live T08d imports retain the old Python-loop behavior unless the env selects
`grouped`.

The fast path preserves the existing GLM parameter tensors and state-dict keys:

- `layers.*.mlp.experts.gate_up_proj`: `[E, 2F, D]`
- `layers.*.mlp.experts.down_proj`: `[E, D, F]`

No checkpoint adapter, rank-local HF loader, or LoRA trainable policy key names
change.

## Implemented

Files changed:

- `torchtitan/experiments/glm53_flash/flavors.py`
  - added `Glm53FlashConfig.experts_impl`, defaulting from
    `GLM53_FLASH_EXPERTS_IMPL` and validating `{loop, grouped}`.
- `torchtitan/experiments/glm53_flash/model.py`
  - kept the original per-expert loop as `_forward_loop`;
  - added `_forward_grouped`, which flattens token-slot assignments, sorts by
    expert on tensor ops, builds cumulative offsets, runs grouped gate/up and
    down projections, applies the same SwiGLU clamp semantics, and combines with
    `index_add_`;
  - uses `torch._grouped_mm` for CUDA bf16/fp16 and a tensorized `bmm` fallback
    for CPU/fp32 tests.
- `torchtitan/experiments/glm53_flash/tests/test_glm53_flash.py`
  - added env/default validation;
  - added grouped-vs-loop forward/backward parity for fp32 and bf16;
  - covered top-8 routing with zero-token experts;
  - checked full `GlmMoE` output and gradients, including router/shared-expert
    gradients.
- `torchtitan/experiments/glm53_flash/benchmark_grouped_experts.py`
  - added a synthetic full-width one-layer `GlmMoE` benchmark.

## Parity

Evidence file:
`.scratch/glm53-flash-sft-ladder/evidence/task-17/grouped_experts_parity.json`

Focused expert parity, top-8 routing, 11 experts with experts 9 and 10 receiving
zero tokens:

| Case | Forward max abs delta | x grad max abs delta | gate/up grad max abs delta | down grad max abs delta | unused expert grad |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU fp32 | 1.4901161193847656e-08 | 1.4901161193847656e-08 | 7.450580596923828e-08 | 7.450580596923828e-08 | 0.0 |
| CUDA bf16 | 0.00048828125 | 0.000244140625 | 0.0 | 0.0 | 0.0 |

The full `GlmMoE` regression test also passed in fp32 with tight tolerance
(`1e-5`) for output, input gradients, router gradients, shared expert gradients,
and routed base expert gradients.

## Benchmark

Evidence file:
`.scratch/glm53-flash-sft-ladder/evidence/task-17/grouped_experts_benchmark.json`

Command ran one released-width `GlmMoE` layer on 1 visible B200, bf16, 4096
tokens, 288 experts, top-8, frozen weights, fwd+bwd to input `x`:

```bash
CUDA_VISIBLE_DEVICES=7 TORCHTITAN_ROOTFS_ENV=glm53_train \
  scripts/rootfs/enter_rootfs.sh -- bash -lc '
    unset CC CXX
    cd /workspace/torchtitan
    python -m torchtitan.experiments.glm53_flash.benchmark_grouped_experts \
      --tokens 4096 --warmup 1 --iters 3 --device cuda
  '
```

GPU state immediately before launch showed all eight B200s busy at 96-100% util,
so this is `shared_gpu` timing, not clean matched performance evidence. Peak
memory remained below the ticket cap.

| Impl | ms/iter | Peak memory | Speedup |
| --- | ---: | ---: | ---: |
| loop | 1352.5309903294935 | 15.031059265136719 GiB | 1.0x |
| grouped | 34.2193393347164 | 16.340137004852295 GiB | 39.52533908091311x |

## Verification

All commands below ran inside the rootfs with `TORCHTITAN_ROOTFS_ENV=glm53_train`
and `unset CC CXX`.

Focused CPU tests:

```bash
pytest -q torchtitan/experiments/glm53_flash/tests/test_glm53_flash.py \
  -k "experts_impl or grouped_experts_match_loop_fp32 or grouped_moe_matches_loop"
```

Result: `3 passed, 24 deselected, 14 warnings`.

Focused CUDA bf16 parity:

```bash
CUDA_VISIBLE_DEVICES=7 pytest -q \
  torchtitan/experiments/glm53_flash/tests/test_glm53_flash.py \
  -k "grouped_experts_match_loop_bf16"
```

Result: `1 passed, 26 deselected, 14 warnings`.

Existing GLM Flash suite:

```bash
CUDA_VISIBLE_DEVICES=7 pytest -q \
  torchtitan/experiments/glm53_flash/tests/test_glm53_flash.py
```

Result after final formatting pass: `26 passed, 1 skipped, 14 warnings`.

Focused hooks:

```bash
pre-commit run ufmt --files \
  torchtitan/experiments/glm53_flash/flavors.py \
  torchtitan/experiments/glm53_flash/model.py \
  torchtitan/experiments/glm53_flash/tests/test_glm53_flash.py \
  torchtitan/experiments/glm53_flash/benchmark_grouped_experts.py
pre-commit run flake8 --files <same files>
pre-commit run pydoclint --files <same files>
pre-commit run codespell --files <same files>
```

Result: all four hooks passed. I did not run repo-wide `pyrefly`, per
`env-facts.md` known pre-existing failures/auto-edits.

## Enablement

Default remains:

```bash
GLM53_FLASH_EXPERTS_IMPL=loop
```

Enable grouped for rung 4 by exporting the env before
`experiments/glm53_flash_sft_ladder/run_rung4_glm.sh` builds the config:

```bash
GLM53_FLASH_EXPERTS_IMPL=grouped \
  experiments/glm53_flash_sft_ladder/run_rung4_glm.sh train
```

Or, if editing the rung-4 runner later, add
`GLM53_FLASH_EXPERTS_IMPL='${GLM53_FLASH_EXPERTS_IMPL:-grouped}'` to the
`rootfs_cmd` export block before `torchrun`.

For T09 4-GPU packages that inherit/build on
`torchtitan.experiments.glm53_flash`, pass the same env into the launch:

```bash
GLM53_FLASH_EXPERTS_IMPL=grouped \
CUDA_VISIBLE_DEVICES=<four-gpu-list> \
TORCHTITAN_ROOTFS_ENV=glm53_train \
scripts/rootfs/enter_rootfs.sh -- bash -lc '
  unset CC CXX
  cd /workspace/torchtitan
  torchrun --nproc_per_node=4 -m torchtitan.train \
    --module torchtitan.experiments.glm53_flash_4gpu_stream.config_registry \
    --config <t09-stream-config> \
    --output-dir <output-dir>
'
```

Use the same leading `GLM53_FLASH_EXPERTS_IMPL=grouped` for the T09 lean/bf16
package (`torchtitan.experiments.glm53_flash_4gpu_lean.config_registry`) or any
other T09 config package that constructs the base GLM config. This checkout's
`--override` flag is for registered config override factories, not dotted field
assignment, so the exact activation for this ticket is the env var above.

## Notes

- No full-model GLM training was launched.
- Benchmark inputs were synthetic tensors only; no dataset, tokenizer, model
  checkpoint, or resume checkpoint input was used, so the HDFS-strict input rule
  was not implicated by the micro-benchmark.
- The grouped path removes the per-expert Python loop and the `.nonzero()` /
  `torch.where` host-sync pattern for selected grouped execution. The preserved
  loop path remains available for debugging and as the default.
