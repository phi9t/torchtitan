# T12 report: GLM EP phase 2 CPU/fake-backend tickets 1-3

## Result

T12 is complete for the CPU/fake-backend portion of GLM EP phase 2. I added an isolated package,
`torchtitan/experiments/glm53_flash_ep/`, with a GLM-compatible adapter onto the common MoE token-dispatch EP path,
focused tests, a rank-local EP HF safetensors loader, trainable-only checkpoint round-trip coverage, four GPU follow-up
briefs, and an update to the EP research doc.

No commit was made. I did not run real-GPU training. I did not make T12 edits under
`torchtitan/experiments/glm53_flash/` or `torchtitan/models/common/`. The worktree already had unrelated untracked files
under `torchtitan/experiments/glm53_flash/`; they were left untouched.

## Implemented

New package:

- `torchtitan/experiments/glm53_flash_ep/__init__.py`
- `torchtitan/experiments/glm53_flash_ep/flavors.py`
- `torchtitan/experiments/glm53_flash_ep/moe.py`
- `torchtitan/experiments/glm53_flash_ep/sharding.py`
- `torchtitan/experiments/glm53_flash_ep/model.py`
- `torchtitan/experiments/glm53_flash_ep/state_dict_adapter.py`
- `torchtitan/experiments/glm53_flash_ep/sharded_hf_load.py`
- `torchtitan/experiments/glm53_flash_ep/checkpoint.py`
- `torchtitan/experiments/glm53_flash_ep/config_registry.py`
- `torchtitan/experiments/glm53_flash_ep/validate_ep_matrix.py`
- `torchtitan/experiments/glm53_flash_ep/tests/test_glm53_flash_ep.py`

Behavior:

- `Glm53FlashEPForCausalLM` subclasses the existing GLM model package boundary and replaces sparse GLM MoE layers with
  `GlmEPMoE`.
- `GlmEPMoE` preserves GLM router semantics, sigmoid correction bias, shared expert path, and LoRA trainable policy, but
  stores routed experts as common `GroupedLinear` `w13`/`w2` weights so they can use the common token-dispatch EP path.
- EP sharding rejects TP, CP, and PP for this prototype, validates expert divisibility, shards `w13`/`w2` on `Shard(0)`,
  and keeps sparse FSDP on `edp_shard`.
- `Glm53FlashEPStateDictAdapter` maps common EP expert keys back to released GLM HF expert keys.
- `load_rank_local_ep_hf_safetensors()` loads only local EP expert HF keys, skips LoRA tensors, and records
  `expert_keys_loaded`.
- `Glm53FlashEPLoRACheckpointManager` uses the EP rank-local loader and preserves trainable-only LoRA checkpointing.
- Configs added: `glm53_flash_ep_debugmodel`, `glm53_flash_ep_debugmodel_lora`, and `glm53_flash_ep_mini`.
- `GLM53_FLASH_EP_DEGREE` controls the prototype EP degree.

## Validation

All Python/model/test/static checks below ran inside the rootfs with `TORCHTITAN_ROOTFS_ENV=glm53_train` and
`unset CC CXX`.

Import/build smoke:

```bash
TORCHTITAN_ROOTFS_ENV=glm53_train scripts/rootfs/enter_rootfs.sh -- bash -lc '
  unset CC CXX
  cd /workspace/torchtitan
  python - <<PY
from torchtitan.experiments.glm53_flash_ep.config_registry import glm53_flash_ep_debugmodel
from torchtitan.experiments.glm53_flash_ep.moe import GlmEPMoE

config = glm53_flash_ep_debugmodel()
config.model.set_sharding_(config.parallelism)
model = config.model.build()
assert isinstance(model.layers["1"].mlp, GlmEPMoE)
print(model.layers["1"].mlp.experts.w13.weight.shape)
PY
'
```

Result: model built; sparse layer `layers["1"].mlp` is `GlmEPMoE`; `w13.weight.shape` printed
`torch.Size([8, 2, 128, 256])`.

Focused EP tests:

```bash
TORCHTITAN_ROOTFS_ENV=glm53_train scripts/rootfs/enter_rootfs.sh -- bash -lc '
  unset CC CXX
  cd /workspace/torchtitan
  pytest -q torchtitan/experiments/glm53_flash_ep/tests/test_glm53_flash_ep.py
'
```

Result: `10 passed, 14 warnings in 6.04s`.

The focused tests cover registry wiring, EP mesh ladder expectations, unsupported-axis validation, expert sharding
placement, router parity, bounded GLM-vs-common MoE deltas, state-dict mapping, rank-local EP loader behavior, and
trainable-only checkpoint round-trip.

Original GLM regression slice:

```bash
TORCHTITAN_ROOTFS_ENV=glm53_train scripts/rootfs/enter_rootfs.sh -- bash -lc '
  unset CC CXX
  cd /workspace/torchtitan
  pytest -q torchtitan/experiments/glm53_flash/tests/test_glm53_flash.py \
    -k "lora_debug_model_only_lora_params_require_grad or trainable_model_wrapper_exposes_only_lora_tensors or grouped_expert_loader_reads_only_target_local_slice or state_dict_adapter_restores_rank_local_expert_subset"
'
```

Result: `4 passed, 19 deselected, 14 warnings in 6.02s`.

EP matrix validation:

```bash
TORCHTITAN_ROOTFS_ENV=glm53_train scripts/rootfs/enter_rootfs.sh -- bash -lc '
  unset CC CXX
  cd /workspace/torchtitan
  python -m torchtitan.experiments.glm53_flash_ep.validate_ep_matrix --world-size 4 --ep 1 --ep 2 --ep 4
  python -m torchtitan.experiments.glm53_flash_ep.validate_ep_matrix --world-size 8 --ep 1 --ep 2 --ep 4 --ep 8
  if python -m torchtitan.experiments.glm53_flash_ep.validate_ep_matrix --world-size 4 --ep 8 >/tmp/glm53_ep_invalid.out 2>/tmp/glm53_ep_invalid.err; then
    echo "unexpected invalid matrix success"
    exit 1
  fi
  cat /tmp/glm53_ep_invalid.err
'
```

Result:

- world size 4 debug config: `dp_shard=4`, `edp_shard=4/2/1`, local experts `8/4/2` for EP `1/2/4`;
- world size 4 mini config: `dp_shard=4`, `edp_shard=4/2/1`, local experts `64/32/16` for EP `1/2/4`;
- world size 8 debug config: `dp_shard=8`, `edp_shard=8/4/2/1`, local experts `8/4/2/1` for EP `1/2/4/8`;
- world size 8 mini config: `dp_shard=8`, `edp_shard=8/4/2/1`, local experts `64/32/16/8` for EP `1/2/4/8`;
- invalid world size 4, EP 8 failed with `ValueError: expert_parallel_degree (8) must divide dp_shard * cp * tp (4)`.

Fake CLI config path:

```bash
TORCHTITAN_ROOTFS_ENV=glm53_train scripts/rootfs/enter_rootfs.sh -- bash -lc '
  unset CC CXX
  cd /workspace/torchtitan
  for ws in 4 8; do
    for ep in 1 2 4 8; do
      if [ "$ws" = 4 ] && [ "$ep" = 8 ]; then
        continue
      fi
      GLM53_FLASH_EP_DEGREE=$ep NGPU=$ws COMM_BACKEND=fake \
        MODULE=torchtitan.experiments.glm53_flash_ep.config_registry \
        CONFIG=glm53_flash_ep_debugmodel \
        ./run_train.sh --print-config >/tmp/glm53_ep_print_config_${ws}_${ep}.txt
    done
  done
'
```

Result: all current fake-backend config-print checks completed for EP `{1,2,4}` at logical world size 4 and EP
`{1,2,4,8}` at logical world size 8. This checkout's launcher uses `COMM_BACKEND=fake`; the older `COMM_MODE=fake_backend`
and `COMM_MODE=local_tensor` wording is stale. Local tensor behavior was covered through CPU DTensor placement tests,
direct `_local_tensor` loader assertions, and the CPU mesh validator without launching real training.

Focused hooks on the new EP package:

```bash
TORCHTITAN_ROOTFS_ENV=glm53_train scripts/rootfs/enter_rootfs.sh -- bash -lc '
  unset CC CXX
  cd /workspace/torchtitan
  files=$(find torchtitan/experiments/glm53_flash_ep -name "*.py" -print)
  pre-commit run ufmt --files $files
  for hook in flake8 pydoclint codespell; do
    pre-commit run "$hook" --files $files
  done
'
```

Result: `ufmt`, `flake8`, `pydoclint`, and `codespell` passed. I did not run repo-wide `pyrefly` because
`env-facts.md` says it has known pre-existing failures and can auto-edit unrelated files.

## Numerical notes

Router parity is exact for the focused fixed-input test: top-k expert IDs and routing weights match the current GLM
router.

Full MoE CPU parity is bounded rather than bitwise:

- output max delta: `< 3.0e-5`;
- input gradient max delta: `< 1.0e-8`;
- router weight gradient max delta: `< 1.0e-8`;
- expert `w13`/`w2` gradient max delta: `< 5.0e-8`.

Reason: the EP adapter intentionally uses the common grouped expert implementation while the current GLM path uses its
own local expert loop. The operation order is not identical even when weights and routes match.

## Loader and checkpoint notes

The tiny safetensors loader fixture proved that fake rank 2 of 4 reads only these HF expert keys:

- `model.language_model.layers.0.mlp.experts.2.gate_proj.weight`
- `model.language_model.layers.0.mlp.experts.2.up_proj.weight`
- `model.language_model.layers.0.mlp.experts.2.down_proj.weight`

LoRA tensors remain initialized and are skipped by the frozen-base loader. The trainable-only checkpoint test saves only
keys ending in `.lora_a` or `.lora_b`, excludes `.experts.w13.` and `.experts.w2.`, and reloads those trainable tensors
into a fresh EP LoRA model.

## GPU follow-up briefs

Added:

- `.scratch/glm53-flash-sft-ladder/sdd/task-13-brief.md`: EP-4 truncated-depth full-width numerical parity.
- `.scratch/glm53-flash-sft-ladder/sdd/task-14-brief.md`: EP-5 full-depth 8-GPU EP ladder.
- `.scratch/glm53-flash-sft-ladder/sdd/task-15-brief.md`: EP-6 full-depth 4-GPU feasibility with the selected memory
  direction.
- `.scratch/glm53-flash-sft-ladder/sdd/task-16-brief.md`: EP-7 optional Dist-MoE evaluation.

These briefs include exact rootfs commands for the intended GPU work, but they also name GPU-only configs and summary
helpers that those tickets must add before execution.

## Research doc update

Updated `docs/research/2026-10-14-glm-5.3-flash-expert-parallelism.md` section 6 with:

- T12 CPU/fake-backend status;
- EP-1, EP-2, and EP-3 results;
- links to the four GPU follow-up briefs;
- updated recommendation that the next ticket is EP-4/T13.

## Remaining limitations

- No real all-to-all GPU training was run in T12.
- The prototype does not yet provide full-depth or truncated-depth full-width Countdown GLM EP training configs.
- The native in-module routed expert parameter names change to common `w13`/`w2`; compatibility is handled through the
  EP state-dict adapter. If preserving native GLM keys inside the module becomes required, that should be a deliberate
  follow-up design change.
- TP, CP, and PP remain unsupported for GLM EP. Therefore 4-GPU EP=8 remains invalid until one of those axes is added.
- Routed-expert LoRA under expert-axis sharding is not added beyond preserving the existing GLM LoRA trainable policy.
