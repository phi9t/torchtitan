# S04 report: GLM element fairness/fast-path pass and Step-3 gate status

Date: 2026-10-14

## Verdict

`S04_ELEMENTS_GATE=PASS`.

`S04_STEP3_GATE=BLOCKED`: the S01 integration seam still launches B0 only, and
`.scratch/glm53-arch-speedrun/sdd/s01-report.md` is missing. I ran the focused
element gates and two bounded B0 smoke runs, but I did not fabricate A1-A6/G
overfit curves.

## Sources used

- GLM HF/reference behavior is summarized in
  `docs/research/2026-10-14-glm-5.3-flash-architecture-and-finetuning.md`:
  indexer scoring/top-k and `@torch.no_grad()` behavior at lines 256-281,
  IndexPool at 283-297, KDA at 299-319, sparse MLA dense-mask reference and
  vLLM sparse path at 321-334, and repo mapping at 340-350.
- GLM training-time behavior is only partially public: the local source audit
  records GLM's dense warmup plus sparse adaptation statement and RL indexer
  freeze at lines 358-381, and points to upstream `pytorch/torchtitan#4835`
  (`[Deepseek V4] Dsv4 sparse indexer loss`) for the sparse-indexer aux-loss
  shape. The exact GLM SFT/pretraining loss recipe is not present in this
  checkout, so S04 declares the implemented KL objective and warmup as dose
  knobs instead of calling it exact GLM.
- GLM mHC init in the verified GLM implementation is
  `fn ~ N(0, 0.02)`, `base = 0`, `scale = 1`
  (`torchtitan/experiments/glm53_flash/model.py:1349-1393`); S04 makes that
  A1's default and keeps S02's exact identity mode as `mhc_init=gated_identity`.

## Implemented

- Indexer training objective:
  `LightningIndexer.forward()` remains HF-style no-grad selection, while
  `score_pools()` is differentiable (`elements/attention.py:260-400`).
  `indexer_auxiliary_loss()` implements
  `KL(stop_grad(head-summed dense attention teacher) || softmax(indexer scores))`
  over either selected pools or all visible causal pools
  (`elements/attention.py:429-567`). Inputs to the indexer objective are
  detached from the main model graph.
- Dense warmup and aux-loss hooks:
  `TopKSparseCausalAttention` exposes `indexer_aux_loss_weight`,
  `indexer_aux_target`, `indexer_dense_warmup_steps`, a cached
  `last_indexer_aux_loss`, and an element-side collector
  (`elements/attention.py:570-672`, `elements/adapter.py:56-70`).
  S01 does not yet consume this collector.
- Sparse attention fast path:
  kept S02's dense-mask oracle as `sparse_causal_attention_reference`, and made
  `sparse_causal_attention()` a chunked gather path with checkpointed chunks, so
  it does not allocate a full `[B, L, L]` attention matrix
  (`elements/attention.py:74-197`).
- KDA fast path:
  added `kda_attention_gym()` over the shared Attention Gym KDA kernel and
  `kda_impl in {auto, attn_gym, chunked, recurrent}` with oracle fallback for
  `auto` (`elements/kda.py:223-372`).
- mHC init:
  `SpeedrunElementsConfig` now has `mhc_init in {glm, gated_identity}` and A1
  defaults to `glm`; `gated_identity` remains a dose knob
  (`elements/config.py:27-48`, `elements/config.py:130-178`,
  `elements/adapter.py:30-43`).

## Focused verification

Command:

```bash
TORCHTITAN_ROOTFS_ENV=glm53_train scripts/rootfs/enter_rootfs.sh -- bash -lc \
  'unset CC CXX; cd /workspace/torchtitan && export PYTHONPATH="$PWD:${PYTHONPATH:-}" && pytest -q torchtitan/experiments/glm53_arch_speedrun/elements/tests/test_elements.py -vv'
```

Result: `19 passed, 14 warnings in 10.23s`.

Coverage added/kept:

| Gate | Seq lens | Result |
| --- | ---: | --- |
| Indexer aux KL improves dense top-k recall on toy task | CPU toy | PASS: recall 0.409091 -> 0.954545, hidden input grad stayed `None` |
| mHC gated-identity dose still equals plain residual, fwd+bwd finite | 1024, 4096 | PASS |
| top-k >= L dense sparse reference equals dense causal, fwd+bwd finite | 1024, 4096 | PASS |
| sparse gather path equals S02 dense-mask reference, fwd+bwd finite | 1024, 4096 | PASS |
| IndexPool pool=1 reduces to A4 raw-key path, fwd+bwd finite | 1024, 4096 | PASS |
| KDA chunked reference equals recurrent oracle, fwd+bwd finite | 1024, 4096 | PASS |
| Attention Gym KDA equals recurrent oracle, fwd+bwd finite | 1024, 4096 | PASS |
| NoPE removes RoPE only, fwd+bwd finite | 1024, 4096 | PASS |
| Aux-loss collector, dense warmup path, sparse post-warmup path | unit | PASS |
| A1 default `mhc_init=glm` and dose knobs exposed | unit | PASS |

S01 compatibility check:

```bash
TORCHTITAN_ROOTFS_ENV=glm53_train scripts/rootfs/enter_rootfs.sh -- bash -lc \
  'unset CC CXX; cd /workspace/torchtitan && export PYTHONPATH="$PWD:${PYTHONPATH:-}" && pytest -q torchtitan/experiments/glm53_arch_speedrun/tests/test_s01_b0_harness.py -vv'
```

Result: `5 passed, 14 warnings in 6.12s`.

Lint/format:

```bash
pre-commit run ufmt --files <S04 element files>
pre-commit run flake8 --files <S04 element files>
pre-commit run pydoclint --files <S04 element files>
pre-commit run codespell --files <S04 element files>
git diff --check -- torchtitan/experiments/glm53_arch_speedrun/elements
```

Result: PASS.

## Step-3 gate attempt

Scheduler dry run:

```bash
experiments/glm53_arch_speedrun/scheduler.sh --once --dry-run --no-wait
```

Result:

```text
action=scheduler_start cells=93 dry_run=1 once=1 poll_sec=120
action=gate_wait waiting missing=.scratch/glm53-arch-speedrun/sdd/s01-report.md
```

Runner arm check:

```bash
experiments/glm53_arch_speedrun/run_arm.sh A1
```

Result:

```text
S01 runner supports B0 only; got A1
```

Therefore the requested per-arm Step-3 table is blocked at the S01 seam. The
S04 elements package now provides adapters and aux-loss collection, but the
landed S01 model/config still constructs only B0 and raises for A1-A6 in
`model.py`, while the shell runner exits for non-B0 arms.

## B0 smoke evidence

These are smoke/plumbing runs only, not CE-collapse proof. Both used one GPU,
`GLM53_SPEEDRUN_GPU_MEMORY_CAP_GIB=24`, staged FineWeb GPT-2 bin shards, and
`environment_class=shared_gpu`.

| Arm | Seq | Steps | Train loss | Val loss | Step-2 tokens/s | Peak memory | Run |
| --- | ---: | ---: | --- | --- | ---: | ---: | --- |
| B0 | 1024 | 2 | 10.97078 -> 10.29926 | 10.4694 -> 10.0389 | 9,174 | 2.26 GiB | `s04-b0-smoke-20261014T151143Z` |
| B0 | 4096 | 2 | 10.98947 -> 10.26178 | 10.2771 -> 9.6544 | 35,368 | 4.13 GiB | `s04-b0-seq4096-smoke-20261014T151254Z` |

Published bundles:

- `hdfs://harunava/user/tiger/waystone/torchtitan/runs/glm53-arch-speedrun/s04-b0-smoke-20261014T151143Z`
- `hdfs://harunava/user/tiger/waystone/torchtitan/runs/glm53-arch-speedrun/s04-b0-seq4096-smoke-20261014T151254Z`

## Step-3 status table

| Arm | CE collapse | lr=0 flat | Arms diverge | Seq-1024 tokens/s | Seq-4096 tokens/s | Peak memory | Status |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| B0 | SMOKE ONLY | not run | blocked | 9,174 | 35,368 | 2.26 / 4.13 GiB | runnable for B0 only |
| A1 mHC | blocked | blocked | blocked | n/a | n/a | n/a | S01 runner/model do not support A1 |
| A2 NoPE | blocked | blocked | blocked | n/a | n/a | n/a | S01 runner/model do not support A2 |
| A3 KDA hybrid | blocked | blocked | blocked | n/a | n/a | n/a | S01 runner/model do not support A3 |
| A4 sparse attention | blocked | blocked | blocked | n/a | n/a | n/a | S01 runner/model do not support A4 |
| A5 IndexPool | blocked | blocked | blocked | n/a | n/a | n/a | S01 runner/model do not support A5 |
| A6 MTP | blocked | blocked | blocked | n/a | n/a | n/a | S01 runner/model do not support A6 |
| G all elements | blocked | blocked | blocked | n/a | n/a | n/a | no combined-arm S01 integration |

## Remaining seam for S01

To make S04's Step-3 gate runnable, S01 needs to call the element adapters:

- build per-layer attention with `build_attention(layer_idx, elements_cfg)`;
- wrap attention/MLP branches with `maybe_wrap_mhc()`;
- add `maybe_build_mtp()` and `collect_element_aux_losses()` to the training
  loss path; and
- pass a trainer step into sparse attention or call `set_train_step()` so dense
  warmup switches at the declared step.

No commits were made.
