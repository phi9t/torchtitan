# S02 report: GLM-5.3-Flash element modules for the speedrun substrate

Date: 2026-10-14

## Scope

Owned files changed:

- `torchtitan/experiments/glm53_arch_speedrun/elements/config.py`
- `torchtitan/experiments/glm53_arch_speedrun/elements/adapter.py`
- `torchtitan/experiments/glm53_arch_speedrun/elements/attention.py`
- `torchtitan/experiments/glm53_arch_speedrun/elements/kda.py`
- `torchtitan/experiments/glm53_arch_speedrun/elements/mhc.py`
- `torchtitan/experiments/glm53_arch_speedrun/elements/mtp.py`
- `torchtitan/experiments/glm53_arch_speedrun/elements/positional.py`
- `torchtitan/experiments/glm53_arch_speedrun/elements/tests/test_elements.py`

Did not edit the verified GLM implementation, DeepSeek-V4 mHC, or any S01-owned
speedrun model/harness files.

## Implemented elements

| Arm | Status | Notes |
|---|---|---|
| B0 seam | Done in `SpeedrunElementsConfig` and `build_attention()` | Speedrun width defaults: d768, 6 heads, head dim 128, 12 layers. |
| A1 mHC | Done | `HcPre`/`HcPost` use GLM post-mix axis `comb_ij` source -> target. `MHCResidualBlock` has exact identity-at-init via a learned zero scalar around the learned mHC path. |
| A2 NoPE | Done | `maybe_apply_rope(..., rope_enabled=False)` returns q/k unchanged and leaves attention otherwise identical. |
| A3 KDA hybrid | Done | `KDALayer`, `kda_recurrent`, `kda_chunked_reference`, and config schedule `(layer_idx + 1) % 4 == 0` -> softmax, others -> KDA. |
| A4 sparse attention | Done | `LightningIndexer` + `sparse_causal_attention`; top-k selection is deterministic, detached, and reference/HF-style no-grad. No GLM/DSA aux indexer objective is implemented in this slice. |
| A5 IndexPool | Done | `IndexPool(pool_size=4)` weighted per-channel pooling; `pool_size=1` reduces to A4 raw keys. |
| A6 MTP | Done | One-layer `MTPHead` with configurable `mtp_aux_loss_weight`; adapter exposes `maybe_build_mtp()`. |
| Combination/dose knobs | Done | `compose_arms()` supports combined factor configs and dose overrides for top-k, pool size, MTP weight, mHC Sinkhorn iters, and KDA conv width. |

## Identity/oracle tests

Command:

```bash
TORCHTITAN_ROOTFS_ENV=glm53_train scripts/rootfs/enter_rootfs.sh -- bash -lc 'unset CC CXX; cd /workspace/torchtitan && export PYTHONPATH="$PWD:${PYTHONPATH:-}" && pytest -q torchtitan/experiments/glm53_arch_speedrun/elements/tests/test_elements.py -vv'
```

Result: 13 passed in 5.03s.

After S01 landed, I also ran the combined seam check:

```bash
TORCHTITAN_ROOTFS_ENV=glm53_train scripts/rootfs/enter_rootfs.sh -- bash -lc 'unset CC CXX; cd /workspace/torchtitan && export PYTHONPATH="$PWD:${PYTHONPATH:-}" && pytest -q torchtitan/experiments/glm53_arch_speedrun/tests/test_s01_b0_harness.py torchtitan/experiments/glm53_arch_speedrun/elements/tests/test_elements.py -vv'
```

Result: 18 passed, 14 warnings in 8.54s. The warnings are torch.jit
deprecation warnings from the installed stack.

| Gate | Seq lens | Result |
|---|---:|---|
| mHC identity init == plain residual, fwd+bwd finite | 1024, 4096 | PASS |
| indexer top-k >= L returns dense causal indices and sparse attention == dense causal, fwd+bwd finite | 1024, 4096 | PASS |
| IndexPool pool=1 == A4 raw-key path, fwd+bwd finite | 1024, 4096 | PASS |
| KDA chunked reference == recurrent oracle, fwd+bwd finite | 1024, 4096 | PASS |
| NoPE removes RoPE only, fwd+bwd finite | 1024, 4096 | PASS |
| 3:1 hybrid layout and MTP aux-loss hook | unit | PASS |
| one-knob arms plus composed arms/dose knobs | unit | PASS |

The long tests set the per-process CUDA memory fraction to at most 24 GiB before
allocating tensors.

## Lint/format

Command shape:

```bash
TORCHTITAN_ROOTFS_ENV=glm53_train scripts/rootfs/enter_rootfs.sh -- bash -lc 'unset CC CXX; cd /workspace/torchtitan && export PYTHONPATH="$PWD:${PYTHONPATH:-}" && pre-commit run <hook> --files <S02 files>'
```

Results:

- `ufmt`: PASS
- `flake8`: PASS
- `pydoclint`: PASS
- `codespell`: PASS
- `git diff --check -- torchtitan/experiments/glm53_arch_speedrun/elements .scratch/glm53-arch-speedrun/sdd/s02-report.md`: PASS

## Overfit and throughput gate

Status: BLOCKED on S01 arm integration, not executed.

S01 landed while this ticket was in progress, and its B0-focused tests pass
together with the S02 element tests. The per-arm overfit/throughput matrix is
still blocked because the landed S01 seam is B0-only:

- `experiments/glm53_arch_speedrun/run_arm.sh A1` exits with
  `S01 runner supports B0 only; got A1` and status 2.
- `SpeedrunResidualBlock` raises `NotImplementedError` for mHC, KDA/sparse
  attention, and MTP rather than calling the S02 factories.
- `config_registry._speedrun_model_config()` constructs `b0_elements_config()`
  and does not yet apply queue or environment arm overrides to
  `model.elements.*`.

Because the arm model path and runner are not wired, I did not fabricate CE
curves, tokens/s, or memory measurements. Current GPU state from `nvidia-smi`
shows GPUs 2, 3, 6, and 7 nearly full, and GPUs 0, 1, 4, 5 with about 44 GiB
used, so any future timing should be labelled `shared_gpu`.

| Arm | CE collapse | lr=0 flat | arms diverge | seq1024 tokens/s | seq4096 tokens/s | memory |
|---|---|---|---|---:|---:|---:|
| B0 | BLOCKED: missing S01 harness | BLOCKED | BLOCKED | n/a | n/a | n/a |
| A1 mHC | BLOCKED: missing S01 harness | BLOCKED | BLOCKED | n/a | n/a | n/a |
| A2 NoPE | BLOCKED: missing S01 harness | BLOCKED | BLOCKED | n/a | n/a | n/a |
| A3 KDA hybrid | BLOCKED: missing S01 harness | BLOCKED | BLOCKED | n/a | n/a | n/a |
| A4 sparse attention | BLOCKED: missing S01 harness | BLOCKED | BLOCKED | n/a | n/a | n/a |
| A5 IndexPool | BLOCKED: missing S01 harness | BLOCKED | BLOCKED | n/a | n/a | n/a |
| A6 MTP | BLOCKED: missing S01 harness | BLOCKED | BLOCKED | n/a | n/a | n/a |

## Slow-path caveats and S01 hook needed

- `sparse_causal_attention()` materializes a `[B, L, L]` mask and scores matrix.
  This is an O(L^2) correctness/reference path, suitable for identity tests but
  too slow for the full speedrun matrix. A production S01 integration should
  replace it with the repo's Attention Gym `gather_attn` path or an equivalent
  sparse kernel.
- `LightningIndexer.forward()` is `@torch.no_grad()`, matching the HF GLM
  reference and the existing repo DSv4 detached indexer behavior. If the
  campaign wants the GLM/DSA auxiliary indexer training objective, that is a
  separate hook and loss term; it is not in this S02 slice.
- `kda_chunked_reference()` implements the GLM chunk transform in Python/Torch
  for correctness tests. It is not the intended training-speed kernel. The S01
  model should call into Attention Gym/FLa KDA when running the matrix.
- S01 integration should route each block through the element factories:
  `build_attention(layer_idx, elements_cfg)`, `maybe_wrap_mhc(branch,
  elements_cfg)`, and `maybe_build_mtp(elements_cfg)`. The runner/config path
  should also translate arm names and queue `model.elements.*` overrides into a
  composed `SpeedrunElementsConfig`.

