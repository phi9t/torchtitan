# Fast Weight Attention (Falcon) — paper deep dive

Status: research note, not an approved implementation spec.

Sources:

- Paper: Zhang et al., *Fast Weight Attention for Continual Learning*, arXiv:2608.27763v1 (27 Aug 2026 / dated 9 Mar 2026). [arXiv PDF](https://arxiv.org/pdf/2608.27763), [project page](https://github.com/yifanzhang-pro/fast-weight-attention).
- Official repo contents as of 2026-09-05: HTML project page, PDF, image, and an empty `model/` stub. No released kernels, trainers, or configs.
- Local attachment map: TorchTitan GDN (`torchtitan/models/qwen3_5/`), Mini-K3 KDA (`torchtitan/experiments/mini_kimi_k3/kda.py`), experiment rules in `torchtitan/experiments/README.md` and `docs/extension.md`.

## What the paper actually proposes

Falcon is a family of **fast-weight sequence mixers**, not a training recipe, dataset, or RL algorithm. Recurrent / SSM / linear-attention models compress context into a fixed matrix state `S_t`. The paper treats that state update as an **online continual-learning rule** and fixes two conventions that most DeltaNet / GDN implementations leave implicit:

1. **Read-after-write (RAW)** autoregressive timing: at step `t`, write first, then read `o_t = S_t^T phi(q_t)` to predict token `t+1`.
2. **Prefix / next-latent pairing**: the local training pair is `(x_t, y_t) = (phi(k_{t-1}), v_t)`, with boundary `x_1 := 0`, `eta_1 := 0`. The common same-step pair `(phi(k_t), v_t)` is still causal, but it is a **different internal objective**.

Two local objectives then yield six rules:

| Objective | Scalar `eta_t` | Per-column `eta_t in R^{d_v}` | Sliding window of size `B` |
| --- | --- | --- | --- |
| Squared-error / NLMS ridge | Falcon-1 | Falcon-2 | Falcon-3 |
| Negative inner-product (Hebbian write) | Falcon-1A | Falcon-2A | Falcon-3A |

Shared semantics:

- `beta_t` is a dimensionless plasticity gain (learned / context-conditioned).
- `lambda_t` is the actual ridge / shrinkage coefficient (optionally scale-coupled as `lambda_t = lambda_bar_t * E_t`).
- `eta_t` is the induced step size, not a free learning rate.
- Regression writes a residual `r_t = v_t - S_{t-1}^T x_t`.
- Inner-product ("A") writes the target `v_t` directly.
- Carry is `gamma_t = 1 - eta_t lambda_t`, clamped to stay strictly positive (`gamma_t >= eps_gamma`).

Canonical Falcon-1 (regression, scalar NLMS):

```text
eta_t = beta_t / (||x_t||^2 + lambda_t + eps)
S_t   = (1 - eta_t lambda_t) S_{t-1} + eta_t x_t r_t^T
o_t   = S_t^T phi(q_t)
```

Canonical Falcon-1A (inner-product):

```text
eta_t = beta_t / (||x_t||^2 + lambda_t + eps)
S_t   = (1 - eta_t lambda_t) S_{t-1} + eta_t x_t v_t^T
```

Falcon-3 / 3A replace the rank-one write with a mini-batch over the last `B` causal pairs. Falcon-3's normalizer uses `lambda_max` of the window Gram; Falcon-3A uses mean write energy.

The paper also gives three computationally equivalent views: recurrent scan, masked-parallel attention, and SSD-style chunk-parallel (WY / Gram / ParallelFlow), plus **positive-decay renormalization** so ridge (`lambda_t > 0`) reduces to a no-ridge kernel after rescaling.

Default implementation choices in the paper:

- QK-RMSNorm (not L2) before forming features; values are not normalized by default.
- Optional short causal conv on q/k/v (orthogonal to the update rule).
- Denominator-free reads for signed features; normalized linear-attention reads are only "attention-like" with nonnegative features.
- Context-conditioned `beta` / `eta` / `lambda` (`ctx-beta`, `ctx-eta`, `ctx-lambda`).

## Empirical claims (small, not frontier)

Section 5 trains **124M–130M** LLaMA-style models on **FineWeb-Edu**, seq len **1024**, global batch **480**, **100k** AdamW steps (~**49.2B** tokens), bf16, muP-style width scaling, lr `1e-3`, cosine, 2k warmup, betas `(0.9, 0.95)`, wd `0.1`, clip `1.0`, single 4x H100/H200 node (Appendix B).

Benchmarked variants: Falcon-1A.1/1A.2/1A.3, Falcon-3A.3, and regression ablation Falcon-1.3. **Falcon-2, 2A, and 3 are defined but not in the main tables.**

Reported FineWeb-Edu val PPL (lower better):

- Transformer 17.38
- Gated DeltaNet 17.32 (best prior recurrent)
- Falcon-1A.3 17.40
- Falcon-1.3 **17.10** (best overall)

Downstream: competitive, not a uniform win. Stronger result is **variable-digit addition length extrapolation** (train 1–32 digits, eval 33–48): Falcon-3A.3 mean acc **87.2**, Falcon-1A.3 **85.9**, vs Transformer **65.8** and Falcon-1.3 (regression) **68.8**. Inner-product writes help storage/carry more than regression in that diagnostic.

Honest reading: Falcon is a **principled mixer paper**. It is competitive with GDN at ~130M / 50B tokens and better on a synthetic memory test. It is not a new training stack and does not claim large-model or post-training gains.

## How this differs from code already in this checkout

TorchTitan already has two gated-delta mixers:

- Core Qwen3.5 `GatedDeltaNet` (`torchtitan/models/qwen3_5/model.py`) with FLA `chunk_gated_delta_rule` / `fused_recurrent_gated_delta_rule`.
- Experiment Mini-K3 `MiniK3DeltaAttention` (`torchtitan/experiments/mini_kimi_k3/kda.py`).

Both use **same-timestep** `k_t` and a gated delta write:

```text
S <- g * S
S <- S + k * ((v - k^T S) * beta)^T
o <- S^T q
```

Falcon is the same *family* (fixed `S` in `R^{d_x x d_v}`, outer-product writes) but not a drop-in:

- write feature is `phi(k_{t-1})`, not `k_t`
- `eta` is NLMS-normalized from `beta`, not a raw sigmoid gate
- ridge carry is `1 - eta lambda`, not a separate log-decay `exp(g)`
- A-variants write `v` not residual
- 3/3A need a sliding window and (for Falcon-3) `lambda_max`
- chunk-parallel uses WY / Gram / renormalization, not the existing FLA GDN kernel

There is **no** Falcon / NLMS / RWKV / Mamba code in this checkout.

## Can we implement it as a TorchTitan experiment?

Yes. This is the documented experiment type: a new sequence mixer with its own package, tests, and optional deps, reusing core `Trainer` / `ModelSpec`. Official code is not released, so the implementation source of truth is the paper plus a CPU oracle.

What is **easy / high leverage**:

- Falcon-1 and Falcon-1A recurrent CPU oracles (rank-one, scalar `eta`).
- Numerics tests copied from `tests/unit_tests/test_qwen3_5_deltanet.py`.
- A tiny LLaMA-style decoder + `config_registry` + identity parallelize, following Mini-K3.
- Addition diagnostic (short sequences, cheap, paper's clearest win).
- Later: chunk-parallel for 1/1A after recurrent vs masked-parallel match.

What is **hard / later**:

- Falcon-2/2A: per-channel `eta` implies `d_v` triangular solves per chunk.
- Falcon-3: `lambda_max` of the window Gram each step.
- Exact chunk-parallel + positive-decay renormalization (Appendices E–G).
- TP `local_map` around a custom kernel (Qwen3.5 GDN pattern).
- Context parallel: already rejected for Qwen3.5 GDN.
- Paper-scale 50B-token FineWeb-Edu run: budget, disk, and eval harness; local `/data02` has been ~30 GB free.
- Swapping Falcon into Qwen3.5 hybrid GDN slots: possible, but changes HF state-dict and VLM/CP contracts.

Recommended first experiment (if approved):

1. `torchtitan/experiments/falcon/` mixer package with CPU recurrent Falcon-1 and Falcon-1A.
2. Tests: delayed pairing, `eta_1=0`, recurrent vs masked-parallel, ridge clamp.
3. Tiny Trainer config via `MODULE=falcon`.
4. Optional cheap addition smoke.
5. Defer 2/3/A-window, FLA kernels, Qwen hybrid, and 50B FineWeb until numerics land.

Out of scope for a first experiment: official 130M/50B reproduction, Falcon-2/3 production kernels, promoting into `torchtitan/models/`, RL/batch-invariance, or changing core Qwen3.5.

## Approaches considered

1. **New experiment mixer (recommended).** Isolated, follows Mini-K3, does not pollute GDN/Kimi fidelity, proves the paper's actual contribution (alignment + NLMS).
2. **Swap Falcon into Qwen3.5 GDN slots.** Fastest path to a hybrid stack, but couples Falcon to VLM, FLA import, CP ban, and HF adapters.
3. **Paper-scale reproduction.** Only after (1) is green; treat as a separate FineWeb campaign, not the mixer ticket.

## Open product question

What is the first success criterion: mixer numerics + tiny smoke, Qwen3.5 hybrid swap, or paper-scale FineWeb/addition reproduction?
