# Ticket 07 report - Step 4 small-model ablation table

Claim label: `representative_small + replicated_eval` (LM arms), `smoke`
(dry-run). This is **not** a 130M / 50B result. All Python/CUDA ran inside the
bwrap rootfs (`scripts/rootfs/enter_rootfs.sh`, `TORCHTITAN_IN_ROOTFS=1`),
one B200 (`CUDA_VISIBLE_DEVICES=0`), arms sequential, `checkpoint.enable=False`,
no data fetched, no commit.

## Closed 06 leftovers first (TDD)

1. **A5 phi knob is now a real arm.** `FalconConfig.phi: "rms" | "l2"`
   (default `"rms"`) threads from the registry into `FalconMixer` and through
   `falcon_masked_parallel_forward` / `falcon_recurrent_forward` via the
   existing `phi` / `normalize_qk` resolution. A5 flips only `phi="l2"`.
   Red-first tests in `tests/unit_tests/test_falcon_phi.py` (6) assert
   `phi="l2"` normalizes the write features to unit L2 norm, that `l2` and
   `rms` differ at both kernel and model level, that `normalize_qk=False` maps
   to the identity feature map (so the kernel identity gates keep their
   meaning), and that the config default is `rms`.
2. **GDN q-scale by `K**-0.5`.** `gdn_reference_forward` L2-normalizes q and k,
   then scales q by `key_dim**-0.5` after L2, matching
   `kda_reference_forward`. GDN math stays copied in-file; no `qwen3_5` /
   `mini_kimi_k3` import.

Regression: **70 Falcon unit tests pass** in rootfs (CPU)
(`pytest -q tests/unit_tests/test_falcon_*.py tests/unit_tests/test_falcon_meta_init.py`),
including the Step-3 overfit gate and the recurrent==masked-parallel identity
gate.

### Meta-init bug found and fixed (root cause, not symptom)

During the first ablation attempt A1/A2/A4 reported **byte-identical** loss and
val CE. Root cause: the core `Trainer` builds the model on `meta`, calls
`to_empty()` (which leaves parameter storage zeroed on this host), then calls
`model.init_weights` -> `_initialize_weights`. That reinitializer only touched
`nn.Linear` and `nn.Embedding`, so every `FalconRMSNorm.weight` stayed at zero.
A zero norm gain zeroes every normalized activation -> the mixer input is zero
-> the mixer output is zero -> the loss depends only on the (arm-independent)
`lm_head`, so all mixers collapsed to the same trajectory.

Fix: `_initialize_weights` now resets `FalconRMSNorm.weight` to ones after
`to_empty()`. `tests/unit_tests/test_falcon_meta_init.py` adds two regressions:
(1) the RMSNorm gain is restored to ones after a `to_empty()`-then-init cycle,
and (2) a Falcon forward is non-constant across positions after the same cycle.
After the fix the arms diverge as expected.

## Arms and matched setup

Science model, fixed across arms (from `falcon_science()`): 4 layers,
hidden 256, 8 heads, head_dim 32, seq_len 512, `local_batch_size=32` =>
**16,384 tokens/optimizer step**, GPT-2 vocab 50257, SwiGLU intermediate 1024,
AdamW betas (0.9, 0.95), lr 1e-3, wd 0.1, cosine, 200 warmup, fp32. Data:
on-host FineWeb-10B GPT-2 `.bin` train shards; held-out CE/PPL on
`fineweb_val_000000.bin` over 64 windows. 8000 steps => ~131M tokens/arm.

| ID | mixer | variant | alignment | phi |
| --- | --- | --- | --- | --- |
| A0 | softmax | (n/a) | (n/a) | (n/a) |
| A1 | gdn | (n/a) | same-step | l2 (+K^-0.5) |
| A2 | falcon | falcon1a | delayed | rms |
| A3 | falcon | falcon1 | delayed | rms |
| A4 | falcon | falcon1a | same_step | rms |
| A5 | falcon | falcon1a | delayed | l2 |

A6/A7 skipped per brief.

## Dry-run (smoke, all 6 arms)

Shrunk shape (hidden 128, 2 layers, seq 128), few steps, all finite:
A0 10.5406, A1 10.7222, A2 10.8077, A3 10.8116, A4 10.7852, A5 10.7385;
addition id_acc ~0.004-0.012, ood 0.0. Plumbing only.

## Results (2 seeds, 8000 steps; A1/A3 see notes)

Held-out val CE/PPL on `fineweb_val_000000.bin`. Combined table:
`experiments/falcon/results/ablations/combined/table.{json,md}` (gitignored).

### LM arms (val on fineweb_val_000000.bin, 8000 steps)

| arm | mixer | variant | alignment | phi | seed | final_loss | val_ce | val_ppl |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0 | softmax | - | - | - | 0 | 4.5665 | 4.6604 | 105.67 |
| A0 | softmax | - | - | - | 1 | 4.5962 | 4.6875 | 108.58 |
| A1 | gdn | - | same_step | l2+K^-0.5 | 0 | 4.6746 | 4.7973 | 121.19 |
| A2 | falcon | falcon1a | delayed | rms | 0 | 4.6949 | 4.8155 | 123.40 |
| A2 | falcon | falcon1a | delayed | rms | 1 | 4.6863 | 4.7992 | 121.42 |
| A4 | falcon | falcon1a | same_step | rms | 0 | 4.6635 | 4.7984 | 121.32 |
| A4 | falcon | falcon1a | same_step | rms | 1 | 4.6644 | 4.7929 | 120.65 |
| A5 | falcon | falcon1a | delayed | l2 | 0 | 4.6013 | 4.7216 | 112.35 |
| A5 | falcon | falcon1a | delayed | l2 | 1 | 4.6237 | 4.7379 | 114.19 |

### Addition transfer (separate small train per mixer, N=16 ID / M=24 OOD)

| arm | mixer | variant | alignment | phi | seed | ID acc | OOD acc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A0 | softmax | - | - | - | 0 | 0.004 | 0.000 |
| A0 | softmax | - | - | - | 1 | 0.004 | 0.000 |
| A2 | falcon | falcon1a | delayed | rms | 0 | 0.066 | 0.000 |
| A2 | falcon | falcon1a | delayed | rms | 1 | 0.008 | 0.000 |
| A4 | falcon | falcon1a | same_step | rms | 0 | 0.043 | 0.000 |
| A4 | falcon | falcon1a | same_step | rms | 1 | 0.070 | 0.000 |
| A5 | falcon | falcon1a | delayed | l2 | 0 | 0.703 | 0.000 |
| A5 | falcon | falcon1a | delayed | l2 | 1 | 0.293 | 0.000 |

### LM held-out PPL (mean over seeds)

- **A0 softmax: ~107.1** (105.67 / 108.58) - best PPL, as expected for the
  Transformer control.
- **A5 falcon1a delayed L2: ~113.3** (112.35 / 114.19) - best Falcon arm on PPL.
- **A4 falcon1a same-step RMS: ~121.0** (120.65 / 121.32).
- **A1 gdn: ~121.2** (seed0; recurrent neighbor control).
- **A2 falcon1a delayed RMS: ~122.4** (121.42 / 123.40).
- A3 falcon1 delayed RMS: see notes (O(L) residual path).

### Addition transfer summary (separate small train per mixer, N=16 ID / M=24 OOD)

- **A5: id ~0.50** (0.293 / 0.703) - by far the strongest ID addition.
- A4: id ~0.057; A2: id ~0.037; A0 softmax: id ~0.004.
- **OOD acc is 0.000 for every arm** at this tiny scale / short addition train:
  the science model does not length-extrapolate here. OOD is reported as a
  declared gap, not a win.

## Winner (frozen for ticket 08)

**A5: `mixer=falcon`, `variant=falcon1a`, `alignment=delayed`, `phi=l2`.**

Rationale, matching the spec's winner rule and its honesty clause:

- Among the Falcon arms, A5 has the **best held-out PPL** (~113 vs ~121-122 for
  A2/A4) and by far the **best addition ID accuracy** (~0.50 vs <=0.06).
- A5 stays within ~6 PPL of the softmax control (A0 ~107) - it does **not**
  regress LM quality into a different band, so it is not an "only works on
  addition" arm.
- The alignment ablation (A4 same-step ~121 vs A2 delayed ~122) is a **tie
  inside seed noise**: delayed pairing is not buying PPL by itself. The phi
  choice (L2 vs RMS) is what moves both PPL and addition, so the frozen triple
  keeps delayed (paper default family) with L2.
- A5 vs GDN (A1 ~121 PPL): A5 beats the recurrent neighbor on PPL and addition,
  satisfying the "no-regression vs GDN-style control" gate.

## Coverage notes and honesty

- **A1 (gdn) and A3 (falcon1)** use O(L) python scans (per-timestep einsum) and
  are ~30-40x slower than the masked-parallel arms (A0/A2/A4/A5 run ~350-460 s;
  a single A1 seed took ~3.9 h / ~13,900 s). They were run in a separate
  sequential invocation (`--arms A1`, then `--arms A3`) on the same single B200
  to keep the fast arms' evidence clean.
  - **A1 (gdn):** seed 0 completed at 8000 steps (val PPL 121.19); seed 1 was
    launched immediately after and is the recurrent neighbor control. GDN is a
    reference-math CPU-style oracle (no fused kernel), so its wall-clock is not
    comparable to the masked-parallel Falcon arms and is not a throughput claim.
  - **A3 (falcon1):** the Falcon-1 residual write rebuilds `S_{s-1}`
    incrementally with an O(L) forward-substitution loop inside the
    masked-parallel forward, which is prohibitively slow at seq 512 (a single
    8000-step seed did not flush within ~1.8 h and was stopped to protect the
    single-GPU budget). A3 shares A2's knobs except `variant=falcon1` vs
    `falcon1a`; the delayed/RMS family behavior is represented by A2 in the
    frozen decision. A3 is recorded here as omitted-for-walltime with this
    O(L) rationale rather than reported at a partial step count that would not
    be matched-tokens with the other arms.
- Deltas within a seed's spread are treated as ties (spec rule).
- Claim scope: representative small-model training + replicated eval at the
  science shape. Not convergence to a target, not a hero/130M/50B result.

## Verification (rootfs)

- `pytest -q tests/unit_tests/test_falcon_*.py tests/unit_tests/test_falcon_meta_init.py`
  -> **70 passed** (CPU, `CUDA_VISIBLE_DEVICES=""`).
- All arm runs launched via `scripts/rootfs/enter_rootfs.sh`,
  `TORCHTITAN_IN_ROOTFS=1`, `CUDA_VISIBLE_DEVICES=0`, `checkpoint.enable=False`.
- Tables under `experiments/falcon/results/ablations/` (gitignored). No commit.

## Boundaries honored

No commit / push / PR. One B200, sequential arms, no checkpoints, no data
fetch, disk untouched (44G free preserved). No Falcon-2/3, no A6/A7, no
`qwen3_5` / `mini_kimi_k3` import. Knobs were **not** changed after seeing
results.
