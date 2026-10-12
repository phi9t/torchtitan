# Ticket 06 report - Step 4 science setup

Setup only. No A0-A5 campaign, no 8000-step run, no commit. TDD, rootfs
pytest. Claim label: `smoke`.

## What landed

All new Python lives under `torchtitan/experiments/falcon/`; the smoke driver
lives under `experiments/falcon/`. Dependency direction stays
`experiments -> core`. No `qwen3_5` / `mini_kimi_k3` import or edit.

### 1. Mixer switch in the same decoder shell

`torchtitan/experiments/falcon/model.py`

- `FalconConfig.mixer: "falcon" | "gdn" | "softmax"`, default `"falcon"`.
  The Falcon default keeps delayed pairing + QK-RMSNorm (`variant=falcon1a`,
  `alignment=delayed`). `FalconDecoderLayer` is unchanged: Pre-Norm RMSNorm +
  SwiGLU, mixer selected by `build_mixer(config)`.
- `falcon`: existing `FalconMixer` -> `falcon_recurrent_forward`.
- `gdn`: `FalconGDNMixer` -> `gdn_reference_forward`, a Gated DeltaNet-style
  same-step residual write reimplemented in-file:
  `S_t = g_t S_{t-1} + k_t ((v_t - S_{t-1}^T k_t) beta_t)^T`, `o_t = S_t^T q_t`,
  with L2-normed q/k. The math mirrors `mini_kimi_k3/kda.py::kda_reference_forward`
  but is copied, not imported.
- `softmax`: `FalconSoftmaxMixer` -> causal `scaled_dot_product_attention`
  (`is_causal=True`) with interleaved RoPE on q/k, same head layout.

Tests (`tests/unit_tests/test_falcon_mixer_switch.py`, 9): default is
`falcon`; each mixer returns finite `(1, L, vocab)` logits; the three mixers
produce distinct outputs; unknown mixer raises `ValueError`; GDN and softmax
are causal (perturbing the last token leaves earlier positions unchanged);
the Falcon mixer still threads the `alignment` switch.

### 2. nanogpt `.bin` reader + dataloader

`torchtitan/experiments/falcon/bin_reader.py`

- `read_nanogpt_bin`: 256-int32 header (magic `20240520`, version `1`, token
  count), then uint16 tokens -> int64. Fails loud on magic/version/count
  mismatch. Confirmed against the real FineWeb 10B shard on host
  (`magic 20240520 version 1 ntok 100000000`, min 0 / max 50256).
- `write_nanogpt_bin`: support helper for synthetic bins.
- `NanoGptBinDataLoader`: single-rank contiguous-window loader; packs
  `(local_batch_size, seq_len)` inputs with next-token-shift labels; cycles
  shards forever; rejects out-of-vocab token ids. `Config.build` resolves a
  glob or literal (`_resolve_bins` handles absolute paths that `Path.glob`
  rejects).

Tests (`tests/unit_tests/test_falcon_bin_reader.py`, 9): raw-format roundtrip,
write/read roundtrip, bad magic / bad version / truncated stream rejection,
shifted next-token labels (not a roll), shard wrap, over-vocab rejection,
registry `Config.build`.

### 3. Addition task (paper section 5.2)

`torchtitan/experiments/falcon/addition.py`

- Prompt `digits_n(a)+digits_n(b)=`, target reversed `(n+1)`-digit sum
  (LSD first), loss on the answer suffix only.
- Character-level `AdditionVocab` over `0-9 + =` plus a pad symbol (index 0),
  chosen so digit positions stay aligned (no GPT-2 BPE digit fragmentation).
- `build_addition_dataset(n, m, ...)` with ID widths `{1..n}` and OOD
  `{n+1..m}`. Ticket locks **N=16, M=24**; the declared OOD gap is widths
  17-24 (never seen in training).
- `AdditionDataset.collate` right-pads to input/labels/loss_mask tensors; the
  loss mask marks only the answer positions in the label frame.

Tests (`tests/unit_tests/test_falcon_addition.py`, 8): reversed-LSD `(n+1)`
width, prompt+suffix layout, suffix-only contiguous mask, ID/OOD split by
width, declared `N=16 M=24` gap disjointness, char-vocab roundtrip,
arithmetic correctness of the encoded sum, batch collation shape/mask.

### 4. Science Trainer config

`torchtitan/experiments/falcon/config_registry.py`

- `falcon_science()` via ConfigManager (`--module falcon --config
  falcon_science`). Flavor `science`; identity parallelize. Locked shape:
  4 layers, hidden 256, 8 heads, head_dim 32, seq_len 512,
  local_batch_size 32 (tokens/step = 16,384), `vocab_size=50257`,
  intermediate 1024. Delayed Falcon default mixer. AdamW betas (0.9, 0.95),
  lr 1e-3, wd 0.1, cosine, 200 warmup. `steps=8000` registered but **not**
  executed here. FineWeb glob points at the on-host train shards
  `fineweb_train_00000[1-9].bin`.

Tests extend `tests/unit_tests/test_falcon_config_registry.py` (5 total):
locked science shape + Falcon/delayed default, ConfigManager resolution with
`global_vocab_size=50257`, tokens/step = 16,384.

### 5. Few-step science smoke

`experiments/falcon/science_smoke.py`

Builds `falcon_science` through the core `Trainer` under `fake_backend`,
overrides to a tiny shape by default (hidden 64, 2 layers, seq 64, batch 4)
so a CPU smoke is cheap while keeping the science surface (same mixer, delayed
alignment, GPT-2 vocab, SwiGLU). Writes a small synthetic nanogpt `.bin` to a
temp dir (never the 1.9 GB corpus), runs N>1 steps, asserts finite loss at
every logged step. `--full-shape` runs the registered 4x256 seq-512 shape.

Rootfs runs:

- `python -m experiments.falcon.science_smoke --steps 3`
  -> `OK steps=3 shape=tiny vocab=50257 initial_loss=10.82491
  final_loss=10.82495 all_finite=True`
- `python -m experiments.falcon.science_smoke --steps 2 --full-shape`
  -> `OK steps=2 shape=full vocab=50257 initial_loss=10.82491
  final_loss=10.82490 all_finite=True`

Loss ~10.82 ~= ln(50257), the expected fresh-init CE for a 50257 vocab on
random tokens. This is plumbing evidence, not convergence.

## Verification (rootfs)

- All Falcon unit tests: `pytest -q tests/unit_tests/test_falcon_*.py`
  -> **60 passed** (31 pre-existing + 29 new: 9 mixer + 9 bin + 8 addition +
  3 net-new config).
- Step 3 overfit gate: `test_falcon_overfit.py` still green (falcon1a and
  falcon1 both overfit the fixed bank; lr=0 still fails the gate).
- Kernel identity tests (recurrent == masked-parallel) still green.
- `pre-commit run --files <changed>`: flake8, ufmt, pydoclint, codespell,
  pyrefly, license, EOF all pass (only `no-commit-to-branch` skipped, as this
  ticket does not commit).

## Boundaries honored

- No commit / push / PR. No 8000-step ablation or hero budget. No A0-A5 runs.
- No Falcon-2/3, chunk-parallel, FLA, CP.
- No `qwen3_5` / `mini_kimi_k3` import or edit (GDN math copied in-file).
- No new corpus fetched; reused the on-host FineWeb 10B GPT-2 bins.
- Falcon delayed default unchanged; 31 existing tests remain green.

## Notes for ticket 07 (ablations)

- Arms map to config knobs already present: A0 = `mixer="softmax"`,
  A1 = `mixer="gdn"`, A2/A3 = `mixer="falcon"` with `variant` falcon1a/falcon1
  (delayed, QK-RMSNorm), A4 = `alignment="same_step"`. A5 (QK-L2 phi) needs a
  `normalize_qk` / phi-selection knob threaded from config into the Falcon
  mixer; the recurrent kernel already accepts `normalize_qk`, so A5 is a small
  follow-up, not a new kernel.
- The science config runs 8000 steps at 16,384 tokens/step = ~131M
  tokens/arm, ~0.15 epoch of the 900M train slice, matching ticket 05.
- GDN and softmax mixers are CPU-reference (no fused kernels); acceptable for
  Step 4 per the spec (identity/health over throughput).
