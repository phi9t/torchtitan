# Ticket 06 brief

Read first, in order:

1. `.scratch/falcon-fast-weight-attention/issues/06-step4-science-setup.md`
2. `.scratch/falcon-fast-weight-attention/spec.md` (Step 4 science model + addition protocol)
3. `.scratch/falcon-fast-weight-attention/sdd/task-05-report.md` (corpus + knobs)
4. Ticket 04 is done: `falcon_masked_parallel_forward` exists; 31 falcon tests green.

This ticket is **setup only**. Do not run A0-A5 campaigns (ticket 07).

## Knobs (locked by 05)

- Corpus: FineWeb 10B GPT-2 bins at
  `experiments/modded_nanogpt_b200/sources/modded-nanogpt-b200-sdpa/data/fineweb10B/`
  (`fineweb_train_000001.bin`..`000009.bin` train, `fineweb_val_000000.bin` val).
- Format: nanogpt `.bin` = 256 int32 header then uint16 tokens. Typical header:
  magic / version / ntok. Confirm against a real file; do not guess a different
  layout if the file disagrees.
- Model `vocab_size=50257` on the FineWeb path.
- Science shape (smoke may use fewer steps): 4 layers, hidden 256, heads 8,
  head_dim 32, seq_len 512, local_batch_size 32, tokens/step 16384, AdamW
  betas (0.9, 0.95). Register the full 8000-step science config but do not
  execute 8000 steps.
- Smoke data: synthetic tiny `.bin` in unit tests; real FineWeb bins only if
  a few-step smoke needs them. `fineweb_test` / `c4_test` are OK for
  HuggingFace-path tests.

## Do (TDD, rootfs pytest)

1. **Mixer switch in the same decoder shell** (`FalconDecoderLayer` keeps
   Pre-Norm + SwiGLU). Config field such as `mixer: "falcon"|"gdn"|"softmax"`.
   Default remains Falcon delayed + QK-RMSNorm.
   - Falcon: existing `FalconMixer` / recurrent (or masked-parallel) path.
   - GDN-style: same-step residual write
     `S <- gS + k((v - S^T k) β)^T` then `o = S^T q`. Copy the *math* from
     `torchtitan/experiments/mini_kimi_k3/kda.py::kda_reference_forward`
     (same-timestep, L2-norm q/k). **Do not import** `mini_kimi_k3` or
     `qwen3_5`. Keep it in `torchtitan/experiments/falcon/`.
   - Softmax+RoPE: causal attention + RoPE on q/k, same head layout.
2. **Science Trainer config** e.g. `falcon_science` via ConfigManager.
   Identity parallelize. `vocab_size=50257`. Delayed Falcon default.
3. **nanogpt `.bin` reader** + dataloader for the FineWeb path. Unit-test
   with a tiny synthetic bin you write in the test (do not depend on the
   1.9 GB corpus for unit tests).
4. **Addition task** (paper §5.2): prompt `digits_n(a) + digits_n(b) =`,
   target reversed `(n+1)`-digit sum (LSD first), loss on suffix only.
   Train widths `{1..N}`, OOD `{N+1..M}`. Use **N=16, M=24** and declare
   the OOD gap. Character-level encoder is preferred (do not fight GPT-2
   BPE on digits). Unit tests for encoding, suffix mask, ID vs OOD split.
5. **Few-step science smoke** (pattern: `experiments/falcon/smoke_trainer.py`):
   ConfigManager builds `falcon_science` (or a debug override with tiny
   steps / maybe smaller width if 4x256x512 is too heavy for CPU smoke),
   N>1 steps, finite loss. Re-run existing `test_falcon_*.py` and keep
   the Step 3 overfit gate green.
6. Report: `.scratch/falcon-fast-weight-attention/sdd/task-06-report.md`
   and an `## Answer` on the issue file.

## Do not

- Commit, push, or open a PR.
- Run 8k-step ablations or hero budgets.
- Falcon-2/3, chunk-parallel, FLA, CP.
- Import or edit `qwen3_5` / `mini_kimi_k3`.
- Fetch FineWeb-Edu or any new corpus.
- Change the Falcon delayed default or break the 31 existing tests.

## Suggested test files

- `tests/unit_tests/test_falcon_bin_reader.py`
- `tests/unit_tests/test_falcon_addition.py`
- `tests/unit_tests/test_falcon_mixer_switch.py` (or extend model tests)
- extend `test_falcon_config_registry.py` for `falcon_science`

ASCII comments only. Rootfs for pytest and any Python/CUDA work.
