# Step 4 setup: science model and addition data

Type: task
Status: resolved
Blocked by: 03, 04, 05
Parent: ../spec.md

## Requirements

- Register a science-scale Falcon Trainer config (4–8 layers, hidden
  256–512 unless ticket 05 says otherwise) with the delayed default
  mixer and identity or minimal parallelize.
- Add a GDN-style same-step residual mixer **in the same decoder
  shell** (not `qwen3_5` / Mini-K3 imports) and a softmax+RoPE
  attention switch for controls.
- Add the paper §5.2 addition task: prompt `digits_n(a) + digits_n(b) =`,
  target reversed `(n+1)`-digit sum, loss on suffix only. Choose N/M
  from ticket 05 (keep a declared OOD gap).
- Wire FineWeb-Edu or C4 from ticket 05 into a science dataloader.
- Re-run Step 3 overfit on the science model at reduced steps if the
  mixer surface changed.

## Exclusions

- Ablation GPU/CPU campaigns (ticket 07)
- Hero token budgets (ticket 08)
- Falcon-2/3, chunk-parallel, lm-eval-harness 8-task suite

## Verification

- Config manager resolves the science config.
- Unit tests for addition encoding/masking and mixer switches.
- One-step (or few-step) science smoke in rootfs, finite loss.
- Claim label: `smoke` only until ticket 07 runs.

## Answer

Done, setup only, no commit. Claim label: `smoke`. Full report:
`sdd/task-06-report.md`.

Landed (TDD, rootfs pytest, `experiments -> core`, no `qwen3_5`/`mini_kimi_k3`
import or edit):

- Mixer switch in the same decoder shell: `FalconConfig.mixer =
  "falcon"|"gdn"|"softmax"` (default `falcon`, delayed + QK-RMSNorm).
  `gdn` is a same-step Gated DeltaNet residual write reimplemented in-file
  (`state <- gS + k((v - S^T k) beta)^T`, L2 q/k); `softmax` is causal SDPA +
  RoPE. `torchtitan/experiments/falcon/model.py`.
- nanogpt `.bin` reader + single-rank window dataloader
  (`torchtitan/experiments/falcon/bin_reader.py`), confirmed against the real
  FineWeb 10B shard (magic 20240520, version 1, uint16, vocab 50257).
- Addition task (`torchtitan/experiments/falcon/addition.py`): char-level,
  reversed `(n+1)`-digit sum LSD-first, suffix-only loss, ID `{1..16}` / OOD
  `{17..24}` declared gap (N=16, M=24).
- Science config `falcon_science` via ConfigManager: 4x256, 8 heads,
  head_dim 32, seq 512, batch 32 (16,384 tok/step), vocab 50257, delayed
  Falcon default, AdamW (0.9, 0.95); `steps=8000` registered, not executed.
- Few-step smoke `experiments/falcon/science_smoke.py`: builds `falcon_science`
  through the core Trainer under `fake_backend`, tiny synthetic `.bin`, N>1
  finite-loss steps. Verified in rootfs (`OK steps=3 ... all_finite=True`,
  loss ~ ln(50257)).

Verification (rootfs): `pytest -q tests/unit_tests/test_falcon_*.py` ->
**60 passed** (31 pre-existing + 29 new). Step 3 overfit gate and kernel
identity tests remain green. `pre-commit` clean on changed files
(`no-commit-to-branch` skipped; this ticket does not commit).

Not done (deferred): A0-A5 ablation runs (ticket 07), hero budgets (08),
Falcon-2/3, chunk-parallel, FLA, CP. A5 (QK-L2 phi) needs one small config
knob threaded to the Falcon mixer; noted in the report.
