# T07a brief: GLM-5.3-Flash model code in TorchTitan (experiment seam), debug + mini flavors, reference parity

Repo: /data02/home/philip.yang/workspace/torchtitan (branch ultron/mainline). Do NOT commit.
Read first (in order): `.scratch/glm53-flash-sft-ladder/sdd/env-facts.md` (MANDATORY),
`docs/research/2026-10-14-glm-5.3-flash-architecture-and-finetuning.md` (sections 1-4 in full: model contract, mechanism
deep dive with tensor shapes and HF line refs, repo mapping table 3.1, indexer behaviour 3.2, mHC finding 3.4),
`.claude/CLAUDE.md`, `.claude/rules/models.md`, `torchtitan/models/README.md`, `torchtitan/experiments/README.md`.
Study as building blocks: `torchtitan/models/kimi_k3/` (KDA hybrid, MLA, MoE, state-dict adapter, sharding),
`torchtitan/models/deepseek_v4/` (indexer/compressor, gather attention, mHC, MoE), `torchtitan/models/common/attention/kda.py`,
`torchtitan/models/common/moe.py`, `torchtitan/models/common/lora.py`, `torchtitan/config/transform/lora.py`.

## Reference implementation
The canonical reference is the upstream HF Transformers modeling file for GLM-5.3-Flash (the research doc cites it as `HF:<line>`;
find the exact file/commit it used in the doc's source list). Download the .py (and config.json from
`zai-org/GLM-5.3-Flash` rev 91b69215804f2d2036a8867f6ce44848e278dcda) into a fresh empty dir
`/data02/home/philip.yang/devx/tmp/t07a_ref/`. If a transformers release that contains this model can be installed into a
SEPARATE venv inside the rootfs (`uv venv /workspace/torchtitan/.cache/venvs/hf-glm53 --system-site-packages` is NOT ok since it
would see rootfs torch... it IS ok to reuse rootfs torch via `--system-site-packages`; just never modify system transformers 5.15),
use it as an executable oracle on RANDOM tiny weights. Otherwise transcribe the reference math into a pure-torch oracle in the tests.
Never import downloaded code from the download dir directly.

## Requirements
Create `torchtitan/experiments/glm53_flash/` (experiment seam; do NOT edit core files -- if a core component needs a
generalisation, implement the GLM-specific variant inside the experiment and note the proposed core change in the report):
- `flavors.py`: `glm53_flash` (exact config from the research doc: 45 layers, KDA on layers where (i+1)%4!=0, sparse MLA+indexer
  on (i+1)%4==0, 4-stream mHC, MoE 288 experts top-8 + 1 shared, layers 0-2 dense, no positional embeddings, vocab, norms,
  eps), `debugmodel` (e.g. 4-8 layers keeping the 3:1 pattern, small hidden, 8 experts top-2, real head dims where kernels require
  them: KDA head_dim 128, MLA latent 512 if the gather kernel requires it), and `mini` (~0.5-1.5B total, same pattern, for rung 3).
  MTP: skip for SFT (document; adapter should ignore/skip mtp keys explicitly).
- `model.py`: decoder with mHC pre/post using the CORRECT post-mix `comb^T @ residual` (sum over source stream i), GLM eps
  placement (eps after softmax), final unweighted mean over streams; KDA layers with the low-rank output gate (g_a/g_b);
  sparse layers: absorbed MLA without rope, IndexPool (`LayerNorm(wk(x))` pooled in groups of 4 with softmax(gate(x)) weights,
  per-document via cu_seqlens / varlen), indexer scoring sum_h w*relu(q.k), top-k pools (512 in full config), expand pools to raw
  token indices + causal tail, gather attention over raw latent tokens (no window, no sink), scale per reference. Indexer runs
  under no_grad / detached (frozen; research 3.2). Use attn_gym `gather_attn`/`lightning_indexer` where shapes allow, and keep a
  pure-torch reference path (`attn_impl="reference"`) usable on CPU and for parity tests.
  Use the repo's logical-shape suffix naming (x_BLD, q_TNH, ...) with a per-module legend; ASCII-only comments.
- `state_dict_adapter.py`: HF <-> TorchTitan names for the BF16 repo (`zai-org/GLM-5.3-Flash-BF16`, prefix
  `model.language_model.layers.N.*`; per-expert -> grouped experts; separate q/k/v conv1d; dt_bias reshape; skip
  `weight_scale_inv`; vision tower and MTP keys skipped explicitly with a logged count). Round-trip test (to_hf(from_hf(x)) == x)
  on a random debug state dict built with the real key schema (derive the key list from `model.safetensors.index.json` of the
  BF16 repo -- fetch only the index json).
- `sharding.py`: FSDP (+ EP for experts) at minimum; TP optional (document if skipped). Validate mesh axes explicitly.
- `config_registry.py`: `glm53_flash_debugmodel` (synthetic/c4_test data, few steps), `glm53_flash_mini_countdown_lora`
  placeholder wired to the countdown dataset via env vars like the Qwen3 countdown recipe (T02 will finalise data paths),
  and `glm53_flash_countdown_lora` (full model, LoRA targets per research 3.1: attention q/kv/o projections, KDA projections,
  shared-expert w13/w2; rank 16, alpha 64), with FullAC.
- `tests/` (under the experiment or `tests/unit_tests/test_glm53_flash_*.py`): (1) mHC post-mix vs oracle; (2) IndexPool vs oracle
  incl. document boundaries; (3) indexer top-k + causal mask vs oracle; (4) sparse attention fast path vs reference path on GPU
  (bf16 tolerance) and reference path vs dense causal attention when top-k covers all tokens; (5) KDA layer vs oracle incl. low-rank
  gate; (6) whole debug model forward vs reference oracle (random weights) -- executable HF oracle if available; (7) finite
  forward+backward at seq 512 and 4096 on GPU (no NaN/Inf in grads; Falcon lesson: forward allclose is not enough);
  (8) meta-init -> to_empty() -> init_weights leaves no zero norm gains / no identical outputs across layers; (9) state-dict round trip.
- Training smoke: `glm53_flash_debugmodel` on 2 GPUs (CUDA_VISIBLE_DEVICES of your choice) for 20 steps, finite decreasing loss;
  with LoRA transform too. Use run_train.sh MODULE=torchtitan.experiments.glm53_flash.config_registry.
All Python in rootfs. GPUs are shared with another user's job (env-facts): keep memory < 60 GiB/GPU.

## Exclusions
No core edits (report proposed core generalisations instead). No real GLM weights download (index json only). No commit.
Do not change the DSv4 model (T04 handles its HcPost bug separately).

## Report
`.scratch/glm53-flash-sft-ladder/sdd/task-07a-report.md`: files, reference source+commit, every test with result, training smoke
losses, which attention path runs on B200 (attn_gym kernel vs reference), known gaps (CP, TP, MTP, perf), proposed core changes,
claim_label `deterministic_comparison` for parity tests and `smoke` for training. No commit.
