# T07c brief: real-weight parity gate for GLM-5.3-Flash (truncated model vs executable HF oracle)

Repo: /data02/home/philip.yang/workspace/torchtitan (branch ultron/mainline). Do NOT commit.
Read first: `.scratch/glm53-flash-sft-ladder/sdd/env-facts.md` (MANDATORY), `.scratch/glm53-flash-sft-ladder/sdd/task-07a-report.md`,
`torchtitan/experiments/glm53_flash/*`, research doc `docs/research/2026-10-14-glm-5.3-flash-architecture-and-finetuning.md` sec 1-2.

## Inputs
- Real BF16 checkpoint, staged locally by the T03 ingest (HDFS system of record:
  `hdfs://harunava/user/tiger/waystone/torchtitan/model-weights/zai-org/GLM-5.3-Flash-BF16/a5b45eb41df6402735dedc900be14a42e8d5e538`):
  `/data05/philip.yang/glm53-ladder-cache/model-weights/zai-org/GLM-5.3-Flash-BF16`. The ingest may still be running: wait
  (poll every 2 min) until every shard you need (per `model.safetensors.index.json`) is present with the size in the HF API
  and, ideally, `.staged.json` exists. Never modify files in that directory.
- Executable oracle: HF Transformers at commit `e0eb2810ac1b2caf6d082f247d8d37162ecf67a0` (has `models/glm5_next`). Install it
  into a SEPARATE rootfs env so neither the system nor glm53_train is changed:
  `TORCHTITAN_ROOTFS_ENV=glm53_ref TORCHTITAN_ROOTFS_NETWORK=networked scripts/rootfs/enter_rootfs.sh -- bash -lc 'uv venv --system-site-packages --python /usr/bin/python3 /project/venvs/b200-runtime && uv pip install --python /project/venvs/b200-runtime/bin/python "transformers @ git+https://github.com/huggingface/transformers.git@e0eb2810ac1b2caf6d082f247d8d37162ecf67a0"'`
  (constrain torch to the system 2.13 build; if that transformers needs a newer torch, install the same nightly as glm53_train).

## Requirements
1. Truncated parity (claim `deterministic_comparison`): build both models with `num_hidden_layers=4` (layers 0-3 cover dense MLP +
   KDA, and layer 3 sparse MLA + indexer + MoE -- verify against the real config), real embeddings/final norm/lm_head and the real
   weights for those layers, loaded through `torchtitan/experiments/glm53_flash/state_dict_adapter.py` for TorchTitan and through
   the HF loader (or manual state dict) for the oracle. Disable MTP. Inputs: 4 countdown prompts rendered with the REAL GLM chat
   template/tokenizer from the staged dir, plus one 4096-token sequence so the sparse top-k actually selects (top-k must be
   smaller than the number of pools). Compare in fp32 first (expect ~1e-4 rel), then bf16 (report max abs diff, top-1
   agreement, KL). Compare per-layer hidden states to localise any mismatch. Fix discrepancies in the experiment seam (not core)
   and re-run until fp32 matches. Indices chosen by the indexer must match exactly in fp32.
2. Repeat for a random 8-layer slice from the middle (e.g. layers 20-27) to cover more sparse layers.
3. Full-checkpoint key coverage: run the adapter over the full real index: every HF key mapped or explicitly skipped
   (vision/MTP/scale), zero unknown keys, shapes match the `glm53_flash` flavor.
4. Tokenizer + chat template: record how the GLM chat template renders the countdown user/assistant turns (thinking tags?), and
   add a GLM countdown data config `glm53_flash_countdown_lora` data path using the real tokenizer and an assistant-only loss
   mask; verify the mask on 2 samples (print token/label alignment).
5. Write the parity script as `torchtitan/experiments/glm53_flash/hf_parity.py` (reusable) and a test that is skipped when the
   checkpoint is absent.

## Exclusions
No core edits, no commit, do not modify the staged checkpoint, GPUs shared (<60 GiB/GPU; prefer CPU for fp32 parity if memory
allows -- the host has a lot of RAM; check `free -g`). Never kill other processes.

## Report
`.scratch/glm53-flash-sft-ladder/sdd/task-07c-report.md`: env used, transformers commit, slices tested, per-layer diffs fp32/bf16,
index agreement, key coverage counts, chat template rendering, fixes made, claim labels, no commit.
