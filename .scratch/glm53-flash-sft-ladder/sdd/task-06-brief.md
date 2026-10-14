# T06 brief: rung 2 -- Qwen3 30B-A3B (MoE) LoRA SFT on countdown-sft/v1

Repo: /data02/home/philip.yang/workspace/torchtitan (branch ultron/mainline). Do NOT commit.
Read first: `.scratch/glm53-flash-sft-ladder/sdd/env-facts.md` (MANDATORY), `spec.md`, `sdd/task-05-report.md` (template),
`experiments/glm53_flash_sft_ladder/run_rung.sh`, `torchtitan/models/qwen3/{flavors.py,sharding.py,state_dict_adapter.py}`,
`torchtitan/models/common/moe.py`, `torchtitan/config/transform/lora.py` (grouped-expert LoRA: rank must be divisible by 8).

## Why this rung
GLM-5.3-Flash is a 288-expert MoE. This rung de-risks MoE SFT mechanics at 1/10 the cost: HF MoE checkpoint load, FSDP and
expert parallelism (EP) for training, LoRA on attention + (optionally) experts, router behaviour under SFT, and vLLM MoE LoRA eval.

## Requirements
1. Config `qwen3_30b_a3b_countdown_lora` with the frozen knobs (rank 16, alpha 64, lr 1e-4, 64x512 tokens/step, seq 512,
   94 steps). LoRA targets: attention wqkv/wo always; arm B adds routed+shared experts if the LoRA transform supports grouped
   experts. Run arm A first (2 seeds); arm B one seed if supported (report if unsupported, with the reason).
2. Parallelism: 4 or 8 GPUs, FSDP + EP (if the Qwen3 MoE path supports EP in this checkout; validate mesh axes); otherwise FSDP
   only. Record memory per GPU -- the shared job holds ~84 GiB/GPU; use CPU offload (`training.enable_cpu_offload`) if needed
   and say so.
3. Gates: HF logits parity (parity.py, MoE routing in fp32), LoRA SFT 2 seeds, resume-from-HDFS gate for one seed, steady
   throughput (`shared_gpu`), router stats before/after (expert load histogram on dev prompts -- detect collapse), adapter
   export + vLLM dev eval base vs adapter (does vLLM 0.27 serve MoE LoRA for Qwen3-MoE? if not, evaluate with a merged
   checkpoint or TorchTitan-native sampling and say which), publish to HDFS.
4. Extend the scale table from T05 with 30B-A3B.
Staged weights: `/data05/philip.yang/glm53-ladder-cache/model-weights/Qwen/Qwen3-30B-A3B`
(HDFS `.../model-weights/Qwen/Qwen3-30B-A3B/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`).

## Exclusions
No knob search, no core edits unless a clear bug blocks MoE LoRA (then stop and report), no commit, never kill others.

## Report
`.scratch/glm53-flash-sft-ladder/sdd/task-06-report.md` per contract.
