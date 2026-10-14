# S01 brief: speedrun harness + B0 baseline + data on HDFS
Read: `../spec.md` (binding), `.scratch/glm53-flash-sft-ladder/sdd/env-facts.md` (MANDATORY env facts),
`torchtitan/experiments/falcon/{HARNESS.md,bin_reader.py,config_registry.py,lm_eval.py}`, `experiments/modded_nanogpt_b200/
{upstream_reference.md,prepare_data.sh}`, `torchtitan/components/optim/` (Muon). Repo on branch ultron/mainline; do NOT commit.

You own: `torchtitan/experiments/glm53_arch_speedrun/` (new: `__init__.py`, `model.py` B0 only plus a clean seam
where S02 plugs element modules -- coordinate by keeping block/attention/residual/pos-enc/head as small replaceable
classes selected by config fields), `config_registry.py`, `tests/`, and `experiments/glm53_arch_speedrun/` (host runner
`run_arm.sh`, results/ and logs/ gitignored). S02 works concurrently in `.../glm53_arch_speedrun/elements/` -- do not
edit that directory.

Requirements:
1. Data: locate existing FineWeb10B GPT-2 shards on host (falcon config references them) or fetch via the upstream
   script inside the rootfs; publish to HDFS `torchtitan/datasets/fineweb10B-gpt2/<rev>` with MANIFEST + md5 via
   `experiments/glm53_flash_sft_ladder/publish.sh` (or lib/waystone.sh) if not already there; stage locally from HDFS.
2. B0 model + config: modded-nanogpt-classic recipe from spec D1, TorchTitan Trainer (FSDP/DDP 1-2 GPUs), Muon for 2D
   hidden matrices, AdamW for embeddings/head/scalars, trapezoid/WSD LR, seq len configurable (default 1024 x batch to
   ~0.5M tokens/step via grad accum), val CE on the FineWeb val shard every N steps (fixed 10M tokens or a declared
   subset), torch.compile on. Per-process GPU memory cap 24 GiB.
3. Tests: label shift (not roll), Muon/Adam param partition covers every param exactly once, to_empty()+init leaves no
   zero norm gains, config contract.
4. Runs: B0 seeds 0,1,2 at a declared token budget (start ~500M tokens; shrink if too slow -- report tokens/s); record
   val CE curve, tokens-to-3.28 if reached (else the calibrated target = B0 mean final CE), seed std. Also run seq 4096
   variant of B0 (1 seed) since KDA/indexer arms need longer context.
5. Artifact schema per spec constraints; publish run bundles to HDFS `torchtitan/runs/glm53-arch-speedrun/<run_id>/`.
Report `.scratch/glm53-arch-speedrun/sdd/s01-report.md`: harness note, data URIs, B0 table (per seed + mean/std),
throughput (shared_gpu), memory, exact launch commands, claim labels.
