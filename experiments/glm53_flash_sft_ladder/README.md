# GLM-5.3-Flash SFT Ladder Storage Runners

This directory holds host-side orchestration scripts for the GLM-5.3-Flash SFT
ladder. The scripts use the Waystone Rust CLI for HDFS storage and keep Python,
CUDA, model, and dataset business logic out of the host shell.

## Storage Contract

The system of record is the TorchTitan child of the shared Waystone prefix:

```bash
/data02/home/philip.yang/workspace/waystone/scripts/waystone layout-profile --project torchtitan --json
```

Child paths are resolved with `waystone storage-prefix --child`; scripts accept
paths under the TorchTitan project such as `datasets/countdown-sft/v1` and also
accept the explicit `torchtitan/...` spelling used in the campaign brief.

Authentication is maintained outside these scripts by:

```bash
~/.local/bin/waystone-auth-keepalive.sh
```

The scripts intentionally do not pass `--refresh-token`, because that path can
prompt for a password in non-interactive runs.

## Commands

Publish the fixed Countdown SFT v1 dataset:

```bash
experiments/glm53_flash_sft_ladder/publish_dataset_v1.sh
```

Ingest a Hugging Face model revision into HDFS:

```bash
experiments/glm53_flash_sft_ladder/hf_ingest.sh \
  --repo Qwen/Qwen3-0.6B \
  --revision main
```

Limit the ingest to selected file globs:

```bash
experiments/glm53_flash_sft_ladder/hf_ingest.sh \
  --repo Qwen/Qwen3-0.6B \
  --revision main \
  --include '*.json' \
  --include '*.safetensors'
```

Stage a completed HDFS tree into local disposable cache:

```bash
experiments/glm53_flash_sft_ladder/stage.sh \
  hdfs://harunava/user/tiger/waystone/torchtitan/model-weights/Qwen/Qwen3-0.6B/<sha> \
  /data05/philip.yang/glm53-ladder-cache/model-weights/Qwen/Qwen3-0.6B
```

Publish a generic artifact directory, such as checkpoints, adapters, eval
outputs, or run bundles:

```bash
experiments/glm53_flash_sft_ladder/publish.sh \
  /path/to/local/artifact \
  torchtitan/runs/glm53-ladder/<run_id>/<attempt_id>
```

All publish paths write `MANIFEST.jsonl` and `_COMPLETE` after file uploads.
`stage.sh` refuses to stage a tree without `_COMPLETE` and verifies every local
file against the manifest MD5 before writing `.staged.json`.

## Local Cache

The default cache root is:

```bash
/data05/philip.yang/glm53-ladder-cache
```

Override it with `GLM_LADDER_CACHE` when needed. The repo-local `cache/` and
`results/` directories are ignored by Git for small local scratch outputs.
