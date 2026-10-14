# T01 report: waystone storage wrappers + dataset v1 publish

Date: 2026-10-14
claim_label: smoke

## Outcome

Implemented the repo-local GLM-5.3-Flash ladder storage runner under
`experiments/glm53_flash_sft_ladder/` and exercised it against live Waystone
HDFS storage.

This is a repo-local research runner surface only. No core TorchTitan code was
changed and no commit was made.

## Files created

- `experiments/glm53_flash_sft_ladder/lib/waystone.sh`
- `experiments/glm53_flash_sft_ladder/hf_ingest.sh`
- `experiments/glm53_flash_sft_ladder/stage.sh`
- `experiments/glm53_flash_sft_ladder/publish.sh`
- `experiments/glm53_flash_sft_ladder/publish_dataset_v1.sh`
- `experiments/glm53_flash_sft_ladder/README.md`
- `.scratch/glm53-flash-sft-ladder/sdd/task-01-report.md`

The required ignore rules for `experiments/glm53_flash_sft_ladder/results/` and
`experiments/glm53_flash_sft_ladder/cache/` are present in `.gitignore`. The
existing `.gitignore` entry also ignores `experiments/glm53_flash_sft_ladder/logs/`.

Generated verification logs were written under the ignored directory
`experiments/glm53_flash_sft_ladder/results/task-01/`.

## HDFS outputs

Dataset v1:

```text
hdfs://harunava/user/tiger/waystone/torchtitan/datasets/countdown-sft/v1
```

Qwen3-0.6B model weights:

```text
hdfs://harunava/user/tiger/waystone/torchtitan/model-weights/Qwen/Qwen3-0.6B/c1899de289a04d12100db370d81485cdf75e47ca
```

Staged local model cache:

```text
/data05/philip.yang/glm53-ladder-cache/model-weights/Qwen/Qwen3-0.6B
```

## Dataset v1 publish

Source directory:

```text
experiments/countdown_search_distill/sweeps/formatting_replication/formatting_seed43_train2000_rank16/data
```

Published files:

| file | rows | bytes | md5 |
|---|---:|---:|---|
| `train/formatting.jsonl` | 931 | 442776 | `640245bc9b13d240464f50f130c666f0` |
| `train/problems.jsonl` | 2000 | 1078178 | `0844d1c7d4fc9c13280cacc213422481` |
| `dev/problems.jsonl` | 500 | 269582 | `6c83e0a60e2cb29b43061a9d3c9a3a70` |
| `iid_test/problems.jsonl` | 1000 | 539112 | `5ea8d81cd90c5a244d7316b69f6e7bd6` |
| `ood_test/problems.jsonl` | 500 | 272990 | `6dd2c15b17bb5dba2664621c858d4a86` |
| `calibration/problems.jsonl` | 500 | 269578 | `26ca06b69244bd6a910a7c333be87c0d` |
| `split_registry.json` | 4041 | 93112 | `970ad1d26db9e770027c45bf7c595664` |
| `README.md` | 15 | 890 | `f5299404e2fecc26b2846b5059f1fc9f` |

`publish_dataset_v1.sh` output:

```text
HDFS_URI=hdfs://harunava/user/tiger/waystone/torchtitan/datasets/countdown-sft/v1
FILES=8
HDFS_PUT_MBPS=0.15782224522773455
```

HDFS listing confirmed `_COMPLETE`:

```text
Found 9 items in hdfs://harunava/user/tiger/waystone/torchtitan/datasets/countdown-sft/v1
Name                       Size      Type  Modified
-----------------------------------------------------------
calibration                   -       DIR  58792-334
dev                           -       DIR  58792-334
iid_test                      -       DIR  58792-334
ood_test                      -       DIR  58792-334
train                         -       DIR  58792-334
MANIFEST.jsonl             693B      FILE  58792-334
README.md                  890B      FILE  58792-334
_COMPLETE                  186B      FILE  58792-334
split_registry.json      90.9KB      FILE  58792-334
```

## Qwen3-0.6B ingest and stage

Resolved `Qwen/Qwen3-0.6B` revision `main` to commit:

```text
c1899de289a04d12100db370d81485cdf75e47ca
```

Initial ingest output:

```text
HDFS_URI=hdfs://harunava/user/tiger/waystone/torchtitan/model-weights/Qwen/Qwen3-0.6B/c1899de289a04d12100db370d81485cdf75e47ca
FILES=10
SKIPPED=0
HF_DOWNLOAD_MBPS=71.90941216478467
HDFS_PUT_MBPS=21.057057427455582
```

HDFS listing confirmed `_COMPLETE`:

```text
Found 12 items in hdfs://harunava/user/tiger/waystone/torchtitan/model-weights/Qwen/Qwen3-0.6B/c1899de289a04d12100db370d81485cdf75e47ca
Name                          Size      Type  Modified
--------------------------------------------------------------
.gitattributes               1.5KB      FILE  58792-336
LICENSE                     11.1KB      FILE  58792-336
MANIFEST.jsonl                985B      FILE  58792-337
README.md                   13.6KB      FILE  58792-336
_COMPLETE                     114B      FILE  58792-337
config.json                   726B      FILE  58792-336
generation_config.json        239B      FILE  58792-336
merges.txt                   1.6MB      FILE  58792-336
model.safetensors           1.40GB      FILE  58792-337
tokenizer.json              10.9MB      FILE  58792-336
tokenizer_config.json        9.5KB      FILE  58792-336
vocab.json                   2.6MB      FILE  58792-336
```

Stage output:

```text
STAGED_DIR=/data05/philip.yang/glm53-ladder-cache/model-weights/Qwen/Qwen3-0.6B
SOURCE_URI=hdfs://harunava/user/tiger/waystone/torchtitan/model-weights/Qwen/Qwen3-0.6B/c1899de289a04d12100db370d81485cdf75e47ca
FILES=10
SKIPPED=0
HDFS_GET_MBPS=18.195449178611025
STAGED_JSON=/data05/philip.yang/glm53-ladder-cache/model-weights/Qwen/Qwen3-0.6B/.staged.json
```

Independent staged-dir manifest verification:

```text
MD5 check passed for 10 files in /data05/philip.yang/glm53-ladder-cache/model-weights/Qwen/Qwen3-0.6B
```

No-op ingest rerun:

```text
HDFS_URI=hdfs://harunava/user/tiger/waystone/torchtitan/model-weights/Qwen/Qwen3-0.6B/c1899de289a04d12100db370d81485cdf75e47ca
FILES=10
SKIPPED=10
HF_DOWNLOAD_MBPS=0
HDFS_PUT_MBPS=0
```

The no-op rerun also printed:

```text
[waystone] remote tree already complete; skipped MANIFEST.jsonl and _COMPLETE upload
```

## Commands run

```bash
sed -n '1,240p' .scratch/glm53-flash-sft-ladder/sdd/task-01-brief.md
sed -n '1,220p' docs/agents/multi-agent-collab.md
sed -n '1,260p' .scratch/glm53-flash-sft-ladder/spec.md
sed -n '1,260p' .claude/CLAUDE.md
sed -n '261,620p' .claude/CLAUDE.md
sed -n '1,260p' /data02/home/philip.yang/workspace/waystone/AGENTS.md
sed -n '261,620p' /data02/home/philip.yang/workspace/waystone/AGENTS.md
sed -n '1,260p' /data02/home/philip.yang/workspace/waystone/docs/shared-hdfs-storage.md
sed -n '1,260p' /data02/home/philip.yang/workspace/waystone/docs/hdfs-cli-contract.md
git status --short
find experiments/glm53_flash_sft_ladder -path '*/logs/*' -prune -o -type f -print | sort
find experiments/countdown_search_distill/sweeps/formatting_replication/formatting_seed43_train2000_rank16/data -maxdepth 3 -type f -print | sort
/data02/home/philip.yang/workspace/waystone/scripts/waystone storage-prefix --child torchtitan/tmp
/data02/home/philip.yang/workspace/waystone/scripts/waystone layout-profile --project torchtitan --json
/data02/home/philip.yang/workspace/waystone/scripts/waystone ls hdfs://harunava/user/tiger/waystone/torchtitan
/data02/home/philip.yang/workspace/waystone/scripts/waystone --output json ls hdfs://harunava/user/tiger/waystone/torchtitan
/data02/home/philip.yang/workspace/waystone/scripts/waystone raw test -e hdfs://harunava/user/tiger/waystone/torchtitan
curl -L --fail --retry 2 -sS 'https://huggingface.co/api/models/Qwen/Qwen3-0.6B/revision/main' | jq '{sha, sibling_count: (.siblings|length), sample: [.siblings[:5][] | {rfilename, size, lfs}]}'
curl -L --fail --retry 2 -sS 'https://huggingface.co/api/models/Qwen/Qwen3-0.6B/tree/main?recursive=1' | jq -r '.[0:5][] | @json'
curl -L --fail --retry 2 -sS 'https://huggingface.co/api/models/Qwen/Qwen3-0.6B/tree/main?recursive=1' | jq -r '.[] | select(.path=="model.safetensors") | @json'
command -v jq
command -v shellcheck
command -v timeout
command -v curl
command -v md5sum
df -h /data05 /data02
chmod +x experiments/glm53_flash_sft_ladder/hf_ingest.sh experiments/glm53_flash_sft_ladder/stage.sh experiments/glm53_flash_sft_ladder/publish.sh experiments/glm53_flash_sft_ladder/publish_dataset_v1.sh experiments/glm53_flash_sft_ladder/trae_worker.sh
bash -n experiments/glm53_flash_sft_ladder/lib/waystone.sh experiments/glm53_flash_sft_ladder/hf_ingest.sh experiments/glm53_flash_sft_ladder/stage.sh experiments/glm53_flash_sft_ladder/publish.sh experiments/glm53_flash_sft_ladder/publish_dataset_v1.sh experiments/glm53_flash_sft_ladder/trae_worker.sh
shellcheck experiments/glm53_flash_sft_ladder/lib/waystone.sh experiments/glm53_flash_sft_ladder/hf_ingest.sh experiments/glm53_flash_sft_ladder/stage.sh experiments/glm53_flash_sft_ladder/publish.sh experiments/glm53_flash_sft_ladder/publish_dataset_v1.sh experiments/glm53_flash_sft_ladder/trae_worker.sh
experiments/glm53_flash_sft_ladder/publish_dataset_v1.sh 2>&1 | tee experiments/glm53_flash_sft_ladder/results/task-01/publish_dataset_v1.log
/data02/home/philip.yang/workspace/waystone/scripts/waystone ls hdfs://harunava/user/tiger/waystone/torchtitan/datasets/countdown-sft/v1 2>&1 | tee experiments/glm53_flash_sft_ladder/results/task-01/dataset_v1_ls.log
experiments/glm53_flash_sft_ladder/hf_ingest.sh --repo Qwen/Qwen3-0.6B --revision main 2>&1 | tee experiments/glm53_flash_sft_ladder/results/task-01/hf_ingest_qwen3_0_6b.log
/data02/home/philip.yang/workspace/waystone/scripts/waystone ls hdfs://harunava/user/tiger/waystone/torchtitan/model-weights/Qwen/Qwen3-0.6B/c1899de289a04d12100db370d81485cdf75e47ca 2>&1 | tee experiments/glm53_flash_sft_ladder/results/task-01/qwen3_0_6b_hdfs_ls.log
experiments/glm53_flash_sft_ladder/stage.sh hdfs://harunava/user/tiger/waystone/torchtitan/model-weights/Qwen/Qwen3-0.6B/c1899de289a04d12100db370d81485cdf75e47ca /data05/philip.yang/glm53-ladder-cache/model-weights/Qwen/Qwen3-0.6B 2>&1 | tee experiments/glm53_flash_sft_ladder/results/task-01/stage_qwen3_0_6b.log
manifest=/data05/philip.yang/glm53-ladder-cache/stage-tmp/20261014T084936Z-4147786/MANIFEST.jsonl
staged=/data05/philip.yang/glm53-ladder-cache/model-weights/Qwen/Qwen3-0.6B
while IFS= read -r record; do file="$(jq -r '.file' <<<"$record")"; expected="$(jq -r '.md5' <<<"$record")"; actual="$(md5sum "${staged}/${file}" | awk '{print $1}')"; [[ "$actual" == "$expected" ]] || exit 1; done < "$manifest"
experiments/glm53_flash_sft_ladder/hf_ingest.sh --repo Qwen/Qwen3-0.6B --revision main 2>&1 | tee experiments/glm53_flash_sft_ladder/results/task-01/hf_ingest_qwen3_0_6b_rerun_fixed.log
```

## Failures and notes

- An initial `publish_dataset_v1.sh` attempt failed before HDFS writes because a
  generated README line used `printf` with a leading `-` format string. Fixed by
  printing those README bullet lines through `%s`.
- A preliminary scratch helper smoke uploaded/downloaded a tiny file
  successfully, then the ad hoc shell command failed while printing an unset
  checksum variable. The scratch HDFS directory was removed and the smoke was
  rerun successfully.
- The Waystone docs mention `raw -stat`, but this installed native wrapper
  expects the native subcommands exposed by `hdfs --help` and does not expose a
  `stat` command. The scripts use `raw test -e`, JSON `ls`, and manifest MD5s
  instead.
