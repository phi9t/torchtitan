# S08 report: FineWeb-Edu sample-10BT GPT-2 shards

## Result

Surface: repo-local experiment data prep. S08 prepared Hugging Face
`HuggingFaceFW/fineweb-edu` config `sample-10BT` as modded-nanogpt/S01
compatible GPT-2 `.bin` shards and published the bin-only dataset to HDFS.

Pinned dataset revision:

`87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`

Short dataset version:

`87f0914-v1`

HDFS dataset URI:

`hdfs://harunava/user/tiger/waystone/torchtitan/datasets/fineweb-edu-10BT-gpt2/87f0914-v1`

Local staged root:

`experiments/glm53_arch_speedrun/results/data/fineweb-edu-10BT-gpt2/87f0914-v1`

The HDFS directory has `_COMPLETE`. The local staged directory has `_COMPLETE`,
the HDFS manifest, the richer token manifest, tokenizer metadata, raw-download
metadata, and the final staged-path verification result.

## Counts

- Raw parquet: 14 files, 28,518,193,415 bytes.
- Documents: 9,672,101.
- GPT-2 tokens including EOT-before-document delimiters: 9,953,989,344.
- Validation shard: 1 shard, 100,000,000 tokens,
  `fineweb_edu_val_000000.bin`.
- Training shards: 99 shards, 9,853,989,344 tokens,
  `fineweb_edu_train_000001.bin` through `fineweb_edu_train_000099.bin`.
- Total bin bytes: 19,908,081,088.
- Last shard: `fineweb_edu_train_000099.bin`, 53,989,344 tokens.

Shard format matches S01/modded-nanogpt:

- 256 little-endian `int32` header values.
- `header[0] = 20240520`.
- `header[1] = 1`.
- `header[2] = token_count`.
- Flat `uint16` GPT-2 token stream after the header.
- GPT-2 EOT token id `50256` before every document.

## Manifests

Local staged files:

- `MANIFEST.jsonl`: rich token manifest with `file`, `size`, `md5`,
  `num_tokens`, `split`, and `shard_index`.
- `HDFS_MANIFEST.jsonl`: exact manifest layout published by
  `experiments/glm53_flash_sft_ladder/publish.sh`.
- `METADATA.json`: tokenizer/run metadata and document/token counts.
- `VERIFY.json`: final S01-reader verification result for the staged path.
- `RAW_METADATA.json` and `RAW_MANIFEST.jsonl`: raw parquet provenance and size
  metadata preserved before local raw parquet deletion.

HDFS manifest:

`hdfs://harunava/user/tiger/waystone/torchtitan/datasets/fineweb-edu-10BT-gpt2/87f0914-v1/MANIFEST.jsonl`

The HDFS manifest and local manifests match by filename, size, and md5. The
HDFS manifest is lexicographically sorted by filename, while the rich token
manifest records the validation shard first.

## Timings

- Hugging Face metadata + parquet download: 352.59 s.
- GPT-2 tokenization and shard writing: 256.75 s
  (37,671 docs/s, 38,769,251 tokens/s).
- Initial generated-directory verification: 37.54 s.
- HDFS publish wall time: 2026-10-14T17:57:26Z to 2026-10-14T18:01:17Z.
- Publish helper reported `HDFS_PUT_MBPS=22.320877162989508`.
- Final staged-path verification: 37.48 s.

The raw parquet was deleted locally after successful HDFS publish and local
staging because the rootfs-backed `/data05` path consumed `/data02`, which was
at 99% during the job. Raw metadata was preserved in the staged directory.

## Verification

Focused tooling checks:

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && pytest -q torchtitan/experiments/glm53_arch_speedrun/data_prep/tests/test_fineweb_edu.py'
```

Result: `1 passed, 14 warnings`.

Targeted hooks on new S08 files:

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc '
cd /workspace/torchtitan
for hook in flake8 ufmt pydoclint codespell; do
  pre-commit run "$hook" --files \
    torchtitan/experiments/glm53_arch_speedrun/data_prep/__init__.py \
    torchtitan/experiments/glm53_arch_speedrun/data_prep/fineweb_edu.py \
    torchtitan/experiments/glm53_arch_speedrun/data_prep/tests/__init__.py \
    torchtitan/experiments/glm53_arch_speedrun/data_prep/tests/test_fineweb_edu.py \
    experiments/glm53_arch_speedrun/data_fineweb_edu/prepare.sh \
    experiments/glm53_arch_speedrun/data_fineweb_edu/stage_from_hdfs.sh \
    experiments/glm53_arch_speedrun/data_fineweb_edu/README.md
done'
```

Result: flake8, ufmt, pydoclint, and codespell passed.

Final staged-path reader verification:

```bash
TIKTOKEN_CACHE_DIR=/data05/philip.yang/glm53-ladder-cache/tiktoken-cache \
  experiments/glm53_arch_speedrun/data_fineweb_edu/prepare.sh verify \
  --data-dir experiments/glm53_arch_speedrun/results/data/fineweb-edu-10BT-gpt2/87f0914-v1 \
  --spot-tokens 128
```

Result:

- `num_shards`: 100.
- `total_tokens`: 9,953,989,344.
- Reader:
  `torchtitan.experiments.falcon.bin_reader.read_nanogpt_bin`.
- Spot-check file:
  `fineweb_edu_val_000000.bin`.
- Spot-check prefix starts with EOT and decodes to:
  `<|endoftext|>The Independent Jane...`

HDFS/local verification:

- HDFS `_COMPLETE`: present.
- Local `_COMPLETE`: present after staging from HDFS manifest.
- HDFS rows: 100.
- Local staged rows: 100.
- HDFS/local bytes: 19,908,081,088.
- HDFS/local manifest match by `file`, `size`, and `md5`: true.
- Staged validation md5: `a485789508a20529e701db3a6b68f233`.

## Reader Config

Use the FineWeb-Edu filename patterns directly. The current S01 helper globs in
`torchtitan/experiments/glm53_arch_speedrun/config_registry.py` are
`fineweb_train_*.bin` and `fineweb_val_000000.bin`, so `GLM53_SPEEDRUN_DATA_DIR`
alone is not enough for this dataset unless the config registry is updated to
accept dataset-specific glob prefixes.

Training dataloader fields:

```python
FineWebBinDataLoader.Config(
    bin_glob=(
        "experiments/glm53_arch_speedrun/results/data/"
        "fineweb-edu-10BT-gpt2/87f0914-v1/fineweb_edu_train_*.bin"
    ),
    vocab_size=50257,
    repeat=True,
    num_mtp_layers=1 if model.elements.mtp_enabled else 0,
)
```

FineWeb-Edu validation dataloader fields:

```python
FineWebBinDataLoader.Config(
    bin_glob=(
        "experiments/glm53_arch_speedrun/results/data/"
        "fineweb-edu-10BT-gpt2/87f0914-v1/fineweb_edu_val_000000.bin"
    ),
    vocab_size=50257,
    repeat=True,
    num_mtp_layers=0,
)
```

Epoch accounting:

- FineWeb-Edu train tokens available: 9,853,989,344.
- Use `steps = ceil(9_853_989_344 / tokens_per_step)` for at-least-one-epoch
  runs, rounded by the run's fixed global tokens per optimizer step.
- Continue to run the original S01 FineWeb validation separately for
  comparability:
  `experiments/glm53_arch_speedrun/results/data/fineweb10B-gpt2/ecbb586-fineweb10b-gpt2-binonly-v1/fineweb_val_000000.bin`.

## Commands

Download:

```bash
TORCHTITAN_ROOTFS_NETWORK=networked \
  experiments/glm53_arch_speedrun/data_fineweb_edu/prepare.sh download \
  --revision 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9 \
  --raw-dir /data05/philip.yang/glm53-ladder-cache/datasets-raw/fineweb-edu/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9 \
  --download-workers 8
```

Tokenize:

```bash
TORCHTITAN_ROOTFS_NETWORK=networked \
TIKTOKEN_CACHE_DIR=/data05/philip.yang/glm53-ladder-cache/tiktoken-cache \
  experiments/glm53_arch_speedrun/data_fineweb_edu/prepare.sh tokenize \
  --revision 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9 \
  --raw-dir /data05/philip.yang/glm53-ladder-cache/datasets-raw/fineweb-edu/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9 \
  --output-dir /data05/philip.yang/glm53-ladder-cache/datasets-tokenized/fineweb-edu-10BT-gpt2/87f0914-v1 \
  --shard-size 100000000 \
  --batch-rows 2048 \
  --num-processes 64 \
  --pool-chunksize 1 \
  --overwrite
```

Publish:

```bash
experiments/glm53_flash_sft_ladder/publish.sh --parallel 4 \
  scripts/rootfs/rootfs/data05/philip.yang/glm53-ladder-cache/datasets-publish/fineweb-edu-10BT-gpt2/87f0914-v1-binonly \
  torchtitan/datasets/fineweb-edu-10BT-gpt2/87f0914-v1
```

