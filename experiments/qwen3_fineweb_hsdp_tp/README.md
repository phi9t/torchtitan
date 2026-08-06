# Hermetic Local Qwen3 Training: FineWeb subset, HSDP + TP

A repeatable, **offline** experiment that trains a tiny Qwen3 debug model on a
small slice of [FineWeb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
using **HSDP + TP** on an 8-GPU node (e.g. 8x B200).

## Why

Torchtitan ships a bundled `c4_test` local dataset, but FineWeb is not
registered and normally streams from HuggingFace. This experiment does a
**one-time online prefetch** of a FineWeb-edu slice to local JSON, registers it
as a local (non-streaming) dataset, and then trains fully offline. This keeps
the training run reproducible and free of network dependencies.

## Layout

```
experiments/qwen3_fineweb_hsdp_tp/
  README.md              # this file
  prefetch_fineweb.py    # ONE-TIME online prefetch -> local JSON
  run.sh                 # offline launch wrapper around ../../run_train.sh
  .gitignore             # ignores data/ (generated artifact)
  data/fineweb_test/data.json   # ~2000 lines {"text": ...} (generated)
```

## How to run

### 1. Prefetch the data (once, online, from repo root)

```bash
python experiments/qwen3_fineweb_hsdp_tp/prefetch_fineweb.py
```

Streams the first 2000 documents (override with `--num-docs`) from
`HuggingFaceFW/fineweb-edu` (`sample-10BT`) and writes
`data/fineweb_test/data.json` as `{"text": ...}` JSON-lines, mirroring
`tests/assets/c4_test/data.json`. Streaming with an early break avoids
downloading a full shard. This is the only network step.

### 2. Train (offline)

```bash
experiments/qwen3_fineweb_hsdp_tp/run.sh
```

Exports `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`
and launches `run_train.sh` with `NGPU=8 MODULE=qwen3
CONFIG=qwen3_debugmodel_fineweb`.

`run.sh` also defaults two env vars if the caller has not set them, to keep the
launch self-contained on hosts where the defaults do not work:

- `HF_HOME` -> `<repo>/.hf_cache`. `datasets` writes a small on-disk cache when
  materializing the local JSON; the system default cache may not be writable.
- `CC` / `CXX` -> `/usr/bin/gcc` / `/usr/bin/g++`. `torch.compile` / Triton
  shell out to a C compiler; a gcc that lacks `/usr/include/x86_64-linux-gnu`
  on its search path (e.g. a nix-profile gcc) cannot find the multiarch
  `pyconfig.h`. Override either var to use a different compiler.

## Mesh math

```
dp_replicate(2) x dp_shard(2) x tp(2) = 8 = NGPU
```

`dp_replicate` x `dp_shard` is HSDP (hybrid sharded data parallel); `tp` is
tensor parallel. The run hard-fails at startup if this product does not equal
world size, so a clean startup itself proves the mesh is correct.

## Config

`qwen3_debugmodel_fineweb` (in
`torchtitan/models/qwen3/config_registry.py`) mirrors `qwen3_debugmodel`
(dim 256, 8 layers, vocab 2048, tokenizer `./tests/assets/tokenizer`,
seq_len 2048, local_batch_size 8, 10 steps, lr 8e-4) but swaps the dataset to
`fineweb_test` and enables checkpointing (`qwen3_debugmodel` leaves it off).
Parallelism is left at defaults so `run.sh` supplies the mesh.

The `fineweb_test` dataset is registered in
`torchtitan/hf_datasets/text_datasets.py`, reusing the `c4_test` local-loader
pattern and `_process_c4_text` (FineWeb also has a `text` field).

## Expected output

- Device mesh / parallel dims showing `dp_replicate=2`, `dp_shard=2`, `tp=2`
  and an `Applied HSDP to the model` line.
- A `Preparing fineweb_test dataset from experiments/...` log line.
- 10 steps logged (log_freq=1) with loss trending down (e.g. ~8.0 -> ~3.9).
- A checkpoint written at step 10 (interval=10) under
  `outputs/checkpoint/step-10/` (DCP shards + `.metadata`).
- No `Downloading` / hub-request lines during training (fully offline).
