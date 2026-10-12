# Modded NanoGPT Upstream Reference

Status: reference
Inspected: 2026-08-14

## Snapshot

- Repository: `https://github.com/kellerjordan/modded-nanogpt`
- Commit: `ecbb586296d3dac36fd206211f25d63bad4a6b35`

This reference records the upstream metadata used by the B200 benchmark design.
Future implementation should refresh this file only when intentionally moving
to a different upstream commit.

## Benchmark Target

The upstream README defines the primary speedrun as training on 8 NVIDIA H100
GPUs to 3.28 FineWeb cross-entropy validation loss:
<https://github.com/kellerjordan/modded-nanogpt/blob/ecbb586296d3dac36fd206211f25d63bad4a6b35/README.md#L1-L11>

The current upstream claim at this snapshot is under 75 seconds and under 400M
tokens:
<https://github.com/kellerjordan/modded-nanogpt/blob/ecbb586296d3dac36fd206211f25d63bad4a6b35/README.md#L9-L11>

## Technique Bundle

The inspected README lists the active technique bundle, including Muon, FP8
head and MLP paths, Flash Attention 3, long-short sliding window attention,
batch and max-sequence schedules, bigram hash embedding, MUDD, and prefix-token
auxiliary loss:
<https://github.com/kellerjordan/modded-nanogpt/blob/ecbb586296d3dac36fd206211f25d63bad4a6b35/README.md#L13-L44>

## Standard Run Path

The upstream run instructions install requirements, download the first 900M
FineWeb training tokens with `python data/cached_fineweb10B.py 9`, and launch
`./run.sh`:
<https://github.com/kellerjordan/modded-nanogpt/blob/ecbb586296d3dac36fd206211f25d63bad4a6b35/README.md#L68-L82>

`run.sh` launches exactly 8 local processes:
<https://github.com/kellerjordan/modded-nanogpt/blob/ecbb586296d3dac36fd206211f25d63bad4a6b35/run.sh#L1>

Upstream notes that `torch.compile` adds about 7 minutes of first-run latency:
<https://github.com/kellerjordan/modded-nanogpt/blob/ecbb586296d3dac36fd206211f25d63bad4a6b35/README.md#L80-L80>

## Container Metadata

The README presents Docker as a way to standardize CUDA, NCCL, CUDNN, and
Python:
<https://github.com/kellerjordan/modded-nanogpt/blob/ecbb586296d3dac36fd206211f25d63bad4a6b35/README.md#L84-L99>

The inspected Dockerfile starts from CUDA 12.6.2 with CUDNN on Ubuntu 24.04,
builds Python 3.12.7, installs `requirements.txt`, then upgrades to a
pre-release PyTorch wheel from the `cu126` index:
<https://github.com/kellerjordan/modded-nanogpt/blob/ecbb586296d3dac36fd206211f25d63bad4a6b35/Dockerfile#L1-L33>

## Competition Rules

The upstream rules require:

- do not modify train or validation data pipelines;
- attain mean validation loss <= 3.28;
- do not use extra `torch._inductor.config` or `torch.compile` flags on the
  main track;
- run faster than the prior record when baselined on the same hardware.

Source:
<https://github.com/kellerjordan/modded-nanogpt/blob/ecbb586296d3dac36fd206211f25d63bad4a6b35/README.md#L226-L249>

The README also clarifies that the target metric is cross-entropy loss on the
FineWeb validation set:
<https://github.com/kellerjordan/modded-nanogpt/blob/ecbb586296d3dac36fd206211f25d63bad4a6b35/README.md#L251-L257>

## Trainer Facts

The trainer initializes CUDA and distributed state from `LOCAL_RANK`:
<https://github.com/kellerjordan/modded-nanogpt/blob/ecbb586296d3dac36fd206211f25d63bad4a6b35/train_gpt.py#L58-L67>

The trainer uses FP8 `_scaled_mm` paths:
<https://github.com/kellerjordan/modded-nanogpt/blob/ecbb586296d3dac36fd206211f25d63bad4a6b35/train_gpt.py#L71-L120>

It loads Flash Attention 3 through the `kernels` package and calls
`flash_attn_varlen_func`:
<https://github.com/kellerjordan/modded-nanogpt/blob/ecbb586296d3dac36fd206211f25d63bad4a6b35/train_gpt.py#L1069-L1141>

The inspected hyperparameters fix validation at 10,485,760 tokens, use a
validation batch size of `4 * 64 * 1024 * 8`, and define 1270 scheduled plus 15
extension iterations:
<https://github.com/kellerjordan/modded-nanogpt/blob/ecbb586296d3dac36fd206211f25d63bad4a6b35/train_gpt.py#L1880-L1904>

The script logs Python, PyTorch, CUDA, Triton, `nvidia-smi`, sampled warmup
steps, validation loss, `train_time`, `step_avg`, and peak CUDA memory:
<https://github.com/kellerjordan/modded-nanogpt/blob/ecbb586296d3dac36fd206211f25d63bad4a6b35/train_gpt.py#L2220-L2382>

The timed region excludes the compile/kernel warmup section and starts after
model reset and `torch.cuda.synchronize()`:
<https://github.com/kellerjordan/modded-nanogpt/blob/ecbb586296d3dac36fd206211f25d63bad4a6b35/train_gpt.py#L2264-L2319>
