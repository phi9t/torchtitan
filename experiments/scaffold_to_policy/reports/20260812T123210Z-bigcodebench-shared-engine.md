# BigCodeBench-Hard Shared-Engine Runner Fix

Run IDs:

- `20260812Tbigcode-hard-isolated-gpu0`
- `20260812Tbigcode-hard-shared-engine`
- `20260812Tbigcode-hard-shared-engine-verified`

This is a coding-lane infrastructure report, not a BigCodeBench-Hard capability
claim.

## What Changed

`run_bigcodebench_hard_public_vllm_smoke.sh` now evaluates dev and OOD splits
with one vLLM engine through `evaluate-coding-style-vllm-splits`.

The previous runner launched one engine per split. On the B200 rootfs path, the
first split could complete and the immediate second vLLM initialization could
observe stale GPU memory state, including negative available KV-cache memory.
That made the runner look benchmark-blocked even after the underlying split
could run in a fresh process.

## Evidence

`20260812Tbigcode-hard-isolated-gpu0` ran the previous split-by-split path on
`CUDA_VISIBLE_DEVICES=0`:

- Dev canonical preflight: selected `1/1`, `BigCodeBench/13`.
- OOD canonical preflight: selected `1/1`, `BigCodeBench/267`.
- GPU preflight: selected at `GPU_MEMORY_UTILIZATION=0.05`.
- Dev model evaluation completed with pass@1 through pass@32 of `0.0`.
- Immediate OOD model evaluation failed at second vLLM engine initialization:
  available KV-cache memory was reported as `-58.78 GiB`.

The same OOD problem then succeeded when launched as a fresh rootfs process:

```bash
CUDA_VISIBLE_DEVICES=0 scripts/rootfs/enter_rootfs.sh -- \
  python -m torchtitan.experiments.scaffold_to_policy.cli \
  evaluate-coding-style-vllm \
  --problems experiments/scaffold_to_policy/data/bigcode_hard_isolated_gpu0/ood_test.jsonl \
  --model ./assets/hf/Qwen3-1.7B \
  --output experiments/scaffold_to_policy/results/bigcode_hard_isolated_gpu0/eval/ood_test_evaluations_fresh.jsonl \
  --summary experiments/scaffold_to_policy/results/bigcode_hard_isolated_gpu0/eval/ood_test_summary_fresh.json \
  --num-rollouts 1 \
  --max-new-tokens 256 \
  --prompt-variant chat \
  --temperature 0.2 \
  --top-p 0.95 \
  --gpu-memory-utilization 0.05 \
  --timeout-seconds 10
```

Fresh-process OOD summary:

- `num_problems=1`
- `total_rollouts=1`
- pass@1 through pass@32: `0.0`
- failure breakdown: `assertion failure=1`

This isolates the failure to the runner lifecycle, not the imported task or
canonical verifier.

## Example Failures

Dev `BigCodeBench/13` asks for FTP download behavior with mocked FTP and
subprocess calls. Qwen3 produced a plausible direct-FTP implementation, but it
downloaded to `/ftp/test/file1.txt`, did not return the file list, and did not
wrap connection/login/cwd errors with the exact expected messages. The released
tests failed with one file-path error and four assertion failures.

OOD `BigCodeBench/267` asks for adding key `a`, building a numeric signal from
dictionary values, returning an FFT, and labeling a Matplotlib axis with exact
strings. Qwen3 attempted `fftpack.fft(data)` directly on the dictionary,
returned a magnitude array instead of `(fft, ax)`, and used different plot
labels. The released tests failed with five type errors.

Both examples are valid verifier failures, not harness errors.

## Verified Shared-Engine Run

After GPU 0 became free, the updated shared-engine runner completed as
`20260812Tbigcode-hard-shared-engine-verified`:

- `DEV_PROBLEMS=1`, `OOD_PROBLEMS=1`, `NUM_ROLLOUTS=1`
- `MAX_NEW_TOKENS=256`
- `GPU_MEMORY_UTILIZATION=0.05`
- `CUDA_VISIBLE_DEVICES=0`
- `INSTALL_BIGCODEBENCH_DEPS=0`

The run wrote:

- stage manifest:
  `experiments/scaffold_to_policy/results/bigcode_hard_shared_engine_verified/manifests/20260812Tbigcode-hard-shared-engine-verified.jsonl`
- report input:
  `experiments/scaffold_to_policy/results/bigcode_hard_shared_engine_verified/manifests/report_input_20260812Tbigcode-hard-shared-engine-verified.json`

The report checks selected:

```text
split_registry_selected=true
summaries_present=true
summary_split_counts_match=true
preflights_present=true
preflight_split_counts_match=true
preflight_canonical_solutions_pass=true
artifact_provenance_labeled=true
```

Both split summaries were benchmark-clean but unsolved:

| Split | Problems | Rollouts | pass@1 | pass@32 | Failure |
| --- | ---: | ---: | ---: | ---: | --- |
| dev | 1 | 1 | 0.0 | 0.0 | assertion failure |
| ood_test | 1 | 1 | 0.0 | 0.0 | assertion failure |

The stage manifest confirms that one `evaluate_splits` stage invoked
`evaluate-coding-style-vllm-splits`, rendered two prompts in one vLLM engine,
and returned code 0.

## Superseded Blocker Attempt

Before the verified rerun, `20260812Tbigcode-hard-shared-engine` stopped before
model execution because the GPU preflight found only `5.74 GiB` free on visible
GPU 0 versus `8.92 GiB` required at `GPU_MEMORY_UTILIZATION=0.05`.

Host `nvidia-smi` then showed all eight B200s occupied by unrelated
`sglang::scheduler` processes. The updated runner therefore still needs a clean
GPU rerun before the code change can be counted as end-to-end BigCodeBench-Hard
shell-run validation. The later verified run above cleared this blocker.

## Interpretation

The shared-engine change removes a real lifecycle hazard in the BigCodeBench
runner while preserving benchmark semantics:

- released BigCodeBench-Hard rows are still imported from the pinned source;
- canonical solutions still preflight through the executable verifier;
- model candidates are still checked by released tests;
- no LLM judge or alternate scoring path is introduced.

Next step: use the verified shared-engine runner for the larger preflight-clean
BigCodeBench-Hard slice, then investigate prompting or repair rather than vLLM
runner lifecycle.
