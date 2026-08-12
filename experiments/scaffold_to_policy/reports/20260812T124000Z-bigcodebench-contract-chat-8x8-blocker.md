# BigCodeBench-Hard Contract-Chat 8x8 Blocker Refresh

Run IDs:

- `20260812Tbigcode-hard-contract-chat-shared-engine-8x8`
- `20260812Tbigcode-hard-contract-chat-shared-engine-8x8-faker`
- `20260812Tbigcode-hard-contract-chat-shared-engine-8x8-faker-lowmem`

This is a coding-lane blocker and dependency-repair report, not a completed
BigCodeBench-Hard model result.

## Scope

The target slice was:

- `bigcode/bigcodebench-hard`, split `v0.1.4`
- dev offset `0`, `8` problems
- OOD offset `32`, `8` problems
- `PROMPT_VARIANT=contract_chat`
- `NUM_ROLLOUTS=4`
- shared-engine evaluator: `evaluate-coding-style-vllm-splits`
- rootfs execution through `run_bigcodebench_hard_public_vllm_smoke.sh`

## Dependency Repair

The first 8x8 run stopped before model execution because two OOD canonical
solutions failed with:

```text
ModuleNotFoundError: No module named 'faker'
```

Affected tasks:

- `BigCodeBench/287`
- `BigCodeBench/313`

`faker==37.5.3` was added to the BigCodeBench-Hard runner's rootfs dependency
bootstrap and installed into the current rootfs.

After that repair, `20260812Tbigcode-hard-contract-chat-shared-engine-8x8-faker-lowmem`
passed canonical preflight for both splits:

| Split | Passed | Problems |
| --- | ---: | ---: |
| dev | 8 | 8 |
| ood_test | 8 | 8 |

This clears the slice's canonical-solution dependency blocker.

## Remaining Blocker

The repaired 8x8 run did not produce model scores.

At `GPU_MEMORY_UTILIZATION=0.12`, the run reached vLLM initialization but failed
with negative available KV-cache memory:

```text
Available KV cache memory: -84.97 GiB
ValueError: No available memory for the cache blocks.
```

At `GPU_MEMORY_UTILIZATION=0.05`, the subsequent run stopped earlier at vLLM
GPU-memory preflight:

```text
free_gib=6.91864013671875
required_gib=8.917559813708067
selected=false
```

Host `nvidia-smi` showed all eight B200s occupied by unrelated
`sglang::scheduler` processes immediately after the low-memory blocker:

```text
GPU 0 free: 0.98 GiB
GPU 1-6 free: 3.53 GiB each
GPU 7 free: 5.57 GiB
```

## Interpretation

The 8x8 `contract_chat` path is now cleaner than before:

- the hard coding runner uses one vLLM engine for dev and OOD;
- the selected 8x8 offset-32 OOD slice passes canonical preflight once `faker`
  is available;
- the current failure is GPU availability / vLLM KV-cache initialization under
  shared-machine contention, not a released-test or import failure.

Next step: rerun the same command when one B200 is actually free, using
`GPU_MEMORY_UTILIZATION=0.05` first. Only a run that writes dev and OOD summary
artifacts should be treated as the larger `contract_chat` model result.
