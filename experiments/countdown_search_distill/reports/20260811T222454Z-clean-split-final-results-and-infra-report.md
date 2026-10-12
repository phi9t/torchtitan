# Countdown Clean-Split Final Results And Infrastructure Report

Date: 2026-08-11

## Executive Summary

The Countdown scaffold-to-policy experiment now has a clean-split full
evaluation. All real setup, generation, training, export, validation, and
evaluation ran through the TorchTitan bwrap rootfs entrypoint:

```bash
scripts/rootfs/enter_rootfs.sh -- <command>
```

The key scientific result survives the split repair. On the clean OOD split,
the best arm, `clean`, improves Qwen3-1.7B from 2.4% to 18.0% pass@1 and from
48.4% to 89.0% pass@32. Every full arm improves both pass@1 and pass@32.

| Arm | OOD pass@1 | OOD pass@32 | pass@1 delta | pass@32 delta |
| --- | ---: | ---: | ---: | ---: |
| base | 0.024 | 0.484 | - | - |
| raw | 0.126 | 0.870 | +0.102 | +0.386 |
| clean | 0.180 | 0.890 | +0.156 | +0.406 |
| hindsight | 0.142 | 0.864 | +0.118 | +0.380 |
| curriculum | 0.142 | 0.878 | +0.118 | +0.394 |

The strongest claim is no longer limited to OOD-only evidence. The split
registry reports no problem-key overlap across train, dev, IID, and OOD:

```text
experiments/countdown_search_distill/data/split_registry.json
```

The clean full run completed all manifest stages with return code 0. The final
base and adapter evaluations are fresh clean-split evaluations. The full
training checkpoints and PEFT/vLLM adapter exports were reused from the earlier
full run, then evaluated against the repaired splits.

## Scientific Question

The experiment tests whether verified best-of-32 Countdown traces from a frozen
Qwen3-1.7B model can be compressed into better first-sample policy behavior by
TorchTitan LoRA SFT.

The intended evidence chain is:

1. Select a Countdown regime where base pass@1 is low and pass@32 is high enough
   to provide verified demonstrations.
2. Generate train/dev/IID/OOD problem splits without number-target key overlap.
3. Collect best-of-32 base rollouts and verify arithmetic exactly.
4. Train LoRA adapters from several scaffold-to-policy target constructions.
5. Export TorchTitan LoRA checkpoints to PEFT/vLLM adapter directories.
6. Evaluate base and adapters with the same verifier and pass@k metrics.
7. Compare pass@1 lift, pass@32 retention, bucket movement, and failure modes.

## Clean-Split Repair

The original pilot revealed a split hygiene problem: train, dev, and IID were
generated from the same seed/regime and overlapped by problem key. The repaired
path adds:

- `split_registry.py` for split manifests, problem-key hashes, overlap checks,
  and JSON output.
- `validate-splits` in the experiment CLI.
- `--exclude-problems` support during problem generation.
- `run_validate_splits.sh` as the rootfs-managed validation wrapper.
- reduced/full launcher validation after collection and before training.
- mode-scoped adapter eval roots to avoid stale reduced/full summary mixing.

The generation order now excludes prior split keys:

| Split | Excludes |
| --- | --- |
| train | none |
| dev | train |
| iid_test | train, dev |
| ood_test | train, dev, iid_test |

The final split registry reports:

| Check | Result |
| --- | --- |
| `selected` | true |
| `no_problem_key_overlap` | true |
| overlap count | 0 |

## Runtime And Hermeticity

The experiment used the bwrap rootfs for every real Python/GPU operation. The
rootfs contract mattered for three reasons:

- It isolated Python dependencies from host Python.
- It exposed the same CUDA/NVIDIA device path to TorchTitan and vLLM.
- It made failures reproducible through repo-local script entrypoints.

The repaired rootfs imported:

| Package | Import status |
| --- | --- |
| `torch` | true |
| `vllm` | true |
| `datasets` | true |
| `transformers` | true |
| `spmd_types` | true |

Qwen3-1.7B assets were present under:

```text
assets/hf/Qwen3-1.7B
```

vLLM handled frozen-base and LoRA adapter generation. TorchTitan handled LoRA
SFT. TorchTitan DCP checkpoints were not passed directly to vLLM; they were
exported first to PEFT/vLLM adapter directories:

```text
experiments/countdown_search_distill/results/adapters/full/{raw,clean,hindsight,curriculum}/
```

Each exported adapter has `adapter_config.json`, `adapter_model.safetensors`,
and `export_summary.json`. The exports report rank 32, alpha 64, and target
modules `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`,
`down_proj`, and `lm_head`.

## Run Manifest

The completed clean full run wrote:

```text
experiments/countdown_search_distill/results/manifests/current.jsonl
```

Tail-stage status:

| Stage | Return code | Duration |
| --- | ---: | ---: |
| `validate_splits` | 0 | 1s |
| `train_raw` | 0 | 28s |
| `train_clean` | 0 | 28s |
| `train_hindsight` | 0 | 28s |
| `train_curriculum` | 0 | 28s |
| `base_eval_dev` | 0 | 110s |
| `base_eval_iid_test` | 0 | 192s |
| `base_eval_ood_test` | 0 | 109s |
| `export_adapters` | 0 | 0s |
| `eval_adapters` | 0 | 2742s |

Important interpretation: the short full train stages loaded existing full
checkpoints and completed; adapter export skipped existing valid exports. The
fresh evidence in this run is the clean split validation plus base and
mode-scoped adapter evaluations.

## Full Metrics

Base Qwen3-1.7B:

| Split | Problems | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 | easy | elicitable | unreached |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dev | 500 | 0.026 | 0.046 | 0.084 | 0.150 | 0.284 | 0.468 | 13 | 221 | 266 |
| iid_test | 1000 | 0.033 | 0.059 | 0.099 | 0.191 | 0.310 | 0.509 | 33 | 476 | 491 |
| ood_test | 500 | 0.024 | 0.052 | 0.092 | 0.194 | 0.308 | 0.484 | 12 | 230 | 258 |

Full adapter metrics:

| Split | Arm | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 | easy | elicitable | unreached |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dev | raw | 0.162 | 0.310 | 0.466 | 0.658 | 0.816 | 0.876 | 81 | 357 | 62 |
| dev | clean | 0.180 | 0.344 | 0.522 | 0.694 | 0.836 | 0.904 | 90 | 362 | 48 |
| dev | hindsight | 0.162 | 0.268 | 0.424 | 0.630 | 0.794 | 0.880 | 81 | 359 | 60 |
| dev | curriculum | 0.182 | 0.310 | 0.492 | 0.674 | 0.812 | 0.890 | 91 | 354 | 55 |
| iid_test | raw | 0.169 | 0.317 | 0.506 | 0.699 | 0.840 | 0.915 | 169 | 746 | 85 |
| iid_test | clean | 0.208 | 0.375 | 0.574 | 0.739 | 0.858 | 0.922 | 208 | 714 | 78 |
| iid_test | hindsight | 0.152 | 0.263 | 0.426 | 0.619 | 0.796 | 0.890 | 152 | 738 | 110 |
| iid_test | curriculum | 0.163 | 0.296 | 0.503 | 0.699 | 0.837 | 0.921 | 163 | 758 | 79 |
| ood_test | raw | 0.126 | 0.224 | 0.404 | 0.624 | 0.770 | 0.870 | 63 | 372 | 65 |
| ood_test | clean | 0.180 | 0.316 | 0.470 | 0.670 | 0.816 | 0.890 | 90 | 355 | 55 |
| ood_test | hindsight | 0.142 | 0.242 | 0.414 | 0.578 | 0.764 | 0.864 | 71 | 361 | 68 |
| ood_test | curriculum | 0.142 | 0.260 | 0.384 | 0.596 | 0.762 | 0.878 | 71 | 368 | 61 |

Adapter deltas versus base:

| Split | Arm | delta pass@1 | delta pass@32 | delta elicitable |
| --- | --- | ---: | ---: | ---: |
| dev | raw | +0.136 | +0.408 | +136 |
| dev | clean | +0.154 | +0.436 | +141 |
| dev | hindsight | +0.136 | +0.412 | +138 |
| dev | curriculum | +0.156 | +0.422 | +133 |
| iid_test | raw | +0.136 | +0.406 | +270 |
| iid_test | clean | +0.175 | +0.413 | +238 |
| iid_test | hindsight | +0.119 | +0.381 | +262 |
| iid_test | curriculum | +0.130 | +0.412 | +282 |
| ood_test | raw | +0.102 | +0.386 | +142 |
| ood_test | clean | +0.156 | +0.406 | +125 |
| ood_test | hindsight | +0.118 | +0.380 | +131 |
| ood_test | curriculum | +0.118 | +0.394 | +138 |

## Interpretation

The clean-split run supports the scaffold-to-policy hypothesis. The adapters do
not merely trade pass@32 for pass@1; they improve both. That means SFT is moving
probability mass into valid arithmetic traces while also making more problems
reachable within 32 samples.

The `clean` arm is the best overall full arm. It is best on IID and OOD pass@1,
best on all three pass@32 metrics, and strongest on OOD, the split that most
directly checks target-range generalization. `curriculum` is narrowly highest
on dev pass@1, but does not dominate elsewhere.

The reduced pilot was useful as a gate and integration check, but its smaller
split sizes produce noisier arm ordering. It correctly predicted that the full
run was worth launching: reduced arms improved pass@1 substantially while
preserving or improving pass@32.

## Examples

### Base Miss, Same Problem Later Solved By Adapter

Problem:

```text
numbers: 11, 90, 43
target: 36
canonical solution:
90 - 43 = 47
47 - 11 = 36
FINAL: 36
```

Base sample 0 failed with `missing FINAL line`. The model searched locally but
did not emit a verifier-complete solution:

```text
To solve the problem, you need to find a sequence of operations that combines
the numbers 11, 90, and 43 using addition, subtraction, multiplication, and
division (with exact division) to reach the target of 36.

Let's see what combinations we can make.
First, let's try combining 11 and 90. 11 + 90 = 101, which is not helpful.
...
```

Full `clean` adapter sample 1 solved the same problem:

```text
11 + 43 = 54
90 - 54 = 36
Final
```

Verifier state:

```text
[11, 43, 90] -> [54, 90] -> [36]
success: true
```

The adapter's final text says `Final` rather than `FINAL: 36`, but the verifier
accepted the arithmetic trace and final value from the parsed operations. Before
treating this as benchmark-grade, the output contract should be tightened so
`FINAL: <target>` is required consistently.

### Clean OOD Success

OOD problem:

```text
numbers: 49, 35, 82
target: 68
canonical solution:
82 - 49 = 33
33 + 35 = 68
FINAL: 68
```

Full `clean` adapter sample 0:

```text
82 - 49 = 33
33 + 35 = 68
FINAL: 68
```

Verifier state:

```text
[35, 49, 82] -> [33, 35] -> [68]
success: true
```

### Remaining Failure Mode

On an OOD unreached problem:

```text
numbers: 1, 98, 21
target: 77
```

One full `clean` rollout reached a valid trace prefix but wrong final value:

```text
1 + 98 = 99
99 + 21 = 120
```

Verifier result:

```text
final value 120 does not match target 77
```

Other failures still cluster around missing `FINAL` lines, unavailable operands,
wrong final values, and occasional invalid intermediate arithmetic. The largest
product gap is now output-contract discipline plus search depth/generalization,
not environment setup.

## Execution Trace Review

What worked well:

- The bwrap rootfs made dependency repair and GPU execution reproducible once
  `vllm`, `datasets`, `transformers`, and `spmd_types` were installed inside it.
- Running smoke first caught rootfs and import issues before committing the
  eight-GPU path.
- Reduced mode correctly acted as a gate before full evaluation.
- The split registry caught a real validity problem and made the repaired
  result auditable.
- Mode-scoped adapter evaluation roots prevented stale reduced/full artifacts
  from contaminating matrix validation.
- Exporting TorchTitan LoRA checkpoints to PEFT/vLLM directories preserved the
  correct boundary between training checkpoints and inference adapters.

What needs improvement:

- Manifests should be timestamped by `RUN_ID`; writing only `current.jsonl`
  makes later audit harder.
- Training and export stages should report whether they performed fresh work or
  reused existing artifacts.
- The verifier/reporting contract should separate strict `FINAL: <target>`
  compliance from arithmetic-trace success.
- Top-level eval paths should also be mode/run scoped, as adapter evals now are.
- Runtime preflight should be the first stage in every mode and should record
  package versions, CUDA device names, GPU count, rootfs marker variables, and
  vLLM backend settings.
- Split registry files should include compact split summaries in addition to
  full key lists, so reports do not need to inspect large JSON blobs.

## Infrastructure Assessment

The current infra is good enough for local experimental iteration on this host:

- bwrap rootfs is the right isolation boundary for TorchTitan plus vLLM.
- TorchTitan can train Qwen3 LoRA adapters on the local 8x B200 setup.
- vLLM can evaluate base and PEFT/vLLM LoRA adapters under the same rootfs.
- The verifier produces useful pass@k, bucket, validity, and trajectory
  artifacts.
- The experiment can resume expensive stages, but resumability needs clearer
  provenance fields.

The current infra is not yet benchmark-harness grade:

- It needs immutable run directories and pinned manifests for every evaluation
  object.
- It needs explicit artifact provenance: split hash, dataset hash, checkpoint
  hash, adapter hash, evaluator commit, rootfs build id, and command env.
- It needs a strict output-format metric alongside arithmetic success.
- It needs report generation from a single structured result file, not manual
  extraction across summaries.

## Follow-Up Experiments

Reasoning-first next steps:

1. Run a matched base-elicitable analysis: restrict comparisons to problems
   where base pass@32 succeeds and measure conversion to pass@1.
2. Add stricter format SFT/eval variants that require `FINAL: <target>` exactly.
3. Sweep train size and LoRA rank for `clean`, since it is the best full arm.
4. Add harder Countdown regimes: more numbers, wider targets, deeper minimum
   solution depth, and OOD number-range shifts.
5. Add adjacent static reasoning benchmarks before agentic environments:
   arithmetic/programmatic puzzles, AIME-style exact-answer math, GPQA-style
   expert QA, ARC-AGI-style structured output, and long-context reasoning.

Agentic expansion should come after the reasoning ladder has a stable harness:

- Use terminal-bench/Harbor for hermetic terminal-state tasks.
- Add tau2-bench or tau-bench lineage for stateful policy-governed tool use.
- Preserve upstream benchmark definitions, metrics, tool access, and version
  pins; do not turn no-tool or GUI-only benchmarks into broader harness runs.

The immediate engineering task before expanding benchmark scope is to turn the
Countdown run contract into a reusable eval registry: pinned splits, pinned
evaluator, pinned rootfs, run-scoped artifacts, and provenance-aware reports.
