# Full Benchmark Matrix Run

Date: 2026-08-14
Run prefix: `20260814T071500Z-full-matrix`
Runtime doctor: `20260814T070000Z-full-matrix-runtime-doctor`
Host: 8x NVIDIA B200, TorchTitan bwrap rootfs, Qwen3-1.7B through vLLM for model smokes.

This report captures the fresh end-to-end execution of every currently local
`experiments/scaffold_to_policy` benchmark or harness entrypoint. It does not
claim that all 47 benchmarks in `agentic_eval_benchmarks.md` are implemented
locally. That document remains the adoption roadmap. This run covers the local
runnable surface: fixture reasoning, public static reasoning/coding smokes,
typed blockers, and external-harness compatibility probes.

## Runtime Gate

The full-matrix runtime doctor passed before the benchmark matrix:

- `selected=true`
- bwrap rootfs active for real execution
- rootfs Python imports `torch`, `vllm`, `datasets`, and `transformers`
- 8 GPUs visible
- artifact:
  `experiments/scaffold_to_policy/results/full_matrix_runtime_doctor/manifests/runtime_doctor_20260814T070000Z-full-matrix-runtime-doctor.json`

The matrix then ran the host and rootfs/vLLM preflight entrypoints:

- `preflight_host`: completed, host static profile ready
- `preflight_rootfs_vllm`: completed, `rootfs_cpu`, `vllm_1gpu`, and
  `reasoning` profiles ready

## Result Classes

The matrix has three result classes:

- Typed lifecycle results: attempt bundles with `outcome.json` plus derived
  report inputs. These are the strongest local execution records.
- Legacy report-only smokes: scripts that wrote report inputs and passed, but
  do not yet write typed `outcome.json`.
- Compatibility probes: external-harness install/import/scorer/oracle probes.
  These validate integration plumbing, not model capability on those upstream
  benchmarks.

## Typed Public And Transfer Results

All completed rows below are smoke-sized measurements. `pass@8` is shown only
where the report schema emitted it; for runs with fewer than 8 actual samples,
the value is a compatibility field and should not be promoted as a real
pass@8 claim.

| Runner | Outcome | Split | n | rollouts | pass@1 | pass@2 | pass@8 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `arithmetic_words` fixture | completed | dev | 8 | 24 | 1.000 | 1.000 | 1.000 |
| `arithmetic_words` fixture | completed | ood_test | 8 | 24 | 1.000 | 1.000 | 1.000 |
| `modular_sequences_transfer` base | completed | dev | 8 | 64 | 0.500 | 0.500 | 0.500 |
| `modular_sequences_transfer` base | completed | ood_test | 8 | 64 | 0.500 | 0.750 | 0.875 |
| `modular_sequences_transfer` raw adapter | completed | adapter_raw_dev | 8 | 64 | 0.625 | 0.625 | 0.625 |
| `modular_sequences_transfer` raw adapter | completed | adapter_raw_ood_test | 8 | 64 | 0.625 | 0.625 | 0.750 |
| `gsm8k_public` | completed | dev | 8 | 32 | 0.625 | 0.625 | 0.750 |
| `gsm8k_public` | completed | ood_test | 8 | 32 | 0.625 | 0.750 | 0.750 |
| `math_public` | completed | dev | 8 | 32 | 0.750 | 0.750 | 0.750 |
| `math_public` | completed | ood_test | 8 | 32 | 0.750 | 0.750 | 0.875 |
| `humaneval_public` | completed | dev | 4 | 16 | 0.750 | 0.750 | 0.750 |
| `humaneval_public` | completed | ood_test | 4 | 16 | 0.250 | 0.250 | 0.250 |
| `mbpp_public` | completed | dev | 4 | 16 | 1.000 | 1.000 | 1.000 |
| `mbpp_public` | completed | ood_test | 4 | 16 | 0.250 | 0.250 | 0.250 |
| `aime_public` | completed | dev | 8 | 32 | 0.125 | 0.125 | 0.125 |
| `aime_public` | completed | ood_test | 8 | 32 | 0.000 | 0.000 | 0.000 |
| `mmlu_pro_public` | completed | dev | 8 | 32 | 0.250 | 0.250 | 0.250 |
| `mmlu_pro_public` | completed | ood_test | 8 | 32 | 0.750 | 0.750 | 0.750 |
| `livecodebench_public` | completed | dev | 2 | 4 | 0.000 | 0.000 | 0.000 |
| `livecodebench_public` | completed | ood_test | 2 | 4 | 0.000 | 0.000 | 0.000 |
| `arc_agi2_public` | completed | dev | 2 | 4 | 0.000 | 0.500 | 0.500 |
| `arc_agi2_public` | completed | ood_test | 2 | 4 | 0.000 | 0.000 | 0.000 |
| `bigcodebench_hard_public` | completed | dev | 2 | 4 | 0.000 | 0.000 | 0.000 |
| `bigcodebench_hard_public` | completed | ood_test | 2 | 4 | 0.000 | 0.000 | 0.000 |

GPQA public also ran through the typed lifecycle and stopped cleanly as:

- outcome: `blocked`
- reason: gated GPQA import/no rootfs-visible Hugging Face authorization or
  raw cache
- artifact:
  `experiments/scaffold_to_policy/results/gpqa_public_vllm_smoke/runs/20260814T071500Z-full-matrix-gpqa_public/attempt-01/outcome.json`

## Legacy Report-Only Smokes

These entrypoints completed and wrote report inputs, but they are not yet on
the typed run-attempt lifecycle:

| Runner | Split | n | rollouts | pass@1 | pass@2 | pass@8 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `gsm_style` fixture | dev | 3 | 12 | 1.000 | 1.000 | 1.000 |
| `gsm_style` fixture | ood_test | 3 | 12 | 1.000 | 1.000 | 1.000 |
| `modular_sequences` fixture | dev | 8 | 24 | 1.000 | 1.000 | 1.000 |
| `modular_sequences` fixture | ood_test | 8 | 24 | 1.000 | 1.000 | 1.000 |
| `arithmetic_words_vllm` | dev | 4 | 16 | 1.000 | 1.000 | 1.000 |
| `arithmetic_words_vllm` | ood_test | 4 | 16 | 1.000 | 1.000 | 1.000 |
| `gsm_style_vllm` | dev | 3 | 12 | 1.000 | 1.000 | 1.000 |
| `gsm_style_vllm` | ood_test | 3 | 12 | 1.000 | 1.000 | 1.000 |
| `modular_sequences_vllm` | dev | 8 | 64 | 0.000 | 0.000 | 0.000 |
| `modular_sequences_vllm` | ood_test | 8 | 64 | 0.000 | 0.000 | 0.000 |

The modular-sequences vLLM zero is a useful plumbing-negative result: the
generation and verifier path ran, but Qwen3-1.7B did not satisfy the strict
`FINAL: <integer>` recurrence contract under this prompt/configuration.

## External Harness Probes

These completed as infrastructure probes:

- `external_harness_dry_run_smoke`: report-only dry-run compatibility path.
- `external_harness_preflight_smoke`: package/import/CLI checks for external
  harness integrations.
- `tau2_mock_score_smoke`: tau2 package and scorer path completed on a
  deterministic fixture trajectory.
- `tau2_execution_probe`: tau2 task execution probe completed.
- `terminal_bench_oracle_probe`: Harbor/Terminal-Bench oracle probe completed.

These rows do not count as Terminal-Bench or tau2 model-agent capability
results yet. They prove that the harnesses can be installed and driven from
the rootfs-compatible local workflow.

## Examples And Failure Modes

Representative observed outcomes:

- ARC-AGI-2 dev had one success out of four rollouts, two unparsable final-grid
  JSON failures, and one exact-grid mismatch. OOD had four unparsable final-grid
  JSON failures. This points first at output-contract control, not search depth.
- BigCodeBench-Hard dev and OOD each had four assertion failures. The runner,
  canonical preflight, and executable test path were healthy; the model samples
  failed the public tests.
- LiveCodeBench public dev failed with one `IndexError: list index out of range`
  and three indentation errors. OOD failed with two indentation errors and two
  wrong answers.
- Modular transfer raw adapter improved dev pass@1 from `0.500` to `0.625` and
  adapter-heldout pass@1 from `0.500` to `0.625` in this smoke. This is still a
  single smoke matrix, not a replicated transfer claim.

## Artifact Index

Primary matrix logs:

- `experiments/scaffold_to_policy/results/20260814T071500Z-full-matrix/matrix.log`
- `experiments/scaffold_to_policy/results/20260814T071500Z-full-matrix/vllm_matrix.log`

Typed attempt examples:

- `experiments/scaffold_to_policy/results/gsm8k_public_vllm_smoke/runs/20260814T071500Z-full-matrix-gsm8k_public/attempt-01/outcome.json`
- `experiments/scaffold_to_policy/results/math_public_vllm_smoke/runs/20260814T071500Z-full-matrix-math_public/attempt-01/outcome.json`
- `experiments/scaffold_to_policy/results/humaneval_public_vllm_smoke/runs/20260814T071500Z-full-matrix-humaneval_public/attempt-01/outcome.json`
- `experiments/scaffold_to_policy/results/arc_agi2_public_vllm_smoke/runs/20260814T071500Z-full-matrix-arc_agi2_public/attempt-01/outcome.json`
- `experiments/scaffold_to_policy/results/bigcodebench_hard_public_vllm_smoke/runs/20260814T071500Z-full-matrix-bigcodebench_hard_public/attempt-01/outcome.json`

Report inputs:

- `experiments/scaffold_to_policy/results/*/manifests/report_input_20260814T071500Z-full-matrix-*.json`
- `experiments/scaffold_to_policy/results/*/runs/20260814T071500Z-full-matrix*/attempt-01/derived/report_input.json`

## Conclusions

The local scaffold-to-policy runnable matrix is now exercised end to end under
the bwrap rootfs contract. The runtime doctor, host/rootfs preflights, typed
public reasoning/coding runners, typed GPQA blocker, and external harness
compatibility probes all completed without requiring runner code changes after
the matrix launch.

The strongest current local measurements are static reasoning/coding smokes
and the modular-sequences transfer smoke. The hardest coding and abstract
reasoning smokes are mostly hard negatives, which is expected at this scale and
is useful for calibrating prompt contracts and future training/eval loops.

Next implementation work:

1. Migrate remaining legacy report-only smokes to typed run-attempt lifecycle.
2. Add rootfs-visible GPQA credential/cache handling, then rerun only after the
   earlier GPQA answer-ordering bug remains repaired and permutation-tested.
3. Replace tau2 and Terminal-Bench oracle/fixture probes with bounded
   model-agent runs that preserve upstream scorers.
4. Add the next bounded tool-use and long-context runners from the benchmark
   roadmap without reporting them as covered until local entrypoints exist.
5. Replicate promising transfer results over independent draws and seeds before
   promoting any training claim.
