# Scaffold-To-Policy Completion Audit Refresh

Audit timestamp: `2026-08-13T00:25:00Z`

Objective audited: implement `experiments/scaffold_to_policy/spec.md` to the
end and ensure the experiments are all carried out under the hermetic bwrap
rootfs discipline.

## Concrete Completion Criteria

The objective is complete only if all of the following are true:

- the scaffold-to-policy registry/reporting contract exists and is exercised by
  rootfs-managed run families;
- Countdown follow-up evidence includes the champion arm, formatting arm,
  replication, and rank/size sweep called for in the spec;
- local exact-verifier reasoning runs precede public benchmark expansion;
- public no-tool reasoning and coding lanes have reportable artifacts that
  preserve benchmark semantics and do not use generic judges;
- Terminal-Bench/Harbor and tau2-bench enter as external-harness smokes with
  upstream scoring boundaries preserved;
- GPQA Diamond either runs through an authenticated or public benchmark-preserving
  source path, or is explicitly blocked by a current rootfs artifact proving no
  benchmark-preserving access exists;
- generated data, checkpoints, caches, rollouts, and result trees remain out of
  git;
- rootfs validation and shell syntax checks cover the current runner/source
  changes.

## Prompt-To-Artifact Checklist

| Requirement | Evidence inspected | Status |
| --- | --- | --- |
| Use bwrap rootfs for real Python, GPU generation/evaluation, and external harness work | `run_common.sh` rootfs re-entry is used by hard-lane shell runners; refreshed GPQA, MMLU-Pro, and BigCodeBench-Hard reports record `runtime.rootfs.active=true`; Harbor/tau2 report inputs record `all_rootfs_selected=true` | Covered for current executed lanes |
| Run-scoped manifests and report inputs | Fresh hard-lane manifests under `mmlu_pro_public_vllm_16x16_runtime`, `bigcodebench_hard_contract_chat_8x8_runtime`, and `gpqa_simple_evals_16x16_labeled` record stage command, rootfs state, timing, return code, roots, and `work_status=executed`; report inputs are run-scoped | Covered for hard lanes |
| Split registries and no-overlap checks | `experiments/scaffold_to_policy/data/mmlu_pro_public_vllm_16x16_runtime/split_registry.json` has 16 dev, 16 OOD, `selected=true`, and no overlaps; `experiments/scaffold_to_policy/data/gpqa_simple_evals_16x16_labeled/split_registry.json` has 16 dev, 16 OOD, `selected=true`, and no overlaps; `experiments/scaffold_to_policy/data/bigcodebench_hard_contract_chat_8x8_runtime/split_registry.json` has 8 dev, 8 OOD, `selected=true`, and no overlaps | Covered for latest reasoning/coding lanes |
| Artifact provenance, runtime metadata, and blocker reporting | Latest hard-lane report inputs have `artifact_provenance_labeled=true` and `runtime_metadata_present=true`; old live-HF GPQA blocker artifacts remain useful for the gated loader path, but the public simple-evals GPQA run now completed benchmark task execution | Covered |
| Countdown champion, formatting arm, replication, and rank/size sweep | `spec.md` lists the Countdown reports: clean split final, formatting arm, base-elicitable examples, replication/rank-size sweep, and formatting replication | Covered for checkpoint |
| Local exact-verifier reasoning before public benchmarks | Reports exist for arithmetic words, modular sequences, modular transfer, GSM-style local fixtures, and vLLM smokes under `experiments/scaffold_to_policy/reports/` | Covered |
| Public no-tool reasoning expansion | GSM8K, MATH, AIME, ARC-AGI-2, MMLU-Pro, and GPQA Diamond public-cache reports exist; latest MMLU-Pro run `20260812T230000Z-mmlu-pro-16x16-runtime` and GPQA run `20260813T003000Z-gpqa-simple-evals-16x16-labeled` preserve exact final-letter scoring and record runtime metadata | Covered as calibration, not leaderboard claims |
| GPQA Diamond lane | Live Hugging Face access remains gated without `HF_TOKEN`, but the OpenAI simple-evals GPQA Diamond CSV was cached through rootfs and the existing GPQA runner completed 16 dev / 16 OOD no-tool vLLM evaluation with offsets 0 and 64 | Covered through public simple-evals cache path |
| Public coding expansion | HumanEval, MBPP, LiveCodeBench, and BigCodeBench-Hard reports exist; latest BigCodeBench-Hard run `20260812T231500Z-bigcodebench-hard-8x8-runtime` passed canonical preflight, completed model evaluation, and records runtime metadata | Covered as executable/public-test smokes and hard negatives |
| Terminal-Bench/Harbor harness smoke | `experiments/scaffold_to_policy/results/terminal_bench_harbor_qwen_headless_official_verifier/manifests/report_input_20260812Tqwen-headless-terminal-official-verifier.json` records installed preflight, rootfs selection, one completed official verifier trial, and unsuccessful task reward | Covered as model-policy execution plumbing, not solved task |
| tau2-bench harness smoke | `experiments/scaffold_to_policy/results/tau2_qwen_model_policy/manifests/report_input_20260812Ttau2-qwen-model-policy.json` records installed preflight, rootfs selection, one completed upstream mock-domain simulation, and unsuccessful task reward | Covered as model-policy execution plumbing, not solved task |
| Preserve upstream semantics and avoid generic judges | Reports label no-tool, public-test-only, oracle/scripted/model-policy, and blocker conditions; exact verifiers or upstream harness scorers are used | Covered |
| Do not commit generated data/results/checkpoints/caches | Current git status shows only pre-existing `.claude/CLAUDE.md` modified; generated `experiments/scaffold_to_policy/data/` and `results/` are ignored | Covered |
| Current validation | `bash -n` on GPQA/MMLU-Pro/BigCodeBench-Hard runners passed before the GPQA cache patch; rootfs unit test `python -m pytest tests/unit_tests/test_scaffold_to_policy.py -q` passed with 86 tests and 14 warnings; metadata-backed MMLU-Pro, BigCodeBench-Hard, and GPQA public-cache runs completed through rootfs | Covered before this audit doc edit; rerun required after committing this doc |

## Fresh GPQA Evidence

The live Hugging Face GPQA path is still gated without credentials. Earlier
credential/cache inspection found:

```text
host HF_TOKEN=
repo token file missing
rootfs HF_TOKEN=
rootfs repo token file missing
HF_HOME token file missing at /token
```

That explains the old blocker artifacts, but a benchmark-preserving public CSV
source is now available through OpenAI simple-evals:

```text
https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv
```

The rootfs cache command wrote 198 rows to:

```text
experiments/scaffold_to_policy/data/gpqa_simple_evals/raw/gpqa_diamond.jsonl
```

Cache hash:

```text
312b80837f84194c3d18675a3f4cd3cabea6d767ef2d1cebacb62d8b0df49d7e
```

Completed metadata-backed command:

```bash
RUN_ID=20260813T003000Z-gpqa-simple-evals-16x16-labeled \
DATA_ROOT=experiments/scaffold_to_policy/data/gpqa_simple_evals_16x16_labeled \
RESULTS_ROOT=experiments/scaffold_to_policy/results/gpqa_simple_evals_16x16_labeled \
DATASET=openai/simple-evals-gpqa \
DATASET_SUBSET=gpqa_diamond \
DATASET_REVISION=main \
SOURCE_SPLIT=gpqa_diamond \
OFFLINE=1 \
DEV_RAW_CACHE=experiments/scaffold_to_policy/data/gpqa_simple_evals/raw/gpqa_diamond.jsonl \
OOD_RAW_CACHE=experiments/scaffold_to_policy/data/gpqa_simple_evals/raw/gpqa_diamond.jsonl \
DEV_PROBLEMS=16 OOD_PROBLEMS=16 DEV_OFFSET=0 OOD_OFFSET=64 \
NUM_ROLLOUTS=4 MAX_NEW_TOKENS=512 GPU_MEMORY_UTILIZATION=0.05 \
experiments/scaffold_to_policy/run_gpqa_public_vllm_smoke.sh
```

The older access-only preflight command remains the diagnostic for the gated
Hugging Face path:

```bash
RUN_ID=20260812T234000Z-gpqa-access-preflight \
DATA_ROOT=experiments/scaffold_to_policy/data/gpqa_access_preflight_current \
RESULTS_ROOT=experiments/scaffold_to_policy/results/gpqa_access_preflight_current \
DEV_PROBLEMS=1 OOD_PROBLEMS=1 \
experiments/scaffold_to_policy/run_gpqa_access_preflight.sh
```

Observed rootfs error on the live-HF path:

```text
datasets.exceptions.DatasetNotFoundError: Dataset 'Idavidrein/gpqa' is a gated
dataset on the Hub. You must be authenticated to access it.
```

The completed public-cache runner wrote:

```text
experiments/scaffold_to_policy/results/gpqa_simple_evals_16x16_labeled/manifests/report_input_20260813T003000Z-gpqa-simple-evals-16x16-labeled.json
experiments/scaffold_to_policy/reports/20260813T003000Z-gpqa-simple-evals-hard-calibration.md
```

The selected split registry records:

```json
{
  "selected": true,
  "overlaps": [],
  "task": "multiple_choice",
  "verifier": "multiple_choice_final_letter_v1"
}
```

The registry contains 16 dev problems and 16 OOD problems.

The latest GPQA report input records:

```json
{
  "checks": {
    "artifact_provenance_labeled": true,
    "split_registry_selected": true,
    "runtime_metadata_present": true
  },
  "run": {
    "lane": "reasoning",
    "scaffold": {
      "type": "no_tool_sampling",
      "budget": 4
    },
    "task": "multiple_choice"
  }
}
```

It also records `runtime.rootfs.active=true`, CUDA availability with eight B200
devices visible, package versions for `torch`, `vllm`, `datasets`,
`transformers`, and `spmd_types`, the local Qwen3-1.7B asset path, and vLLM
backend settings.

The command-stage manifest records the explicit stage work-status field and
completed both model-evaluation stages:

```json
[
  {
    "stage": "evaluate_dev",
    "return_code": 0,
    "rootfs_active": true,
    "work_status": "executed"
  },
  {
    "stage": "evaluate_ood_test",
    "return_code": 0,
    "rootfs_active": true,
    "work_status": "executed"
  }
]
```

Key GPQA public-cache metrics:

| Split | Problems | Rollouts | pass@1 | pass@2 | pass@4 | pass@32 | Buckets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| dev | 16 | 64 | 0.4375 | 0.5625 | 0.5625 | 0.5625 | easy 7, elicitable 2, unreached 7 |
| OOD | 16 | 64 | 0.3125 | 0.3750 | 0.4375 | 0.4375 | easy 5, elicitable 2, unreached 9 |

## Latest Hard-Lane Evidence

The latest MMLU-Pro 16/16 calibration report input:

```text
experiments/scaffold_to_policy/results/mmlu_pro_public_vllm_16x16_runtime/manifests/report_input_20260812T230000Z-mmlu-pro-16x16-runtime.json
```

Key metrics:

| Split | Problems | Rollouts | pass@1 | pass@2 | pass@4 | pass@32 | Buckets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| dev | 16 | 64 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | easy 8, elicitable 0, unreached 8 |
| OOD | 16 | 64 | 0.5625 | 0.6250 | 0.6875 | 0.6875 | easy 9, elicitable 2, unreached 5 |

The latest BigCodeBench-Hard `contract_chat` report input:

```text
experiments/scaffold_to_policy/results/bigcodebench_hard_contract_chat_8x8_runtime/manifests/report_input_20260812T231500Z-bigcodebench-hard-8x8-runtime.json
```

Key metrics:

| Split | Problems | Rollouts | pass@1 | pass@8 | pass@32 | Buckets |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| dev | 8 | 64 | 0.000 | 0.000 | 0.000 | easy 0, elicitable 0, unreached 8 |
| OOD | 8 | 64 | 0.000 | 0.000 | 0.000 | easy 0, elicitable 0, unreached 8 |

Canonical preflight passed for all 16 BigCodeBench-Hard selected tasks before
model scoring.

Both latest hard-lane report inputs record `runtime_metadata_present=true`,
`runtime.rootfs.active=true`, CUDA device count `8`, package versions, model
asset path, and vLLM backend settings. The companion report is:

```text
experiments/scaffold_to_policy/reports/20260812T232000Z-hard-runtime-metadata-refresh.md
```

## Audit Result

The objective is materially complete for the benchmark ladder requested in this
spec: Countdown follow-ups, local reasoning, public hard reasoning, hard coding,
Terminal-Bench/Harbor, and tau2 all have rootfs-managed artifacts, report
inputs, or explicit upstream-harness outcomes. The previous GPQA blocker is now
closed for the public simple-evals GPQA Diamond source path.

Residual caveats:

- The GPQA live Hugging Face loader path is still gated without `HF_TOKEN`;
  future official comparisons should state whether they use the HF loader or the
  OpenAI simple-evals CSV cache.
- Public reasoning/coding numbers here are calibration slices, not official
  benchmark submissions.
- The GPQA runner still initializes one vLLM engine per split, which is wasteful
  but not semantically wrong.

The helper for the live Hugging Face access path remains:

```bash
experiments/scaffold_to_policy/setup_gpqa_access_wizard.sh
```
