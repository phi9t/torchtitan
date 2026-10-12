# Countdown Experiment Completion Audit

Date: 2026-08-11

## Objective

Design and implement the Countdown scaffold-to-policy experiments, ensure the
design is scientifically meaningful, carry the experiments end to end under the
hermetic bwrap rootfs, and improve the process so future runs are preflighted,
resumable, and auditable.

## Prompt-To-Artifact Checklist

| Requirement | Evidence | Status |
| --- | --- | --- |
| Use the bwrap rootfs for all real Python/GPU work | `run_common.sh` re-execs through `scripts/rootfs/enter_rootfs.sh -- ...`; final validation invoked every command via `scripts/rootfs/enter_rootfs.sh -- bash -lc ...`; `results/runtime_preflight.json` has `rootfs_active=true` | complete |
| Repair and verify runtime dependencies | `results/runtime_preflight.json` has `torch`, `vllm`, `datasets`, `transformers`, and `spmd_types` all true | complete |
| Verify model assets | `results/runtime_preflight.json` has Qwen3 config, tokenizer files, and safetensors checks all true | complete |
| Preserve scientific smoke -> reduced -> full progression | `run_full_pilot.sh` runs preflight, calibration sweep, calibration, collection, preflight gate, training, base eval, adapter export, and adapter eval; reports record smoke, reduced, and full outcomes | complete |
| Gate non-smoke runs scientifically | `preflight-reduced` checks calibration selection plus train/dev matched coverage; `data/reduced_preflight.json` and `data/full_preflight.json` record gate decisions | complete |
| Train all intended full arms | Full checkpoints exist under `results/train/full/{raw,clean,hindsight,curriculum}/checkpoint/step-94/`; adapter eval report records all four arms | complete |
| Do not pass TorchTitan checkpoints directly to vLLM | `run_export_adapters.sh` exports to `results/adapters/{mode}/{arm}`; `run_eval_adapters.sh` uses those exported adapter directories | complete |
| Export adapters in vLLM/PEFT format | `results/adapters/full/*/{adapter_config.json,adapter_model.safetensors,export_summary.json}` exist; export summaries report rank 32 and alpha 64 | complete |
| Evaluate full adapter matrix | `results/eval/{dev,iid_test,ood_test}/{raw,clean,hindsight,curriculum}/summary.json` exist and are validated by `results/eval/adapter_matrix_full.json` | complete |
| Record scientific results | `RESULTS.md` and `reports/20260811T120736Z-rootfs-full-adapter-eval.md` include base and adapter metrics with interpretation | complete |
| Review and improve process | `reports/20260811T120736Z-agentic-execution-trace-review.md` records what worked, what failed, and follow-up process changes | complete |
| Encode process improvements | `run_preflight.sh`, `run_export_adapters.sh`, `run_eval_adapters.sh`, `validate-eval-matrix`, and stage manifest logging are implemented and documented | complete |
| Verify implementation | Final rootfs validation passed: shell syntax, 30 unit tests, runtime preflight, matrix validation, export/eval skip-mode wrappers, and manifest smoke | complete |

## Final Validation Command

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'set -euo pipefail; bash -n experiments/countdown_search_distill/run_*.sh; python -m pytest tests/unit_tests/test_countdown_search_distill.py -q; experiments/countdown_search_distill/run_preflight.sh; python -m torchtitan.experiments.countdown_search_distill.cli validate-eval-matrix --eval-root experiments/countdown_search_distill/results/eval --decision experiments/countdown_search_distill/results/eval/adapter_matrix_full.json --splits dev iid_test ood_test --arms raw clean hindsight curriculum --expected-problems dev=500 iid_test=1000 ood_test=500 --num-rollouts 32; MODE=full experiments/countdown_search_distill/run_export_adapters.sh; MODE=full experiments/countdown_search_distill/run_eval_adapters.sh; source experiments/countdown_search_distill/run_common.sh; countdown_setup_env; tmp=$(mktemp); TORCHTITAN_COUNTDOWN_MANIFEST=${tmp} countdown_run_stage manifest_smoke true; MANIFEST_PATH=${tmp} python - <<PY
import json, os
from pathlib import Path
row = json.loads(Path(os.environ["MANIFEST_PATH"]).read_text())
assert row["stage"] == "manifest_smoke"
assert row["return_code"] == 0
assert row["command"] == ["true"]
PY'
```

Result: `30 passed, 14 warnings`; all shell, preflight, matrix, wrapper, and
manifest checks completed with return code 0.

## Remaining Risks

The current results are from one deterministic problem-generation/evaluation
configuration and one full training run per arm. The next scientific extension
would be seed replication and a solved/unsolved failure analysis, but those are
new experiments rather than missing requirements for this completed end-to-end
pilot.
