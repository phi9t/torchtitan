# BigCodeBench-Hard Expanded Calibration

Run ID: `20260812T173000Z-bigcodebench-hard-public-vllm-expanded`

This expands the earlier two-task BigCodeBench-Hard smoke to an 8-task dev and
8-task OOD calibration slice. It is a rootfs-managed, no-tool executable-coding
calibration run for Qwen3-1.7B. It is not a BigCodeBench leaderboard result.

## Command

```bash
RUN_ID=20260812T173000Z-bigcodebench-hard-public-vllm-expanded \
DATA_ROOT=experiments/scaffold_to_policy/data/bigcodebench_hard_public_vllm_expanded_clean \
RESULTS_ROOT=experiments/scaffold_to_policy/results/bigcodebench_hard_public_vllm_expanded_clean \
DEV_PROBLEMS=8 OOD_PROBLEMS=8 DEV_OFFSET=0 OOD_OFFSET=72 \
NUM_ROLLOUTS=4 MAX_NEW_TOKENS=1024 TIMEOUT_SECONDS=60 \
experiments/scaffold_to_policy/run_bigcodebench_hard_public_vllm_smoke.sh
```

## Environment And Provenance

- Execution boundary: `scripts/rootfs/enter_rootfs.sh`.
- Model: `./assets/hf/Qwen3-1.7B`.
- Runtime: vLLM inside the bwrap rootfs.
- Dataset: `bigcode/bigcodebench-hard`, source split `v0.1.4`, revision `main`.
- Verifier: released BigCodeBench unit tests executed in an isolated Python
  subprocess. No LLM judge or replacement tests were used.
- Report input:
  `experiments/scaffold_to_policy/results/bigcodebench_hard_public_vllm_expanded_clean/manifests/report_input_20260812T173000Z-bigcodebench-hard-public-vllm-expanded.json`.

## Canonical Preflight

The expanded slice initially exposed rootfs package gaps and version-sensitive
canonical failures. The final selected slice required these additional rootfs
packages for released canonical solutions to run:

- `flask==3.1.3`
- `flask-login==0.6.3`
- `flask-wtf==1.3.0`
- `pycryptodome==3.23.0`
- `rsa==4.9.1`
- `seaborn==0.13.2`
- `wordcloud==1.9.6`

The runner now installs those packages before canonical preflight. The final
preflight result was clean:

| Split | Problems | Canonical pass | Failure breakdown |
| --- | ---: | ---: | --- |
| dev | 8 | 8/8 | success 8 |
| OOD | 8 | 8/8 | success 8 |

Rejected OOD offset `64` was intentionally not used because two released
canonical solutions failed under the current rootfs library versions:
`BigCodeBench/503` uses removed pandas `DataFrame.applymap`, and
`BigCodeBench/511` fails in a matplotlib/numpy dtype path. This is exactly why
canonical preflight remains a gate before model scoring.

## Results

| Split | Problems | Rollouts/problem | pass@1 | pass@2 | pass@4 | Buckets |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| dev | 8 | 4 | 0.000 | 0.000 | 0.000 | easy 0, elicitable 0, unreached 8 |
| OOD | 8 | 4 | 0.000 | 0.000 | 0.000 | easy 0, elicitable 0, unreached 8 |

Failure breakdown:

| Split | Rollouts | Assertion failures |
| --- | ---: | ---: |
| dev | 32 | 32 |
| OOD | 32 | 32 |

The result is environment-clean but model-negative: every canonical solution in
the selected slice passed, and every sampled model solution failed the released
unit tests.

## Examples

### Flask App Wiring Failure

Problem: `BigCodeBench/82`

The task provides a Flask/LoginManager skeleton and expects a complete login
application. The first model rollout created routes and initialized
`LoginManager`, but did not register the required `user_loader` and missed route
behavior expected by the tests.

```text
Failure: Missing user_loader or request_loader.
Test failures: login page returned 500; logout route did not expose the
expected logged-in page content.
```

This is a benchmark-semantic failure rather than a formatting failure: the code
is syntactically plausible, but it does not satisfy the released app-state
contract.

### Cryptography API Failure

Problem: `BigCodeBench/583`

The task imports `rsa`, `Crypto.Random`, `AES`, and base64 helpers. The first
model rollout generated keys and tried AES ECB encryption directly on a literal
message.

```text
Failure: ValueError: Data must be aligned to block boundary in ECB mode.
```

All six released tests errored because the implementation did not handle AES
padding, output files, or the exact artifact contract expected by the task.

## Interpretation

BigCodeBench-Hard is a useful harder coding benchmark for the scaffold-to-policy
ladder because it stresses library composition, side effects, and exact
executable semantics. The expanded run does not yet show model capability on
this slice, but it does advance the infrastructure:

- canonical-solution preflight catches missing packages and incompatible task
  slices before vLLM generation;
- generated code is scored only by released executable tests;
- task failures are now cleanly separated from rootfs/package failures;
- the selected 16-task slice is hard enough that the current prompt/model
  condition has no pass@4 successes.

The next coding step should improve the prompt/extraction strategy or add a
repair pass, then rerun the same preflight-clean slice before expanding further.
