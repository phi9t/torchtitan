# ARC-AGI-2 And BigCodeBench-Hard Smokes

Run IDs:

- `20260812T090000Z-arc-agi2-public-vllm-smoke`
- `20260812T091500Z-bigcodebench-hard-public-vllm-smoke`

These are rootfs-managed no-tool smokes for harder reasoning and coding lanes.
They are infrastructure and calibration evidence, not leaderboard-grade
benchmark claims.

## Environment

- All import, generation, scoring, and report-input steps ran through
  `scripts/rootfs/enter_rootfs.sh`.
- Model: `./assets/hf/Qwen3-1.7B`.
- Runtime: vLLM inside the bwrap rootfs.
- ARC source: `https://github.com/arcprize/ARC-AGI-2.git` at
  `f3283f727488ad98fe575ea6a5ac981e4a188e49`.
- BigCodeBench-Hard source: `bigcode/bigcodebench-hard`, split `v0.1.4`.
- BigCodeBench-Hard required one rootfs repair: install `matplotlib`. The first
  attempt exposed the missing package on an OOD task; the repaired run removed
  that environment failure and left only benchmark-test assertion failures.

## ARC-AGI-2 Exact-Grid Smoke

Command:

```bash
RUN_ID=20260812T090000Z-arc-agi2-public-vllm-smoke \
DATA_ROOT=experiments/scaffold_to_policy/data/arc_agi2_public_vllm_smoke_min \
RESULTS_ROOT=experiments/scaffold_to_policy/results/arc_agi2_public_vllm_smoke_min \
DEV_PROBLEMS=2 OOD_PROBLEMS=2 NUM_ROLLOUTS=2 \
experiments/scaffold_to_policy/run_arc_agi2_public_vllm_smoke.sh
```

Verifier:

- Prompt includes released train examples and one test input.
- Output contract is exactly `FINAL: <json-grid>`.
- Score is exact JSON grid equality.
- No partial credit, no symbolic inference, no LLM judge.

Results:

| Split | Problems | Rollouts/problem | pass@1 | pass@2 | Buckets |
| --- | ---: | ---: | ---: | ---: | --- |
| dev | 2 | 2 | 0.000 | 0.500 | easy 0, elicitable 1, unreached 1 |
| OOD | 2 | 2 | 0.000 | 0.000 | easy 0, elicitable 0, unreached 2 |

Failure breakdown:

- dev: 1 success, 1 wrong-grid failure, 2 JSON parse failures.
- OOD: 4 JSON parse failures.

Example that worked on the second rollout:

```text
Problem: ARC-AGI-2/00576224/0
Test input: [[3,2],[7,8]]
Expected: [[3,2,3,2,3,2],[7,8,7,8,7,8],[2,3,2,3,2,3],
           [8,7,8,7,8,7],[3,2,3,2,3,2],[7,8,7,8,7,8]]
Rollout 1: emitted a grid copied from training colors -> wrong grid.
Rollout 2: emitted the exact expected JSON grid -> success.
```

Representative format failure:

```text
Problem: ARC-AGI-2/007bbfb7/0
Failure: could not parse final grid JSON: Expecting value
Observed pattern: the model ended with `FINAL: <json-grid>` on one line and
placed the grid on the next line, so the verifier correctly rejected it.
```

Interpretation:

ARC-AGI-2 is a useful next hard reasoning lane because it stresses exact
structured output and latent transformation inference. The first signal is
mostly a formatting and abstraction failure, not a runtime failure.

## BigCodeBench-Hard Executable Smoke

Command:

```bash
RUN_ID=20260812T091500Z-bigcodebench-hard-public-vllm-smoke \
DATA_ROOT=experiments/scaffold_to_policy/data/bigcodebench_hard_public_vllm_smoke_min \
RESULTS_ROOT=experiments/scaffold_to_policy/results/bigcodebench_hard_public_vllm_smoke_repaired \
DEV_PROBLEMS=2 OOD_PROBLEMS=2 NUM_ROLLOUTS=2 \
experiments/scaffold_to_policy/run_bigcodebench_hard_public_vllm_smoke.sh
```

Verifier:

- Importer preserves `code_prompt`, `entry_point`, and released `unittest`
  suite.
- The suite is wrapped into `check(candidate)`.
- Candidate code runs in an isolated Python subprocess with timeout.
- No generic judge or synthetic replacement tests are used.

Results after rootfs dependency repair:

| Split | Problems | Rollouts/problem | pass@1 | pass@2 | Failure breakdown |
| --- | ---: | ---: | ---: | ---: | --- |
| dev | 2 | 2 | 0.000 | 0.000 | 4 assertion failures |
| OOD | 2 | 2 | 0.000 | 0.000 | 4 assertion failures |

Representative dev failure:

```text
Problem: BigCodeBench/13
Task: FTP file listing/downloading with mocked ftplib/subprocess behavior.
Model output: connected and logged in, then used os.listdir('/ftp/test').
Failure: benchmark tests expected FTP nlst(), subprocess wget calls, and
specific wrapped exception messages.
```

Representative OOD failure:

```text
Problem: BigCodeBench/267
Task: mutate a dictionary, generate a signal, compute FFT, and return both FFT
and a matplotlib axis with exact title/labels.
Model output: tried fftpack.fft(data) directly on a dict and returned only a
magnitude array.
Failure: benchmark unittest raised TypeError and assertion failures.
```

Interpretation:

BigCodeBench-Hard is a clear step up from MBPP. The harness now executes its
released tests, including tasks requiring scientific and plotting libraries.
The first repaired smoke is a hard negative for Qwen3-1.7B under this prompting
condition, but it validates the executable-test path.

## Next Improvements

- Add a strict-format ARC prompt variant that avoids the literal
  `FINAL: <json-grid>` copy failure and rerun the same task slices before
  expanding.
- Add dependency preflight for BigCodeBench imported tasks by scanning test
  imports before vLLM generation, so missing packages are repaired before
  scoring.
- Increase slices only after the smoke harnesses remain environment-clean for a
  repeated run.
- Keep Terminal-Bench/Harbor and tau2 full execution blocked until Docker or an
  equivalent Harbor backend and a local model/user provider path are available
  inside the hermetic rootfs.
