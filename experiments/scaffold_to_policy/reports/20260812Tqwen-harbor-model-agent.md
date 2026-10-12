# Qwen Harbor Model-Agent Probe

Run ID: `20260812Tqwen-headless-terminal-official-verifier`

This report records the first Qwen3/vLLM custom-agent execution through
Terminal-Bench / Harbor for the pinned `headless-terminal` task. Harbor owns the
Docker environment, task lifecycle, and official verifier. The custom agent only
generates and writes `/app/headless_terminal.py`.

This is a model-policy execution artifact, not a successful Terminal-Bench
capability result.

## Command

```bash
RUN_ID=20260812Tqwen-headless-terminal-official-verifier \
RESULTS_ROOT=experiments/scaffold_to_policy/results/terminal_bench_harbor_qwen_headless_official_verifier \
HARBOR_AGENT=torchtitan.experiments.scaffold_to_policy.harbor_headless_terminal_agent:HeadlessTerminalQwenAgent \
TIMEOUT_SECONDS=1200 \
SCAFFOLD_TO_POLICY_HARBOR_GPU_MEMORY_UTILIZATION=0.05 \
experiments/scaffold_to_policy/run_terminal_bench_oracle_probe.sh
```

The agent runs in Harbor's installed-agent process, but the Harbor venv is kept
small and does not install model packages. For generation, the agent spawns
rootfs `/usr/bin/python`, where `vllm`, `transformers`, and `torch` are already
available.

Generation settings:

| Field | Value |
| --- | --- |
| Model | `./assets/hf/Qwen3-1.7B` |
| Python | rootfs `/usr/bin/python` |
| vLLM attention backend | `TRITON_ATTN` |
| FlashInfer sampler | `VLLM_USE_FLASHINFER_SAMPLER=0` |
| GPU memory utilization | `0.05` |
| Max model length | `2048` |
| Max generated tokens | `768` |
| Temperature | `0.2` |

## Result

Harbor's own job result:

| Field | Value |
| --- | ---: |
| `n_total_trials` | 1 |
| `stats.n_completed_trials` | 1 |
| `stats.n_errored_trials` | 0 |
| `stats.evals.torchtitan-headless-terminal-qwen__adhoc.n_trials` | 1 |
| `stats.evals.torchtitan-headless-terminal-qwen__adhoc.n_errors` | 0 |
| `stats.evals.torchtitan-headless-terminal-qwen__adhoc.metrics[0].mean` | 0.0 |
| verifier reward | 0.0 |

The scaffold report input records:

```text
task_execution_probes_completed=true
task_execution_probes_succeeded=false
all_rootfs_selected=true
```

This means the model agent reached official Harbor verifier execution and was
scored as a failed task, rather than failing at the wrapper or Docker boundary.

## Failure Mode

The generated `headless_terminal.py` imported successfully, but the released
tests failed six of seven checks. The first failure was:

```text
TypeError: HeadlessTerminal.__init__() missing 1 required positional argument: 'session_id'
```

The model-generated implementation required an explicit constructor argument,
while the benchmark tests instantiate `HeadlessTerminal()` with no arguments.
The verifier then failed non-interactive command handling, interactive Vim
handling, Ctrl-C handling, startup-file behavior, persistent shell state, and
background command behavior.

The trial result has `exception_info=null` and `verifier_result.rewards.reward=0.0`,
so this is an official verifier failure, not an infrastructure exception.

## Boundary Fixes Made

Several earlier Qwen Harbor attempts produced useful blocker evidence before
the final verifier-owned result:

| Run | Failure | Fix |
| --- | --- | --- |
| `20260812Tqwen-headless-terminal-smoke` | Harbor venv lacked `vllm` | Spawn rootfs `/usr/bin/python` for generation |
| `20260812Tqwen-headless-terminal-rootfs-python` | FlashInfer CUDA header/compiler mismatch | Use `attention_backend=TRITON_ATTN` and `VLLM_USE_FLASHINFER_SAMPLER=0` |
| `20260812Tqwen-headless-terminal-triton` | vLLM logs interleaved with JSON on stdout | Write generation result to a temp JSON file |
| `20260812Tqwen-headless-terminal-verifier` | Multiline generated code broke the container write command | Transfer generated code with base64 |

The final agent preserves benchmark semantics: it does not read Harbor's oracle
solution, does not use the verifier as feedback, and does not substitute a
scripted fallback when Qwen emits bad code.

## Artifacts

```text
experiments/scaffold_to_policy/results/terminal_bench_harbor_qwen_headless_official_verifier/manifests/report_input_20260812Tqwen-headless-terminal-official-verifier.json
experiments/scaffold_to_policy/results/terminal_bench_harbor_qwen_headless_official_verifier/runs/20260812Tqwen-headless-terminal-official-verifier/result.json
experiments/scaffold_to_policy/results/terminal_bench_harbor_qwen_headless_official_verifier/runs/20260812Tqwen-headless-terminal-official-verifier/headless-terminal__zjm6XXg/result.json
experiments/scaffold_to_policy/results/terminal_bench_harbor_qwen_headless_official_verifier/runs/20260812Tqwen-headless-terminal-official-verifier/headless-terminal__zjm6XXg/verifier/test-stdout.txt
```

## Next Steps

- Keep this result as the Harbor model-agent baseline for the pinned
  `headless-terminal` task.
- Improve the policy prompt or add bounded self-repair only if the repair loop
  is reported as part of the agent harness and does not use hidden verifier
  feedback.
- Repeat the same model-agent pattern on tau2 only after a tau2 model-policy
  interface is implemented; current tau2 successes remain deterministic mock or
  noop baselines.
