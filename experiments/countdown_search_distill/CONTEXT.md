# Countdown Search Distillation

This context defines the experiment language for compressing Countdown
best-of-N search behavior into one-shot Qwen3 policy behavior with TorchTitan
LoRA SFT.

## Language

**Search Scaffold**:
Inference-time sampling budget that exposes a verified solution the model does
not reliably produce on the first sampled completion.
_Avoid_: MCTS, search policy, teacher policy

**Elicitable Problem**:
A Countdown problem not solved by the first sampled rollout but solved at least
once within the sampled rollout budget.
_Avoid_: Hard problem, lucky problem

**Search Compression**:
An improvement in pass@1 after SFT that preserves some of the base model's
best-of-N success without requiring the original rollout budget.
_Avoid_: Distillation win, reasoning improvement

**Hindsight Hint**:
A prompt-side scaffold synthesized from verified or near-miss rollouts and used
to construct guided SFT targets.
_Avoid_: Teacher answer, chain-of-thought label

**Curriculum Stage**:
The training row phase that controls whether a hindsight hint is present,
dropped out, or absent while preserving the verified target answer.
_Avoid_: Epoch, split, condition

**Reduced Pilot**:
The first non-smoke evidence tier: large enough to compare arms and pass@k
curves, but smaller than the full pilot so calibration and training failures
are caught before spending the full GPU budget.
_Avoid_: Smoke, full run

**Matched Coverage**:
The count of train or evaluation problems that have verified targets for every
arm included in the comparison.
_Avoid_: Dataset size, retained rows

**Base-Elicitable Gate**:
The reduced-pilot success check measured on problems that were elicitable for
the frozen base model.
_Avoid_: Dev score, all-problem lift

**Problem Key**:
The canonical identity of a Countdown problem for split hygiene: the ordered
input numbers together with the target.
_Avoid_: Prompt ID, row ID

**Heldout Same-Regime Split**:
An evaluation split generated from the same problem regime as training but with
no `Problem Key` overlap with train or other evaluation splits.
_Avoid_: IID, dev, validation set

**OOD Target-Range Split**:
An evaluation split whose targets are outside the training target range and
whose `Problem Key`s are disjoint from every other split.
_Avoid_: Test set, hard split

**Champion Arm**:
The currently strongest training arm that new experiment variants must beat on
clean held-out evidence.
_Avoid_: Winner, best model

**Formatting Arm**:
A training or decoding intervention whose primary purpose is to improve
verifier-compliant output shape rather than arithmetic search behavior.
_Avoid_: Reasoning arm, cleanup

**Regime Ladder**:
An ordered set of Countdown problem regimes used to test whether search
compression survives increasing difficulty or distribution shift.
_Avoid_: Benchmark list, sweep

**Experiment Registry**:
A machine-readable record of regimes, split policy, arms, budgets, and expected
artifacts for a run family.
_Avoid_: Config file, notes

**Reasoning Lane**:
The first expansion lane after Countdown split repair, covering tasks whose
answers can be checked by exact or task-specific symbolic verifiers.
_Avoid_: Math benchmarks, evals

**Agentic Task Lane**:
A later expansion lane for multi-step tool, user, or terminal interaction
benchmarks such as tau2-bench and Terminal-Bench.
_Avoid_: Coding benchmark, tool benchmark

**Harness Boundary**:
The integration layer where an external benchmark runner owns task execution
while the experiment registry records models, adapters, budgets, and artifacts.
_Avoid_: Wrapper, adapter
