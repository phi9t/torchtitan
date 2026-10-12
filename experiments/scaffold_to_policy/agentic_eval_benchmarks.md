# General Agentic Evaluation Benchmarks

This note maps benchmark families that matter for the scaffold-to-policy
program. It separates three objects that are often conflated:

1. **Upstream benchmark**: task distribution, environment, release, and canonical
   metric.
2. **Agent harness**: tools, context policy, retries, action space, and time
   budget.
3. **Evaluation protocol**: judge, sampling policy, filtering, aggregation, and
   reliability reporting.

The goal is not to make one universal leaderboard. The goal is to choose a
scientific ladder where each benchmark adds a capability axis without changing
the meaning of earlier results.

## Complexity Model

Operational complexity:

| Level | Name | Definition |
| --- | --- | --- |
| L0 | Static answer | One prompt, no tools, no mutable environment, short final answer. |
| L1 | Static executable or structured output | Code, strict formatting, localization, grid generation, or another output with deterministic checking. |
| L2 | Large or multimodal context | Charts, documents, video, long context, or persistent-memory content, but no broadly mutable world. |
| L3 | Bounded tools or retrieval | Narrow function calling, search, retrieval, or memory operations with limited state. |
| L4 | Stateful closed world | Domain APIs, user simulation, persistent database state, and policy-governed actions. |
| L5 | Broad operational environment | Web apps, desktop, repositories, terminals, multiple MCP servers, or long cross-tool workflows. |
| L6 | Dynamic or adversarial environment | Asynchronous events, other agents, changing state, prompt injection, safety attacks, or time-dependent behavior. |
| L7 | Open-ended production or research | Professional or scientific deliverables with many valid trajectories, rich artifacts, and rubric-based evaluation. |

Reasoning depth:

| Level | Definition |
| --- | --- |
| R1 | Recognition, extraction, or direct lookup. |
| R2 | Local composition, formatting, constraint satisfaction, or straightforward API selection. |
| R3 | Multi-step general reasoning, planning, and error recovery. |
| R4 | Expert, cross-source, temporal, or policy-intensive synthesis. |
| R5 | Frontier specialist reasoning, novel abstraction, or research-level work. |

These axes should be reported together. FrontierMath is roughly L0/R5, while
OSWorld v1 is roughly L5/R3; a single easy-to-hard ordering hides that
difference.

## Capability Ladders

### Reasoning Lane

This is the first expansion lane after Countdown split repair.

| Benchmark | Why it matters | Level |
| --- | --- | --- |
| Synthetic exact-verifier reasoning | Preserves Countdown-style exact verification while exercising more reasoning shapes. | L0-L1/R2-R4 |
| GSM8K or GSM-style subset | Recognizable arithmetic reasoning baseline with parsed final-answer checking. | L0/R3 |
| MATH subset | Harder symbolic/math reasoning with answer normalization challenges. | L0/R4 |
| AIME | Contest math, exact integer answers, small sample size. | L0/R4 |
| GPQA Diamond | Expert science multiple choice, deep domain reasoning. | L0/R5 |
| FrontierMath | Research-level math; useful later, not for first scaffold plumbing. | L0/R5 |
| ARC-AGI-2 | Exact structured grid output and abstract rule induction. | L1/R5 |

Recommended order:

1. Synthetic exact-verifier reasoning.
2. GSM8K or GSM-style subset.
3. Small MATH subset.
4. ARC-AGI-2 or AIME as a harder static-output branch.

### Tool Use And State Management

| Benchmark | Capability added | Level |
| --- | --- | --- |
| BFCL | Function selection and argument construction. | L3/R2 |
| ToolSandbox | Stateful tools, clarification, dependencies, forbidden states. | L4/R3 |
| tau-bench | Tool-agent-user interaction with domain APIs and policy rules. | L4/R3 |
| tau-Knowledge / tau-Banking | Stateful interaction plus unstructured domain knowledge. | L5/R4 |
| MCP Atlas | Broad MCP tool ecosystems and multi-server composition. | L5/R3 |
| AppWorld | Cross-application API workflows. | L5/R4 |
| GAIA2 | Dynamic simulated application universes with events and time. | L6/R4 |

Recommended entry point for this program: tau-bench/tau2-bench only after a
reasoning benchmark shows replicated held-out gain and the experiment registry
supports external harness boundaries.

### Coding And Research Engineering

| Benchmark | Capability added | Level |
| --- | --- | --- |
| LiveCodeBench | Fresh contest-style code generation and execution. | L1/R3 |
| BigCodeBench | Library/API composition in function-level code tasks. | L1/R3 |
| SciCode | Scientific programming with numeric tolerances. | L1/R5 |
| SWE-bench Verified | Repository issue repair with executable tests. | L5/R3 |
| SWE-bench Pro | Larger/harder repository repair. | L5/R4 |
| Terminal-Bench | Arbitrary terminal-state tasks in isolated environments. | L5/R4 |
| PaperBench | Reproduce research-paper results against a rubric. | L7/R5 |
| ResearchClawBench | End-to-end autonomous scientific rediscovery. | L7/R5 |

Terminal-Bench should enter through Harbor rather than a native runner. This
repo should own rootfs-managed launch, registry entries, artifact locations, and
result ingestion; Harbor should own task execution.

### GUI And Computer Use

| Benchmark | Capability added | Level |
| --- | --- | --- |
| ScreenSpot-Pro | GUI grounding: identify where to click. | L1/R2 |
| VisualWebArena | Browser workflows requiring visual grounding. | L5/R3 |
| OSWorld v1 | Desktop application workflows. | L5/R3 |
| OSWorld 2.0 | Longer dynamic desktop workflows. | L6/R4 |
| WildClawBench | Multimodal, bilingual, realistic agent containers. | L6/R4 |

These should be treated as agent-harness benchmarks, not model-only evaluations.

### Long Context, Memory, And Research

| Benchmark | Capability added | Level |
| --- | --- | --- |
| RULER | Controlled long-context primitives. | L2/R2 |
| LongBench v2 | Realistic long-input reasoning. | L2/R4 |
| AA-LCR | Long document collections with joins and normalization. | L2/R4 |
| BEAM / BEAM-128K | Coherent long-dialogue memory and state updates. | L2/R4 |
| LongMemEval-V2 | Long-term memory over accumulated agent history. | L3/R4 |
| DeepSearchQA | Exhaustive web research sets and deduplication. | L5/R4 |
| BrowseComp | Hard-to-find web answers through clue chains. | L5/R4 |
| GDPval | Professional work products with rubric or pairwise evaluation. | L7/R4 |

Long-context and research benchmarks should not be converted into RAG tasks
unless the result is explicitly labeled as a RAG-system condition.

### Safety

| Benchmark | Capability added | Level |
| --- | --- | --- |
| CIMemories | Contextual integrity over persistent personal memories. | L2/R4 |
| AgentDojo | Prompt injection in untrusted tool data. | L6/R3 |
| Prompt Siren + AgentDojo | Stronger constructed prompt-injection attacks. | L6/R3 |
| AgentHarm | Harmful multi-step agent objectives. | L6/R3 |

Safety benchmarks require separate reporting of benign utility, attack success,
refusal behavior, and harmful capability.

## Normalization Rules

### Pin The Complete Evaluation Object

A benchmark version is not only a data file. Registry entries must pin:

```yaml
benchmark:
  task_release: exact tag or dataset revision
  task_manifest_hash: sha256
  evaluator_commit: exact git commit
  container_images: immutable digests
  official_metric: exact implementation
```

For agentic evaluations, also pin:

```yaml
agent:
  harness_commit: exact git commit
  system_prompt_hash: sha256
  tool_schema_hash: sha256
  action_budget: exact limit
  context_policy: exact truncation and summarization policy
  retry_policy: exact behavior
```

### Preserve Upstream Semantics

- A no-tools benchmark remains no-tools.
- A screenshot-only benchmark should not receive DOM access.
- A GUI benchmark should not receive an unrestricted shell unless reported as a
  separate condition.
- A long-context benchmark should not become a RAG benchmark unless labeled as
  RAG.
- A benchmark with executable tests should not be replaced by a generic model
  judge.

Canonical metric semantics matter:

| Benchmark family | Preserve this scoring style |
| --- | --- |
| SWE-bench, Terminal-Bench, LiveCodeBench, BigCodeBench, SciCode | executable tests or final environment verifiers |
| OSWorld, WebArena, AppWorld, tau-bench, GAIA2 | final-state verifiers |
| ScreenSpot-Pro | point-in-box scoring |
| DeepSearchQA | set precision, recall, and F1 |
| MCP Atlas | required-claim coverage |
| PaperBench, ResearchClawBench, GDPval | rubric or expert pairwise evaluation |
| AgentDojo and related safety suites | benign utility and attack success separately |

### Separate Model Capability From Harness Capability

Every result should be labeled as:

```text
model + reasoning configuration + agent harness + tools + environment version
```

Scores from Codex, OpenHands, Claude Code, Harbor, or a custom harness are not
model-only measurements.

### Treat Reliability As A First-Class Output

Interactive and agentic tasks should report more than mean success. Capture:

- pass^k or repeated-attempt reliability when upstream defines it;
- per-task trajectories;
- timeouts;
- environment errors;
- token use;
- wall-clock time;
- tool-call counts;
- retry counts.

Two systems with the same mean success can have very different deployment
reliability.

## Adoption Plan For This Repo

### Phase 0: Countdown Repair

Before any benchmark expansion:

- create a canonical split registry;
- enforce global `Problem Key` de-duplication;
- regenerate heldout same-regime splits;
- rerun base and exported adapters;
- update the report with clean held-out evidence.

### Phase 1: Reasoning Lane

Implement shared registry support just far enough to run:

- synthetic exact-verifier reasoning;
- GSM8K or GSM-style subset;
- a small MATH subset.

First scaffold: best-of-N sampling. Tool-assisted scaffolds are a second-tier
condition.

First arms:

- base;
- raw;
- clean;
- formatting.

### Phase 2: Agentic Harness Feasibility

Run rootfs-managed smoke checks only:

- Harbor + Terminal-Bench tiny task or dry-run equivalent;
- tau2-bench tiny task or dry-run equivalent.

No scientific claims from this phase. The output is compatibility evidence and
an artifact-ingestion contract.

### Phase 3: Agentic Lane

After one reasoning benchmark shows replicated held-out gain:

- add pinned Harbor/Terminal-Bench registry entries;
- add pinned tau2-bench registry entries;
- ingest trajectories and scores;
- report benchmark, harness, model, adapter, tools, and environment as one
  labeled evaluation object.

## Version-Pin Watchlist

The following require especially careful labels:

- MCP Atlas: public 500-task split versus full 1,000-task benchmark.
- tau lineage: tau-bench, tau2-bench, tau-Knowledge/tau-Banking, tau3 releases.
- SkillsBench: release version and with-skills versus no-skills condition.
- OSWorld: v1 and 2.0 are different task distributions.
- OmniDocBench: v1.5 data/parser/matcher/aggregation if reporting v1.5.
- HLE: full multimodal release versus text-only filtered condition.
- BEAM: full 100-dialogue benchmark versus the 128K bucket.
- AIME: year, exam split, attempts per problem, and tool policy.

## Immediate Open Decisions

- Which synthetic exact-verifier reasoning task should be first?
- Should GSM8K enter as an imported public dataset or as a local small fixture
  first?
- Which output format should the formatting arm enforce across reasoning tasks?
- What is the minimum Harbor smoke that proves rootfs compatibility without
  consuming meaningful experiment budget?
- What is the minimum tau2-bench smoke that proves harness compatibility without
  making benchmark claims?
