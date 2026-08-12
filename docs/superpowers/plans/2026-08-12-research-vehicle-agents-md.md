# Research Vehicle AGENTS.md Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update the root `AGENTS.md` with the approved operating contract for TorchTitan's observability, foundation-training, multimodal, and post-training research program.

**Architecture:** Keep the always-loaded instructions compact and put the complete design and observability rationale behind branch-specific context pointers. Add one co-located research-vehicle section after the existing development-surface classification; preserve the detailed core, RL, experiment, numerical, and testing contracts already in `AGENTS.md`.

**Tech Stack:** Markdown, `rg`, Git, repository pre-commit hooks

## Global Constraints

- The implementation modifies only `AGENTS.md`; the approved design and research note remain reference sources.
- Preserve the three existing execution surfaces and dependency direction `experiments -> core`.
- Integrate observability into the core `Trainer` first; later consumers reuse the evidence contract without becoming modes of core training.
- Keep repo-local run-attempt artifacts canonical and TensorBoard as the compact scalar dashboard; remote services remain optional exporters.
- Tier 0 always-on observability must stay within 1% median steady-state throughput regression and report memory, CPU, artifact volume, and detection latency separately.
- V1 captures evidence before continuing performance anomalies or aborting fatal correctness and unrecoverable distributed failures; it does not automatically recover, retry, or quarantine hardware.
- Preserve the 2/4/8-GPU Qwen3 0.6B/1.7B -> 8B -> 30B-A3B promotion ladder.
- Keep SigLIP2 and external model-family packages out of core dependencies.
- Keep native VLM integrations distinct from the SigLIP2 vision-encoder/projector plugin path.
- Preserve explicit checkpoint, sample, verifier, reward, and policy lineage across SFT, rejection sampling, and online RL.
- Leave unrelated dirty-worktree changes untouched and stage only `AGENTS.md` for the implementation commit.

---

## File structure

- Modify `AGENTS.md`: add the concise, always-loaded research operating contract.
- Read `docs/superpowers/specs/2026-08-12-training-research-vehicle-design.md`: authoritative approved program design and proof gates.
- Read `docs/research/2026-08-12-training-observability-paper-closure.md`: bounded paper closure, tool inventory, capture rationale, and current repo gap analysis.

### Task 1: Add the training research vehicle operating contract

**Files:**
- Modify: `AGENTS.md:28`
- Reference: `docs/superpowers/specs/2026-08-12-training-research-vehicle-design.md`
- Reference: `docs/research/2026-08-12-training-observability-paper-closure.md`

**Interfaces:**
- Consumes: the existing `Classify the Work First` surface boundaries and the two committed design references.
- Produces: one `## Training Research Vehicle` section with branch-specific context pointers, program order, observability gates, model promotion, multimodal boundaries, and post-training lineage.

- [ ] **Step 1: Verify the insertion anchor and reference targets**

Run:

```bash
test -f docs/superpowers/specs/2026-08-12-training-research-vehicle-design.md
test -f docs/research/2026-08-12-training-observability-paper-closure.md
rg -n '^## (Classify the Work First|Configuration and Entrypoints)$' AGENTS.md
```

Expected: both `test` commands succeed; `rg` reports `Classify the Work First`
before `Configuration and Entrypoints` with no existing `Training Research
Vehicle` heading between them.

- [ ] **Step 2: Insert the approved operating contract after the surface classification**

Apply this exact patch:

```diff
@@
 Keep the dependency direction `experiments -> core`. Core must remain usable
 without optional experiment dependencies. Read
 `torchtitan/experiments/README.md` before changing an experiment's boundary or
 copying core machinery into an experiment.

+## Training Research Vehicle
+
+Use `docs/superpowers/specs/2026-08-12-training-research-vehicle-design.md`
+before changing program sequencing, model promotion gates, shared observability,
+multimodal interfaces, or post-training lineage. For observability tool
+selection, capture policy, incident evidence, hangs, stragglers, or hardware
+diagnostics, also read
+`docs/research/2026-08-12-training-observability-paper-closure.md`.
+
+Develop the research vehicle in this gated order without collapsing the three
+execution surfaces above:
+
+1. **Observability foundation.** Instrument the core `Trainer` first with a
+   versioned, repo-local run-attempt evidence contract. Let RL and offline
+   programs adopt the schema later.
+2. **Foundation training.** Certify Qwen3 0.6B and 1.7B on 2, 4, and 8 GPUs,
+   then promote Qwen3 8B and Qwen3 30B-A3B MoE.
+3. **Multimodal training.** Develop native Qwen3-VL and Qwen3.5/3.6 paths
+   separately from a SigLIP2 encoder plus trained vision-token projector for
+   Qwen3, DeepSeek-v4, and GLM-5.2.
+4. **Post-training.** Join SFT, rejection sampling/distillation, online RL, and
+   evaluation through explicit data and checkpoint lineage.
+
+### Observability Evidence Contract
+
+- Treat an immutable repo-local run-attempt bundle as canonical. Join kernel,
+  module, collective, job, checkpoint, and correctness evidence with stable
+  run/attempt, process/rank/mesh, device, clock, step, phase, and lineage keys.
+  TensorBoard is the compact scalar dashboard; remote services are optional
+  exporters.
+- Tier capture: Tier 0 is always-on structured metrics, bounded Flight Recorder,
+  checkpoint lineage, and low-rate DCGM evidence; Tier 1 is scheduled PyTorch
+  Profiler and memory capture; Tier 2 is anomaly-triggered Flight Recorder,
+  NCCL RAS, py-spy, and evidence-tail capture; Tier 3 is stopped-job EUD,
+  `nccl-tests`, SuperBench, deterministic replay, and Nsight diagnosis.
+- Tier 0 must stay within 1% median steady-state throughput regression. Report
+  GPU-memory delta, host CPU, artifact bytes per GPU-hour, and detection latency
+  separately.
+- The v1 fault suite covers collective hangs, rank death, compute and dataloader
+  stragglers, NaN/Inf, checkpoint corruption/interruption, inconsistent per-rank
+  config, and DCGM/EUD hardware drills. Capture before abort where safe,
+  continue bounded performance warnings, and abort fatal correctness or
+  unrecoverable distributed faults. V1 diagnoses and preserves evidence;
+  automatic retry/recovery and fleet quarantine remain outside TorchTitan.
+
+### Research Promotion Evidence
+
+- Keep global batch and tokens per optimizer step fixed across the 2/4/8-GPU
+  ladder. Promote Qwen3 8B or 30B-A3B only after the smaller models pass
+  configuration, deterministic numerical, checkpoint-resume, observability,
+  convergence, and matched-performance gates.
+- For native VLMs, preserve each family's processor, image-token, position,
+  mask, state-dict, and checkpoint contracts. For the pluggable path, fine-tune
+  SigLIP2, train the projector with frozen backbones, and prove processor,
+  forward, state-dict, checkpoint, and task compatibility before broader
+  unfreezing. Verify public model contracts before adding Qwen3.6, DeepSeek-v4,
+  or GLM-5.2 integrations.
+- Preserve sample provenance from base checkpoint -> SFT -> candidate generation
+  -> verification/reward -> rejection-selected data -> SFT/distillation -> RL
+  -> evaluation. Keep deterministic RL parity evidence separate from
+  production-style asynchronous throughput, reward, policy-age, and stability
+  evidence.
+- Classify claims precisely: smoke runs prove plumbing; deterministic comparison
+  proves promised numerical identity; representative training proves
+  convergence; matched steady-state runs prove performance; fault injection
+  proves detection/diagnosis; replicated evaluation proves quality improvement.
+
 ## Configuration and Entrypoints
```

Expected: the new section follows the existing surface classification, and the
original `Configuration and Entrypoints` section starts immediately after it.

- [ ] **Step 3: Verify the context pointers and single-source structure**

Run:

```bash
test "$(rg -c '^## Training Research Vehicle$' AGENTS.md)" = 1
test "$(rg -c '^### Observability Evidence Contract$' AGENTS.md)" = 1
test "$(rg -c '^### Research Promotion Evidence$' AGENTS.md)" = 1
test -f docs/superpowers/specs/2026-08-12-training-research-vehicle-design.md
test -f docs/research/2026-08-12-training-observability-paper-closure.md
rg -n 'Qwen3 0\.6B and 1\.7B|30B-A3B|SigLIP2|DeepSeek-v4|GLM-5\.2|within 1%|rank death|rejection-selected' AGENTS.md
```

Expected: every `test` succeeds. `rg` finds every required research branch in
the new section, with no duplicate research-vehicle headings elsewhere.

- [ ] **Step 4: Inspect the isolated documentation diff**

Run:

```bash
git diff --check -- AGENTS.md
git diff --stat -- AGENTS.md
git diff -- AGENTS.md
```

Expected: `git diff --check` is silent; the stat names only `AGENTS.md`; the
diff adds the research section without changing existing core, RL, experiment,
test, numerical, style, or completion instructions.

- [ ] **Step 5: Run the documentation hooks**

Run:

```bash
SKIP=no-commit-to-branch pre-commit run --files AGENTS.md
```

Expected: all applicable content hooks pass. Only the branch-protection hook is
skipped; no formatting hook rewrites the file.

- [ ] **Step 6: Commit only the operating-contract update**

Run:

```bash
git diff --cached --name-only
git add -- AGENTS.md
git diff --cached --name-status
git commit -m "Document training research vehicle contract"
```

Expected: the first command is empty before staging; the staged diff names only
`AGENTS.md`; the commit succeeds on an implementation branch. If unrelated
paths were already staged, stop before `git add` and preserve the user's index.
