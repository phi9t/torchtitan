# Core Run Evidence Closeout and Handoff Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` to implement this closeout plan task-by-task.
> Use `superpowers:subagent-driven-development` only to dispatch the bounded
> review/fix work named below, and continue the existing
> `.superpowers/sdd/2026-08-12-core-run-evidence-foundation/` ledger rather than
> initializing a second SDD workspace for this handoff plan. Use
> `superpowers:systematic-debugging` for any new smoke/test failure,
> `superpowers:verification-before-completion` before completion claims, and
> `superpowers:finishing-a-development-branch` only after the final review is
> clean. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close, independently review, and hand off the core TorchTitan
run-evidence foundation without overstating GPU, Qwen, overhead, fault-tolerance,
or external-tool readiness.

**Architecture:** The feature branch adds one process-local `RunEvidence`
recorder installed only by core `torchtitan.train`. Existing native producers
register lifecycle transitions through a no-op-safe facade, while launchers own
shared run/attempt identity. The remaining work is review and documentation,
not a new observability feature: gate the deterministic config snapshot fix,
rerun one fresh CPU bootstrap smoke, commit/review the prepared operator docs,
then run a whole-branch standards/spec/readiness review.

**Tech Stack:** Python 3.11, PyTorch, TorchTitan `ConfigManager`, JSON/JSONL,
`torchrun`, Bash/Slurm launchers, pytest, pre-commit, Pyrefly, git worktrees.

## Global Constraints

- Work only in
  `/data02/home/philip.yang/workspace/torchtitan/.worktrees/core-run-evidence`
  on branch `codex/core-run-evidence`.
- Use binaries from
  `/data02/home/philip.yang/workspace/torchtitan/.venv/bin`; the linked
  worktree does not contain its own `.venv`.
- Preserve all four prepared documentation changes and both existing smoke
  bundles. Do not clean, reset, overwrite, or delete them.
- Stage files explicitly. Other worktrees share the repository, and the main
  checkout is currently on unrelated branch
  `scaffold-wave-f0-execution-foundation`.
- The main checkout is also dirty with three modified and four untracked files
  under `experiments/scaffold_to_policy/`. Those changes belong to another
  effort. Do not checkout, merge, clean, reset, or otherwise mutate the main
  checkout at `/data02/home/philip.yang/workspace/torchtitan`.
- Base this feature review on `main` commit
  `cec1afd76ec7935040e149583ef7294defc9172a`.
- Keep the dependency direction `experiments -> core`; core must not acquire an
  optional experiment or external observability dependency.
- Keep native log, profiler, memory, TensorBoard, Flight Recorder, and
  checkpoint formats and compatibility paths unchanged.
- Preserve active training/producer exceptions when evidence cleanup also
  fails. Healthy-path evidence contract/write failures remain fatal.
- Use strict TDD for every code fix: regression RED, minimal implementation,
  focused GREEN, owning suite, separate commit, independent re-review.
- The final claim is a CPU/static-verified evidence-contract foundation only.
  It is not 2/4/8-GPU certification, a Qwen training result, a Tier-0 overhead
  result, or a hang/fault-recovery result.
- The configured pre-commit Pyrefly adapter is known to miss its console
  executable or exclude the hidden `.worktrees` path. Record that limitation
  and run the installed module directly. Require zero diagnostics in every
  changed production file, including TorchFT.

---

## Frozen Handoff State

### Repository state

At handoff-plan creation, before the plan-only commit:

```text
worktree: /data02/home/philip.yang/workspace/torchtitan/.worktrees/core-run-evidence
branch:   codex/core-run-evidence
implementation HEAD: 797ef563595029acf40cbada657355f6625f13d1
base:     cec1afd76ec7935040e149583ef7294defc9172a (main)
```

Before this plan is committed, it is a fifth untracked file. The current
session must commit only this plan as `Plan core run evidence handoff`. After
that commit, the new session must see exactly these uncommitted files, all
owned by the unfinished documentation task:

```text
 M CONTEXT.md
 M docs/debugging.md
 M torchtitan/observability/structured_logger/README.md
?? docs/run_evidence.md
```

If the status differs, inspect and attribute every extra path before editing.
Do not assume an extra change is disposable.

### Required reading order

Read these files completely before resuming:

1. `AGENTS.md` -- current development rules and research-vehicle roadmap.
2. `docs/superpowers/specs/2026-08-12-training-research-vehicle-design.md` --
   phase order and promotion gates.
3. `docs/research/2026-08-12-training-observability-paper-closure.md` --
   bounded ByteRobust/FT-HSDP paper and tool closure.
4. `docs/superpowers/plans/2026-08-12-core-run-evidence-foundation.md` --
   original implementation plan.
5. This handoff plan.
6. `.superpowers/sdd/2026-08-12-core-run-evidence-foundation/progress.md` --
   task review ledger.
7. `.superpowers/sdd/2026-08-12-core-run-evidence-foundation/deterministic-config-report.md`
   -- latest shared-config root cause, tests, audit, and boundaries.
8. The four uncommitted documentation files listed above.

### Completed implementation commits

The repository guide requested by the original work is already in the fixed
base through commit `86dab35eb` (`Document training research vehicle contract`);
`AGENTS.md` resolves to that guide through `.claude/CLAUDE.md`. Do not create a
second research-roadmap configuration path in this feature branch.

These commits are already on the feature branch:

```text
0c6d352e2 Plan core run evidence foundation
a47e865c3 Add core run evidence recorder
749db40e5 Harden core run evidence recorder
957eeae1b Correlate core structured training evidence
d2134df05 Harden structured evidence cleanup
67878f578 Index native training diagnostic artifacts
9974d5872 Fix native diagnostic artifact evidence paths
64d844772 Record checkpoint artifact lifecycles
cb6618570 Cover checkpoint load preparation evidence
4082cbc70 Assign shared run evidence identity at launch
800951f72 Abort invalid Slurm evidence identity
8f4a0404f Honor Slurm identity generation failures
fcbc680f4 Represent unknown source dirtiness explicitly
3bc53d9ed Stabilize relative structured evidence paths
eeb7bbe09 Make config snapshots deterministic
797ef5635 Make config snapshots deterministic
```

Task-scoped reviews are clean through launcher identity, source provenance, and
relative structured paths. Commit `eeb7bbe09` received a review that found
injectivity/global-binding gaps; `797ef5635` is the fix round and still requires
the focused re-review in Task 1 below.

The latest task-local serializer verification recorded 129 owner tests and 16
RL/TorchFT tests passing. Two fresh processes produced byte-identical,
address-free config JSON for llama3 debug; Qwen3 0.6B, 1.7B, 30B-A3B, and
non-NVFP4 `sft_qwen3_8b_math`; deepseek_v3 debug; gpt_oss debug; flux debug;
TorchFT llama3; and RL alphabet_sort. Llama4 is absent in this checkout, and
the optional Qwen3 8B NVFP4 registry cannot load because torchao is absent.
These are recorded task results, not final branch verification or model
training evidence; Tasks 1, 2, and 5 must still run their gates.

A prior direct Pyrefly probe reported two diagnostics in the changed TorchFT
checkpoint file (`save` return override and `dcp_save` union return type). They
are historical context, not a waiver. If the final direct command reproduces
them, treat them as blocking findings and resolve them without changing runtime
checkpoint behavior.

### Implemented contract

The branch currently implements:

- launcher-owned `TORCHTITAN_RUN_ID` and `TORCHTITAN_ATTEMPT_ID`, with elastic
  restart suffixing in the recorder and run ID used as rendezvous ID;
- one immutable shared `manifest.json`, an exclusive append-only per-process
  artifact index, and an immutable per-process outcome;
- structured-event run/attempt/process/rank/mesh/device/clock/step/phase
  correlation;
- lifecycle indexing for structured JSONL, PyTorch Profiler traces, CUDA memory
  snapshots, TensorBoard event streams, Flight Recorder dump prefixes, and
  DCP/Hugging Face checkpoint inputs/outputs;
- async checkpoint completion only after its durable upload future resolves;
- no-op behavior outside an installed recorder, with core-entrypoint TorchFT
  compatibility but no indexing of TorchFT's private per-replica dataloader
  checkpoint;
- deterministic, JSON-compatible config snapshots for supported config values,
  including callable code/default/closure/direct-global identity, with explicit
  failure for unsupported state.

### Preserved smoke evidence

Do not delete either directory:

```text
outputs/run_evidence_local_tensor_smoke.DVEqcq
outputs/run_evidence_local_tensor_smoke.sxUZMD
```

- `DVEqcq` is the expected failed diagnostic bundle that exposed relative
  structured-path identity mismatch. Its outcome is failed and its structured
  artifact remains declared.
- `sxUZMD` is a successful bootstrap bundle after the path fix. Its review then
  exposed process-address-bearing callable representations in the manifest.
  It predates the final deterministic serializer and is not the final smoke.

Both paths are ignored by `.gitignore` through `outputs/`.

### Explicitly deferred work

This branch does not implement or certify:

- DCGM telemetry/EUD, py-spy, NCCL RAS, `nccl-tests`, SuperBench, Nsight, or
  external-tool adapters;
- anomaly triggers, capture policy, incident classification, retention policy,
  module/kernel ranges, or offline joined analyzers;
- FT-HSDP membership, commit/recovery state, automatic retry, or quarantine;
- fault injection, hang/straggler drills, semantic checkpoint lineage, or
  cross-process outcome aggregation;
- the 1% Tier-0 overhead gate, artifact-volume budgets, or detection latency;
- real 2/4/8-GPU execution, Qwen numerics/convergence/checkpoint-resume, or
  Qwen3 8B/30B-A3B promotion.

---

### Task 1: Re-review Deterministic Config Snapshot Hardening

**Files:**
- Review: `torchtitan/config/configurable.py`
- Review: `tests/unit_tests/test_configurable.py`
- Reference: `tests/unit_tests/observability/test_run_evidence.py`
- Reference: `.superpowers/sdd/2026-08-12-core-run-evidence-foundation/deterministic-config-report.md`

**Interfaces:**
- Consumes: `Configurable.Config.to_dict() -> dict` and RunEvidence canonical
  manifest hashing.
- Produces: a review-clean, rank-stable config snapshot contract that Task 2's
  final smoke can trust.

- [ ] **Step 1: Confirm the frozen state before review**

Run:

```bash
cd /data02/home/philip.yang/workspace/torchtitan/.worktrees/core-run-evidence
git branch --show-current
git rev-parse HEAD
git status --short
```

Expected: branch `codex/core-run-evidence`; HEAD at or descended from
`797ef5635`; only the four prepared docs are uncommitted.

- [ ] **Step 2: Generate the exact fix-round review package**

Run:

```bash
bash /data02/home/philip.yang/.codex/plugins/cache/openai-curated-remote/superpowers/6.2.0/skills/subagent-driven-development/scripts/review-package \
  docs/superpowers/plans/2026-08-12-core-run-evidence-foundation.md \
  eeb7bbe098761e2af307019dc7773708df0e041c \
  797ef563595029acf40cbada657355f6625f13d1 \
  .superpowers/sdd/2026-08-12-core-run-evidence-foundation/review-eeb7bbe..797ef56.diff
```

Completion criterion: the package contains only the review-fix commit and its
tests; the prepared docs are absent.

- [ ] **Step 3: Dispatch one read-only focused reviewer**

Give the reviewer the exact package and the original unresolved findings:

```text
1. Lists and tuples must remain distinct after JSON encoding.
2. Internal sentinels/descriptors cannot collide with ordinary user dicts.
3. Dictionary keys must not collapse under JSON; supported user keys are strings.
4. Direct LOAD_GLOBAL bindings must contribute bounded deterministic identity.
5. Callable/global traversal must terminate on cycles and remain bounded.
6. Unsupported state must fail clearly rather than use raw repr().
7. Existing core/RL/TorchFT config consumers must stay JSON-compatible.
8. The audit must include non-NVFP4 qwen3/sft_qwen3_8b_math.
```

Require file:line evidence and a verdict of `Approved` or `Needs fixes`. The
review is read-only and must not rerun broad suites.

- [ ] **Step 4: Resolve every Critical or Important finding through TDD**

For each valid finding, use the existing serializer tests as the owning seam.
Run the complete focused file for both RED and GREEN so no dynamically named
test is accidentally omitted:

```bash
/data02/home/philip.yang/workspace/torchtitan/.venv/bin/pytest -q \
  tests/unit_tests/test_configurable.py
```

Capture RED, implement the smallest serializer fix, rerun GREEN and the owning
suites, commit separately, generate a new delta package, and request a focused
re-review. Do not edit documentation in this task.

- [ ] **Step 5: Record the review result in the SDD ledger**

If the review approves `797ef5635` unchanged, append this exact line to
`.superpowers/sdd/2026-08-12-core-run-evidence-foundation/progress.md`:

```text
Deterministic config hardening: complete (commits eeb7bbe..797ef56, review clean)
```

If another fix was required, record the actual fix commit hash and finding
count in prose, then record the exact reviewed range from `eeb7bbe09` through
the literal output of `git rev-parse --short=9 HEAD`. Do not leave symbolic
hashes or a provisional finding count in the ledger.

Completion criterion: no Critical or Important review finding remains open.

---

### Task 2: Finish Operator Documentation and Run a Fresh Bootstrap Smoke

**Files:**
- Modify: `CONTEXT.md`
- Create: `docs/run_evidence.md`
- Modify: `docs/debugging.md`
- Modify: `torchtitan/observability/structured_logger/README.md`
- Create local report:
  `.superpowers/sdd/2026-08-12-core-run-evidence-foundation/task-6-report.md`
  (ignored handoff evidence; never stage it)

**Interfaces:**
- Consumes: the final reviewed schema, launcher identity, deterministic config
  snapshot, producer lifecycle, and bootstrap behavior.
- Produces: the canonical operator contract plus a post-hardening CPU smoke
  bundle.

- [ ] **Step 1: Reconcile the prepared docs with final code**

Create the local Task 6 report with `apply_patch`, starting with the current
branch, HEAD, expected four-file status, and the two preserved smoke paths. Add
commands and exact results as the steps below complete; do not stage the report.

Read all four files and verify these exact facts:

- layout is
  `dump_folder/<run_evidence.folder>/<run_id>/<attempt_id>/`, with
  `run_evidence` only the configurable default;
- manifest and outcomes are immutable, artifact indexes are append-only, and
  the bundle as a whole is not immutable;
- failed Git collection is `revision="unknown", dirty=null`;
- config snapshots use deterministic tagged callable/value descriptors and
  reject unsupported state; direct callable globals are bounded snapshots, not
  proof of arbitrary later mutation or dependency equivalence;
- `event_seq`, structured logger `seq_id`, and `artifact_seq` are distinct;
- structured events do not contain `artifact_id`;
- artifact paths are `dump_relative`, `external_absolute`, or `uri`;
- `local_tensor` returns before `Trainer` construction and proves bootstrap
  only;
- Flight Recorder is a declared native prefix, not proof of a completed dump;
- only core `torchtitan.train` installs the recorder; TorchFT's core entrypoint
  inherits it, while RL/Forge/offline/direct construction remain no-op;
- every deferred item listed in Frozen Handoff State remains explicit.

Completion criterion: no documentation statement promises behavior beyond the
implemented schema or the final serializer boundary.

- [ ] **Step 2: Run the full CPU-capable owning suite after Task 1**

Run exactly:

```bash
/data02/home/philip.yang/workspace/torchtitan/.venv/bin/pytest -q \
  tests/unit_tests/test_configurable.py \
  tests/unit_tests/test_config_manager.py \
  tests/unit_tests/observability/test_run_evidence.py \
  tests/unit_tests/observability/test_structured_logging.py \
  tests/unit_tests/test_train_run_evidence.py \
  tests/unit_tests/test_profiler.py \
  tests/unit_tests/test_metrics_run_evidence.py \
  tests/unit_tests/test_distributed_run_evidence.py \
  tests/unit_tests/test_checkpoint.py \
  tests/unit_tests/test_run_evidence_launchers.py \
  tests/unit_tests/test_train_spec.py \
  torchtitan/experiments/torchft/tests/test_torchft_checkpoint.py \
  torchtitan/experiments/rl/tests/test_async_controller.py
```

Record the exact pass/warning count. Treat new warnings as findings; identify
the known PyTorch/config deprecation warnings separately.

- [ ] **Step 3: Run and validate a new unique post-hardening smoke**

Run this as one shell block so the generated path is passed directly to the
validator. The sentinel is intentionally present only in the training process
environment; neither its name nor value may enter the evidence bundle.

```bash
(
set -euo pipefail
cd /data02/home/philip.yang/workspace/torchtitan/.worktrees/core-run-evidence
mkdir -p outputs
smoke_dir="$(mktemp -d ./outputs/run_evidence_local_tensor_smoke.XXXXXX)"
printf 'smoke_dir=%s\n' "$smoke_dir"
TORCHTITAN_RUN_EVIDENCE_SENTINEL=run-evidence-smoke-secret-must-not-appear \
PATH="/data02/home/philip.yang/workspace/torchtitan/.venv/bin:$PATH" \
  MODULE=llama3 CONFIG=llama3_debugmodel NGPU=2 COMM_MODE=local_tensor \
  ./run_train.sh --dump_folder="$smoke_dir"
/data02/home/philip.yang/workspace/torchtitan/.venv/bin/python \
  - "$smoke_dir" <<'PY'
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path


root = Path(sys.argv[1])


def exactly_one(pattern: str) -> Path:
    matches = list(root.glob(pattern))
    assert len(matches) == 1, (pattern, matches)
    return matches[0]


manifest_path = exactly_one("run_evidence/*/*/manifest.json")
index_path = exactly_one("run_evidence/*/*/indexes/artifacts.*.jsonl")
outcome_path = exactly_one("run_evidence/*/*/processes/*/outcome.json")
structured_path = exactly_one("structured_logs/*.jsonl")
all_files = {path for path in root.rglob("*") if path.is_file()}
assert all_files == {
    manifest_path,
    index_path,
    outcome_path,
    structured_path,
}, all_files

manifest_text = manifest_path.read_text(encoding="utf-8")
bundle_text = "\n".join(
    path.read_text(encoding="utf-8") for path in sorted(all_files)
)
manifest = json.loads(manifest_text)
artifacts = [
    json.loads(line)
    for line in index_path.read_text(encoding="utf-8").splitlines()
]
outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
events = [
    json.loads(line)
    for line in structured_path.read_text(encoding="utf-8").splitlines()
]

assert manifest["schema_version"] == 1
canonical_config = json.dumps(
    manifest["config"]["normalized"],
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
)
assert hashlib.sha256(canonical_config.encode("utf-8")).hexdigest() == (
    manifest["config"]["sha256"]
)
assert re.search(r"<[^>\n]* at 0x[0-9a-fA-F]+>", manifest_text) is None
assert "TORCHTITAN_RUN_EVIDENCE_SENTINEL" not in bundle_text
assert "run-evidence-smoke-secret-must-not-appear" not in bundle_text

assert len(artifacts) == 2
assert [row["state"] for row in artifacts] == ["declared", "complete"]
assert {row["artifact_id"] for row in artifacts} == {
    artifacts[0]["artifact_id"]
}
assert {row["path_type"] for row in artifacts} == {"dump_relative"}
expected_structured_path = structured_path.relative_to(root).as_posix()
assert {row["path"] for row in artifacts} == {expected_structured_path}
assert events

run_id = manifest["run_id"]
attempt_id = manifest["attempt_id"]
for record in [*artifacts, outcome]:
    assert record["schema_version"] == 1
    assert record["evidence_schema_version"] == 1

for event in events:
    assert event["evidence_schema_version"] == 1
    assert "schema_version" not in event

for record in [*artifacts, *events, outcome]:
    assert record["run_id"] == run_id
    assert record["attempt_id"] == attempt_id
    assert record["process_id"] == "trainer.core.global_rank_000000"

for record in [*artifacts, *events]:
    assert isinstance(record["wall_time_ns"], int)
    assert isinstance(record["monotonic_ns"], int)
assert artifacts[0]["monotonic_ns"] <= min(
    event["monotonic_ns"] for event in events
)
assert max(event["monotonic_ns"] for event in events) <= artifacts[1][
    "monotonic_ns"
]
assert outcome["outcome"] == "succeeded"
assert isinstance(outcome["elapsed_monotonic_ns"], int)
assert outcome["elapsed_monotonic_ns"] >= 0
assert "wall_time_ns" not in outcome
assert "monotonic_ns" not in outcome

print(
    json.dumps(
        {
            "smoke_dir": str(root),
            "run_id": run_id,
            "attempt_id": attempt_id,
            "artifact_id": artifacts[0]["artifact_id"],
            "structured_path": expected_structured_path,
            "event_count": len(events),
            "outcome": outcome["outcome"],
        },
        sort_keys=True,
    )
)
PY
find "$smoke_dir" -type f -print | sort
)
```

Do not reuse `DVEqcq` or `sxUZMD`. Do not delete any smoke directory. Record the
validator's smoke path, artifact path, IDs, event count, and outcome in the Task
6 report, but do not commit the bundle.

- [ ] **Step 4: Run documentation and changed-file checks**

Run:

```bash
bash -n run_train.sh
bash -n multinode_trainer.slurm
git diff --check
PATH="/data02/home/philip.yang/workspace/torchtitan/.venv/bin:$PATH" \
/data02/home/philip.yang/workspace/torchtitan/.venv/bin/pre-commit run --files \
  CONTEXT.md \
  docs/run_evidence.md \
  docs/debugging.md \
  docs/superpowers/plans/2026-08-12-core-run-evidence-foundation.md \
  docs/superpowers/plans/2026-08-12-core-run-evidence-closeout-handoff.md \
  torchtitan/observability/structured_logger/README.md \
  torchtitan/config/configurable.py \
  torchtitan/observability/run_evidence.py \
  torchtitan/observability/structured_logger/structured_logging.py \
  torchtitan/observability/structured_logger/jsonl_handler.py \
  torchtitan/observability/structured_logger/__init__.py \
  torchtitan/train.py \
  torchtitan/trainer.py \
  torchtitan/tools/profiler.py \
  torchtitan/components/metrics.py \
  torchtitan/components/checkpoint.py \
  torchtitan/distributed/utils.py \
  torchtitan/experiments/torchft/checkpoint.py \
  run_train.sh \
  multinode_trainer.slurm \
  tests/unit_tests/test_configurable.py \
  tests/unit_tests/observability/test_run_evidence.py \
  tests/unit_tests/observability/test_structured_logging.py \
  tests/unit_tests/test_train_run_evidence.py \
  tests/unit_tests/test_profiler.py \
  tests/unit_tests/test_metrics_run_evidence.py \
  tests/unit_tests/test_distributed_run_evidence.py \
  tests/unit_tests/test_checkpoint.py \
  tests/unit_tests/test_config_manager.py \
  tests/unit_tests/test_run_evidence_launchers.py \
  torchtitan/experiments/torchft/tests/test_torchft_checkpoint.py
```

If the Pyrefly adapter alone fails for the known environment reason, run:

```bash
/data02/home/philip.yang/workspace/torchtitan/.venv/bin/python -m pyrefly check \
  torchtitan/config/configurable.py \
  torchtitan/observability/run_evidence.py \
  torchtitan/observability/structured_logger/structured_logging.py \
  torchtitan/observability/structured_logger/jsonl_handler.py \
  torchtitan/observability/structured_logger/__init__.py \
  torchtitan/train.py \
  torchtitan/trainer.py \
  torchtitan/tools/profiler.py \
  torchtitan/components/metrics.py \
  torchtitan/components/checkpoint.py \
  torchtitan/distributed/utils.py \
  torchtitan/experiments/torchft/checkpoint.py \
  --disable-project-excludes-heuristics=true \
  --project-excludes does-not-match \
  --python-interpreter-path \
    /data02/home/philip.yang/workspace/torchtitan/.venv/bin/python \
  --site-package-path \
    /data02/home/philip.yang/workspace/torchtitan/.venv/lib/python3.11/site-packages \
  --remove-unused-ignores \
  --summary=full
```

Require zero diagnostics in every listed production file. Record the complete
command and output in the report. Any diagnostic is a blocking finding; use
Task 1's TDD/re-review discipline for a code fix.

- [ ] **Step 5: Commit only the prepared docs**

Run:

```bash
git add \
  CONTEXT.md \
  docs/run_evidence.md \
  docs/debugging.md \
  torchtitan/observability/structured_logger/README.md
git diff --cached --check
git diff --cached --stat
git commit -m "Document core run evidence contract"
```

Completion criterion: the documentation commit contains exactly four files;
the branch worktree is otherwise clean except ignored smoke/report files.

---

### Task 3: Independently Review the Documentation Task

**Files:**
- Review: the four Task 2 documentation files
- Reference: the final implementation and fresh smoke report

**Interfaces:**
- Consumes: the Task 2 documentation commit and smoke evidence.
- Produces: a task-scoped spec/quality verdict before broad branch review.

- [ ] **Step 1: Generate a docs-only review package**

Task 2's docs commit must be the current tip. Generate its exact one-commit
package:

```bash
docs_head="$(git rev-parse HEAD)"
docs_base="$(git rev-parse HEAD^)"
git diff --name-only "$docs_base..$docs_head" | sort
bash /data02/home/philip.yang/.codex/plugins/cache/openai-curated-remote/superpowers/6.2.0/skills/subagent-driven-development/scripts/review-package \
  docs/superpowers/plans/2026-08-12-core-run-evidence-foundation.md \
  "$docs_base" \
  "$docs_head" \
  .superpowers/sdd/2026-08-12-core-run-evidence-foundation/review-task-6-docs.diff
```

The name-only output and review package must contain exactly `CONTEXT.md`,
`docs/debugging.md`, `docs/run_evidence.md`, and
`torchtitan/observability/structured_logger/README.md`. Stop and reconcile the
range if any other path appears.

- [ ] **Step 2: Dispatch a read-only docs reviewer**

Require checks for:

```text
actual schema fields by record type
identity precedence and elastic suffixing
config callable snapshot boundary
append-only versus immutable wording
path normalization and lifecycle transitions
native producer kinds and async checkpoint durability
local_tensor and Flight Recorder limitations
adoption scope and TorchFT nuance
security/secret caveats
complete deferred-tool/Tier-0/GPU/Qwen boundary
working relative links
```

The reviewer must cite file:line evidence and return `Approved` or
`Needs fixes`.

- [ ] **Step 3: Fix and re-review every blocking finding**

Make documentation-only corrections, rerun link/codespell/diff checks, commit a
separate fix, and give only the fix delta to the same reviewer. Use
`git diff --check` and the exact pre-commit command from Task 2 Step 4 so the
`codespell` and `lychee-link-checker` hooks run on the corrected docs.
Completion criterion: no Critical or Important docs finding remains.

---

### Task 4: Whole-Branch Standards and Spec Review

**Files:**
- Review: every path in
  `git diff --name-status cec1afd76ec7935040e149583ef7294defc9172a..HEAD`
- Compare: `AGENTS.md`
- Compare: `docs/superpowers/specs/2026-08-12-training-research-vehicle-design.md`
- Compare: `docs/research/2026-08-12-training-observability-paper-closure.md`
- Compare: both core run-evidence plans

**Interfaces:**
- Consumes: all reviewed task commits and Task 2 smoke evidence.
- Produces: independent Standards and Spec verdicts for the whole milestone.

- [ ] **Step 1: Use the `code-review` skill from the fixed base**

Run a two-axis review over:

```text
base: cec1afd76ec7935040e149583ef7294defc9172a
head: current codex/core-run-evidence HEAD
```

Standards axis must check:

```text
core versus experiment dependency direction
config ownership and all to_dict callsites
no optional external dependency in core
structured logger cleanup and compatibility
append-only concurrency and outcome immutability
checkpoint/TorchFT compatibility and exception precedence
launcher shell quoting/fail-closed identity
native producer path/format preservation
test quality and maintainability of the large serializer/evidence additions
```

Spec axis must check:

```text
shared identity and manifest equality
per-process index/outcome schema
event/artifact correlation
all required native artifact producers
async checkpoint durability semantics
disabled/no-op behavior
operator documentation and security caveats
explicit exclusion of external tools/triggers/overhead/GPU/Qwen certification
```

- [ ] **Step 2: Resolve all Critical and Important branch findings**

Use a separate TDD fix commit for code and a separate docs fix commit for prose.
Rerun the owning suite for each fix and request a scoped re-review before
returning to the branch gate.

- [ ] **Step 3: Record minor findings explicitly**

Either fix each minor finding or record it in the SDD ledger with a precise
reason and next milestone owner. Do not silently omit reviewer minors.

Completion criterion: both Standards and Spec axes approve the current HEAD.

---

### Task 5: Fresh Final Verification

**Files:**
- Verify: all changed production, test, launcher, and documentation paths

**Interfaces:**
- Consumes: review-clean branch HEAD.
- Produces: fresh completion evidence and a clean integration choice.

- [ ] **Step 1: Re-run the Task 2 CPU owning suite**

Run the exact pytest command from Task 2 Step 2 after all review fixes. Read the
complete output and record exact pass, skip, warning, and failure counts.

- [ ] **Step 2: Run a fresh final-HEAD bootstrap smoke**

Run the exact shell block and inline validator from Task 2 Step 3 again after
all Task 3/4 review fixes. It must allocate another unique
`outputs/run_evidence_local_tensor_smoke.XXXXXX` directory; do not reuse or
delete any earlier bundle. Record the literal new path and validator summary as
the final-HEAD smoke in the Task 6 report.

- [ ] **Step 3: Re-run shell and whitespace checks**

Run:

```bash
bash -n run_train.sh
bash -n multinode_trainer.slurm
git diff --check cec1afd76ec7935040e149583ef7294defc9172a..HEAD
git status --short
```

Expected: both syntax checks and diff check exit zero; tracked worktree is
clean.

- [ ] **Step 4: Re-run full changed-file pre-commit**

Use the exact pre-commit and direct Pyrefly commands from Task 2 Step 4. If the
adapter cannot run, record that as a verification limitation and require zero
diagnostics from the direct check.
If any review fix introduced a path absent from that command, discover it with
`git diff --name-only cec1afd76ec7935040e149583ef7294defc9172a..HEAD`
and include it in this final pre-commit run.

- [ ] **Step 5: Validate the final claims checklist**

The handoff may claim only:

```text
core run-evidence schema/lifecycle implemented
existing native artifacts indexed
launcher propagation behavior tested for NGPU 2/4/8 under stubs
CPU local_tensor bootstrap smoke succeeded
available config families serialize deterministically across independent processes
owning unit/static checks passed with exact reported limitations
```

It must also state:

```text
no real GPU training was run
no 2/4/8 scale or performance certification exists
no Qwen numerical/convergence/checkpoint-resume result exists
no Tier-0 overhead measurement exists
no external diagnostic adapters/triggers/incidents/fault drills exist
```

- [ ] **Step 6: Update the SDD ledger and session plan**

Record Task 6 documentation/review completion, config-hardening review
completion, final verification counts, and the final whole-branch review
verdict. Record every preserved smoke attempt, including at minimum:

```text
DVEqcq: failed; exposed relative structured-path identity mismatch
sxUZMD: succeeded; exposed process-address-bearing config serialization
the literal directory printed by Task 2 Step 3: post-797ef56 smoke succeeded
the literal directory printed by Task 5 Step 2: final-HEAD smoke succeeded
with the final schema/serializer checks
```

Update the executing session's plan state; do not edit either committed plan
merely to mirror checkbox state. Mark the milestone complete only when every
preceding criterion is satisfied.

---

### Task 6: Present the Branch Integration Choice

**Files:**
- Preserve: feature branch and worktree until the user chooses

**Interfaces:**
- Consumes: review-clean, freshly verified branch.
- Produces: an explicit human-selected merge, PR, or preservation action.

- [ ] **Step 1: Detect the worktree environment**

Run:

```bash
git_dir="$(cd "$(git rev-parse --git-dir)" && pwd -P)"
git_common="$(cd "$(git rev-parse --git-common-dir)" && pwd -P)"
worktree_path="$(git rev-parse --show-toplevel)"
printf 'git_dir=%s\ngit_common=%s\nworktree=%s\n' \
  "$git_dir" "$git_common" "$worktree_path"
git -C /data02/home/philip.yang/workspace/torchtitan branch --show-current
git -C /data02/home/philip.yang/workspace/torchtitan status --short
```

This is a named feature branch in a linked worktree. The repository's main
checkout is currently occupied by unrelated branch
`scaffold-wave-f0-execution-foundation` and contains seven uncommitted
scaffold-to-policy files. Do not switch, merge, stash, clean, or reset that
checkout.

- [ ] **Step 2: Present exactly the finishing choices**

Present:

```text
Implementation complete. What would you like to do?

1. Merge back to main locally
2. Push and create a Pull Request
3. Keep the branch as-is (I'll handle it later)

Which option?
```

Wait for the user's explicit choice. Do not merge, push, remove the worktree,
or delete the branch merely because verification passed.

For choice 1, "merge back to main locally" never means checking out `main` in
the dirty root checkout. The `superpowers:using-git-worktrees` skill correctly
declines to nest another isolation worktree from this already isolated checkout,
so use this explicit existing-branch path only after the user chooses option 1:

```bash
(
set -euo pipefail
integration_path=/data02/home/philip.yang/workspace/torchtitan/.worktrees/core-run-evidence-main-integration
case "$integration_path" in
  /data02/home/philip.yang/workspace/torchtitan/.worktrees/*) ;;
  *) printf 'unsafe integration path: %s\n' "$integration_path" >&2; exit 1 ;;
esac
if test -e "$integration_path"; then
  printf 'integration path already exists: %s\n' "$integration_path" >&2
  exit 1
fi
git show-ref --verify --quiet refs/heads/main
integration_base="$(git rev-parse refs/heads/main)"
if ! git merge-base --is-ancestor \
  cec1afd76ec7935040e149583ef7294defc9172a "$integration_base"; then
  printf 'main is not descended from the reviewed base: %s\n' \
    "$integration_base" >&2
  exit 1
fi
git worktree add "$integration_path" main
test -z "$(git -C "$integration_path" status --porcelain)"
test "$(git -C "$integration_path" rev-parse HEAD)" = "$integration_base"
printf 'clean integration worktree=%s base=%s\n' \
  "$integration_path" "$integration_base"
)
```

If `integration_base` differs from
`cec1afd76ec7935040e149583ef7294defc9172a`, review the intervening `main`
changes for compatibility before merging. Then run:

```bash
cd /data02/home/philip.yang/workspace/torchtitan/.worktrees/core-run-evidence-main-integration
git merge --no-ff codex/core-run-evidence
```

Remain in that exact integration-worktree directory while rerunning the Task 5
pytest, shell, diff, pre-commit, and direct-Pyrefly gates. For the Task 5 Step 2
smoke, copy the Task 2 Step 3 block but replace its hard-coded `cd` line with:

```bash
cd /data02/home/philip.yang/workspace/torchtitan/.worktrees/core-run-evidence-main-integration
```

Leave every other smoke/validator line unchanged. This ensures the post-merge
checks exercise merged `main`, not the feature worktree. Do not remove the
integration worktree until the merge result is verified and the user approves
cleanup. If any guard fails, stop and ask the user to choose push/PR or branch
preservation.

---

## Next Milestone After This Branch Lands

Create a separate plan and branch for GPU certification; do not append it to
this evidence-contract branch. Its order is:

1. Complete the typed incident/progress envelope that was deliberately
   deferred from this run-artifact foundation.
2. Establish Qwen3 0.6B/1.7B one-step validation, then matched 2/4/8-GPU
   numerical, checkpoint-resume, artifact-volume, and Tier-0 overhead gates,
   keeping global batch and tokens per optimizer step fixed.
3. Add stable phase/module/collective correlation and repo-local offline
   analyzers for per-rank phase skew, Flight Recorder/mesh joins, incident
   timelines, checkpoint lineage, and cumulative/windowed ETTR.
4. Add launcher-owned adapters for DCGM/job evidence, py-spy, NCCL RAS/logs,
   and stopped-job diagnostics; index results locally and keep fleet action
   outside TorchTitan.
5. Run the full perturbation gate: hangs, rank death, compute/dataloader
   stragglers, NaN/Inf, checkpoint interruption/corruption, restore
   equivalence, inconsistent per-rank config, and DCGM/EUD hardware drills.
6. Promote to Qwen3 8B and 30B-A3B MoE only after the small-model evidence,
   analyzer, adapter, overhead, and perturbation gates pass.
7. Only then begin the separate multimodal program, followed by the separate
   full post-training program.

The first GPU plan must treat launcher stub coverage as plumbing evidence only;
it starts with no existing GPU certification claim.

Preserve the capture-first escalation model established by the bounded paper
closure:

```text
Tier 0, always on: structured evidence, Flight Recorder buffers, DCGM fields
Tier 1, scheduled: PyTorch Profiler traces and CUDA memory snapshots
Tier 2, anomaly-triggered: Flight Recorder dumps, py-spy, NCCL RAS evidence
Tier 3, stopped-job/deep-dive: DCGM EUD, nccl-tests, SuperBench, Nsight, replay
```

ByteRobust informs the incident/evidence control-plane pattern. Fault-tolerant
HSDP informs later per-step membership, commit, and recovery state. Neither
paper is evidence that those mechanisms exist on this branch.

The larger research-vehicle program remains ordered as follows; each phase
needs a separate design, plan, and promotion gate:

1. Foundation-model training: validate Qwen3 0.6B/1.7B, then promote to Qwen3
   8B and 30B-A3B MoE on the 2/4/8-GPU ladder.
2. Multimodal: establish Qwen3-VL, then investigate the requested Qwen3.5/3.6
   native vision-language path; separately fine-tune SigLIP2 and train vision
   token projection plugins for Qwen3 and, where feasible, DeepSeek-v4 or
   GLM-5.2 model families.
3. Post-training: build SFT, rejection-sampling, and RL stages with immutable
   dataset/checkpoint/evaluation lineage. Keep the existing online RL
   controller distinct from offline generation/SFT/evaluation programs.
