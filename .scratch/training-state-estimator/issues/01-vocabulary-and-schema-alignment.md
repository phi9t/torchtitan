# Issue 01: Vocabulary and Schema Alignment

Status: open
Type: task
Blocked by:

## Intent

Define the first durable vocabulary and derived-artifact schema for the
training state estimator, grounded in the existing `run_evidence` contract.

This issue should make the estimator language precise before implementation:
belief state, evidence graph, topology epoch, observation, sensor health, fault
mode, transaction risk, and probe recommendation.

## Acceptance Criteria

- Add a concise schema/vocabulary document under the tracker or docs tree that
  maps estimator terms to existing `docs/run_evidence.md` fields.
- Define the initial derived artifact names and top-level JSON fields for:
  `evidence_graph.json`, `timeline.jsonl`, `entities.json`, `quality.json`,
  `belief_summary.json`, and `diagnosis.md`.
- Explicitly distinguish raw evidence from derived artifacts.
- Explicitly distinguish missing, unknown, uncollected, malformed, and
  contradictory evidence.
- Record the safety boundary: estimator output is advisory and may not change
  optimizer commit, checkpoint validity, membership, or recovery behavior.

## Non-Goals

- Implementing analyzer code.
- Adding learned-model interfaces.
- Changing the core `run_evidence` recorder.
- Changing training control flow.

## Verification

Run:

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 -m py_compile scripts/rootfs/verify_runtime_env.py'
```

If only Markdown files change, also run a lightweight static check such as:

```bash
git diff --check
```
