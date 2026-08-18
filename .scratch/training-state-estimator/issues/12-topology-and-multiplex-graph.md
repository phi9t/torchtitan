# Issue 12: Topology And Multiplex Graph

Status: resolved
Type: task
Blocked by: 11

## Intent

Deepen the derived evidence graph into a multiplex graph with distinct
physical, logical, and control layers plus topology epochs.

## Acceptance Criteria

- Add `torchtitan/observability/state_estimator/topology.py`.
- Represent physical, logical, and control graph layers separately.
- Add host, process, rank, device, mesh-axis, process-group, artifact,
  checkpoint, incident, and sensor entities when evidence exists.
- Add topology epoch support and deterministic entity/edge ordering.
- Encode known relationships without guessing absent relationships.
- Emit `unknown` or `uncollected` quality findings for missing topology and
  state which diagnoses are blocked by that missing evidence.
- Preserve existing v1 graph outputs for consumers that do not use layers.

## Non-Goals

- Real-time topology discovery.
- Mandatory DCGM, NCCL, or external topology tool dependency.
- Dense graph library dependency.
- Fault-mode inference.

## Subagent Trace Requirement

Execute with fresh implementer, reviewer, and finalizer subagents. The finalizer
must inspect both previous traces for topology ambiguity, missing producer
assumptions, and whether the graph boundary should be split further.

## Verification

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 -m pytest -q tests/unit_tests/observability/state_estimator/test_topology.py tests/unit_tests/observability/state_estimator/test_graph.py'
```

## Comments

### 2026-08-18 Issue 12 execution

- Implementer: `01a01569-a70b-7062-b397-59c33e36b668` (`Goodall`).
- Reviewer: `01a01570-923d-7be2-83ba-79bee3a418e6` (`Sagan`), requested changes for same-file observation key collisions, dropped topology epochs, and duplicated loader quality.
- Fix implementer: `01a01572-80fa-7142-899b-b354c9c300b3` (`Mencius`).
- Scoped re-reviewer: `01a01577-c006-7821-b493-f1efc95a737d` (`Laplace`), approved the three fixes.
- Finalizer: `01a0157a-5641-74c0-8fd1-6a18ecb0e97c` (`Hypatia`), accepted Issue 12.
- Verification: `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 -m pytest -q tests/unit_tests/observability/state_estimator/test_topology.py tests/unit_tests/observability/state_estimator/test_graph.py'` -> `13 passed in 0.17s`.
- Finalizer note: multiplex output is additive through `build_multiplex_evidence_graph`; `write_evidence_graph()` remains legacy v1 unless a later schema migration ticket changes that. Future consumers should clarify whether layer edges may cross-reference entities owned by another layer.
