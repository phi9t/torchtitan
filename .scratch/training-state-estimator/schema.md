# Training State Estimator Schema

Status: implementation contract

## Raw Versus Derived Evidence

Raw evidence is the immutable run-attempt bundle described in
`docs/run_evidence.md`. Derived evidence is regenerated analyzer output under
`derived/state_estimator/`.

## Evidence State Values

- `present`: the field or artifact exists and parsed successfully.
- `missing`: the expected field or artifact was absent.
- `unknown`: the analyzer cannot infer the value from available records.
- `uncollected`: the producing tool was not enabled or did not claim to collect
  the signal.
- `malformed`: a record or artifact was present but failed schema validation.
- `contradictory`: two immutable records disagree about the same identity or
  event.

## Derived Artifacts

- `evidence_graph.json`: entity and relationship graph.
- `entities.json`: normalized entity index.
- `timeline.jsonl`: ordered observation and transition timeline.
- `quality.json`: missing, malformed, unknown, uncollected, and contradictory
  evidence findings.
- `belief_summary.json`: advisory estimator output.
- `diagnosis.md`: human-readable diagnosis and probe recommendations.
