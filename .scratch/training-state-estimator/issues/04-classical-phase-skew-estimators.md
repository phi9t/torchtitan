# Issue 04: Classical Phase and Peer-Skew Estimators

Status: open
Type: task
Blocked by: 03

## Intent

Add the first deterministic estimator pass over the evidence graph:
phase-duration summaries and peer-relative skew findings.

This ticket is the first useful diagnostic output. It should classify symptoms
conservatively and avoid pretending heuristic scores are calibrated
probabilities.

## Acceptance Criteria

- Read derived evidence graph and timeline artifacts.
- Emit `derived/state_estimator/belief_summary.json` with:
  - phase-duration summaries;
  - per-rank progress and liveness summary;
  - peer-relative skew findings;
  - evidence-quality warnings;
  - advisory `claim_calibration="heuristic"` or equivalent.
- Emit `derived/state_estimator/diagnosis.md` with human-readable findings.
- Cover fixtures for:
  - no skew;
  - one late rank;
  - failed process outcome;
  - missing timing fields.
- Do not identify a root cause when available observations cannot distinguish
  competing explanations.

## Non-Goals

- Collective-internal flow diagnosis.
- Active probing.
- Learned likelihoods.
- Automatic remediation.

## Verification

Run:

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 -m pytest -q tests/unit_tests/observability/state_estimator/test_estimators.py'
```
