# 20 — Keep host paths out of the evidence identity chain

**What to build:** Record stage commands in a form that does not embed absolute
host paths, so a sealed bundle can be scrubbed or published without breaking its
own digests.

**Blocked by:** none.

**Status:** resolved -- `HEAD_COMMIT_PATHS_EVIDENCE_PATH` keeping host paths out
of the identity chain landed in `47ebae4d9`. This line read `ready-for-agent`
until a status audit corrected it; the work had been committed for some time and
the ledger was advertising it as available.

## The problem

`qfv_append_stage` (`experiments/qwen3_formal_verifier/runner_lib.sh`)
records the full stage command, which contains absolute host paths. Verified in
the current bundle:

```
checker/stages.tsv         8
manifests/stages.json      8
```

`stage_id` is computed over those stage records and `evidence_id` over
`stage_id`, so the host path is load-bearing for the identity chain. Scrubbing
the bundle invalidates it; keeping it unscrubbed means it can never be
published.

`scripts/check_no_pii.py` treats `/data02/home/<user>` as exactly the token to
scrub, and commit `dbebc877f` exists to keep it out of the evidence tree. So the
evidence contract and the PII guard currently pull in opposite directions.

Nothing is leaked today -- `.gitignore:6` ignores `outputs` and
`git ls-files outputs` is empty -- which is why this is a design fix rather than
an incident.

## Acceptance criteria

- [ ] Stage commands are recorded relative to a declared root, or with the host
  prefix replaced by a stable placeholder, before the digest is computed. The
  recorded form must still identify what ran.
- [ ] `check_no_pii.py` passes over a sealed bundle.
- [ ] Digests are stable across two runs from different host paths with
  everything else equal. That is the real test: if the identity chain still
  moves when only the checkout location changes, the fix is incomplete.
- [ ] The exit-status column in `stages.tsv` is either recorded from the real
  status or removed. It is currently the literal `0`
  (`printf '%s\t0\t%s\t%s\n'`). The journal fails closed anyway, because
  `run_logged` returns before appending when a stage fails and
  `_read_stage_journal` requires every declared stage to be present -- so this
  is not a hole, but a reader of `stages.tsv` sees a status that was never
  measured.
