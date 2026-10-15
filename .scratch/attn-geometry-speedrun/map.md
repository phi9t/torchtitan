# Map
- Step 1 Harness: A01/A01b green (arm implementation + equivalence tests + small-model config + recall probe)
- Step 2 Systems: A01b green (host-verified FineWeb, all-arm dry-run, finite loss, timing fields)
- Step 3 Overfit: A01b green (fixed-bank overfit per arm, lr=0 controls, recall-after-train execution)
- Step 4 Ablate: A02 Revision 2 (36 non-anchor screen + 2 anchor, optional rescue, then 4 arms x 3 confirmation),
  now unblocked by A01b
- Step 5/6: scale-up / transfer -- not opened

# Frontier
- A01b is green. Report: `sdd/a01b-report.md`.
- A02 can start from `sdd/a02-brief.md`: implement Revision-2 pre-spend requirements, rerun focused tests and all-arm
  50-step dry-runs, then run A02a screen/rescue/A02b confirmation per the brief.

# Blockers / Fog
- Waystone remains host-side only; keep using host `hdfs_strict.sh --quick` before each launch.
- FineWeb tokenizer fetch failed under DNS during A01b; artifacts record `offline_fallback` for FineWeb runs.
- A01b recall probes ran after overfit but scored 0/16 at distances 128, 1024, and 1536 for every arm; treat this as
  behavior evidence for A02, not as a closed recall-quality claim.
