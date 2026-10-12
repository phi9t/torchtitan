# Ticket 08 report -- Step 5 local hero (complete)

Claim label: `representative_training`. This is **not**
Table 1 / 49.2B. No commit.

## What ran

- Science shape, frozen Step 4 knobs, FineWeb 10B GPT-2 bins, seq 512,
  GBS 32, **16,384 tok/step**, 20,000 optimizer steps = **328M tokens/arm**.
- Three B200s in one rootfs-launched process each (`--share-pid`):
  A5 GPU0, A0 GPU1, A1 GPU2. Tag `hero_20k`.
- Checkpoints on: `interval=500`, `keep_latest_k=2`,
  `last_save_model_only=False`. Disk floor 5G (host had 44G at launch).
- Pause/resume: `{dump}/PAUSE` or SIGINT/SIGTERM finishes the current
  step and writes a full train-state checkpoint. Re-run the same command
  to resume. Proven by
  `tests/unit_tests/test_falcon_hero_resume.py` (1 passed, rootfs).
- Dry-run smoke (tiny shape, 4 steps, all finite, losses diverge):
  A5 10.84->10.74, A0 10.84->10.54, A1 10.85->10.72.

## Exit

| arm | mixer | steps | train loss | val CE | val PPL | wall | status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| A0 | softmax | 20000 | 10.86->4.14 | 4.4731 | 87.63 | 894.08s | done |
| A5 | falcon1a delayed L2 | 20000 | 10.88->4.23 | 4.5543 | 95.04 | 1166.27s | done |
| A1 | gdn | 20000 | 10.88->4.28 | 4.6000 | 99.48 | 36525.88s | done |

Artifacts: `experiments/falcon/results/hero/hero_20k/{arm}_seed0/`.

## Frozen decision

Nothing new frozen. A5 knobs stay the Step 4 winner. A5 hero PPL is
still in-band with softmax (~7 PPL behind, same direction as Step 4's
~6 PPL gap at 8k). Do not retune on this disagreement.

## Honesty

- A1 is O(L) and took 31.32x as long as A5. It nevertheless completed, so
  Ticket 08 is resolved.
- One seed only (ticket 08). Closure owns multi-region checkpoint evaluation
  and replicated addition draws; no new hero training is implied.
- `keep_latest_k=2` plus a final last-step save can leave three step
  dirs at completion (~1G/arm). Acceptable vs the 5G floor.

## Completed checkpoint inspection

Each arm has `step-19000`, `step-19500`, and `step-20000` after retention plus
the final save. The outcome and status JSONs report `completed_steps=20000`,
`running=false`, and `paused=false`.

## Evidence limitations discovered at recap

The compact historical JSON predates the full run-attempt contract and omits
`lane`, `mode`, `evidence_tier`, `run_id`, `attempt_id`, and
`environment_class`. Raw files remain unchanged. The approved closure work
will import them into a canonical ledger with hashes and explicit
`legacy_import` gaps rather than fabricate missing provenance.
