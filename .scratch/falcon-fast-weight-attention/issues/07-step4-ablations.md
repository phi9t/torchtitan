# Step 4: small-model ablation table

Type: task
Status: claimed
Blocked by: 06
Parent: ../spec.md

## Requirements

- Run arms A0–A5 from the spec on the science model with matched
  data, global batch, tokens/step, and steps.
- Optional A6/A7 only if A2/A3 are healthy and cheap.
- Metrics: held-out CE/PPL; addition ID and OOD teacher-forced acc
  from separate small addition trains (spec default).
- Replicate 2 seeds minimum (3 if wall-time allows). Deltas inside
  seed noise are ties.
- Freeze one mixer+alignment+`phi` triple for ticket 08. Record the
  table even if Falcon ties GDN on PPL and wins only on addition.

## Exclusions

- Changing knobs after seeing hero-scale results
- Calling the table a 130M / 50B result
- New mixer variants not in A0–A7

## Verification

- Ablation report with `claim_label=representative_small` and
  `replicated_eval`.
- Winner written into the ticket Answer and `map.md`.
- Step 3 overfit still green on the delayed default.

## Answer

Ablation table run and frozen. Claim `representative_small + replicated_eval`
(LM arms), one B200, sequential arms, `checkpoint.enable=False`, all
Python/CUDA in the bwrap rootfs, no data fetched, no commit. Full report:
`../sdd/task-07-report.md`; tables (gitignored) under
`experiments/falcon/results/ablations/`.

Closed 06 leftovers first (TDD): the `phi` (`rms`/`l2`) knob is threaded from
`FalconConfig` into the mixer (A5 is now a real arm; `test_falcon_phi.py`), and
GDN `q` is scaled by `K**-0.5` after L2 to match `kda_reference_forward`.
Found and fixed a meta-init bug: `_initialize_weights` left
`FalconRMSNorm.weight` at zero after the Trainer's `to_empty()`, which
collapsed every mixer to the same (lm_head-only) trajectory and produced
identical losses across arms; the reinitializer now restores the RMSNorm gain
to ones (`test_falcon_meta_init.py`). 70 Falcon unit tests green in rootfs.

Held-out val PPL on `fineweb_val_000000.bin` (mean over seeds, 8000 steps):
A0 softmax ~107 (control), **A5 falcon1a/delayed/L2 ~113 (best Falcon arm)**,
A4 same-step/RMS ~121, A1 gdn ~121, A2 delayed/RMS ~122. Addition ID acc
(N=16/M=24): **A5 ~0.50** vs A4 ~0.06, A2 ~0.04, A0 ~0.004; OOD acc 0.000 for
all arms at this scale (declared gap, not a win). Alignment (A4 vs A2) is a tie
inside seed noise, so delayed pairing alone buys nothing; the phi choice moves
both PPL and addition.

**Winner frozen for ticket 08: `mixer=falcon, variant=falcon1a,
alignment=delayed, phi=l2` (A5).** It has the best Falcon PPL, by far the best
addition ID accuracy, and stays within ~6 PPL of the softmax control (no
"only works on addition" regression) and beats the GDN-style neighbor on both
PPL and addition.

Coverage note: A1 (gdn) and A3 (falcon1) use O(L) per-timestep scans and are
~30-40x slower than the masked-parallel arms; they were run in a separate
sequential single-B200 invocation. See the report for their exact seed/step
coverage.
