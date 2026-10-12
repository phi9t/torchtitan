# Step 4: small-model ablation table

Type: task
Status: resolved
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

Ablation table run and frozen. Claim `representative_small`; `replicated_eval`
applies only to the completed A0/A2/A4/A5 LM and addition subsets. One B200,
sequential arms, `checkpoint.enable=False`, all
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

Held-out val PPL on `fineweb_val_000000.bin` (mean over completed seeds,
8000 steps):
A0 softmax ~107 (control), **A5 falcon1a/delayed/L2 ~113 (best Falcon arm)**,
A4 same-step/RMS ~121, A1 gdn ~121 (one seed), A2 delayed/RMS ~122. The
reported addition ID subset overlaps the finite training bank and is therefore
training-bank accuracy, not held-out transfer: **A5 ~0.50** vs A4 ~0.06,
A2 ~0.04, A0 ~0.004. OOD widths are disjoint and accuracy is 0.000 for every
completed arm (declared gap, not a win). Alignment (A4 vs A2) is a tie inside
seed noise, so delayed pairing alone buys nothing; the phi choice moves LM PPL
and training-bank addition fit.

**Winner frozen for ticket 08: `mixer=falcon, variant=falcon1a,
alignment=delayed, phi=l2` (A5).** It has the best Falcon PPL, by far the best
training-bank addition accuracy, and stays within ~6 PPL of the softmax
control. It beats the single completed GDN seed on LM PPL. No GDN addition
result exists, so no Falcon-versus-GDN addition claim is supported.

Coverage note: A1 (gdn) and A3 (falcon1) use O(L) per-timestep scans and are
~30-40x slower than the masked-parallel arms; they were run in a separate
sequential single-B200 invocation. A1 has one completed LM seed and no addition
result; A3 has no completed full run. On 2026-09-07 the research owner accepted
these as explicit performance-based omissions rather than authorizing more
matched-token work on slow reference implementations. This is a documented
scope downgrade from the original literal A0-A5/two-seed requirement, not a
claim that the missing runs completed. See the report for exact coverage.
