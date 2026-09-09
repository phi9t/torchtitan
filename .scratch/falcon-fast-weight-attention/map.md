# Falcon FWA campaign map

## Notes

- Paper: arXiv:2608.27763, mixer family Falcon-1/2/3 + A variants.
- Playbook: Mercor × SkyRL 6-step de-risk / ablate / hero / transfer
  (`.claude/skills/running-experiment-campaigns/SKILL.md`).
- Domain shift: agentic RL → causal LM mixer. Keep invariants, drop Harbor/SkyRL/TITO-chat.
- Selected campaign: **B** process-faithful local (approved 2026-09-05).

## Decisions-so-far

- Attachment is `torchtitan/experiments/falcon/`, not Mini-K3 and not core Qwen3.5.
- Step 3 overfit done: CE 3.47 → 0.011 on an 8-sequence bank for Falcon-1 and 1A; `lr=0` fails.
- Campaign B locked: local hero at preflight-approved budget. Paper-scale 49.2B is out of this campaign.
- Spec approved: `spec.md`. Campaign B tickets `03`-`13` and separate
  mechanism tickets `01`-`05` are filed.
- Step 2 identity: recurrent ≡ masked-parallel for Falcon-1/1A × delayed/same-step (fp32). Falcon-1 residual writes stay an O(L) oracle; chunk-parallel is still excluded.
- Step 4 winner (ticket 07): **`mixer=falcon, variant=falcon1a, alignment=delayed, phi=l2` (A5)** -- best completed Falcon PPL (~113), within ~6 PPL of softmax control. Its ~0.50 addition result is training-bank accuracy, not held-out transfer. Frozen for ticket 08.
- A1 has one complete Step-4 LM seed and no addition run; A3 has no complete full run. The research owner accepted both as explicit performance-based omissions rather than authorizing more O(L) reference compute.
- Step 5 completed all three 20k / 327.68M-token local heroes: A0 PPL 87.63, A5 95.04, and A1 GDN 99.48. A5 remains the frozen local Falcon winner; the hero evidence is one seed and is not paper-scale.
- Historical raw evidence stays immutable. Closure will create a canonical import ledger with source hashes and explicit `legacy_import` gaps.
- New scale, alignment, and state-dynamics work belongs to the separate approved `../falcon-fwa-mechanism/spec.md`; it does not retune Campaign B.
- Audited recap: `sdd/2026-09-07-campaign-recap.md`. Primary-source alignment:
  `../../docs/research/2026-09-07-falcon-fast-weight-attention-paper-alignment.md`.

## Frontier

The approved Step-6 decomposition is published. `09` (versioned evidence and
legacy import) is resolved with independent review and finalizer acceptance.
Campaign B `10` (disjoint addition) and `11` (fixed-region LM evaluation) are
the unblocked frontiers; `12` blocks on `10`. In parallel, mechanism `01` may
prove the scale transform without GPU science work. Final closeout `13` blocks
on `10`-`12` plus the human-enforced mechanism `05` terminal decision.

## Fog

- Step 5 A5 is ~7.4 PPL behind softmax, the same direction as Step 4. This is a
  finding, not a retuning invitation.
- Existing addition ID results overlap training; no held-out transfer claim is
  supported. Every valid width-OOD result is zero.
- The A5-versus-RMS mechanism remains ambiguous because RMS and L2 also change
  effective feature energy, lambda, write/read scale, and memory horizon.
- The checkout is closest to the paper's Falcon-1A.1 recurrence family, not its
  strongest `ctxeta-ctxlambda` or Falcon-3A variants.
- The filesystem had about 24G free on 2026-09-07. Evidence normalization must
  avoid duplicating raw checkpoints, and every GPU ticket needs an artifact-size
  preflight.
