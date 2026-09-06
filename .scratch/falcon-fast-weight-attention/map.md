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
- Spec approved: `spec.md`. Tickets `03`–`09` filed.
- Step 2 identity: recurrent ≡ masked-parallel for Falcon-1/1A × delayed/same-step (fp32). Falcon-1 residual writes stay an O(L) oracle; chunk-parallel is still excluded.
- Step 4 winner (ticket 07): **`mixer=falcon, variant=falcon1a, alignment=delayed, phi=l2` (A5)** — best Falcon PPL (~113), best addition ID acc (~0.50), within ~6 PPL of softmax control, beats GDN neighbor. Frozen for ticket 08.

## Frontier

`07` (Step 4 ablations) **resolved**: A5 knob + GDN `K**-0.5` landed via TDD;
meta-init RMSNorm-gain bug fixed (was collapsing arms to identical loss);
70 Falcon tests green. 2-seed 8000-step table on one B200 — A0/A2/A4/A5 clean;
A1 (gdn) and A3 (falcon1) are O(L) neighbor/residual paths run separately.
Winner A5 frozen. `08` (hero run) is next: freeze A5 knobs, scale only.

## Fog

- A5 phi knob and GDN `K**-0.5` q-scale are 07 leftovers from 06.
- Step 4: 8k steps, seq 512, GBS 32, 16,384 tok/step, 131M tokens/arm.
- Step 5: 20k-40k steps, same tokens/step, 328-655M tokens/run.
- Disk 44G free: no checkpoints on ablation dumps.
