---
name: running-experiment-campaigns
description: Use when starting, steering, or closing a TorchTitan research campaign, or when the user mentions Mercor playbook, 6-step de-risk, science ablations, hero run, transfer eval, claim labels, campaign tickets, or map/progress ledgers.
---

# Running experiment campaigns

Mercor-shaped **de-risk → ablate → freeze → scale → transfer**.
Keep the invariants. Drop the instance (Harbor, SkyRL, Megatron, APEX)
unless this campaign *is* that instance.

A campaign spec owns domain specialization. This skill owns sequencing,
claim honesty, and the dynamic next-ticket loop. If they conflict on
*what to measure*, the spec wins. If they conflict on *when to spend
science compute*, this skill wins.

**Core rule:** Steps 1–3 spend almost no science compute. An ablation or
hero while the harness or overfit gate is red is a process bug.

## When to use

- New mixer / recipe / post-training campaign.
- "Follow Mercor", "science table", "hero run", "which ticket next".
- A table looks too good (identical losses) or too expensive (one arm
  blocks the rest).
- Someone wants to call a small run "reproduction" or "convergence".

Live GPU attach: `attaching-live-training`. Bitwise drift:
`numerics_debugging`.

## Invariants (from Mercor, kept)

1. Harness and learnability before science tokens.
2. Small fixed set cannot overfit → walk back. Do not buy more tokens.
3. Ablate on the small model. Freeze knobs. Hero is not a search.
4. Eval is noisy. Replicate. Deltas inside seed noise are ties.
5. Read behavior, not only the scalar (traces, state, train vs held-out).
6. Transfer is a first-class gate. Train-harness-only gains do not close.
7. Classify claims. Smoke ≠ representative. Local hero ≠ paper table.

Specialize the *objects* (env, identity gate, overfit set, knobs, eval).
Do not specialize away the order.

## Dynamic loop

The next action is computed from the ledger, not from memory.

1. Find the campaign dir (usually `.scratch/<name>/`).
2. Read `spec.md`, `map.md`, `sdd/progress.md` (and the frontier ticket).
3. Run the workflow in [workflow.md](workflow.md): classify step, test
   that step's exit, then stay / walk back / freeze / open next.
4. Isolated ticket agents get a brief with requirements, exclusions,
   blockers, and verification. They do not invent a new Step 4 on the
   hero path.
5. Update `progress.md` + `map.md` from the report, not from chat.

## Steps 1–6

Exit criteria are binary. Partial tables do not advance the step.

| Step | Goal | Exit (all required) | Allowed `claim_label` |
| --- | --- | --- | --- |
| 1 Harness | Train the thing you think you train | Tests for the pairing/labels/token contract; target-shape dry-run clean; harness note | `smoke`, `deterministic_pairing` |
| 2 Systems | Views and loop are trustworthy | Identity gate (oracle ≡ fast path); finite-loss Trainer smoke; no hang | `deterministic_comparison`, `smoke` |
| 3 Overfit | Architecture can memorize | Fixed bank CE collapses; `lr=0` fails; re-run after any mixer edit | `learnability_smoke` |
| 4 Ablate | Pick knobs at small width | Matched arms, 2+ seeds, held-out + transfer, arms **diverge**, winner frozen | `representative_small`, `replicated_eval` |
| 5 Hero | Scale only | Frozen knobs, matched controls, preflight budget, no new variants | `representative_training` |
| 6 Close | Transfer + no-regression | Replicated held-out + task-swap; written table; campaign claim | `replicated_eval`, `transfer_eval` |

Mercor: "We do not spend significant compute until Step 4." Same here.

## Falcon-hardened rules

These failed in the Falcon FWA campaign. Treat them as campaign law.

**Silent collapse.** `meta` → `to_empty()` → incomplete `init_weights`
left RMSNorm gains at zero. Every mixer then produced the **same**
loss. A table of identical curves is invalid. Before a long table:
one short run per arm must diverge, and a `to_empty()` init test must
exist if the Trainer uses meta-init.

**Identity ≠ train-stable.** Masked-parallel matched the recurrent
oracle and still NaN'd in backward (`inf * 0` on the masked triangle).
Identity tests need a backward/finite-grad check at the science
`seq_len`, not only a forward allclose.

**O(L) arms off the critical path.** Python scans (GDN residual,
Falcon-1 forward-substitution) were 30–40× slower. Vectorize, or run
them as a side table. Do not stall the matched-token table on them.
Omit-for-walltime with a reason; do not report a partial step count
as a matched arm.

**Dry-run the whole matrix** at the science shape (few steps, all
arms) before 8k-step seeds. Plumbing only. Then spend tokens.

**Invalidate and rerun** when a harness/init bug is found. Do not
"interpret" the collapsed table.

**Winner rule is declared before looking.** Falcon froze
`falcon1a + delayed + l2` (A5): best Falcon PPL, best addition ID,
within ~6 PPL of softmax, beat GDN. Alignment (delayed vs same-step)
was a **tie**. Addition ID was seed-noisy (0.70 vs 0.29); OOD was 0
everywhere — reported as a declared gap, not a win. Hero must not
reopen knobs because of that fog.

**Claim the scale you ran.** 131M tokens/arm is
`representative_small + replicated_eval`. It is not Table 1, not
49.2B, not generation quality.

Rootfs for all Python/CUDA. No science fetch. Checkpoints only if
disk preflight allows. No commit unless asked.

## Claim labels

| Forbidden | Say instead |
| --- | --- |
| 10-step Trainer "converged" | `smoke` |
| Overfit gate "quality" | `learnability_smoke` |
| Local hero "Table 1" | `representative_training` at declared tokens |
| Teacher-forced acc "generation" | `transfer_eval`, teacher-forced |
| One seed "replicated" | `replicated_eval` needs 2+ |

## Common mistakes

| Move | Instead |
| --- | --- |
| Start FineWeb while overfit is red | Walk back to Step 1 |
| Search knobs on the hero | New Step 4, freeze again |
| Kitchen-sink ablation | Paper's actual claims, one knob each |
| Single-pass delta as a win | Seed/split mean; else tie |
| Train-task-only close | Transfer gate |
| Copy Mercor's Harbor/SkyRL into a mixer campaign | Keep invariants, drop instance |
| Let chat be the ledger | `map.md` + `progress.md` |

## Sources

- Mercor × SkyRL, *Training frontier knowledge work agents* (2026-09-01).
- Falcon instance: `.scratch/falcon-fast-weight-attention/spec.md`.
- Worked collapse / O(L) / winner:
  `.scratch/falcon-fast-weight-attention/sdd/task-07-report.md`.
- Ticket loop: [workflow.md](workflow.md).
- Claim language: `AGENTS.md` research promotion evidence.
