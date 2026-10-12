# Dynamic campaign workflow

Compute the next ticket from the ledger. Do not resume a remembered
plan if `map.md` disagrees.

## Layout

```text
.scratch/<campaign>/
  spec.md              methodology + specialization (canonical)
  map.md               decisions, frontier, fog
  issues/NN-*.md       one step or leftover each
  sdd/progress.md      ticket status
  sdd/task-NN-brief.md isolated-agent contract
  sdd/task-NN-report.md what ran, what is frozen, claim_label
```

Python/CUDA lives under `torchtitan/experiments/<name>/` (reusable) and
`experiments/<name>/` (runners, gitignored results). Dependency
direction stays `experiments -> core`.

## Classify

Read `progress.md` then `map.md` Frontier. The current step is the
lowest Mercor step whose exit is not recorded as green.

```text
1-3 red?     -> stay in de-risk. No ablation GPU budget.
4 running?   -> dry-run all arms, then matched table. Watch collapse.
4 green?     -> winner frozen in map.md? if no, do not open 5.
5 green?     -> Step 6 tables. Do not add knobs.
6 green?     -> close. New science is a new campaign or a new Step 4.
```

Walk-back triggers (immediate, even mid-table):

- Overfit gate fails after a mixer/harness edit → Step 1.
- Identity or finite-grad gate fails → Step 2.
- All arms share a loss curve → invalidate table, fix init/harness,
  re-dry-run, then rerun Step 4.
- Hero wants a new variant → new Step 4. Hero stays frozen.

## Next-ticket action

Once classified, do exactly one:

| State | Action |
| --- | --- |
| Exit red, work unfinished | Write or refresh `task-NN-brief.md`. Dispatch one isolated agent. |
| Exit red, agent still running | Monitor. Live-attach with `attaching-live-training` if it looks stuck. Do not start the next science step. |
| Exit green, winner not frozen | Freeze knobs in `map.md` with the declared winner rule. |
| Exit green, winner frozen | Open the next issue + brief. Update `progress.md` blockers. |
| Fog only (seed noise, omitted O(L) arm) | Record in map Fog. Do not treat fog as a new knob. |

Parallel agents only on non-overlapping tickets (e.g. data readiness
|| harness tests). Never two writers on the same mixer file or the
same GPU table.

## Brief contract

Every `task-NN-brief.md` states:

1. **Read first:** spec section, issue, this skill.
2. **Requirements:** exit criteria copied from the spec, not invented.
3. **Exclusions:** no new knobs, no paper-scale unless authorized, no
   commit unless the user asked, rootfs for Python/CUDA.
4. **Blockers:** prior tickets that must stay green.
5. **Verification:** the exact pytest / dry-run / table command.
6. **Claim_label** allowed for this ticket.
7. **Report path:** `sdd/task-NN-report.md` + issue Answer + map edit.

Dispatch with a clean context and that brief as the only requirements
file. Session chat is not canonical.

## Report contract

A ticket is not done until the report has:

- What was run (config, tokens/step, steps, seeds, GPU, rootfs).
- Exit criteria: pass/fail each.
- Table or pointer to gitignored artifacts.
- **Frozen decision** or explicit "nothing frozen".
- Honesty notes: omitted arms, invalided prior tables, seed ties.
- `claim_label` used.
- "No commit" unless the user asked.

Then the orchestrator updates `progress.md` and `map.md` Frontier.

## Step 4 execution order (from Falcon)

1. TDD leftovers that make arms *real* (a named knob that does not
   reach the kernel is not an arm).
2. Meta-init / `to_empty()` regression if Trainer builds on meta.
3. Few-step dry-run of **all** arms at science shape. Losses finite
   and not identical.
4. Fast matched-token table (vectorized arms) × ≥2 seeds.
5. Slow O(L) neighbors as a side invocation, or omit-for-walltime.
6. Transfer eval as declared in the spec (separate small trains vs
   eval-only).
7. Freeze one triple. Record ties and gaps.

Matched across arms: data, global batch, tokens/step, steps, eval
protocol. One hypothesized knob per arm.

## Step 5 / 6

Hero: same knobs, scale width and/or tokens, matched controls, same
tokens/step. Preflight data + disk first. Disagreement with the
paper is a finding; do not "fix" knobs on the hero.

Close: replicated held-out + transfer. Teacher-forced addition is
not generation. A transfer that is 0 OOD everywhere is a declared
gap, not a hidden failure.

## Monitor

While a table runs: host attach, do not stop the job. GPU-bound
`cuStreamSynchronize` is healthy compute. Identical losses across
arms is not — stop spending tokens and walk back.
