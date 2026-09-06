# Falcon FWA end-to-end campaign — methodology spec

Status: approved 2026-09-05. Campaign **B** (process-faithful local)
locked. Paper-scale 49.2B hero is out of this campaign. Implementation
proceeds only through the numbered tickets under `issues/`.

Sources of intent:

- Mercor × SkyRL, *Training frontier knowledge work agents* (2026-09-01),
  https://www.mercor.com/blog/training-frontier-knowledge-work-agents-a-397b-rl-training-guide-with-skyrl/
- Zhang et al., *Fast Weight Attention for Continual Learning*,
  arXiv:2608.27763
- Repo claim language in `AGENTS.md` (smoke vs deterministic vs
  representative vs matched-performance vs replicated eval)
- Existing Falcon overfit gate in `torchtitan/experiments/falcon/`

This spec owns the **Falcon instance** of the playbook, not file-and-code
micro-plans. Sequencing, claim honesty, and the ticket loop live in
`.claude/skills/running-experiment-campaigns/SKILL.md`. Tickets after
approval implement one step each.

## Outcome

Run a Mercor-shaped de-risk → science → hero → transfer campaign for
Falcon fast-weight attention inside TorchTitan, without pretending Falcon
is an RL agent stack and without claiming paper-table numbers unless the
paper's budget is actually spent.

Success is a reviewed evidence bundle that says, for a declared scale:

1. the mixer and Trainer harness are trustworthy (Steps 1–3);
2. a small-model ablation picked the knobs (Step 4);
3. one larger run used only those knobs (Step 5);
4. quality was measured on held-out LM and on addition transfer (Step 6).

## What we take from Mercor (invariants)

Keep these even though the domain is different:

1. **Steps 1–3 spend almost no science compute.** Do not start ablations
   or a hero run while the harness or learnability gate is red.
2. **If a small fixed set cannot overfit, walk back.** Do not buy more
   tokens.
3. **Ablate on the small model, freeze knobs, then scale.** The hero run
   is not a search.
4. **Eval is noisy; replicate.** Treat single-pass deltas inside noise as
   ties. Prefer 3-seed or 3-split means.
5. **Read behavior, not only the scalar.** For Falcon that means write
   timing, state magnitude, train vs held-out CE, and addition carry
   errors — not tool-call traces.
6. **Transfer is a first-class gate.** Gains that exist only on the
   training harness/task do not count as the campaign result.
7. **Classify claims.** Overfit ≠ convergence. One FineWeb slice ≠
   Table 1. Addition teacher-forced acc ≠ generation quality.

## What we do not take from Mercor

Harbor, Modal/ECR worlds, MCP, SkyRL async RL, vLLM, Megatron, TITO
chat/completions, GRPO/DPPO, LLM judges, and APEX tasks. Those are the
*instance* of the playbook for knowledge-work agents. Falcon's instance
is a causal LM mixer on next-token prediction.

TITO's *invariant* still applies: the tokens and the internal write pair
the trainer optimizes must be the ones the model actually used. For
Falcon that is delayed pairing plus exact next-token labels, not
re-tokenized chat strings.

## Campaign choice (approaches)

Three ways to "finish" Falcon. This spec picks **B**.

| | A. Paper reproduction | B. Process-faithful local (selected) | C. Mixer-only |
| --- | --- | --- | --- |
| Follow Mercor steps 1–6 | yes | yes | stop after 3 |
| 130M / 49.2B FineWeb-Edu, seq 1024, GBS 480 | required | optional later, separately authorized | no |
| Matched GDN + Transformer controls | yes | yes, at the local budget | no |
| Addition length-extrapolation | yes | yes, primary transfer gate | optional |
| Disk / time risk on this host | high | bounded | low |
| Honest claim | "Table 1 reproduced" | "playbook run at declared scale" | "architecture can memorize" |

**A** is a data-and-budget campaign, not the next coding ticket. **C** is
already partly done and is not end-to-end Mercor. **B** uses Mercor's
sequencing and Falcon's actual scientific questions at a scale this
checkout can finish.

Step 5's token budget is the one remaining scale knob. Default local
hero: the largest FineWeb-Edu (or C4) run that passes a data-readiness
preflight on this host, with global batch and tokens/step fixed across
controls. Do not advertise it as 50B unless the manifest is 50B.

## Specialization map

| Mercor | Falcon FWA |
| --- | --- |
| Environment / sandbox | Mixer + dataloader + Trainer batch contract |
| Harness bugs | Silent same-step writes, wrong `phi`, circular labels, missing doc-boundary state reset |
| TITO | Delayed `(φ(k_{t-1}), v_t)` and `input[:, :-1] → labels[:, 1:]` with no re-encode |
| Eval-at-concurrency, error→0 | Target batch/seq/dtype dry-run; non-model faults → 0 |
| RL systems (Megatron, async split, KV ceiling) | Kernel equivalence, Trainer loop health, optional compile/FSDP |
| Train–inference logprob diff < 0.03 | Recurrent ≡ masked-parallel (and later chunk-parallel) within a stated tol |
| 32-task sync overfit | Fixed sequence bank; CE collapse (done) |
| 35B algorithm ablations | Small Falcon LM ablations (objective, alignment, `phi`, controls) |
| 397B hero | One larger run with frozen knobs |
| Held-out APEX 3-pass | Held-out LM CE/PPL, replicated |
| Harness swap (Archipelago → OpenCode) | Task swap: FineWeb/C4 → variable-digit addition |
| HLE/GPQA no-regression | Softmax/GDN matched CE must not show Falcon as a broken LM |

## Steps 1–3 are de-risking

Mercor: "We do not spend significant compute until Step 4." Same here.

### Step 1 — Environment, harness, token accounting

**Goal.** The thing we train is the paper's mixer, on the labels we
think we are training.

**1.0 What runs where**

```text
experiments/falcon/run.sh          rootfs, reports, launch
torchtitan/experiments/falcon/     mixer, model, dataloader, configs
torchtitan/trainer.py              core Trainer (no Falcon-specific loop)
tests/unit_tests/test_falcon_*     kernel / model / overfit / config
```

No Harbor. No second trainer. Dependency direction stays
`experiments -> core`.

**1.1 Infrastructure robustness**

Analog of timeouts / error classification:

- Rootfs required for any Python/CUDA/training work.
- Fail loud on: NaN/Inf, vocab overflow, seq_len longer than the model
  contract, DP world size ≠ 1 until Step 2 authorizes FSDP.
- Dry-run the intended Step 4 batch/seq/dtype once and drive
  launch/data/runtime errors to zero before ablations.

**1.2 Optimize the harness (read traces)**

Mercor raised base reward a full epoch's worth by fixing the harness
before RL. Falcon's equivalent bugs are mathematical:

- Write feature is `φ(k_{t-1})`, not `k_t`. A same-step implementation
  is a different paper.
- `t = 0` is a no-op write (`x = 0`, `η = 0`).
- Read after write: `o_t = S_t^T φ(q_t)`.
- Default `φ` is QK-RMSNorm; L2 is an ablation, not a silent default.
- Values are not normalized unless an ablation enables VNorm.
- Next-token labels are a shift, not a roll. No wrap-around target.
- If document packing appears later, reset `S` at document boundaries
  (Qwen3.5 GDN already does this via `cu_seqlens`).

Required Step-1 artifact: a short harness note plus tests for delayed
vs same-step pairing and label shift. If a later overfit or ablation
looks dead, read dumps of `η_t`, `‖S_t‖`, and per-position CE before
changing the optimizer.

**1.3 Token / write accounting (TITO analog)**

Pass only if all of these hold:

- Trainer batch is `{input: tokens[:, :-1], positions}`, `labels = tokens[:, 1:]`.
- Kernel tests prove the first write uses `k_0` to associate `v_1`.
- No text encode/decode on the overfit or FineWeb-token path.
- A same-step kernel is available as an **explicit ablation switch**,
  never as the default.

**Step 1 exit.** Tests green; dry-run at Step 4 shape is clean;
harness note checked in. Claim class: **smoke + deterministic pairing**.

### Step 2 — Systems tuning

Mercor tuned Megatron, split rollout vs train, set concurrency, then
checked train–inference logprob match. Falcon has no generator mesh.
The systems object is the **mixer implementation used by Trainer**.

Tune in this order:

1. **Numerical identity of views.** Recurrent scan is the oracle.
   Masked-parallel (and later chunk-parallel with positive-decay
   renormalization) must match it on tiny `(B, L, N, K, V)` cases.
   This is the logprob-diff gate. Default tol: tight fp32 on the CPU
   oracle; documented looser tol if a later CUDA kernel is added.
2. **Trainer loop health.** One-GPU (or fake-backend) `falcon_tiny_overfit`
   and a slightly larger debug config complete N steps without hang,
   with finite loss, and with the dataloader cycling as specified.
   There is no `wait_for_generation_buffer`; the analog is "the step
   time is compute, not a stuck iterator."
3. **Optional compile / FSDP.** Only after (1) and (2). CP is out:
   core Qwen3.5 GDN already rejects it; do not invent Falcon CP in
   this campaign.
4. **Throughput is not a Step-2 pass condition.** Record step time,
   but do not block science on a fast kernel. A slow CPU/GPU recurrent
   path is allowed for Step 4 if identity holds. Chunk-parallel / FLA
   is a later acceleration ticket, not a science blocker.

**Step 2 exit.** Recurrent ≡ masked-parallel tests; Trainer debug run
artifact; written note that no train–inference stack exists yet.
Claim class: **deterministic comparison** (views) + **smoke** (Trainer).

### Step 3 — Overfitting run (done)

Mercor: 32 nonzero-variance tasks, batch 32, n=8, synchronous, one
epoch per step. Fail => usually Step 1.

Falcon analog already landed:

- 8 distinct sequences, vocab 32, input length 16
- `local_batch_size` = bank size
- CE must drop below 50% of step-0 and below 1.0
- `lr = 0` must fail the gate

Measured: Falcon-1 and Falcon-1A both go **3.47 → 0.011** in 80 AdamW
steps. That is learnability, not language-model quality.

If a later harness change (alignment switch, new kernel, packing)
lands, **re-run Step 3** before Step 4. A new write rule that cannot
overfit is a broken Step 1, not a reason to start FineWeb.

**Step 3 exit.** Current tests remain green after any mixer edit.
Claim class: **learnability smoke**.

## Step 4 — Ablations on the small model

Mercor ablated a few hypothesized knobs on 35B, evaluated every arm at
the same checkpoint with a 3-pass held-out mean, and froze the winner
for the hero run. Do the same at **small Falcon width**, not 130M.

### Science model (fixed across arms)

A LLaMA-style decoder that is larger than the overfit toy and smaller
than the paper's 130M. Suggested starting point: 4–8 layers, hidden
256–512, seq 256–1024, SwiGLU, Pre-Norm RMSNorm, no dropout, AdamW
with paper-like betas `(0.9, 0.95)`. Token budget: enough that held-out
CE moves and addition can be trained, not 49B. Exact integers live in
the Step 4 ticket after a data-readiness check.

Every arm shares: data, global batch, tokens/step, steps, seed policy,
and eval protocol. Change one hypothesized knob.

### Required arms

These are the paper's actual claims, not a kitchen sink:

| ID | Knob | Why |
| --- | --- | --- |
| A0 | Softmax Transformer + RoPE | paper Table 1 control |
| A1 | Gated DeltaNet-style same-step residual write | strongest published recurrent neighbor. Implement as a mixer switch in the **same Falcon decoder shell**, using the existing Qwen3.5/Mini-K3 gated-delta *math* (`state <- gS + k((v-k^T S)β)^T` on `k_t`). Do not load `qwen3_5` or Mini-K3. |
| A2 | Falcon-1A, delayed, QK-RMSNorm, ctx-η/λ | paper LM default family |
| A3 | Falcon-1, delayed, QK-RMSNorm, ctx-η/λ | paper's best FineWeb PPL variant |
| A4 | Falcon-1A, **same-step** write | alignment ablation; if A4 ≈ A2, delayed pairing is not buying anything |
| A5 | Falcon-1A, delayed, QK-ℓ2 | paper 1A.1 vs 1A.3 |

Optional, only if A2/A3 are healthy and cheap:

- A6: short conv on/off (paper default on; theory says orthogonal)
- A7: Falcon-3A sliding window (paper's best addition model; skip if
  `λ_max` / window work would delay the campaign)

Out of Step 4: Falcon-2/2A production kernels, FLA, Qwen3.5 hybrid
swap, lm-eval-harness 8-task suite, 50B tokens.

### Metrics (apples-to-apples)

Primary:

- held-out next-token CE / PPL on the same FineWeb-Edu or C4 slice
- variable-digit addition: train widths `{1..N}`, teacher-forced acc
  in-distribution and OOD `{N+1..M}` (paper uses N=32, M=48; the
  ticket may shrink N/M if the science model is tiny, but must keep
  a declared OOD gap)

Secondary (behavior, Mercor-style):

- `‖S‖` growth, mean `η`, fraction of steps with `γ` clamp hits
- train vs held-out CE gap (overfit vs fit)

Replication: 2 seeds minimum, 3 if wall-time allows. Deltas inside
seed noise are ties.

### Winner rule

Pick one mixer+alignment+`phi` triple for Step 5. Record the table
even if the winner is "no better than GDN on PPL, better on addition"
— that is still a result, and it matches the paper's own honesty.

**Step 4 exit.** Ablation table, chosen knobs, claim class
**representative training + replicated eval** at science scale.

## Step 5 — Hero run

Mercor: freeze knobs, scale the model, only systems differ.

Falcon: freeze Step 4 knobs, scale **width and/or tokens**, keep the
algorithm the same. Do not open new mixer variants on the hero path.

**Local hero (default).** One run of the winning Falcon plus the two
controls (Transformer, GDN-style) at the preflight-approved budget.
Same global batch and tokens/step across the three. Single node.

**Paper hero (separate authorization).** 124–130M, FineWeb-Edu,
seq 1024, GBS 480, 100k steps, ~49.2B tokens, bf16, μP-style width
scaling, lr `1e-3`, cosine, 2k warmup, wd 0.1, clip 1.0. Only if a
data-readiness ticket proves the corpus and disk.

If the local hero disagrees with the paper (e.g. delayed pairing
does not help), that is a finding. Do not "fix" it by changing knobs
on the hero run; open a new Step 4.

**Step 5 exit.** Three matched run-attempt bundles (Falcon, GDN-style,
Transformer) or an explicit skip of a control with a reason. Claim
class: **representative training**.

## Step 6 — Evaluation and generalization

Mercor tested (i) held-out APEX on the train harness, (ii) same tasks
on a different harness, (iii) Terminal-Bench, (iv) HLE/GPQA
no-regression.

Falcon specialization:

| Mercor eval | Falcon eval |
| --- | --- |
| Held-out APEX, 3-pass | Held-out LM CE/PPL, 2–3 seeds or 3 shards |
| Harness swap | **Task swap:** addition length extrapolation on the Step 5 checkpoint and/or a dedicated addition train (declare which) |
| Terminal-Bench | Optional later; not required to close the campaign |
| HLE/GPQA no-regression | Falcon held-out LM PPL must stay in the same band as the GDN-style control (no "only works on addition") |

Addition protocol (paper §5.2):

- Prompt: `digits_n(a) + digits_n(b) =`
- Target: reversed `(n+1)`-digit sum (LSD first)
- Train widths uniform in `{1..N}`; loss on the suffix only
- Report ID val acc and OOD teacher-forced suffix acc
- Replicate (seeds or resampled width draws)

Declare in the Step 6 ticket whether addition is:

- **eval-only** on the LM hero checkpoint (harder, cleaner transfer), or
- **separate small addition trains** per mixer (paper's setup, and the
  one that can actually show 80%+ OOD). Default: **separate addition
  trains**, because that is what the paper measured. Optionally probe
  zero-shot addition on the LM checkpoint as a harsher extra.

**Step 6 exit.** Written table: LM PPL + addition ID/OOD for Falcon vs
controls, with replication and claim labels. Campaign closes.

## Evidence and claim labels

Every run-attempt records `lane`, `mode`, `arm`, `claim_label`,
`evidence_tier`, `run_id`, `attempt_id`, `environment_class` when the
shared experiment schema is used; otherwise a compact JSON report with
those fields.

| Step | Allowed claim_label |
| --- | --- |
| 1 | `smoke`, `deterministic_pairing` |
| 2 | `deterministic_comparison`, `smoke` |
| 3 | `learnability_smoke` |
| 4 | `representative_small`, `replicated_eval` |
| 5 | `representative_training` |
| 6 | `replicated_eval`, `transfer_eval` |

Forbidden: calling Step 3 or a 10-step Trainer job "convergence";
calling a local hero "Table 1"; calling teacher-forced addition
"generation".

## Current tree vs this spec

Already done (keep, do not redo unless a mixer change invalidates it):

- Falcon-1 / 1A recurrent CPU kernel + worked-example tests
- Tiny LM + repeating bank + overfit gate (Step 3)
- `MODULE=falcon CONFIG=falcon_tiny_overfit`

Not done:

- Same-step ablation switch and harness note (Step 1 remainder)
- Masked-parallel identity (Step 2)
- Science-scale model, FineWeb/C4 wiring, ablation matrix (Step 4)
- Hero runs (Step 5)
- Addition task + replicated eval (Step 6)
- Falcon-2/3, chunk-parallel, Qwen3.5 swap (out of campaign)

## Ticket decomposition

Filed under `issues/`. Campaign B omits a paper-scale hero ticket.

- `03` Step 1 remainder (unblocked)
- `04` Step 2 systems (blocked by 03)
- `05` Data readiness (unblocked; blocks 06–08)
- `06` Step 4 science config + addition data (blocked by 03, 04, 05)
- `07` Step 4 ablation runs (blocked by 06)
- `08` Step 5 local hero (blocked by 05, 07)
- `09` Step 6 eval close (blocked by 06, 08)

## Exclusions

- Promoting Falcon into `torchtitan/models/`
- Editing Mini-K3 or production Qwen3.5 GDN
- Online RL / batch-invariant path
- Context parallel
- Vendor kernels in-tree (FLA stays an optional dep if used later)
- Commits, pushes, or PRs without a later explicit request

## Authority

Experiment-only. This spec is the durable methodology. Session
file-and-code lists are non-authoritative.
