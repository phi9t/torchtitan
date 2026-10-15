# Attention association-geometry ablation (small modded-nanogpt speedrun)

Question: does the query-key matching rule matter, holding everything else fixed?
Owner request 2026-10-15. Vehicle: modded-nanogpt (pinned upstream ecbb586, see
experiments/modded_nanogpt_b200/upstream_reference.md), scaled down to small models,
single GPU.

## Arms (only the association rule changes)
Fixed across arms: RoPE, Q/K/V/O projections, backbone, data order, optimizer, schedule.
- anchor: unchanged upstream trainer. It already uses gain-free RMS QK norm + RoPE, which
  at matched temperature and negligible eps equals L2 (cosine) normalization. It is the
  anchor, not "raw SDPA".
- dot: no QK norm; logits = s_h * q.k with a learned positive per-head scale s_h.
- cosine: L2-normalize q, k (after RoPE); logits = tau_h * cos(q, k), tau_h = exp(log_tau_h)
  learned per head.
- gauss: unnormalized q, k; logits = -||q - k||^2 / (2 sigma_h^2). The query-only term cancels
  in softmax, so logits = (q.k - ||k||^2 / 2) / sigma_h^2: dot product plus a key-norm penalty.
  Implement with augmented SDPA (q' = [q, 1], k' = [k, -||k||^2 / 2], scale 1/sigma_h^2) so
  the fast attention kernel is kept; learned bandwidth sigma_h per head.
- NOT an arm: unit-normalized RBF (equivalent to cosine after rescaling).
All arms get the same temperature-tuning budget (initial temperature/bandwidth grid).

## Scoreboards
1. Learning: held-out loss vs tokens; associative-recall probe accuracy on fresh key->value
   mappings and at longer retrieval distances than trained.
2. Speedrun: held-out loss after 5 min wall-clock; time to a baseline-defined target loss
   (anchor's 5-min loss). Record compile time, eval time, peak memory separately. Confirm
   finalists at matched token budgets.

## Pilot budget (proposed, gated)
12 screening runs (3 geometries x 2 LR x 2 init temperature), then 6 fresh-seed confirmation
runs (finalists), plus 1 unchanged-anchor run: 19 x 5 min = 95 min timed single-GPU training
plus overhead. Winner rule declared before results: lowest mean 5-min held-out loss over the
confirmation seeds; deltas inside seed spread are ties.

## Out of scope until the pilot closes
Changing RoPE, tying Q/K projections, hard top-k attention. (Otherwise a result cannot be
attributed to matching geometry vs positional info, parameter count, or gradient path.)

## Revision 2 (2026-10-15 ~16:40Z, before any A02 result): fair best-shot protocol
Owner: "try really hard to get them to work". The anchor is upstream-tuned; the other arms are not. A 2x2 screen
cannot tell "geometry is worse" from "geometry is untuned". Replaces the pilot budget above.
1. Screen (A02a): each non-anchor arm gets LR scale {0.5, 0.75, 1.0, 1.5} x init temperature/bandwidth
   {0.5, 1, 2} = 12 runs/arm, 36 total, 5 min each, seed 0. Anchor x 2 seeds in the same screen (noise floor).
   Geometry scalars (log s / log tau / log sigma) sit in their own AdamW group (not Muon), lr declared once for all arms.
   Gauss key penalty computed in fp32 before the cast.
2. Rescue round (only for an arm whose best screen run diverges or is > 0.1 nats behind anchor): <= 6 extra runs from
   this fixed menu, no new mechanisms: geometry-scalar warmup (frozen first 10% of steps), halved geometry-scalar lr,
   LR scale 0.35. Anything outside this menu (softcap, z-loss, QK clip) changes the rule -> out of scope.
3. Confirmation (A02b): best config per arm, all 4 arms (anchor included) x 3 fresh seeds = 12 runs, so every arm gets
   a replicated best-shot number, plus a matched-token rerun of each and the recall probe.
4. Behavior, not only scalars: per-layer max |logit|, attention entropy, learned s/tau/sigma trajectories, ||q||/||k||
   every eval. A failing arm gets a written diagnosis, not just a loss.
Winner rule (unchanged): lowest mean 5-min held-out loss over confirmation seeds; deltas within seed spread are ties.
Budget ~ 38 + <=18 + 12 = <= 68 x 5 min (~6 GPU-h); runs are single-GPU, one run per exclusive GPU, parallel over
whatever GPUs are free.

## Revision 3 (2026-10-15 ~17:35Z, owner): four families, separate leaderboards
Owner expansion: (A) similarity kernels, (B) normalization/selection rules, (C) finite-feature attention, (D) stateful
associative memories. Each family tests a different hypothesis -> its own leaderboard, never one mixed sweep. B-D are
pending owner detail. Common form: a_ij = k(q_i,k_j) / sum_{m<=i} k(q_i,k_m), y_i = sum a_ij v_j; k needs k >= 0 and a
nonzero denominator (PSD/symmetry is NOT required and separate Q/K breaks it anyway).

Key implementation fact: the normalized weights depend only on log k, since softmax(log k) = k / sum k. So any k whose
log is a function of (q.k, ||q||^2, ||k||^2) runs as a FlexAttention score_mod (norms passed as captured per-token
tensors). A zero of k becomes -inf (allowed; guard every row with at least one positive weight, e.g. the diagonal).

Family A arms and implementation tier:
| # | kernel | log k as function of | tier |
| A1 | exp dot, exp(q.k/tau) | q.k | SDPA (= `dot`; anchor = A2 at fixed temp) |
| A2 | cosine / vMF, exp(alpha qhat.khat) | q.k, norms | SDPA (= `cosine`) |
| A3 | Gaussian, exp(-r^2/(2l^2)), variable norms | q.k, ||k|| | SDPA augmented (= `gauss`) |
| A4 | key-norm penalty, alpha q.k - beta ||k||^2, beta >= 0 (beta=alpha/2 -> A3) | q.k, ||k|| | SDPA augmented |
| A5 | diagonal anisotropic Gaussian (then low-rank M) | Mq.k, k'Mk | SDPA augmented; needs a constraint (e.g. normalized q/k or trace-fixed M) or it is absorbed by W_Q/W_K |
| A7 | L2 exponential, exp(-r/l) | r = sqrt(r^2) | flex score_mod (sqrt: clamp r^2 >= eps) |
| A8 | Matern 3/2, 5/2 | r | flex score_mod |
| A9 | rational quadratic / Cauchy, a > 0 | r^2 | flex score_mod |
| A10 | multi-bandwidth Gaussian mixture, pi >= 0 | r^2 (logsumexp over components) | flex score_mod |
| A11 | arc-cosine degree 1 (closed form ||q|| ||k|| (sin t + (pi - t) cos t)/pi) | q.k, norms | flex score_mod |
| A12 | shifted polynomial (1 + qhat.khat)^p, p in {1,2,4} | cos | flex score_mod (k=0 at antipode) |
| A13 | 2nd-order Taylor 1 + s + s^2/2 | s | flex score_mod (positive, minimum 1/2 at s=-1; needs score-scale control) |
| A14 | squared similarity s^2 (ReBased-style, affine-normalized q/k) | s | flex score_mod via 2 log(|s| + eps); antipodal ambiguity is the point |
| A6 | L1 Laplace exp(-||q-k||_1/l) | coordinatewise | not expressible from q.k/norms: explicit O(T^2 D) chunked op -> SIDE TABLE only (Falcon O(L)-arm rule) |
Notes: A6/A7-A12 radial or not -- A6 is not rotation invariant, so it interacts with RoPE; record it. Finite-feature
versions of A11-A13 belong to Family C, not here.

Scoreboards for Family A: wall-clock (5 min) penalizes flex/explicit kernels for kernel speed, not geometry. Primary
Family-A leaderboard = matched-token held-out loss + recall probe; wall-clock speedrun is reported separately and only
compares arms on the same kernel tier.

Ticket plan: A02 (Revision 2, SDPA tier: A1-A3 + anchor) unchanged. A03: implement A4, A5 (SDPA augmented) and
A7-A14 (flex score_mod) + A6 explicit, with unit tests (each arm vs an explicit normalized-kernel reference at seq 1024,
fwd + finite backward; A4 beta=alpha/2 == A3; A4 beta=0 == A1; A12/A13/A14 zero-or-tie handling) -- new files only, no
runner wiring while A01b/A02 own the runner. A04: wire + dry-run + screen Family A tier-2 (LR {0.75, 1.0} x bandwidth
{0.5, 1, 2} per arm), matched tokens, then confirmation of the top 3 by 3 seeds.
