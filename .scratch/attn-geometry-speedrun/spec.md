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
