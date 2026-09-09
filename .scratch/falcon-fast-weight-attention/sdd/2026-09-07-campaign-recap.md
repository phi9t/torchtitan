# Falcon FWA Campaign B evidence recap

Date: 2026-09-07

Status: audited; raw artifacts unchanged

## Executive result

Campaign B completed its three 20,000-step local hero training runs. The local
ranking is Transformer A0, Falcon A5, then GDN A1. A5 is the best completed
Falcon arm but does not match the Transformer control. Its apparent addition
advantage is training-bank accuracy, not held-out transfer, and every valid
width-OOD addition score is zero.

The local Falcon recurrence matches the paper's delayed RAW equations given
beta and lambda inputs. The learned parameterization is not the paper's
strongest `ctxeta-ctxlambda` configuration: the checkout emits
context-conditioned beta and an uncoupled softplus lambda.

## Evidence classification

- Decision-grade LM records: 12 completed run outcomes. Historical attempt
  identity is unavailable for some compact records, so this is not a claim that
  each outcome maps to exactly one launch attempt.
  - Step 4: nine 8,000-step launches, 1,179,648,000 token exposures.
  - Step 5: three 20,000-step launches, 983,040,000 token exposures.
  - Total: 2,162,688,000 token exposures.
- Completed Step-4 addition records: eight 1,500-step runs for A0, A2, A4,
  and A5 across two seeds. Their ID evaluation overlaps the training bank.
- Smoke/tooling evidence: six Step-4 LM dry-runs, six addition dry-runs, three
  hero dry-runs, core Trainer/science smokes, three A5 profiler workloads, and
  one completed 30,000-step tiny-overfit profiler target.
- Invalid or incomplete evidence:
  - the initial collapsed ablation table invalidated by zero RMSNorm gains;
  - A3's stopped O(L) 8,000-step attempt;
  - A1 seed 1, described as launched but without an outcome artifact;
  - the first failed 30,000-step profiler launch with an invalid scheduler.

Derived combined tables, copied evidence tables, and live-attach captures are
not additional training runs.

## Step-4 held-out LM results

All runs use the 4-layer, hidden-256, sequence-512 fp32 science model and
16,384 tokens per optimizer step. Each completed run evaluates 1,048,576 held-
out tokens from the fixed FineWeb validation shard.

| Arm | Configuration | Seed | Steps | Final loss | Val CE | Val PPL | Wall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A0 | softmax | 0 | 8,000 | 4.5665 | 4.6604 | 105.67 | 350.63 s |
| A0 | softmax | 1 | 8,000 | 4.5962 | 4.6875 | 108.58 | 354.58 s |
| A1 | GDN | 0 | 8,000 | 4.6746 | 4.7973 | 121.19 | 13,943.30 s |
| A2 | Falcon-1A, delayed, RMS | 0 | 8,000 | 4.6949 | 4.8155 | 123.40 | 458.78 s |
| A2 | Falcon-1A, delayed, RMS | 1 | 8,000 | 4.6863 | 4.7992 | 121.42 | 458.59 s |
| A4 | Falcon-1A, same-step, RMS | 0 | 8,000 | 4.6635 | 4.7984 | 121.32 | 456.34 s |
| A4 | Falcon-1A, same-step, RMS | 1 | 8,000 | 4.6644 | 4.7929 | 120.65 | 456.30 s |
| A5 | Falcon-1A, delayed, L2 | 0 | 8,000 | 4.6013 | 4.7216 | 112.35 | 458.09 s |
| A5 | Falcon-1A, delayed, L2 | 1 | 8,000 | 4.6237 | 4.7379 | 114.19 | 457.98 s |

Arithmetic means:

| Arm | Seeds | Mean val CE | Mean val PPL |
| --- | ---: | ---: | ---: |
| A0 | 2 | 4.673918 | 107.1264 |
| A5 | 2 | 4.729747 | 113.2706 |
| A4 | 2 | 4.795631 | 120.9812 |
| A1 | 1 | 4.797328 | 121.1862 |
| A2 | 2 | 4.807353 | 122.4112 |

A5 trails A0 by 0.055829 mean CE / 6.1442 mean PPL and beats A2 by
0.077606 mean CE / 9.1406 mean PPL. A2 versus A4 is a tie at this seed
coverage. Replication applies to A0, A2, A4, and A5; it does not apply to A1
or omitted A3.

## Step-5 local hero results

All three runs use seed 0, 20,000 steps, 327,680,000 tokens, and full train-
state checkpoints through `step-20000`.

| Arm | Configuration | Final loss | Val CE | Val PPL | Wall | Outcome |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| A0 | softmax | 4.1391 | 4.4731 | 87.63 | 894.08 s | finite, complete |
| A5 | Falcon-1A, delayed, L2 | 4.2350 | 4.5543 | 95.04 | 1,166.27 s | finite, complete |
| A1 | GDN | 4.2807 | 4.6000 | 99.48 | 36,525.88 s | finite, complete |

A5 trails A0 by 0.081220 CE / 7.4141 PPL and takes 1.304x as long. A5
beats A1 by 0.045669 CE / 4.4411 PPL; A1 takes 31.319x as long as A5.
These are single-seed representative-training observations, not replicated
evaluation or paper-scale results.

## Addition result correction

The Step-4 addition runner builds one 2,048-example ID bank, evaluates its
first 256 examples, and samples training batches from the entire bank. The ID
evaluation subset is therefore eligible for training and is not held out.

Existing two-seed training-bank exact-suffix means are A5 0.4980, A4 0.0566,
A2 0.0371, and A0 0.0039. They show different memorization/optimization
behavior only. A1 and A3 have no full addition records, so the previous claim
that A5 beats GDN on addition is unsupported.

OOD widths 17 through 24 are disjoint from training widths 1 through 16. Every
recorded OOD exact-suffix score is 0.000. Campaign B has not demonstrated
addition length extrapolation or held-out addition transfer.

## Paper-alignment correction

- Exact or close matches: Falcon-1/1A recurrence, delayed pairing, RAW timing,
  QK-RMSNorm, conventional QK-L2, and positive decay.
- Important divergence: local beta is context-conditioned and eta is derived;
  local lambda is uncoupled. The paper's strongest rows use
  `ctxeta-ctxlambda`, whose exact implementation is not published.
- Other divergences: about 29.9M versus 124M-130M parameters, plain FineWeb
  versus FineWeb-Edu, fp32 versus bf16, sequence 512 versus 1,024, separate
  embeddings versus tied, no short convolution, no muP path, and 0.33B versus
  about 49.2B hero tokens.
- Missing variants: efficient Falcon-1 and Falcon-3A.

The complete primary-source comparison is in
`docs/research/2026-09-07-falcon-fast-weight-attention-paper-alignment.md`.

## Mechanistic inference after equation audit

- Fact: away from epsilon, per-head RMS normalization is L2 normalization
  multiplied by the square root of head dimension. The corresponding exact
  fast-weight coordinate transform scales lambda and the NLMS denominator by
  head dimension and scales the stored state inversely by its square root.
- Fact: the current kernel uses one finite epsilon for both QK normalization and
  the NLMS denominator, so it cannot express the complete finite-epsilon
  transform without separating those two roles.
- Approved test: preserve existing defaults, expose separate epsilon roles, and
  require the scale-compensated RMS arm to match L2 outputs, coordinate-adjusted
  state, and gradients before any mechanism training.
- Hypothesis: the observed A5-over-A2 result is a parameterization or
  preconditioning effect caused by uncompensated scale. The five-arm screen
  tests its empirical size and the missing alignment interaction; it does not
  presume a distinct L2 geometry.

## Approved interpretation and next direction

- Accept A1/A3 O(L) omissions explicitly rather than spend matched-token
  budgets on slow reference implementations. Scope replication claims to the
  arms actually replicated.
- Preserve A5 as Campaign B's frozen local hero arm. Do not retroactively
  retune the completed hero.
- Preserve raw artifacts. Import them into a normalized ledger with hashes and
  explicit `legacy_import` gaps rather than rewriting or fabricating history.
- Repair addition evaluation before any transfer claim.
- Run a separate mechanism-first campaign for normalization scale, alignment,
  and state dynamics. Do not add speculative paper variants to its first
  screen.
- Close the scale/preconditioning branch if compensation explains the A5 gain.
- Do not authorize another hero or paper-scale run without a new decision gate.

## Raw evidence pointers

- Step-4 combined results:
  `experiments/falcon/results/ablations/combined/table.json`
- Step-4 dry-run results:
  `experiments/falcon/results/ablations/dry_run/table.json`
- Step-5 outcomes:
  `experiments/falcon/results/hero/hero_20k/{A0,A1,A5}_seed0/hero_outcome.json`
- Step-5 checkpoints:
  `experiments/falcon/results/hero/hero_20k/{A0,A1,A5}_seed0/checkpoint/`
- Tooling-only results: `experiments/falcon/results/tooling/`
- Detailed Step-4 report: `sdd/task-07-report.md`
- Detailed Step-5 report: `sdd/task-08-report.md`
