# Falcon Fast Weight Attention: Paper-to-Checkout Alignment

Date: 2026-09-07

## Scope and source quality

This note compares the Falcon experiment in this checkout with Zhang et al.,
*Fast Weight Attention for Continual Learning*, arXiv:2608.27763. Paper facts
come only from the [arXiv paper](https://arxiv.org/pdf/2608.27763) and the
[authors' official project repository](https://github.com/yifanzhang-pro/fast-weight-attention).
The repository's advertised `model/` directory currently contains only a
[one-line placeholder README](https://github.com/yifanzhang-pro/fast-weight-attention/blob/master/model/README.md),
so there is no released reference implementation, data generator, or experiment
configuration against which to resolve details omitted by the paper.

The comparison labels below mean:

- **Exact match:** the checked-in equation or protocol is the same modulo
  zero-based indexing and tensor layout.
- **Approximation:** the same idea is implemented at a deliberately smaller
  scale or through a non-production computational form.
- **Divergence:** the checkout implements a materially different equation,
  parameterization, dataset, model, or evaluation protocol.
- **Unknown:** the primary sources do not specify enough to decide.

Local evidence was read from the campaign
[specification](../../.scratch/falcon-fast-weight-attention/spec.md),
[Step 4 report](../../.scratch/falcon-fast-weight-attention/sdd/task-07-report.md),
[Step 5 report](../../.scratch/falcon-fast-weight-attention/sdd/task-08-report.md),
current result artifacts, and the Falcon source and tests. No training or Python
job was run for this review.

## 1. Paper contract

### 1.1 Common timing and notation

The paper fixes read-after-write (RAW) semantics. With one-based paper
indexing,

\[
x_t = \phi(k_{t-1}),\qquad y_t=v_t,\qquad x_1=0,\qquad \eta_1=0,
\]

the state is updated to \(S_t\), and the position-\(t\) output is read from that
updated state,

\[
o_t=S_t^\top\phi(q_t),
\]

which is used to predict token \(t+1\). The pre-update fast-memory prediction
\(S_{t-1}^\top x_t\) is therefore distinct from the model output. The paper
calls \((\phi(k_t),v_t)\) the *same-step* or *unshifted* pairing. It remains
causal under RAW, but optimizes a different internal objective. See Section 4's
formula summary (paper pp. 7-8) and Appendix C.5 (pp. 35-36) in the
[paper](https://arxiv.org/pdf/2608.27763).

### 1.2 Falcon-1, Falcon-1A, and Falcon-3A equations

Let

\[
r_t = y_t-S_{t-1}^\top x_t,\qquad E_t=\lVert x_t\rVert_2^2,
\]

and let \(\epsilon>0\) be the step-size stabilizer.

**Falcon-1 (scalar NLMS regression).** The instantaneous objective is squared
error plus ridge regularization. Its normalized scalar update is

\[
\eta_t=\frac{\beta_t}{\lVert x_t\rVert_2^2+\lambda_t+\epsilon},
\qquad \beta_t\in(0,2),
\]

\[
S_t=(1-\eta_t\lambda_t)S_{t-1}+\eta_t x_t r_t^\top.
\]

This is Equations 3.3-3.4 and 4.1-4.3, summarized on pp. 7-10 of the
[paper](https://arxiv.org/pdf/2608.27763).

**Falcon-1A (scalar normalized inner-product write).** Replacing regression
with the negative inner-product objective removes the residual correction:

\[
\eta_t=\frac{\beta_t}{E_t+\lambda_t+\epsilon},
\qquad
S_t=(1-\eta_t\lambda_t)S_{t-1}+\eta_t x_t y_t^\top.
\]

Thus the A variant writes \(v_t\) directly; it is not DeltaNet regression with
a different name. See the Section 4 formula summary and Section 4.3 (paper
pp. 7-8 and 14-15) in the
[paper](https://arxiv.org/pdf/2608.27763).

**Falcon-3A (sliding-window normalized inner-product write).** For nominal
window size \(B\), define

\[
I_t=\{j:\max(2,t-B+1)\le j\le t\},\qquad B_t=|I_t|,
\]

\[
\bar N_t^{(B)}=\frac{1}{B_t}\sum_{j\in I_t}x_jy_j^\top,
\qquad
\bar E_t^{(B)}=\frac{1}{B_t}\sum_{j\in I_t}\lVert x_j\rVert_2^2,
\]

\[
\eta_t=\frac{\beta_t}{\bar E_t^{(B)}+\lambda_t+\epsilon},
\qquad
S_t=(1-\eta_t\lambda_t)S_{t-1}+\eta_t\bar N_t^{(B)}.
\]

For the positive-decay implementation the paper replaces the raw carry with

\[
\alpha_t=\min(\eta_t\lambda_t,1-\epsilon_\gamma),
\qquad \gamma_t=1-\alpha_t,
\qquad S_t=\gamma_tS_{t-1}+\eta_t\bar N_t^{(B)}.
\]

The \(1/B_t\) averages are essential scaling, not cosmetic: in the stationary
\(\lambda=0\) case each item is injected over \(B\) consecutive windows with
total coefficient \(\eta\), independent of \(B\). Section 4.5, Algorithm 4,
and Appendix C.7 give these equations and boundary conditions (paper pp. 22-24
and 38-39) in the [paper](https://arxiv.org/pdf/2608.27763). The appendix says
the experiments use small windows and gives \(B=4\) as an example, but the
released sources do not explicitly bind a window size to the reported
Falcon-3A.3 result; that exact hyperparameter is therefore **unknown**.

For contrast, Falcon-3 uses the same window-average structure but writes
residuals and normalizes by

\[
\mu_t^{(B)}=\lambda_{\max}\left(\frac{1}{B_t}
\sum_{j\in I_t}x_jx_j^\top\right),
\qquad
\eta_t=\frac{\beta_t}{\mu_t^{(B)}+\lambda_t+\epsilon}.
\]

That largest-eigenvalue statistic belongs to Falcon-3, not Falcon-3A; 3A uses
mean write energy. See the summary on pp. 7-8 and Section 4.4 on pp. 18-21 of
the [paper](https://arxiv.org/pdf/2608.27763).

For sequence length \(T\) and per-head key/value widths \(d_k,d_v\), the
Falcon-1/1A online recurrence takes \(O(Td_kd_v)\) arithmetic and carries an
\(O(d_kd_v)\) matrix state, exclusive of ordinary model activations. A literal
dense causal-mask realization instead materializes \(O(T^2)\) sequence
interactions. The paper's chunkwise construction avoids a full \(T\)-by-\(T\)
matrix and is linear in \(T\) for fixed chunk size, although the released
sources do not provide a production kernel whose constants can be benchmarked.
Falcon-3/3A continuation additionally carries a \(B-1\)-pair window tail, or
equivalent sufficient statistics. This computational scaling is separate from
the statistical scaling above: the \(1/B_t\) factors determine update
magnitude, while chunk size determines the parallel-computation tradeoff.

### 1.3 QK-RMSNorm versus QK-L2

The paper explicitly defines its default normalization for a projected vector
\(u\in\mathbb{R}^{d_u}\) as

\[
\operatorname{RMSNorm}(u)=
\frac{u}{\sqrt{\lVert u\rVert_2^2/d_u+\epsilon_{\rm rms}}}.
\]

It applies this independently to Q and K before dot products and outer
products. Therefore its typical squared norm is approximately \(d_u\), not 1.
Values are not normalized unless the optional VNorm ablation is enabled. The
paper contrasts this with QK-L2 variants but does not separately define the L2
operator or its epsilon convention; the natural reading is unit normalization,
\(u/\lVert u\rVert_2\). See Section 4.1 (paper p. 9) in the
[paper](https://arxiv.org/pdf/2608.27763).

Consequently, QK-RMSNorm and QK-L2 differ by approximately \(\sqrt{d_u}\) per
vector away from epsilon, and their QK dot products differ by approximately
\(d_u\). NLMS then cancels much of the K-side scaling in a single normalized
write, but the state/read magnitude and finite-epsilon behavior still differ.
The paper does not apply an additional \(d_u^{-1/2}\) query factor to the
Falcon formulas.

### 1.4 Eta, beta, lambda, and decay

The theoretical recurrence reserves \(\beta_t\in(0,2)\) for a dimensionless
NLMS gain and \(\eta_t\) for the realized normalized step. It allows an actual
ridge coefficient \(\lambda_t\ge0\), with carry
\(\gamma_t=1-\eta_t\lambda_t\). When positive-decay unrolling is required,
\(\eta_t\lambda_t\) is clamped at \(1-\epsilon_\gamma\), and `log1p` is used in
fp32. See Equations 3.4 and 4.18 plus Appendices C.3 and C.6 (paper pp. 6,
22-23, 35, and 37) in the [paper](https://arxiv.org/pdf/2608.27763).

The paper also describes a scale-coupled implementation:

\[
\bar\lambda_t=\lambda_{\rm scale}\sigma(\tilde\lambda_t),
\qquad
\lambda_t^{\rm eff}=\bar\lambda_t E_t,
\]

where \(E_t=\lVert x_t\rVert_2^2\) for non-sliding variants,
\(E_t=\mu_t^{(B)}\) for Falcon-3, and
\(E_t=\bar E_t^{(B)}\) for Falcon-3A. The sliding implementation treats the
statistic as stop-gradient on the shrinkage path while retaining a live
statistic in the step-size denominator. See Appendix C.1 and C.6 (paper pp.
34 and 38) in the [paper](https://arxiv.org/pdf/2608.27763).

There is a primary-source gap around the empirical labels `ctxbeta` and
`ctxeta`. Tables 1-3 say that `ctxbeta`, `ctxeta`, and `ctxlambda` mean
context-conditioned beta, eta, and lambda, but the paper does not specify the
projection, activation, range, initialization, or how a directly emitted eta
interacts with the normalized formula. The official repository contains no
code to answer this. It is safe to distinguish the labels; it is **not** safe
to reconstruct the exact `ctxeta` parameterization.

### 1.5 Fresh sequences, segment continuation, and document resets

The exact paper boundary is a fresh sequence with \(S_0=0\), \(x_1=0\), and
\(\eta_1=0\). If a nonzero state is carried across segments, that sentinel must
not be reintroduced as though the continuation were fresh. Falcon-3/3A also
require the last \(B-1\) causal pairs (or an equivalent FIFO) in addition to
the matrix state for exact continuation; resetting the tail is explicitly a
boundary reset. See Appendix C.5 and C.7 (paper pp. 35 and 38) in the
[paper](https://arxiv.org/pdf/2608.27763).

The paper does **not** state whether FineWeb-Edu documents are packed, whether
state is reset at an end-of-document token inside a 1,024-token sample, or
whether only sample boundaries reset state. Document-boundary behavior beyond
the mathematical fresh-sequence/continuation rules is therefore **unknown**.

## 2. Paper experiments and strongest supported claims

### 2.1 Language-model budget

The paper trains matched 124M-130M-parameter models on FineWeb-Edu for 100,000
optimizer steps, sequence length 1,024, and global batch 480: approximately
49.2B tokens. Runs use bf16 AdamW, tied input/output embeddings, Pre-Norm
RMSNorm, no bias, no dropout, muP-style width scaling, base learning rate
\(10^{-3}\), cosine decay, 2,000 warmup steps, Adam betas (0.9, 0.95), weight
decay 0.1, gradient clipping 1.0, and one four-H100/H200 node. Unless ablated,
fast-weight models use QK-RMSNorm and short causal convolutions on attention
projections. See Section 5.1 and Appendix B.1 (paper pp. 25 and 34) in the
[paper](https://arxiv.org/pdf/2608.27763).

The paper does not publish layer count, hidden size, head count, exact
`ctxeta`/`ctxlambda` heads, tokenizer details, or per-variant parameter counts
beyond the 124M-130M range. These remain **unknown** without released code.

### 2.2 Addition generation and evaluation

The paper prints the prompt as
`digits_n(a) + digits_n(b) =` and the target as the reversed, fixed
\((n+1)\)-digit sum, so the least-significant digit is produced first. It
samples training widths uniformly from 1-32 and optimizes masked next-token
log-likelihood on the target suffix only. It reports an in-distribution
validation accuracy and out-of-distribution teacher-forced target-suffix
accuracy averaged over widths 33-48. See Section 5.2 and Table 3 (paper pp.
25-27) in the [paper](https://arxiv.org/pdf/2608.27763).

The following details are not in the paper and cannot be recovered from the
placeholder official repository:

- operand sampling distribution within each width, leading-zero policy, and
  duplicate rejection;
- number of training/validation/OOD examples, batch size, optimizer, model
  size, and whether examples are generated online or from fixed banks;
- whether ID validation operands are guaranteed disjoint from training
  operands;
- whether "target-suffix accuracy" is exact whole-suffix accuracy or an
  aggregate token accuracy;
- whether spaces in the printed quoted prompt are literal tokenizer inputs;
- model-selection protocol beyond Table 3's reported best checkpoint at step
  1,900 or 2,000.

OOD evaluation is disjoint from training **by width** (33-48 versus 1-32).
Exact operand-level disjointness is **unknown**.

### 2.3 Which variants support which claims

| Claim | Paper variant and evidence |
| --- | --- |
| Best FineWeb-Edu perplexity in Table 1 | **Falcon-1.3**, QK-RMSNorm + `ctxeta-ctxlambda`, 17.10 PPL; better than Gated DeltaNet 17.32 and Transformer 17.38 at the matched 50B-token budget. |
| Best evaluated inner-product LM variant | **Falcon-1A.3**, QK-RMSNorm + `ctxeta-ctxlambda`, 17.40 FineWeb-Edu PPL. |
| Best zero-shot downstream average among listed 124M-130M models | **Falcon-1A.2**, QK-L2 + `ctxeta-ctxlambda`, 49.30. |
| Best recurrent one-shot downstream average | **Falcon-1.3**, 49.54; the Transformer row is 49.67. |
| Best addition length extrapolation | **Falcon-3A.3**, QK-RMSNorm + `ctxeta-ctxlambda`, 87.2 mean accuracy over widths 33-48; Falcon-1A.3 is next at 85.9. |
| L2 + context-beta comparison point | **Falcon-1A.1**, QK-L2 + `ctxbeta-ctxlambda`, 17.70 FineWeb-Edu PPL and 80.6 mean OOD addition accuracy. |

These are Table 1-3 results (paper pp. 25-27) in the
[paper](https://arxiv.org/pdf/2608.27763). The paper explicitly characterizes
the LM result as competitive rather than a uniform win and the arithmetic
experiment as supporting evidence. Falcon-2, Falcon-2A, and Falcon-3 are
defined mathematically but do not support the main empirical claims. Falcon-3A
appears in downstream and addition tables but not the Table 1 perplexity table.

## 3. Checkout alignment

### 3.1 Equation and kernel alignment

| Surface | Classification | Finding |
| --- | --- | --- |
| Falcon-1 recurrence | **Exact match**, given local beta/lambda inputs | [`falcon_recurrent_forward`](../../torchtitan/experiments/falcon/falcon.py) lines 115-159 computes energy, normalized eta, clamped positive carry, pre-update residual, write, then updated-state read. |
| Falcon-1A recurrence | **Exact match**, given local beta/lambda inputs | The same function switches only the write target from the residual to `v`, as the paper requires. |
| Delayed/RAW semantics | **Exact match** | Zero-based local step 0 is the paper's sentinel step 1; local step `t>0` writes `phi(k[t-1]) -> v[t]` and reads the updated state. The worked-example tests pin the first real write to `k0 -> v1`. |
| Same-step ablation | **Exact match to the paper's alternative objective** | Local `same_step` writes `phi(k[t]) -> v[t]`, including step 0, and then reads. It is explicit and not the default. |
| QK-RMSNorm | **Exact match** | [`_rms_normalize`](../../torchtitan/experiments/falcon/falcon.py) lines 28-29 computes `x / sqrt(mean(x^2) + eps)` independently per head for Q and K. Values are left unnormalized. |
| QK-L2 | **Best-supported match; epsilon convention unknown** | Lines 32-33 compute `x / sqrt(sum(x^2) + eps)`, the conventional stabilized unit-L2 map. The paper labels QK-L2 but does not define where epsilon enters. |
| Positive decay | **Exact match** | Local `alpha=min(eta*lambda, 1-eps_gamma)`, `gamma=1-alpha`, and cumulative log-carry match the paper. |
| Masked-parallel Falcon-1A | **Equation match; non-production implementation** | The masked causal sum includes the diagonal for RAW and reconstructs the final state. Campaign tests report fp32 agreement with the recurrent oracle. |
| Masked-parallel Falcon-1 | **Approximation as a systems implementation, exact as an oracle** | Residual targets are still rebuilt with an O(L) Python loop before masked read assembly; it is not the paper's WY/chunk-parallel production algorithm. |
| Falcon-3/Falcon-3A | **Missing/divergence in coverage** | The type and runtime validators accept only `falcon1` and `falcon1a`; no sliding window, window energy, largest-eigenvalue normalizer, or tail state exists. |

The local kernel uses one `eps` for both QK normalization and the eta
denominator, whereas the paper distinguishes \(\epsilon_{\rm rms}\) and
\(\epsilon\). This is an **implementation simplification**; exact equality to
an unreleased paper configuration is unknown.

### 3.2 Parameterization is not the paper's strongest empirical variant

[`FalconMixer`](../../torchtitan/experiments/falcon/model.py) lines 95-124
emits

```text
beta_t   = 2 * sigmoid(beta_proj(h_t))
lambda_t = softplus(lambda_proj(h_t))
eta_t    = beta_t / (||x_t||^2 + lambda_t + eps)
```

This is a valid instance of the paper's theoretical `ctxbeta` rule, and the
beta range is exact. It is **not** a direct context-conditioned eta rule.
Further, local lambda is an uncoupled actual coefficient; it does not implement
the paper's `lambda_bar = lambda_scale * sigmoid(...)` followed by
`lambda_eff = lambda_bar * E_t`, nor the sliding-statistic detach rule.

Accordingly:

- local A5 (Falcon-1A, delayed, L2) is closest to **Falcon-1A.1**, not to
  Falcon-1A.2/1A.3;
- local A2 (Falcon-1A, delayed, RMS) combines RMS with `ctxbeta`, a combination
  not reported as a named main-table variant;
- local A3 is Falcon-1 with RMS but still uses `ctxbeta`, whereas the paper's
  strongest LM row Falcon-1.3 is labeled `ctxeta-ctxlambda`.

Earlier campaign text described A2/A3 as `ctx-eta/lambda`; that label was
**inconsistent with the checked-in model**, which projects beta and derives
eta. The 2026-09-07 evidence amendment and current campaign table correct the
label to context-beta plus uncoupled lambda. This does not make the local
recurrence wrong, but it prevents the local arms from being treated as exact
reproductions of the paper's `.2`/`.3` variants.

### 3.3 Reset and data-boundary behavior

Each local Falcon forward initializes a zero state per batch row, so it exactly
implements a paper-style fresh sequence at the start of every model call. The
FineWeb loader, however, constructs flat contiguous token windows and supplies
no document-boundary metadata; the model ignores `attention_masks`. Therefore:

- state resets at every 512-token training/evaluation row, even if adjacent
  rows are contiguous in the source token stream;
- state does not reset at an end-of-document token inside a row;
- no state or boundary tail is carried between rows.

See [`bin_reader.py`](../../torchtitan/experiments/falcon/bin_reader.py) lines
156-189 and [`model.py`](../../torchtitan/experiments/falcon/model.py) lines
326-336. This is **exact** to a fresh-sequence mathematical call, but
**unknown/divergent at corpus-document level** because the paper does not
publish its packing/reset policy. The campaign harness correctly records
internal document reset as unimplemented, but its assertion that documents
*must* reset is a local methodological choice rather than a documented paper
fact.

### 3.4 Model and training budget comparison

| Property | Paper | Checkout science/hero | Classification |
| --- | --- | --- | --- |
| Corpus | FineWeb-Edu | Plain FineWeb-10B GPT-2 `.bin` shards | **Divergence** |
| Model size | 124M-130M | About 29.9M parameters from the declared 4x256 shape (mixer-dependent by a few thousand) | **Approximation** |
| Embeddings | Tied input/output | Separate embedding and LM-head matrices | **Divergence** |
| Core shell | Pre-Norm RMSNorm, SwiGLU, no dropout | Pre-Norm RMSNorm, SwiGLU, no dropout | **Exact high-level match** |
| Bias | No bias terms | Q/K/V/O and MLP are bias-free; beta/lambda projections have bias | **Divergence** |
| Local convolution | Enabled by default for fast-weight models unless ablated | Not implemented | **Divergence** |
| Width scaling | muP-style | No muP path | **Divergence** |
| Precision | bf16 | fp32 | **Divergence** |
| Hardware | One 4x H100/H200 node | One B200 per arm; hero arms ran concurrently on separate B200s | **Divergence**, no throughput comparison |
| Sequence/global batch | 1,024 / 480 | 512 / 32 | **Approximation** |
| Steps/tokens | 100k / about 49.2B | Step 4: 8k / 131.072M per arm; hero: 20k / 327.68M per arm | **Approximation**, 0.27% and 0.67% of paper tokens |
| Optimizer | AdamW, lr 1e-3, cosine, 2k warmup, betas (0.9,0.95), wd 0.1, clip 1.0 | AdamW, lr 1e-3, cosine, 200 warmup at 8k; betas (0.9,0.95), wd 0.1; core clipping defaults apply | **Partly matched** |
| Held-out LM eval | FineWeb-Edu plus WikiText/LAMBADA and 8 downstream tasks | First 64 batches (1,048,576 tokens) of one plain-FineWeb val shard; no paper downstream suite | **Divergence** |

The current artifacts support a process-faithful local comparison, not a Table
1 reproduction.

### 3.5 Addition protocol and split audit

The local encoder gets the central arithmetic direction right: exact-width
operands, a reversed zero-padded \((n+1)\)-digit sum, next-token shift, and
suffix-only loss. See
[`addition.py`](../../torchtitan/experiments/falcon/addition.py) lines 63-97.
The following differences matter:

| Property | Paper | Checkout | Classification |
| --- | --- | --- | --- |
| Prompt surface | Prints `a + b =` with spaces | Emits `a+b=` without spaces | **Possible divergence**; literal paper spacing is unverified without code |
| Train/ID widths | Train 1-32; ID validation within that regime | One bank spanning 1-16 | **Scaled approximation** |
| OOD widths | 33-48 | 17-24 | **Scaled approximation** |
| Width sampling | Uniform 1-32 | Uniform random choice while constructing a fixed 2,048-example bank | **Approximation** |
| Operand sampling | Not specified | Uniform integer among exactly-n-digit operands; 1 digit includes 0; duplicates allowed | **Unknown versus paper** |
| Addition model | Not specified in paper | About 0.53M parameters, 2x128, four heads | **Unknown/divergent scale** |
| Training | Best checkpoint at 1,900-2,000 steps; other details omitted | 1,500 AdamW steps, batch 128, lr 3e-3, wd 0, final checkpoint | **Divergence** |
| Accuracy | Teacher-forced target-suffix accuracy; exact aggregation unspecified | Whole example counts only if every suffix token is correct | **Plausible match, formally unknown** |
| OOD disjointness | Width-disjoint by construction | Width-disjoint by construction | **Exact** |
| ID validation disjointness | Called validation, but operand-level disjointness unspecified | **Not disjoint:** `id_eval` is the first 256 examples of the same 2,048-example bank sampled during training | **Divergence from held-out evaluation practice** |
| OOD averaging | Reported mean over each width 33-48 | Accuracy over 256 randomly width-mixed examples, not an explicit per-width macro-average | **Divergence** |

The ID-overlap follows directly from
[`ablation_runner.py`](../../experiments/falcon/ablation_runner.py) lines
314-331: `id_eval = ds.id_examples[:256]`, while training indexes the full
`ds.id_examples` list. Existing tests prove only width-set disjointness, not
train/evaluation example disjointness. Local OOD evidence is clean with respect
to widths, but local ID accuracy must not be described as held-out validation.

## 4. What the current evidence does and does not establish

### Verified local evidence already present

- Campaign reports record tiny-bank learnability for Falcon-1 and Falcon-1A
  (CE 3.47 to 0.011 in 80 steps) and failure of the lr=0 control. This is a
  learnability smoke, not LM quality.
- Campaign tests record recurrent-versus-masked-parallel fp32 agreement for
  Falcon-1/1A across delayed/same-step and RMS/identity settings, later extended
  to L2. This validates the local views, not the unreleased paper code.
- The two-seed 8k Step 4 table has complete A0/A2/A4/A5 LM and addition rows,
  one GDN LM seed, and no completed Falcon-1 science row. Its chosen A5 arm has
  mean LM PPL about 113.3 versus softmax 107.1, and mean ID addition exact-match
  about 0.498; all reported OOD accuracies are 0.0. See the
  [combined table](../../experiments/falcon/results/ablations/combined/table.md).
- Current hero artifacts show all three 20k runs complete: softmax A0 PPL
  87.63, Falcon-1A/L2 A5 PPL 95.04, and GDN A1 PPL 99.48, each at 327.68M
  tokens and seed 0. The Step 5 report still says A1 is running, but
  [`A1_seed0/hero_outcome.json`](../../experiments/falcon/results/hero/hero_20k/A1_seed0/hero_outcome.json)
  records 20,000 completed steps; the artifact is the newer source of truth.

### Claims that are not supported by this checkout

- **The paper's strongest LM claim is not tested.** Falcon-1.3 requires
  Falcon-1 + RMS + the paper's `ctxeta-ctxlambda` setup at 130M/49.2B. Local A3
  has no completed science run and uses a context-beta/uncoupled-lambda
  parameterization at about 30M.
- **The paper's strongest addition claim is not testable here.** Falcon-3A.3 is
  absent, the local widths/budget are smaller, and every local OOD result is
  zero.
- **Local A5 is not evidence for Falcon-1A.3.** It is closest to Falcon-1A.1,
  whose paper result is weaker than the `.3` and 3A.3 addition rows.
- **No absolute PPL comparison to the paper is valid.** Corpus, tokenizer,
  model size, precision, sequence/global batch, token budget, embedding tying,
  convolution, and evaluation data differ.
- **The campaign is not yet replicated at hero scale.** Hero results are one
  seed, and the addition ID numbers are not held out from the training bank.

## 5. Concise implications

1. Describe the checkout as a small, paper-inspired Falcon-1/1A experiment
   whose core delayed RAW equations are faithful, not as a reproduction of the
   paper's named `.2`/`.3` variants or tables.
2. Treat A5 as the local winner only. It is closest to Falcon-1A.1 and does not
   cover either strongest paper result.
3. Any future paper-alignment claim would first need an explicit decision about
   `ctxeta` and scale-coupled lambda despite the primary-source gap, plus
   Falcon-3A for the addition claim and genuinely disjoint ID evaluation.
4. Keep document-reset language qualified: fresh-sequence and continuation
   state are specified by the paper; FineWeb document packing/reset policy is
   not.
