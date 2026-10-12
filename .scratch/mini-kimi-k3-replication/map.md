# Mini Kimi K3 Replication Map

Date: 2026-08-21

Source request: deep dive `https://books.vizuara.ai/book/pretraining-a-mini-k3`
and investigate whether the recipe can be replicated in this TorchTitan checkout.

## Classification

This is a repo-local research program investigation, not a core training change
or an online RL change.

- Entrypoint surface: likely top-level `experiments/mini_kimi_k3/` for runners,
  manifests, reports, and run artifacts.
- Reusable Python surface: likely `torchtitan/experiments/mini_kimi_k3/` for
  experiment-owned model, data loader, eval, and config glue.
- Dependency direction: `experiments -> core`. Core TorchTitan must remain
  usable without Mini-K3-specific dependencies.
- Current deliverable: feasibility map and decision sequence. No training,
  GPU run, or implementation was launched.

## Source Access

Primary sources used:

- Vizuara book landing page:
  `https://books.vizuara.ai/book/pretraining-a-mini-k3`
- First-party book metadata API:
  `https://books.vizuara.ai/api/book?slug=pretraining-a-mini-k3`
- First-party content API:
  `POST https://books.vizuara.ai/api/content`
- First-party companion repo:
  `https://github.com/VizuaraAI/mini-kimi-k3`
- Local companion clone:
  `/tmp/mini-kimi-k3-vizuara` at commit
  `ee2012c64c4a4a94df04d6a1f4d2bbd56b6d1b13`

The static `/read/...` pages render a client shell, but the book metadata and
capsule content are available through first-party APIs. A GitHub REST search
found the companion repo; cloning it avoided authenticated code search.

## What The Book Claims Was Trained

The completed r1 run is the reproducible low-budget run, not the larger
flagship plan:

- 1.02B total parameters
- 145M active parameters
- 61M non-embedding parameters
- 5,000,003,584 training tokens
- one H200 GPU
- 38,147 optimizer steps
- 131,072 tokens per step
- sequence length 4,096
- local batch 4 with gradient accumulation 8
- peak learning rate 6.00e-4
- WSD decay from step 32,430
- final loss 2.62
- final dead experts 0.0%
- skipped loss spikes 0
- reported cloud cost $252.35

The companion repo also contains a larger 55B-token flagship Mini-K3 plan. Keep
these claims separate:

- r1 ladder model: hidden 512, 12 layers, 256 experts, top-6 routing,
  MoE intermediate 416, 4 KDA heads, 8 attention heads.
- flagship Mini-K3 plan: hidden 1536, 24 layers, 512 experts, top-6 routing,
  2 shared experts, SITU, K3 tokenizer, about 35.56B total / 1.237B active.

## Recipe Components

The first-party repo describes or implements these major components:

- Kimi/K3-like architecture: a mix of KDA and MLA layers, SITU activation,
  MoE routing with many experts, shared experts, noaux_tc bias correction, and
  parameter-count validation.
- Data pipeline: six-source mixture, benchmark decontamination, K3 tokenizer,
  pretokenized little-endian `uint32` token shards, JSON sidecars, stable and
  decay manifests, and per-rank shard splitting.
- Scheduler: warmup-stable-decay with warmup fraction 0.02, decay fraction
  0.15, and minimum LR fraction 0.1 in the companion code.
- Training loop: spike guard, checkpoint state, monitoring, MFU accounting, and
  in-loop evaluation.
- Distributed path: FSDP2 sharding, replicated router bias handling, activation
  checkpointing, fused experts, fused SITU, and chunked cross entropy.
- Evaluation: multiple-choice subsets for HellaSwag, PIQA, ARC-Challenge,
  WinoGrande, MMLU, plus held-out perplexity over fixed tail shards.

## TorchTitan Fit

TorchTitan already has useful substrate:

- `ConfigManager` selects `--module` and `--config` through model and experiment
  registries.
- `run_train.sh` provides the core `torchrun` launch path.
- `Trainer` already owns model construction, optimizer, scheduler, checkpoint,
  metrics, and distributed setup.
- The LR scheduler supports WSD-like behavior through `decay_ratio`,
  `decay_type`, and `min_lr_factor`.
- Qwen3 dense and MoE configs exist, including debug and real model entries.
- The transformer modeling backend has partial precedent for MoE, MLA-style
  families, FSDP/EP, DeepSeek-style sigmoid router correction bias, and shared
  experts.
- Existing repo-local experiment structure shows how to keep research runners,
  manifests, and reports outside core.

TorchTitan does not currently have a Mini-K3 implementation:

- No `mini_kimi_k3`, Kimi K3, or book-recipe config exists in the checkout.
- Qwen3 MoE is not the Kimi K3 architecture; it does not become Mini-K3 by
  changing a config.
- Current text datasets tokenize Hugging Face text dynamically. They are not a
  faithful loader for the book's pretokenized `uint32` mixture-shard contract.
- Existing FineWeb debug experiment proves plumbing, not Mini-K3 tokenizer,
  corpus, schedule, architecture, or quality replication.
- The existing FineWeb experiment wrapper does not self-reexec through
  `scripts/rootfs/enter_rootfs.sh`; new real experiment entrypoints should.

## Feasibility

Replication is feasible as a TorchTitan repo-local experiment, but it is not a
small config-only task.

The most realistic claim ladder is:

1. Plumbing reproduction: run a tiny Mini-K3-shaped model through TorchTitan's
   trainer on synthetic or tiny pretokenized shards.
2. r1 methodology reproduction: match the r1 scale, token accounting, WSD
   schedule, data-shard semantics, checkpoint/resume, and evaluation harness.
3. Architecture fidelity: prove KDA, MLA, SITU, MoE routing, noaux_tc bias, and
   parameter counts match the first-party recipe.
4. Training evidence: run the 5B-token r1 attempt with immutable manifests,
   checkpoint lineage, eval summaries, and throughput/cost evidence.

A faithful reproduction needs either:

- a TorchTitan-native experiment model that ports the relevant Kimi/K3 modules;
  or
- a carefully bounded transformer-backend adapter around released Moonshot or
  Vizuara modeling code, with explicit compatibility tests.

The backend adapter path may be faster for first smoke runs. The native path is
more likely to give TorchTitan-quality tests, model-family ownership, and
distributed integration if this becomes a durable research vehicle.

## Main Gaps

- Model fidelity: KDA recurrence/kernel behavior, MLA query/value asymmetry,
  SITU, expert dispatch, shared experts, `e_score_correction_bias`, trainable
  router state, and exact parameter-count accounting.
- Data fidelity: K3 tokenizer assets, source manifests, decontamination records,
  pretokenized shard format, rank-split semantics, and checkpointable mixture
  cursor state.
- Schedule and guardrails: exact WSD parameterization, spike guard behavior,
  skip accounting, and loss-state checkpointing.
- Distributed semantics: FSDP2/EP sharding, router-bias replication, activation
  checkpointing, and compatibility with TorchTitan mesh validation.
- Evaluation: length-normalized multiple-choice scoring, held-out perplexity
  over fixed tail shards, and immutable eval manifests.
- Evidence: run-attempt bundle, source provenance, token-count ledger,
  checkpoint lineage, cost/throughput accounting, and claim classification.

## Proposed Decision Sequence

Use this as a wayfinding map before implementation.

### D1: Replication Target

Choose the first target claim:

- `r1-faithful`: reproduce the book's 1.02B/145M-active, 5B-token run.
- `tiny-plumbing`: build a much smaller shape-compatible smoke target first.
- `flagship-plan`: target the companion repo's 55B-token flagship plan.

Recommended first answer: `tiny-plumbing`, then `r1-faithful`.

### D2: Model Route

Decide between:

- experiment-owned TorchTitan-native Mini-K3 model;
- transformer-backend adapter over existing released Kimi/Vizuara code;
- hybrid, where adapter proves the smoke path and native modules are ported
  behind tests.

Recommended first answer: hybrid, with the adapter as a short-lived oracle.

### D3: Data Route

Decide whether to:

- reproduce the book's pretokenized `uint32` shard pipeline and manifests; or
- use existing TorchTitan text dataset plumbing only for initial smoke tests.

Recommended first answer: synthetic/tiny `uint32` shards for the first loader
test, then real manifest reproduction before any r1 claim.

### D4: Evidence Contract

Define the minimal run-attempt bundle before training:

- source commit and book/API snapshot;
- model/config fingerprint;
- tokenizer fingerprint;
- corpus manifest and token-count ledger;
- checkpoint lineage;
- eval manifest and summaries;
- throughput/MFU/cost summary;
- rootfs sentinel and environment facts.

### D5: GPU Budget Gate

Do not spend a full run until all cheap gates pass:

- parameter-count test;
- loader-resume test;
- tiny loss-decrease smoke;
- checkpoint/resume equivalence;
- eval-harness deterministic fixture;
- rootfs-aware runner preflight.

## Candidate Tickets

1. Write `torchtitan/experiments/mini_kimi_k3/README.md` and top-level
   `experiments/mini_kimi_k3/README.md` with scope, claims, and rootfs workflow.
2. Add a rootfs-aware top-level runner that refuses to execute Python unless
   `TORCHTITAN_IN_ROOTFS=1` is present.
3. Add an experiment-owned `uint32` token-shard loader with deterministic
   per-rank splitting and checkpointable cursor state.
4. Add tiny fixture shards and tests for mixture sampling, resume, and exact
   token-count accounting.
5. Add a Mini-K3 model-route spike comparing transformer-backend adapter versus
   native modules on parameter counts and one forward pass.
6. Add a tiny shape-compatible config and smoke test through core `Trainer`.
7. Add checkpoint/resume equivalence for model, optimizer, scheduler, RNG,
   dataloader, and train-step state.
8. Add deterministic eval fixtures for length-normalized multiple-choice
   scoring and held-out perplexity.
9. Add run-attempt evidence bundle generation for Mini-K3 experiment runs.
10. Only after the above, prepare an r1 preflight and costed launch plan.

## Current Recommendation

Proceed, but frame the next phase as wayfinding/spec work, not immediate
training. The repo can host a credible Mini-K3 replication, but the first
engineering milestone should be a tiny, rootfs-governed, experiment-owned
TorchTitan path that validates architecture, data, checkpoint, and evidence
contracts before any 5B-token run is proposed.
