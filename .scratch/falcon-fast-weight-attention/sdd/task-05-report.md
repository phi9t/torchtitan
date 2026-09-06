# Ticket 05 report - data readiness

## Answer`.
- If no corpus fits, the note names the blocker and the smallest
  fetch that would unblock Step 4 — without fetching it in this ticket.

## Answer

Research-only inventory. No data fetched, no Falcon Python edited, no commit.

### Disk

- `df -h /data02` and the repo (same filesystem): **2.9T total, 44G free
  (99% used)**. The repo lives on `/data02`, so corpus + checkpoints share
  this 44G headroom. A local-hero-scale slice fits; a 50B corpus does not
  and is out of scope anyway (Campaign B).

### Corpora already on this host

1. **FineWeb 10B (GPT-2), best fit for Step 4/5.**
   `experiments/modded_nanogpt_b200/sources/modded-nanogpt-b200-sdpa/data/fineweb10B/`
   - `fineweb_train_0000{01..09}.bin` = **900,000,000 train tokens**
     (9 x 100M shards); `fineweb_val_000000.bin` = **100,000,000 val tokens**.
   - Format: nanogpt `.bin` (256 int32 header, then uint16 tokens),
     **GPT-2 tokenizer, vocab 50257**, from HF `kjj0/fineweb10B-gpt2`.
   - On-disk: **1.9 GB total**. Already present; zero fetch.
   - Caveat: this is plain **FineWeb**, not FineWeb-Edu, and it is
     **GPT-2-tokenized bins**, not TorchTitan's `HuggingFaceTextDataset`
     JSON-lines path. Reusing it means either (a) a small nanogpt-style
     `.bin` reader in the Falcon dataloader ticket (06), or (b) retokenizing
     to the Falcon/Qwen tokenizer. Neither is done in this ticket. Vocab
     must match model `vocab_size` (Falcon config currently `vocab_size=32`
     toy; Step 4 model will set a real vocab).

2. **FineWeb-Edu smoke slice (TorchTitan-native, tiny).**
   `experiments/qwen3_fineweb_hsdp_tp/data/fineweb_test/data.json` = **2000
   raw docs (~10 MB)**, `{"text": ...}` JSON-lines. Registered as
   `fineweb_test` in `torchtitan/hf_datasets/text_datasets.py` and consumed
   by `HuggingFaceTextDataset` (tokenizes on the fly with the configured
   tokenizer). This is the drop-in loader path but is **smoke-scale only**
   (~a few M tokens), not enough to move held-out CE at science scale.

3. **C4 smoke slice.** `tests/assets/c4_test/data.json` = 2000 docs (~4.6 MB),
   registered as `c4_test`, same loader path. Smoke only.

Tokenizer present for the native path: `assets/hf/Qwen3-1.7B/tokenizer.json`
and `tests/assets/tokenizer/`.

### Recommendation

**Use FineWeb 10B (corpus #1) as the Step 4/5 science corpus.** 900M train
tokens is ~13x any budget below, so a held-out slice is trivially available
(reserve `fineweb_val_000000.bin`, 100M tokens, as the fixed held-out set for
all arms). The reader-vs-retokenize choice is a Step-6-ticket detail; flag it
for ticket 06 as a prerequisite, not a blocker here.

Corpus #2/#3 (native JSON-lines) are the **preflight/dry-run** datasets for
Step 1's Step-4-shape dry-run and for the Trainer-loop-health smoke in Step 2,
because they need no new reader.

### Proposed matched knobs (fixed across Falcon / GDN-style / Transformer)

Keep global batch and tokens/step identical across all arms. Two clean
options; both hold **tokens/step = 16,384**:

- seq_len 512, global batch 32 seqs -> 16,384 tokens/step (recommended;
  matches Falcon science model 4-8 layers, hidden 256-512, seq 256-1024).
- seq_len 1024, global batch 16 seqs -> 16,384 tokens/step (use if the
  addition transfer task or RoPE range wants longer context).

**Step 4 (ablations, small model, per arm):** 8,000 optimizer steps =
**131M tokens/arm** (0.15 epoch of the 900M slice; no repetition). If
wall-time is tight, 4,000 steps = 65M tokens is the floor that still moves
held-out CE for a 4-8 layer / hidden 256-512 model. 6 required arms
(A0-A5) x 131M = ~786M token-reads, still < 1 epoch total, fully covered.
2 seeds min per arm (spec winner rule) is a step-count multiplier, not a
token-availability problem.

**Step 5 (local hero, per run, 3 runs = winning Falcon + GDN-style +
Transformer):** scale tokens to **328M-655M/run** (20k-40k steps at
16,384 tok/step = 0.36-0.73 epoch), same seq_len and global batch as
Step 4. This stays inside the single 900M train slice with no wrap-around
and no fetch. Do **not** advertise as 50B; the manifest is 900M.

Checkpoint headroom: a 4-8 layer, hidden 256-512, vocab ~50k model is
< 200M params; fp32 optimizer + weights checkpoints are well under 2 GB
each. With 44G free, keep <= ~10 checkpoints per run or prune; fail loud in
the runner if free disk drops below a 5G floor before a save.

### Blocker (named, not fetched)

None blocks Step 4 on *availability*. The one open item for ticket 06 is the
**reader/tokenizer bridge**: FineWeb 10B is GPT-2 `.bin`, while the native
loader is JSON-lines + on-the-fly tokenize. Smallest unblock that stays local
and needs **no download**: add a nanogpt `.bin` reader (uint16, 256-int32
header) to the Falcon dataloader in ticket 06 and set model `vocab_size=50257`
to match GPT-2. If instead a FineWeb-Edu JSON-lines slice at science scale is
required, the smallest fetch would be extending `prefetch_fineweb.py`
(`HuggingFaceFW/fineweb-edu`, `sample-10BT`) to ~300k-600k docs (order ~1-3 GB
on disk, fits in 44G) — **do not fetch in this ticket**.
