# Map
- Step 1 Harness: A01 (arm implementation + equivalence tests + small-model config + recall probe)
- Step 2 Systems: A01 (all-arm dry-run, finite loss, timing fields)
- Step 3 Overfit: A01 (fixed-bank overfit per arm, lr=0 fails)
- Step 4 Ablate: A02 (12 screening + 6 confirmation + 1 anchor), only after A01 exits green
- Step 5/6: scale-up / transfer -- not opened

# Frontier
- A01 remains open. Passing pieces: geometry equivalence tests, recall probe construction, copied runner/variant syntax, and one fixed-bank anchor GPU sanity.
- A02 is blocked until A01 records FineWeb verified data, all-arm science-shape dry-run divergence, per-arm fixed-bank overfit plus lr=0 failure, recall-after-train, and 5-minute anchor timing.

# Blockers / Fog
- `waystone` is not available in rootfs PATH, so FineWeb HDFS verified staging is missing.
- GPUs are shared with large jobs; further matrix runs would risk interfering with them.
- The fixed-bank sanity used an offline prefix-table fallback and is not a FineWeb or speedrun claim.
