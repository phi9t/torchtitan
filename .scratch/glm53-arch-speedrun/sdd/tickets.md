# Ticket index: glm53-arch-speedrun

Each ticket has `sN-brief.md` (what to do, blockers, verification) and `sN-report.md` when done. The ledger is
`progress.md`. Launch order follows the blocking edges; GLM rung 4 (T08d) has GPU priority throughout.

| Ticket | What | Blocked by | GPU | Status |
| --- | --- | --- | --- | --- |
| S01 | B0 harness, FineWeb on HDFS, first calibration | -- | small | done (fce8a52fb) |
| S02 | GLM element modules + identity tests | -- | none | done (fc412ef58) |
| S03 | ablation queue (93 cells), factorial design, scheduler | -- | none | done (ee80e69cc) |
| S04 | trainable indexer objective, fast KDA/sparse paths, GLM mHC init | S02 | small | done (51f3008d2) |
| S05 | B0 recalibration (268M tok, CE 3.9028 +/- 0.0019), elements wired | S01, S04 | 1 GPU | done (04497f671) |
| S06 | Step-3 overfit gate all arms + speed fixes (mHC, indexer) | S05 | 1 GPU | running |
| S07 | model-family ladder L1-L4 configs, gates, 268M comparison | S05 | 1 GPU x3 | running |
| S08 | FineWeb-Edu sample-10BT -> GPT-2 shards on HDFS | -- | none | running |
| S09 | ladder epoch runs L0-L4 x2 seeds on FineWeb-Edu 10BT | S07, S08, H01 | 1-4 GPU, <= 6 total | queued |
| S10 | slice 4a add-one (30 cells) | S06 PASS, H01 | 1 GPU/cell | queued |
| S11 | slice 4b leave-one-out (15 cells) | S10 | 1 GPU/cell | queued |
| S12 | slice 4c fractional factorial (48 cells) | S11 | 1 GPU/cell | queued |
| S13 | slice 4d dose sweeps (36 templates) | S12 | 1 GPU/cell | queued |
| S14 | C1 combined winners: 268M x3 + FineWeb-Edu epoch x2 | S13 (+S08, S09 for epoch) | 1-4 GPU | queued |
| S15 | close-out doc, HDFS index, campaign claim | S09, S14 | none | queued |

Cross-campaign: H01 (HDFS-strict guard + audit) lives in `.scratch/glm53-flash-sft-ladder/sdd/task-h01-*`.
