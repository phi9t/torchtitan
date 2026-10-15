# S10 report: slice 4a add-one via scheduler

## Slice 4a Analysis

S10 analysis generated for slice `4a`.

Effect threshold: 2 x B0 seed std = 0.00388. Negative delta means lower validation CE than B0.

### Full-budget 4a rows

| Arm | Seq | Budget tokens | Done/runs | Val CE mean | Val CE std | Delta vs B0 | Call | Tokens/s | Memory GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| a1_mhc | 1024 | 268435456 | 2/2 | 3.83995 | 0.00742462 | -0.0698167 | better | 54563 | 18.37 |
| a1_mhc | 4096 | 268435456 | 2/2 | 3.81775 | 0.00615183 | -0.08315 | better | 70167 | 18.37 |
| a2_nope | 1024 | 268435456 | 2/2 | 4.1442 | 0.00438406 | 0.234433 | worse | 100806 | 20.14 |
| a2_nope | 4096 | 268435456 | 2/2 | 4.4219 | 0.00806102 | 0.521 | worse | 119452 | 20.14 |
| a3_kda | 1024 | 268435456 | 2/2 | 3.9285 | 0.00212132 | 0.0187333 | worse | 74296 | 13.68 |
| a3_kda | 4096 | 268435456 | 2/2 | 3.893 | 0.00735391 | -0.0079 | better | 116577 | 13.66 |
| a6_mtp | 1024 | 268435456 | 2/2 | 4.0012 | 0.00494975 | 0.0914333 | worse | 71168.5 | 21.29 |
| a6_mtp | 4096 | 268435456 | 2/2 | 4.01425 | 0.000494975 | 0.11335 | worse | 77046 | 23.03 |
| b0 | 1024 | 268435456 | 3/3 | 3.90977 | 0.000750555 | 0 | tie | 113816 | 20.37 |
| b0 | 4096 | 268435456 | 3/3 | 3.9009 | 0.00104403 | 0 | tie | 169652 | 20.37 |

### Reduced-budget slow-arm side table

| Arm | Seq | Budget tokens | Done/runs | Val CE mean | Val CE std | Delta vs B0 | Call | Tokens/s | Memory GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| a4_indexer | 1024 | 33554432 | 2/2 | 5.9708 | 0.02885 | 0.3415 | worse | 16989 | 9.77 |
| a4_indexer | 4096 | 33554432 | 2/2 | 6.19605 | 0.000212132 | 0.376383 | worse | 5391.5 | 21.43 |
| a5_indexpool | 1024 | 33554432 | 2/2 | 5.9434 | 0.00975807 | 0.3141 | worse | 16791.5 | 7.43 |
| a5_indexpool | 4096 | 33554432 | 2/2 | 6.1715 | 0.00438406 | 0.351833 | worse | 4549.5 | 12.25 |

Side-table deltas use B0 validation CE from the full-budget curve at the same 33,554,432-token point.

### Behavior Reads

| Kind | Arm | Seq | Seed | Records | Latest read |
| --- | --- | ---: | ---: | ---: | --- |
| mhc | a1_mhc | 1024 | 0 | 72 | comb_row_sum_mean=1, comb_col_sum_mean=1 |
| mhc | a1_mhc | 1024 | 1 | 72 | comb_row_sum_mean=1, comb_col_sum_mean=1 |
| kda | a3_kda | 1024 | 0 | 27 | gate_mean=-2.248, beta_mean=0.9791, impl=attn_gym |
| kda | a3_kda | 1024 | 1 | 27 | gate_mean=-2.062, beta_mean=0.9755, impl=attn_gym |
| indexer | a4_indexer | 1024 | 0 | 12 | recall_vs_dense_topk_mean=0.3663, path=masked_dense |
| indexer | a4_indexer | 1024 | 1 | 12 | recall_vs_dense_topk_mean=0.3957, path=masked_dense |
| indexer | a5_indexpool | 1024 | 0 | 12 | recall_vs_dense_topk_mean=0.3708, path=masked_dense |
| indexer | a5_indexpool | 1024 | 1 | 12 | recall_vs_dense_topk_mean=0.4141, path=masked_dense |
| mtp | a6_mtp | 1024 | 0 | 3 | aux_loss=0.6127 |
| mtp | a6_mtp | 1024 | 1 | 3 | aux_loss=0.6127 |
| mhc | a1_mhc | 4096 | 0 | 72 | comb_row_sum_mean=1, comb_col_sum_mean=1 |
| mhc | a1_mhc | 4096 | 1 | 72 | comb_row_sum_mean=1, comb_col_sum_mean=1 |
| kda | a3_kda | 4096 | 0 | 27 | gate_mean=-2.223, beta_mean=0.9787, impl=attn_gym |
| kda | a3_kda | 4096 | 1 | 27 | gate_mean=-1.996, beta_mean=0.9601, impl=attn_gym |
| indexer | a4_indexer | 4096 | 0 | 12 | recall_vs_dense_topk_mean=0.3377, path=masked_dense |
| indexer | a4_indexer | 4096 | 1 | 12 | recall_vs_dense_topk_mean=0.4028, path=masked_dense |
| indexer | a5_indexpool | 4096 | 0 | 12 | recall_vs_dense_topk_mean=0.3376, path=masked_dense |
| indexer | a5_indexpool | 4096 | 1 | 12 | recall_vs_dense_topk_mean=0.403, path=masked_dense |
| mtp | a6_mtp | 4096 | 0 | 3 | aux_loss=0.6144 |
| mtp | a6_mtp | 4096 | 1 | 3 | aux_loss=0.6152 |

## Publication

- Slice bundle HDFS URI: `hdfs://harunava/user/tiger/waystone/torchtitan/runs/glm53-arch-speedrun/s10-bundle`
- Bundle publish status: Waystone publish succeeded with `MANIFEST.jsonl` and `_COMPLETE`; manifest file count was 36.
- Timing/environment label: `shared_gpu`.
- Claim label: `representative_small+replicated_eval`; A4/A5 reduced-budget rows are side-table evidence.
