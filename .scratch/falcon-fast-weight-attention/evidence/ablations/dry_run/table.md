# Falcon Step 4 ablation table (claim_label=smoke)

## LM arms (held-out val on fineweb_val_000000.bin)

| arm | mixer | variant | alignment | phi | seed | steps | final_loss | val_ce | val_ppl | finite |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0 | softmax | falcon1a | delayed | rms | 0 | 4 | 10.5406 | 10.5552 | 38377.20 | True |
| A1 | gdn | falcon1a | delayed | rms | 0 | 4 | 10.7222 | 10.7260 | 45526.37 | True |
| A2 | falcon | falcon1a | delayed | rms | 0 | 4 | 10.8077 | 10.8116 | 49594.63 | True |
| A3 | falcon | falcon1 | delayed | rms | 0 | 4 | 10.8116 | 10.8065 | 49340.03 | True |
| A4 | falcon | falcon1a | same_step | rms | 0 | 4 | 10.7852 | 10.7699 | 47566.22 | True |
| A5 | falcon | falcon1a | delayed | l2 | 0 | 4 | 10.7385 | 10.7416 | 46239.15 | True |

## Addition transfer (separate small train per mixer, teacher-forced)

| arm | mixer | variant | alignment | phi | seed | ID acc | OOD acc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A0 | softmax | falcon1a | delayed | rms | 0 | 0.004 | 0.000 |
| A1 | gdn | falcon1a | delayed | rms | 0 | 0.004 | 0.000 |
| A2 | falcon | falcon1a | delayed | rms | 0 | 0.004 | 0.000 |
| A3 | falcon | falcon1 | delayed | rms | 0 | 0.012 | 0.000 |
| A4 | falcon | falcon1a | same_step | rms | 0 | 0.008 | 0.000 |
| A5 | falcon | falcon1a | delayed | l2 | 0 | 0.004 | 0.000 |
