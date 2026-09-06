# Falcon Step 4 ablation table (combined; claim_label=representative_small + replicated_eval)

## LM arms (held-out val on fineweb_val_000000.bin, 8000 steps unless noted)

| arm | mixer | variant | alignment | phi | seed | steps | final_loss | val_ce | val_ppl | finite |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0 | softmax | falcon1a | delayed | rms | 0 | 8000 | 4.5665 | 4.6604 | 105.67 | True |
| A0 | softmax | falcon1a | delayed | rms | 1 | 8000 | 4.5962 | 4.6875 | 108.58 | True |
| A1 | gdn | falcon1a | delayed | rms | 0 | 8000 | 4.6746 | 4.7973 | 121.19 | True |
| A2 | falcon | falcon1a | delayed | rms | 0 | 8000 | 4.6949 | 4.8155 | 123.40 | True |
| A2 | falcon | falcon1a | delayed | rms | 1 | 8000 | 4.6863 | 4.7992 | 121.42 | True |
| A4 | falcon | falcon1a | same_step | rms | 0 | 8000 | 4.6635 | 4.7984 | 121.32 | True |
| A4 | falcon | falcon1a | same_step | rms | 1 | 8000 | 4.6644 | 4.7929 | 120.65 | True |
| A5 | falcon | falcon1a | delayed | l2 | 0 | 8000 | 4.6013 | 4.7216 | 112.35 | True |
| A5 | falcon | falcon1a | delayed | l2 | 1 | 8000 | 4.6237 | 4.7379 | 114.19 | True |

## Addition transfer (separate small train per mixer, teacher-forced)

| arm | mixer | variant | alignment | phi | seed | ID acc | OOD acc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A0 | softmax | falcon1a | delayed | rms | 0 | 0.004 | 0.000 |
| A0 | softmax | falcon1a | delayed | rms | 1 | 0.004 | 0.000 |
| A2 | falcon | falcon1a | delayed | rms | 0 | 0.066 | 0.000 |
| A2 | falcon | falcon1a | delayed | rms | 1 | 0.008 | 0.000 |
| A4 | falcon | falcon1a | same_step | rms | 0 | 0.043 | 0.000 |
| A4 | falcon | falcon1a | same_step | rms | 1 | 0.070 | 0.000 |
| A5 | falcon | falcon1a | delayed | l2 | 0 | 0.703 | 0.000 |
| A5 | falcon | falcon1a | delayed | l2 | 1 | 0.293 | 0.000 |
