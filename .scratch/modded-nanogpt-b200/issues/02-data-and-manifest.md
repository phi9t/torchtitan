# FineWeb data preparation and manifest

Type: task
Status: ready-for-agent
Blocked by: 01

## Requirement

Add the data-preparation slice for the Modded NanoGPT B200 benchmark. The
implementation must preserve upstream train and validation token streams for
competition-comparable claims.

## Scope

Allowed:

- wrap upstream `data/cached_fineweb10B.py`;
- add a smoke-size data path that is clearly classified as `smoke`;
- write checksum manifests under the experiment `results/` or `data/` area;
- record fresh versus reused data status.

Excluded:

- no tokenizer or data-pipeline change for competition-comparable runs;
- no full training run;
- no data files tracked in git.

## Verification Evidence

- A smoke manifest generated from a cheap data-preparation path, or a blocker
  with the exact missing dependency/network condition.
- Manifest records source commit, command, environment, shard paths, sizes, and
  checksums.
