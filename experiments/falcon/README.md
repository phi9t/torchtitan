# Falcon experiment runners

Rootfs-managed commands for the Falcon fast-weight mixer. Reusable Python
lives in `torchtitan/experiments/falcon/`.

```bash
experiments/falcon/run.sh overfit
experiments/falcon/run.sh train
```

`overfit` is the Mercor Step-3 analog: a tiny Falcon-1A LM must memorize a
fixed 8-sequence bank. If that fails, do not start a larger run.
