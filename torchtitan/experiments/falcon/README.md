# Falcon Fast Weight Attention

Experiment-owned Falcon-1 / Falcon-1A mixer from
[arXiv:2608.27763](https://arxiv.org/abs/2608.27763). Optional TorchTitan
code; one-way dependency `experiments -> core`.

The first gate is Mercor Step-3 learnability: a tiny Falcon LM must overfit a
fixed 8-sequence bank. If cross-entropy does not collapse, the mixer or
harness is broken. This is not FineWeb quality evidence.

## Layout

- `falcon.py` — CPU recurrent Falcon-1 / Falcon-1A
- `model.py` — tiny decoder + causal LM
- `repeat_data.py` — repeating sequence bank
- `overfit.py` — synchronous overfit report
- `config_registry.py` — `falcon_tiny_overfit` Trainer config

Launch the Trainer config with:

```bash
experiments/falcon/run.sh overfit
```

or:

```bash
MODULE=falcon CONFIG=falcon_tiny_overfit NGPU=1 COMM_MODE=fake_backend ./run_train.sh
```
