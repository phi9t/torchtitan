# DiT Synthetic-Latent Experiment

This experiment is a minimal TorchTitan integration proof for a
Diffusion Transformer (DiT) training loop. It trains on deterministic synthetic
latents, predicts DDPM noise with MSE, and exercises the normal trainer,
optimizer, metrics, and checkpoint path.

Run from the repo root:

```bash
NGPU=1 experiments/dit/run.sh
```

The default config runs 10 steps and writes a checkpoint at step 10:

```bash
python -m torchtitan.train --module dit --config dit_debug_synthetic
```

The committed proof target is single-GPU training. The experiment package is
under `torchtitan/experiments/dit` and is not a supported core model family.

Out of scope for this v1:

- ImageNet input pipelines
- VAE encode/decode assets
- sampling, EMA, classifier-free guidance, FID, or learned covariance heads
- model/data/tensor/pipeline parallel execution
