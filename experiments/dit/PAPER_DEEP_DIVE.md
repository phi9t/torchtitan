# DiT Paper Deep Dive For The Synthetic-Latent Experiment

This note maps the subset of "Scalable Diffusion Models with Transformers"
implemented in `torchtitan/experiments/dit` to TorchTitan's training path. The
goal is an architecture and trainer proof, not an image-generation benchmark.

Primary sources:

- Peebles and Xie, "Scalable Diffusion Models with Transformers", arXiv:2212.09748, https://arxiv.org/abs/2212.09748
- Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models", arXiv:2112.10752, https://arxiv.org/abs/2112.10752
- Ho et al., "Denoising Diffusion Probabilistic Models", arXiv:2006.11239, https://arxiv.org/abs/2006.11239

## What DiT Changes

DiT replaces the usual convolution-heavy diffusion backbone with a Transformer
operating over latent patches. In the paper setting, an image is encoded by a
pretrained VAE into a spatial latent, that latent is patchified into a token
sequence, and the Transformer predicts the diffusion target conditioned on
timestep and class information.

This experiment implements that training skeleton without external assets:

```text
x0 [B, C, H, W]
  -> add DDPM noise at timestep t
xt [B, C, H, W]
  -> patchify
tokens [B, T, D]
  -> DiT blocks with timestep + class conditioning
noise prediction [B, C, H, W]
  -> MSE against sampled noise
```

The synthetic latents stand in for VAE latents so the job can prove the model,
trainer, metrics, and checkpoint path without ImageNet, VAE weights, image
preprocessing, or network access.

## Implemented Architecture

### Latent Patchify

`DiTModel.forward()` converts `x_BCHW` into non-overlapping latent patches:

```text
[B, C, H, W] -> [B, T, C * patch_size * patch_size] -> [B, T, D]
```

The debug config uses `C=4`, `H=W=16`, and `patch_size=2`, so the transformer
sequence length is `T=(16/2)*(16/2)=64`.

### 2D Sinusoidal Position Embeddings

The patch grid receives fixed 2D sine/cosine position embeddings. Half of the
embedding dimensions encode the row coordinate and half encode the column
coordinate. This follows the ViT/DiT convention of making patch position
available without learning a position table during the tiny synthetic run.

### Timestep And Class Conditioning

The timestep embedding applies sinusoidal timestep features followed by a small
MLP. Class labels come from an embedding table. The two vectors are added to
produce the per-sample conditioning vector:

```text
c_BD = timestep_embed(t_B) + class_embed(y_B)
```

### adaLN-Zero Blocks

Each DiT block uses adaptive LayerNorm modulation driven by `c_BD`. The
conditioning projection emits shift, scale, and gate values for attention and
MLP sublayers:

```text
shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
```

Those modulation projections are initialized to zero. At initialization each
block is residual-safe because both residual branch gates are zero, so the block
starts as an identity function even though the attention and MLP weights are
initialized normally.

### Unpatchify Decoder

The final layer applies another adaLN modulation, projects each hidden token
back to patch pixels, and unpatchifies:

```text
[B, T, D] -> [B, T, C * patch_size * patch_size] -> [B, C, H, W]
```

The final projection is zero-initialized, matching the DiT zero-output training
stabilization pattern at step zero.

## Diffusion Objective

The custom trainer implements the DDPM-style noise-prediction objective:

```text
epsilon ~ N(0, I)
t ~ Uniform({0, ..., T - 1})
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
loss = MSE(model(x_t, t, y), epsilon)
```

`MSELoss` uses sum reduction, and the trainer normalizes by the global number of
latent elements processed by TorchTitan's existing loss interface. Metrics count
tokens as `batch_size * latent_patch_tokens`, which reflects the Transformer
sequence rather than raw latent pixels.

## Explicitly Out Of Scope

This proof does not implement:

- learned covariance or variance prediction heads
- VAE encode/decode
- EMA weights
- classifier-free guidance or sampling
- FID/IS evaluation
- ImageNet or web-scale image data loading
- multi-GPU sharding for DiT parameters

Those pieces are deliberately excluded so the first runnable target remains a
small, hermetic TorchTitan training proof.
