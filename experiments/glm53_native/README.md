# GLM-5.3 Native Artifact Layer

This directory owns L2 native artifacts for GLM-5.3 work. It is intentionally
separate from the rootfs image and the `glm53_ep` Python environment:

- L0 rootfs provides system toolchains, CUDA 13.2, CUTLASS headers, ccache, and
  linkable CUDA libraries.
- L1 `glm53_ep` provides the Python/Torch environment.
- L2 native packages are wheels built against that exact env and published to
  HDFS under `torchtitan/native-wheels/<pkg>/<key>/`.

Builds run inside the `glm53_ep` rootfs. The host wrapper only re-enters the
rootfs and publishes or stages completed artifacts through the existing
Waystone helpers, because the Waystone checkout is not bound inside bwrap.

Typical commands:

```bash
experiments/glm53_native/build_native.sh deepep_v2
experiments/glm53_native/install_native.sh deepep_v2 --key <build-key>
```

The build key is the SHA-256 of source content after patches, patch content,
torch version, CUDA version, target arch, package metadata, and build args.
`--dev` installs an editable source tree inside `glm53_ep` with ccache enabled
and does not publish.

The package lock currently contains:

- `deepep_v2`: pinned DeepEP v2 `ElasticBuffer`.
- `torchao`: source-built torchao after DeepEP. It stays out of L1 because the
  PyPI 0.18.0 wheel trips Triton 3.9 import-time autotune-key validation.
- `glm53_kernel_examples`: in-repo C++/CUDA `torch.library` example package
  used to validate the native-op convention.

