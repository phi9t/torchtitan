# DeepEP v2 Patches

- `0001-add-nccl-library-dir.patch`: add the discovered NCCL wheel `lib`
  directory to `library_dirs` so exact-name linking against `libnccl.so.2`
  resolves when NCCL comes from the `nvidia-nccl-cu13` pip wheel.

