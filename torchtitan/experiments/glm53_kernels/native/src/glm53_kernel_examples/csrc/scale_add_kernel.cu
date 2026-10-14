#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace {

template <typename scalar_t>
__global__ void scale_add_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ y,
    scalar_t* __restrict__ out,
    double alpha,
    int64_t n) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    float x_val = static_cast<float>(x[idx]);
    float y_val = static_cast<float>(y[idx]);
    out[idx] = static_cast<scalar_t>(x_val * static_cast<float>(alpha) + y_val);
  }
}

} // namespace

at::Tensor scale_add_cuda(
    const at::Tensor& x,
    const at::Tensor& y,
    double alpha) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(y.is_cuda(), "y must be a CUDA tensor");
  TORCH_CHECK(x.sizes() == y.sizes(), "x and y must have the same shape");
  TORCH_CHECK(x.scalar_type() == y.scalar_type(), "x and y must have the same dtype");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(y.is_contiguous(), "y must be contiguous");

  const c10::cuda::CUDAGuard device_guard(x.device());
  auto out = at::empty_like(x);
  int64_t n = x.numel();
  if (n == 0) {
    return out;
  }

  constexpr int threads = 256;
  int blocks = static_cast<int>((n + threads - 1) / threads);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      x.scalar_type(),
      "glm53_native_scale_add",
      [&] {
        scale_add_kernel<scalar_t><<<
            blocks,
            threads,
            0,
            at::cuda::getCurrentCUDAStream()>>>(
            x.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            alpha,
            n);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

