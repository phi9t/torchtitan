#include <ATen/ATen.h>
#include <torch/library.h>

namespace {

at::Tensor scale_add_cpu(
    const at::Tensor& x,
    const at::Tensor& y,
    double alpha) {
  TORCH_CHECK(x.device().is_cpu(), "x must be a CPU tensor");
  TORCH_CHECK(y.device().is_cpu(), "y must be a CPU tensor");
  TORCH_CHECK(x.sizes() == y.sizes(), "x and y must have the same shape");
  TORCH_CHECK(x.scalar_type() == y.scalar_type(), "x and y must have the same dtype");
  return at::add(y, x, alpha);
}

} // namespace

TORCH_LIBRARY(glm53_native, m) {
  m.def("scale_add(Tensor x, Tensor y, float alpha) -> Tensor");
}

TORCH_LIBRARY_IMPL(glm53_native, CPU, m) {
  m.impl("scale_add", scale_add_cpu);
}

