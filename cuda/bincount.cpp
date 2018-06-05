#include <torch/torch.h>

#define CHECK_CUDA(x) AT_ASSERT(x.type().is_cuda(), #x " must be a CUDA tensor")

at::Tensor bincount_cuda(at::Tensor src, int64_t size);

at::Tensor bincount(at::Tensor src, int64_t size) {
  CHECK_CUDA(src);
  return bincount_cuda(src, size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bincount", &bincount, "BinCount (CUDA)");
}
