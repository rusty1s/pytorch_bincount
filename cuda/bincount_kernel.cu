#include <ATen/ATen.h>
#include <THC/THCAtomics.cuh>

#define THREADS 1024
#define BLOCKS(N) (N + THREADS - 1) / THREADS

template <typename scalar_t>
__global__ void bincount_cuda_kernel(scalar_t *__restrict__ src, int64_t *out,
                                     size_t numel) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;
  for (ptrdiff_t i = index; i < numel; i += stride) {
    atomicAdd(out + (ptrdiff_t)src[i], 1);
  }
}

at::Tensor bincount_cuda(at::Tensor src, int64_t size) {
  auto out = at::zeros(size, src.type().toScalarType(at::kLong));

  AT_DISPATCH_ALL_TYPES(src.type(), "bincount_cuda_kernel", [&] {
    bincount_cuda_kernel<scalar_t><<<BLOCKS(src.numel()), THREADS>>>(
        src.data<scalar_t>(), out.data<int64_t>(), src.numel());
  });

  return out;
}
