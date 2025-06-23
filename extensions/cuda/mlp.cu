#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void linear_relu_forward_kernel(const float *x, const float *w,
                                           const float *b, float *z, float *out,
                                           int B, int I, int O) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < B * O) {
    int row = i / O;
    int col = i % O;
    float acc = b[col];
    for (int k = 0; k < I; ++k) {
      acc += x[row * I + k] * w[col * I + k];
    }
    z[row * O + col] = acc;
    out[row * O + col] = acc > 0.0f ? acc : 0.0f;
  }
}

__global__ void linear_relu_backward_kernel(const float *grad_out,
                                            const float *z, const float *x,
                                            const float *w, float *grad_x,
                                            float *grad_w, float *grad_b, int B,
                                            int I, int O) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < B * O) {
    int row = i / O;
    int col = i % O;
    float dz = (z[row * O + col] > 0.0f) ? grad_out[row * O + col] : 0.0f;

    // Atomic add for grad_b
    atomicAdd(&grad_b[col], dz);

    for (int k = 0; k < I; ++k) {
      atomicAdd(&grad_w[col * I + k], dz * x[row * I + k]);
      atomicAdd(&grad_x[row * I + k], dz * w[col * I + k]);
    }
  }
}

void mlp_forward_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b,
                      torch::Tensor z, torch::Tensor out) {
  int B = x.size(0), I = x.size(1), O = w.size(0);
  int threads = 256;
  int blocks = (B * O + threads - 1) / threads;
  linear_relu_forward_kernel<<<blocks, threads>>>(
      x.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(),
      z.data_ptr<float>(), out.data_ptr<float>(), B, I, O);
}

void mlp_backward_cuda(torch::Tensor grad_out, torch::Tensor z, torch::Tensor x,
                       torch::Tensor w, torch::Tensor grad_x,
                       torch::Tensor grad_w, torch::Tensor grad_b) {
  int B = x.size(0), I = x.size(1), O = w.size(0);
  int threads = 256;
  int blocks = (B * O + threads - 1) / threads;
  linear_relu_backward_kernel<<<blocks, threads>>>(
      grad_out.data_ptr<float>(), z.data_ptr<float>(), x.data_ptr<float>(),
      w.data_ptr<float>(), grad_x.data_ptr<float>(), grad_w.data_ptr<float>(),
      grad_b.data_ptr<float>(), B, I, O);
}
