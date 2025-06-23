#include <torch/extension.h>

void mlp_forward_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b,
                      torch::Tensor z, torch::Tensor out);
void mlp_backward_cuda(torch::Tensor grad_out, torch::Tensor z, torch::Tensor x,
                       torch::Tensor w, torch::Tensor grad_x,
                       torch::Tensor grad_w, torch::Tensor grad_b);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &mlp_forward_cuda, "MLP forward (CUDA)");
  m.def("backward", &mlp_backward_cuda, "MLP backward (CUDA)");
}
