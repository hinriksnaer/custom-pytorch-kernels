# Custom MLP Kernels For PyTorch

A simple implementation of custom MLP kernels for PyTorch. This includes both a CUDA kernel for fused linear and ReLU operations on a GPU. Currently, the triton implementation uses standard PyTorch operations for backward pass, but it can be extended to use custom backward kernels as well.

