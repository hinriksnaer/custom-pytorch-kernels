# Custom MLP Kernels For PyTorch

A simple implementation of custom MLP kernels for PyTorch. This includes both a CUDA kernel for fused linear and ReLU operations on a GPU. Currently, the triton implementation uses standard PyTorch operations for backward pass, but it can be extended to use custom backward kernels as well.

```

--- Inference Benchmark ---
Baseline MLP (PyTorch)   | Latency: 0.19 ms | Throughput: 5,377,260 samples/sec
Custom CUDA MLP          | Latency: 4.04 ms | Throughput: 251,912 samples/sec
Custom Triton MLP        | Latency: 0.21 ms | Throughput: 4,754,199 samples/sec

--- Training Benchmark ---
Baseline MLP (PyTorch)   | Latency: 0.84 ms | Throughput: 1,203,966 samples/sec | Accuracy: 96.47%
Custom CUDA MLP          | Latency: 53.43 ms | Throughput: 19,032 samples/sec | Accuracy: 96.31%
Custom Triton MLP        | Latency: 0.95 ms | Throughput: 1,067,412 samples/sec | Accuracy: 96.47%
```

