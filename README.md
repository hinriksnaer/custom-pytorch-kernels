# Custom MLP Kernels For PyTorch

A simple implementation of custom MLP kernels for PyTorch. This includes both a CUDA kernel for fused linear and ReLU operations on a GPU. Currently, the triton implementation uses standard PyTorch operations for backward pass, but it can be extended to use custom backward kernels as well.

```

--- Inference Benchmark ---
Baseline MLP (PyTorch)   | Latency: 0.19 ms | Throughput: 5,474,053 samples/sec | Accuracy: 7.87%
Custom CUDA MLP          | Latency: 3.96 ms | Throughput: 256,761 samples/sec | Accuracy: 9.96%
Custom Triton MLP        | Latency: 0.22 ms | Throughput: 4,669,434 samples/sec | Accuracy: 7.89%

--- Training Benchmark ---
Baseline MLP (PyTorch)   | Latency: 0.70 ms | Throughput: 1,453,485 samples/sec | Accuracy: 67.50%
Custom CUDA MLP          | Latency: 56.19 ms | Throughput: 18,099 samples/sec | Accuracy: 66.84%
Custom Triton MLP        | Latency: 0.79 ms | Throughput: 1,287,270 samples/sec | Accuracy: 66.41%
```

