import torch
import torch.nn as nn
import torch.nn.functional as F
from extensions.cuda.bindings import CustomMLPFunctionCUDA
from extensions.triton.bindings import CustomMLPFunctionTriton


class BaselineMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.l1 = nn.Linear(in_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.l1(x))
        return self.l2(x)


class CustomLinearReLU(nn.Module):
    def __init__(self, in_dim, out_dim, backend: str = "cuda"):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_dim))

        if backend == "cuda":
            self.mlp_fn = CustomMLPFunctionCUDA
        elif backend == "triton":
            self.mlp_fn = CustomMLPFunctionTriton
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def forward(self, x):
        return self.mlp_fn.apply(x, self.weight, self.bias)


class CustomCUDAMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.l1 = CustomLinearReLU(in_dim, hidden_dim, backend="cuda")
        self.l2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.l1(x)
        return self.l2(x)


class CustomTritonMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.l1 = CustomLinearReLU(in_dim, hidden_dim, backend="triton")
        self.l2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.l1(x)
        return self.l2(x)
