import torch

import mlp_cuda

forward = mlp_cuda.forward
backward = mlp_cuda.backward

__all__ = ["forward", "backward"]


class CustomMLPFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b):
        B, I = x.shape
        O = w.shape[0]
        z = torch.empty((B, O), device="cuda")
        out = torch.empty((B, O), device="cuda")
        forward(x, w, b, z, out)
        ctx.save_for_backward(x, w, b, z)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, w, b, z = ctx.saved_tensors
        grad_x = torch.zeros_like(x)
        grad_w = torch.zeros_like(w)
        grad_b = torch.zeros_like(b)
        backward(grad_out.contiguous(), z, x, w, grad_x, grad_w, grad_b)
        return grad_x, grad_w, grad_b
