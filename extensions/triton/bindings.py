import torch
from torch.autograd import Function
import triton
import triton.language as tl
from .mlp import fused_linear_relu_kernel, BLOCK_M, BLOCK_N, BLOCK_K


class CustomMLPFunctionTriton(Function):
    @staticmethod
    def forward(ctx, x, w, b):
        M, K = x.shape
        N = w.shape[0]
        y = torch.empty((M, N), device=x.device, dtype=x.dtype)

        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

        fused_linear_relu_kernel[grid](
            x,
            w,
            b,
            y,
            M,
            N,
            K,
            x.stride(0),
            x.stride(1),
            w.stride(0),
            w.stride(1),
            y.stride(0),
            y.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
        )

        ctx.save_for_backward(x, w, b, y)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w, b, y = ctx.saved_tensors
        dy_relu = dy.clone()
        dy_relu[y <= 0] = 0

        dx = dy_relu @ w
        dw = dy_relu.t() @ x
        db = dy_relu.sum(0)

        return dx, dw, db


