import triton
import triton.language as tl


BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 32


@triton.jit
def fused_linear_relu_kernel(
    X_ptr,
    W_ptr,
    B_ptr,
    Y_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_ym,
    stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        x_ptrs = X_ptr + (
            offs_m[:, None] * stride_xm + (k + offs_k)[None, :] * stride_xk
        )
        w_ptrs = W_ptr + (
            offs_n[None, :] * stride_wn + (k + offs_k)[:, None] * stride_wk
        )

        x = tl.load(
            x_ptrs, mask=(offs_m[:, None] < M) & (k + offs_k)[None, :] < K, other=0.0
        )
        w = tl.load(
            w_ptrs, mask=(offs_n[None, :] < N) & (k + offs_k)[:, None] < K, other=0.0
        )

        acc += tl.dot(x, w)

    b = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0)[None, :]
    acc += b
    acc = tl.maximum(acc, 0.0)

    y_ptrs = Y_ptr + (offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn)
    tl.store(y_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
