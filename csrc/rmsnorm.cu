#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

// RMSNorm over the last dimension:
//   ms   = mean(x^2)                      (over the feature dim D)
//   r    = rsqrt(ms + eps)
//   xhat = x * r
//   y    = weight * xhat
//
// This matches src/model.py RMSNorm: x * rsqrt(x.pow(2).mean(-1) + eps) * scale.
// Layout: input is treated as [rows, D] (rows = product of all leading dims),
// one CUDA block per row, threads cooperatively reduce over D.

// ---------------------------------------------------------------------------
// Forward
// ---------------------------------------------------------------------------
__global__
void rmsnorm_forward_kernel(const float* x, const float* weight, const int D,
                            const float eps, float* y, float* rrms) {
    int row = blockIdx.x;              // one block per row
    int tx = threadIdx.x;
    int nthreads = blockDim.x;

    const float* x_row = x + row * D;
    float* y_row = y + row * D;

    // Each thread accumulates a partial sum of squares over a strided slice.
    float local_sumsq = 0.0f;
    for (int i = tx; i < D; i += nthreads) {
        float v = x_row[i];
        local_sumsq += v * v;
    }

    // Block reduction of sum-of-squares via shared memory.
    extern __shared__ float sdata[];
    sdata[tx] = local_sumsq;
    __syncthreads();

    for (int stride = nthreads / 2; stride > 0; stride >>= 1) {
        if (tx < stride) {
            sdata[tx] += sdata[tx + stride];
        }
        __syncthreads();
    }

    // Thread 0 finalizes r = rsqrt(mean(x^2) + eps) and shares it.
    __shared__ float r_shared;
    if (tx == 0) {
        float mean_sq = sdata[0] / (float) D;
        float r = rsqrtf(mean_sq + eps);
        r_shared = r;
        rrms[row] = r;  // cache rsqrt for the backward pass
    }
    __syncthreads();

    float r = r_shared;
    for (int i = tx; i < D; i += nthreads) {
        float xhat = x_row[i] * r;
        y_row[i] = weight[i] * xhat;
    }
}

// ---------------------------------------------------------------------------
// Backward
// ---------------------------------------------------------------------------
// Given dy, x, weight and the cached r = rsqrt(mean(x^2)+eps):
//   xhat    = x * r
//   dxhat   = dy * weight
//   dx      = r * (dxhat - xhat * mean(dxhat * xhat))    (mean over D)
//   dweight = sum_over_rows(dy * xhat)
//
// One block per row computes dx and the row's partial contribution to dweight.
// dweight partials are accumulated across rows with atomicAdd.
__global__
void rmsnorm_backward_kernel(const float* dy, const float* x, const float* weight,
                             const float* rrms, const int D,
                             float* dx, float* dweight) {
    int row = blockIdx.x;
    int tx = threadIdx.x;
    int nthreads = blockDim.x;

    const float* dy_row = dy + row * D;
    const float* x_row = x + row * D;
    float* dx_row = dx + row * D;

    float r = rrms[row];

    // Reduce sum(dxhat * xhat) = sum(dy * weight * x * r) over the feature dim.
    float local_dot = 0.0f;
    for (int i = tx; i < D; i += nthreads) {
        float xhat = x_row[i] * r;
        float dxhat = dy_row[i] * weight[i];
        local_dot += dxhat * xhat;
    }

    extern __shared__ float sdata[];
    sdata[tx] = local_dot;
    __syncthreads();

    for (int stride = nthreads / 2; stride > 0; stride >>= 1) {
        if (tx < stride) {
            sdata[tx] += sdata[tx + stride];
        }
        __syncthreads();
    }

    __shared__ float mean_dot_shared;
    if (tx == 0) {
        mean_dot_shared = sdata[0] / (float) D;  // mean(dxhat * xhat)
    }
    __syncthreads();
    float mean_dot = mean_dot_shared;

    for (int i = tx; i < D; i += nthreads) {
        float xhat = x_row[i] * r;
        float dxhat = dy_row[i] * weight[i];
        dx_row[i] = r * (dxhat - xhat * mean_dot);

        // Accumulate weight gradient across all rows.
        atomicAdd(&dweight[i], dy_row[i] * xhat);
    }
}

// ---------------------------------------------------------------------------
// C++ wrappers
// ---------------------------------------------------------------------------
static int threads_for_dim(int D) {
    // Power-of-two thread count (<= 1024) so the tree reduction is exact.
    int t = 1;
    while (t < D && t < 1024) t <<= 1;
    return t;
}

// Returns {y, rrms}: y has the same shape as x, rrms holds the per-row rsqrt
// (needed by backward). weight is 1-D of length D (the last dim of x).
std::vector<torch::Tensor> rmsnorm_forward(torch::Tensor x, torch::Tensor weight, double eps) {
    const int D = x.size(-1);
    const int rows = x.numel() / D;

    auto y = torch::empty_like(x);
    auto rrms = torch::empty({rows}, x.options());

    const int threads = threads_for_dim(D);
    const int sram_size = threads * sizeof(float);

    rmsnorm_forward_kernel<<<rows, threads, sram_size>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), D, (float) eps,
        y.data_ptr<float>(), rrms.data_ptr<float>()
    );

    return {y, rrms};
}

// Returns {dx, dweight}. rrms is the tensor returned by rmsnorm_forward.
std::vector<torch::Tensor> rmsnorm_backward(torch::Tensor dy, torch::Tensor x,
                                            torch::Tensor weight, torch::Tensor rrms) {
    const int D = x.size(-1);
    const int rows = x.numel() / D;

    auto dx = torch::empty_like(x);
    auto dweight = torch::zeros({D}, weight.options());  // atomicAdd accumulates here

    const int threads = threads_for_dim(D);
    const int sram_size = threads * sizeof(float);

    rmsnorm_backward_kernel<<<rows, threads, sram_size>>>(
        dy.data_ptr<float>(), x.data_ptr<float>(), weight.data_ptr<float>(),
        rrms.data_ptr<float>(), D,
        dx.data_ptr<float>(), dweight.data_ptr<float>()
    );

    return {dx, dweight};
}
