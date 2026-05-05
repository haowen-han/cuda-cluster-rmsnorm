#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>

namespace cg = cooperative_groups;

// ============================================================
// Baseline Kernel: one block per row
// ============================================================
template <int BLOCK_SIZE>
__global__ void rmsnorm_baseline_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    float* __restrict__ y,
    const int m,
    const int k,
    const float eps)
{
    const int row = blockIdx.x;
    if (row >= m) return;

    const float4* x_f4 = reinterpret_cast<const float4*>(x + row * k);
    const float4* g_f4 = reinterpret_cast<const float4*>(gamma);
    float4*       y_f4 = reinterpret_cast<float4*>(y + row * k);
    const int num_f4 = k / 4;

    // ---- Phase 1: compute partial sum of squares ----
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < num_f4; i += BLOCK_SIZE) {
        float4 xv = x_f4[i];
        local_sum += xv.x * xv.x + xv.y * xv.y + xv.z * xv.z + xv.w * xv.w;
    }

    // ---- Phase 2: warp-level reduce, then block-level reduce ----
    constexpr int warp_size = 32;
    constexpr int num_warps = BLOCK_SIZE / warp_size;
    int warp_id = threadIdx.x / warp_size;
    int lane_id = threadIdx.x % warp_size;

    // Warp-level reduce using shuffle
#pragma unroll
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);
    }

    // Each warp's result is now in lane 0, write to smem
    __shared__ float s_sum[num_warps];
    if (lane_id == 0) {
        s_sum[warp_id] = local_sum;
    }
    __syncthreads();

    // First warp reads from smem and does a final warp-level reduce
    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? s_sum[lane_id] : 0.0f;
#pragma unroll
        for (int offset = warp_size / 2; offset > 0; offset /= 2) {
            val += __shfl_xor_sync(0xffffffff, val, offset);
        }
        if (lane_id == 0) {
            s_sum[0] = val;
        }
    }
    __syncthreads();

    float inv_rms = rsqrtf(s_sum[0] / k + eps);

    // ---- Phase 3: normalize ----
    for (int i = threadIdx.x; i < num_f4; i += BLOCK_SIZE) {
        float4 xv = x_f4[i];
        float4 gv = g_f4[i];
        float4 yv;
        yv.x = xv.x * inv_rms * gv.x;
        yv.y = xv.y * inv_rms * gv.y;
        yv.z = xv.z * inv_rms * gv.z;
        yv.w = xv.w * inv_rms * gv.w;
        y_f4[i] = yv;
    }
}

// ============================================================
// Cluster RMSNorm Kernel
// ============================================================
template <int BLOCK_SIZE, int CLUSTER_SIZE>
__global__ void rmsnorm_cluster_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    float* __restrict__ y,
    const int m,
    const int k,
    const float eps)
{
    cg::cluster_group cluster = cg::this_cluster();
    const int block_rank = cluster.block_rank();
    const int row = blockIdx.x / CLUSTER_SIZE;
    if (row >= m) return;

    const float4* x_f4 = reinterpret_cast<const float4*>(x + row * k);
    const float4* g_f4 = reinterpret_cast<const float4*>(gamma);
    float4*       y_f4 = reinterpret_cast<float4*>(y + row * k);
    const int num_f4 = k / 4;

    // ---- Phase 1: each thread accumulates partial sum ----
    float local_sum = 0.0f;
    for (int i = block_rank * BLOCK_SIZE + threadIdx.x;
         i < num_f4;
         i += CLUSTER_SIZE * BLOCK_SIZE) {
        float4 xv = x_f4[i];
        local_sum += xv.x * xv.x + xv.y * xv.y + xv.z * xv.z + xv.w * xv.w;
    }

    // ---- Phase 2: warp-level reduce, then block-level reduce ----
    constexpr int warp_size = 32;
    constexpr int num_warps = BLOCK_SIZE / warp_size;
    int warp_id = threadIdx.x / warp_size;
    int lane_id = threadIdx.x % warp_size;

    // Warp-level reduce using shuffle
#pragma unroll
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);
    }

    // Each warp's result is now in lane 0, write to smem
    __shared__ float s_partial[num_warps];
    if (lane_id == 0) {
        s_partial[warp_id] = local_sum;
    }
    __syncthreads();

    // First warp reads from smem and does a final warp-level reduce
    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? s_partial[lane_id] : 0.0f;
#pragma unroll
        for (int offset = warp_size / 2; offset > 0; offset /= 2) {
            val += __shfl_xor_sync(0xffffffff, val, offset);
        }
        if (lane_id == 0) {
            s_partial[0] = val;
        }
    }
    __syncthreads();
    // s_partial[0] now holds this block's partial sum

    // ---- Phase 3: cross-block reduce via DSM ----
    cluster.sync();

    __shared__ float s_inv_rms;
    if (block_rank == 0 && threadIdx.x == 0) {
        float total_sum = 0.0f;
        for (int r = 0; r < CLUSTER_SIZE; r++) {
            const float* remote = cluster.map_shared_rank(&s_partial[0], r);
            total_sum += *remote;
        }
        s_inv_rms = rsqrtf(total_sum / k + eps);
    }

    cluster.sync();

    // All blocks read inv_rms from block 0 via DSM
    const float inv_rms = *cluster.map_shared_rank(&s_inv_rms, 0);

    // ---- Phase 4: normalize ----
    for (int i = block_rank * BLOCK_SIZE + threadIdx.x;
         i < num_f4;
         i += CLUSTER_SIZE * BLOCK_SIZE) {
        float4 xv = x_f4[i];
        float4 gv = g_f4[i];
        float4 yv;
        yv.x = xv.x * inv_rms * gv.x;
        yv.y = xv.y * inv_rms * gv.y;
        yv.z = xv.z * inv_rms * gv.z;
        yv.w = xv.w * inv_rms * gv.w;
        y_f4[i] = yv;
    }
}

// ============================================================
// Launchers
// ============================================================

torch::Tensor rmsnorm_baseline(torch::Tensor x, torch::Tensor gamma, float eps, int64_t block_size) {
    TORCH_CHECK(x.is_cuda() && gamma.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(x.is_contiguous() && gamma.is_contiguous(), "Inputs must be contiguous");
    TORCH_CHECK(x.dim() == 2, "x must be 2-D [m, k]");
    TORCH_CHECK(gamma.dim() == 1, "gamma must be 1-D [k]");
    TORCH_CHECK(x.size(1) == gamma.size(0), "x.size(1) must equal gamma.size(0)");
    TORCH_CHECK(x.size(1) % 4 == 0, "k must be divisible by 4 (for float4)");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Only float32 is supported");
    TORCH_CHECK(block_size == 256 || block_size == 512 || block_size == 1024,
                "block_size must be 256, 512, or 1024, got ", block_size);

    auto y = torch::empty_like(x);
    const int m = x.size(0);
    const int k = x.size(1);
    dim3 grid(m);

    switch (block_size) {
        case 256:
            rmsnorm_baseline_kernel<256><<<grid, dim3(256), 0, at::cuda::getCurrentCUDAStream()>>>(
                x.data_ptr<float>(), gamma.data_ptr<float>(), y.data_ptr<float>(), m, k, eps);
            break;
        case 512:
            rmsnorm_baseline_kernel<512><<<grid, dim3(512), 0, at::cuda::getCurrentCUDAStream()>>>(
                x.data_ptr<float>(), gamma.data_ptr<float>(), y.data_ptr<float>(), m, k, eps);
            break;
        case 1024:
            rmsnorm_baseline_kernel<1024><<<grid, dim3(1024), 0, at::cuda::getCurrentCUDAStream()>>>(
                x.data_ptr<float>(), gamma.data_ptr<float>(), y.data_ptr<float>(), m, k, eps);
            break;
    }
    C10_CUDA_CHECK(cudaGetLastError());
    return y;
}

torch::Tensor rmsnorm_cluster(torch::Tensor x, torch::Tensor gamma, float eps, int64_t cluster_size) {
    TORCH_CHECK(x.is_cuda() && gamma.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(x.is_contiguous() && gamma.is_contiguous(), "Inputs must be contiguous");
    TORCH_CHECK(x.dim() == 2, "x must be 2-D [m, k]");
    TORCH_CHECK(gamma.dim() == 1, "gamma must be 1-D [k]");
    TORCH_CHECK(x.size(1) == gamma.size(0), "x.size(1) must equal gamma.size(0)");
    TORCH_CHECK(x.size(1) % 4 == 0, "k must be divisible by 4 (for float4)");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Only float32 is supported");
    TORCH_CHECK(cluster_size == 2 || cluster_size == 4 || cluster_size == 8,
                "cluster_size must be 2, 4, or 8, got ", cluster_size);

    auto y = torch::empty_like(x);
    const int m = x.size(0);
    const int k = x.size(1);
    constexpr int BLOCK_SIZE = 256;

    const float* x_ptr   = x.data_ptr<float>();
    const float* g_ptr   = gamma.data_ptr<float>();
    float*       y_ptr   = y.data_ptr<float>();
    int m_val = m, k_val = k;
    float eps_val = eps;

    auto launch = [&](auto kernel) {
        cudaLaunchConfig_t cfg = {};
        cfg.blockDim = {BLOCK_SIZE, 1, 1};
        cfg.dynamicSmemBytes = 0;
        cfg.stream = at::cuda::getCurrentCUDAStream();

        cudaLaunchAttribute attr{};
        attr.id = cudaLaunchAttributeClusterDimension;
        attr.val.clusterDim = {(unsigned int)cluster_size, 1, 1};
        cfg.attrs = &attr;
        cfg.numAttrs = 1;
        cfg.gridDim = {(unsigned int)(m * cluster_size), 1, 1};

        C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel,
            x_ptr, g_ptr, y_ptr, m_val, k_val, eps_val));
    };

    switch (cluster_size) {
        case 2:  launch(rmsnorm_cluster_kernel<BLOCK_SIZE, 2>);  break;
        case 4:  launch(rmsnorm_cluster_kernel<BLOCK_SIZE, 4>);  break;
        case 8:  launch(rmsnorm_cluster_kernel<BLOCK_SIZE, 8>);  break;
        default: TORCH_CHECK(false, "Unsupported cluster_size: ", cluster_size);
    }

    return y;
}

// ============================================================
// Occupancy queries
// ============================================================

int64_t get_max_active_blocks_baseline(int64_t block_size) {
    TORCH_CHECK(block_size == 256 || block_size == 512 || block_size == 1024,
                "block_size must be 256, 512, or 1024, got ", block_size);
    int num_blocks = 0;
    switch (block_size) {
        case 256:
            C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &num_blocks, rmsnorm_baseline_kernel<256>, 256, 0));
            break;
        case 512:
            C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &num_blocks, rmsnorm_baseline_kernel<512>, 512, 0));
            break;
        case 1024:
            C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &num_blocks, rmsnorm_baseline_kernel<1024>, 1024, 0));
            break;
    }
    return num_blocks;
}

int64_t get_max_active_clusters(int64_t cluster_size) {
    TORCH_CHECK(cluster_size == 2 || cluster_size == 4 || cluster_size == 8,
                "cluster_size must be 2, 4, or 8, got ", cluster_size);
    constexpr int BLOCK_SIZE = 256;
    cudaLaunchConfig_t config = {};
    config.blockDim = {BLOCK_SIZE, 1, 1};
    config.dynamicSmemBytes = 0;
    // gridDim must be a multiple of clusterDim for the occupancy API
    config.gridDim = {(unsigned int)cluster_size, 1, 1};

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    config.attrs = attrs;
    config.numAttrs = 1;

    int num_clusters = 0;
    switch (cluster_size) {
        case 2:
            attrs[0].val.clusterDim = {2, 1, 1};
            C10_CUDA_CHECK(cudaOccupancyMaxActiveClusters(
                &num_clusters, (const void*)rmsnorm_cluster_kernel<BLOCK_SIZE, 2>, &config));
            break;
        case 4:
            attrs[0].val.clusterDim = {4, 1, 1};
            C10_CUDA_CHECK(cudaOccupancyMaxActiveClusters(
                &num_clusters, (const void*)rmsnorm_cluster_kernel<BLOCK_SIZE, 4>, &config));
            break;
        case 8:
            attrs[0].val.clusterDim = {8, 1, 1};
            C10_CUDA_CHECK(cudaOccupancyMaxActiveClusters(
                &num_clusters, (const void*)rmsnorm_cluster_kernel<BLOCK_SIZE, 8>, &config));
            break;
        default:
            TORCH_CHECK(false, "Unsupported cluster_size: ", cluster_size);
    }
    return num_clusters;
}
