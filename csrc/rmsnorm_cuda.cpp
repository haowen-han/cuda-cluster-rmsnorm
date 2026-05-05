#include <torch/extension.h>

// Forward declarations (implemented in rmsnorm_kernel.cu)
torch::Tensor rmsnorm_baseline(torch::Tensor x, torch::Tensor gamma, float eps, int64_t block_size);
torch::Tensor rmsnorm_cluster(torch::Tensor x, torch::Tensor gamma, float eps, int64_t cluster_size);
int64_t get_max_active_blocks_baseline(int64_t block_size);
int64_t get_max_active_clusters(int64_t cluster_size);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm_baseline",  &rmsnorm_baseline,  "RMSNorm baseline (one block per row)");
    m.def("rmsnorm_cluster",   &rmsnorm_cluster,   "RMSNorm with Thread Block Cluster");
    m.def("get_max_active_blocks_baseline", &get_max_active_blocks_baseline);
    m.def("get_max_active_clusters",        &get_max_active_clusters);
}
