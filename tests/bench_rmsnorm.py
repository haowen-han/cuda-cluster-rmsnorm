"""Benchmark script for RMSNorm CUDA kernels."""
import torch
import sys
import os
import rmsnorm_cluster


def benchmark(func, *args, warmup=20, repeat=200, **kwargs):
    for _ in range(warmup):
        func(*args, **kwargs)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(repeat):
        func(*args, **kwargs)
    end_event.record()
    torch.cuda.synchronize()

    return start_event.elapsed_time(end_event) / repeat  # ms


def print_occupancy(m, k):
    print("=" * 70)
    print("Occupancy Info (per SM)")
    print("=" * 70)

    for bs in [256, 512, 1024]:
        blocks = rmsnorm_cluster.get_max_active_blocks_baseline(bs)
        print(f"  Baseline bs={bs:4d}: max active blocks/SM = {blocks}, total blocks = {m}")

    for cs in [2, 4, 8]:
        max_clusters = rmsnorm_cluster.get_max_active_clusters(cs)
        total_blocks = m * cs
        print(f"  Cluster  cs={cs}:  max active clusters = {max_clusters}, "
              f"max active blocks = {max_clusters * cs}, "
              f"total blocks = {total_blocks}")
    print()


def run_benchmark(m, k, eps=1e-6):
    print(f"=== Benchmark: m={m}, k={k} ===")

    torch.manual_seed(0)
    x = torch.randn(m, k, device='cuda', dtype=torch.float32)
    gamma = torch.randn(k, device='cuda', dtype=torch.float32)

    effective_bytes = m * k * 4 * 4  # 4 ops * 4 bytes each
    effective_gb = effective_bytes / 1e9

    # Baseline variants
    baseline_results = {}
    for bs in [256, 512, 1024]:
        t = benchmark(rmsnorm_cluster.rmsnorm_baseline, x, gamma, eps, bs)
        bw = effective_gb / (t / 1000)
        baseline_results[bs] = (t, bw)

    # Cluster variants
    cluster_results = {}
    for cs in [2, 4, 8]:
        t = benchmark(rmsnorm_cluster.rmsnorm_cluster, x, gamma, eps, cs)
        bw = effective_gb / (t / 1000)
        cluster_results[cs] = (t, bw)

    # Use baseline 256 as reference
    _, bw_ref = baseline_results[256]

    print(f"{'Kernel':<25} {'Time(ms)':>10} {'GB/s':>10} {'Speedup vs B256':>16}")
    print("-" * 65)
    for bs in [256, 512, 1024]:
        t, bw = baseline_results[bs]
        label = f"Baseline bs={bs}"
        print(f"{label:<25} {t:>10.4f} {bw:>10.2f} {bw/bw_ref:>16.2f}x")
    for cs in [2, 4, 8]:
        t, bw = cluster_results[cs]
        label = f"Cluster cs={cs}"
        print(f"{label:<25} {t:>10.4f} {bw:>10.2f} {bw/bw_ref:>16.2f}x")
    print()


if __name__ == '__main__':
    m, k = 64, 131072

    print_occupancy(m, k)
    run_benchmark(m, k)
    run_benchmark(32, 65536)
