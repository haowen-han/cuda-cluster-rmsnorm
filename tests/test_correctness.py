"""Correctness test for RMSNorm CUDA kernels."""
import torch
import sys
import os
import rmsnorm_cluster


def cpu_rmsnorm(x, gamma, eps=1e-6):
    variance = torch.mean(x ** 2, dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + eps)
    return x * inv_rms * gamma


def test_correctness(m, k, eps=1e-6):
    print(f"=== Testing m={m}, k={k}, eps={eps} ===")

    torch.manual_seed(42)
    x = torch.randn(m, k, device='cuda', dtype=torch.float32)
    gamma = torch.randn(k, device='cuda', dtype=torch.float32)

    y_ref = cpu_rmsnorm(x.cpu(), gamma.cpu(), eps).cuda()

    all_pass = True
    for bs in [256, 512, 1024]:
        y_b = rmsnorm_cluster.rmsnorm_baseline(x, gamma, eps, bs)
        err = (y_b - y_ref).abs()
        max_abs = err.max().item()
        max_rel = (err / (y_ref.abs() + 1e-8)).max().item()
        passed = max_abs < 1e-4
        all_pass = all_pass and passed
        print(f"  Baseline bs={bs:4d}: max_abs_err={max_abs:.2e}, max_rel_err={max_rel:.2e}  "
              f"{'PASS' if passed else 'FAIL'}")

    for cs in [2, 4, 8]:
        y_c = rmsnorm_cluster.rmsnorm_cluster(x, gamma, eps, cs)
        err = (y_c - y_ref).abs()
        max_abs = err.max().item()
        max_rel = (err / (y_ref.abs() + 1e-8)).max().item()
        passed = max_abs < 1e-4
        all_pass = all_pass and passed
        print(f"  Cluster  cs={cs}:  max_abs_err={max_abs:.2e}, max_rel_err={max_rel:.2e}  "
              f"{'PASS' if passed else 'FAIL'}")

    print()
    return all_pass


if __name__ == '__main__':
    ok = test_correctness(m=64, k=131072, eps=1e-6)
    ok = test_correctness(m=32, k=65536, eps=1e-6) and ok
    ok = test_correctness(m=128, k=32768, eps=1e-6) and ok

    print("=" * 50)
    print(f"Overall: {'ALL PASSED' if ok else 'SOME FAILED'}")
    sys.exit(0 if ok else 1)
