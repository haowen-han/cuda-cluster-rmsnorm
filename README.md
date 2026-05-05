# CUDA Thread Block Cluster RMSNorm Tutorial

在 NVIDIA B200 (sm_100) 上使用 CUDA Thread Block Cluster 与 Distributed Shared Memory (DSM) 加速 RMSNorm 的教学项目。

核心论点：当 RMSNorm 输入满足"小 m、大 k"时，传统 one-block-per-row 因 block 数不足无法打满 148 个 SM；Cluster 让多个 block 协作处理一行，提升 SM 利用率，且在单 kernel 内通过 DSM 完成跨 block reduction。

## 安装

```bash
cd CUDA_Cluster_tutorial
python setup.py install
```

## 运行测试

```bash
# 正确性测试
CUDA_VISIBLE_DEVICES=0 python tests/test_correctness.py

# 性能 benchmark
CUDA_VISIBLE_DEVICES=0 python tests/bench_rmsnorm.py
```

## 测试结果
### 环境
```
Blackwell B200
CUDA: 12.9 (nvcc V12.9.86)
PyTorch: 2.9.1+cu129
```
### 结果
```
root@xxx:/ssd1/hanhaowen/# python tests/test_correctness.py
=== Testing m=64, k=131072, eps=1e-06 ===
  Baseline bs= 256: max_abs_err=1.91e-06, max_rel_err=3.44e-07  PASS
  Baseline bs= 512: max_abs_err=1.91e-06, max_rel_err=3.44e-07  PASS
  Baseline bs=1024: max_abs_err=1.91e-06, max_rel_err=3.44e-07  PASS
  Cluster  cs=2:  max_abs_err=1.91e-06, max_rel_err=3.44e-07  PASS
  Cluster  cs=4:  max_abs_err=2.86e-06, max_rel_err=3.58e-07  PASS
  Cluster  cs=8:  max_abs_err=1.91e-06, max_rel_err=3.44e-07  PASS

=== Testing m=32, k=65536, eps=1e-06 ===
  Baseline bs= 256: max_abs_err=2.86e-06, max_rel_err=3.37e-07  PASS
  Baseline bs= 512: max_abs_err=2.86e-06, max_rel_err=3.41e-07  PASS
  Baseline bs=1024: max_abs_err=2.86e-06, max_rel_err=3.41e-07  PASS
  Cluster  cs=2:  max_abs_err=2.86e-06, max_rel_err=3.37e-07  PASS
  Cluster  cs=4:  max_abs_err=2.86e-06, max_rel_err=3.41e-07  PASS
  Cluster  cs=8:  max_abs_err=2.86e-06, max_rel_err=3.37e-07  PASS

=== Testing m=128, k=32768, eps=1e-06 ===
  Baseline bs= 256: max_abs_err=1.91e-06, max_rel_err=3.57e-07  PASS
  Baseline bs= 512: max_abs_err=1.91e-06, max_rel_err=3.51e-07  PASS
  Baseline bs=1024: max_abs_err=1.91e-06, max_rel_err=3.57e-07  PASS
  Cluster  cs=2:  max_abs_err=1.91e-06, max_rel_err=3.57e-07  PASS
  Cluster  cs=4:  max_abs_err=1.91e-06, max_rel_err=3.51e-07  PASS
  Cluster  cs=8:  max_abs_err=1.91e-06, max_rel_err=3.51e-07  PASS

==================================================
Overall: ALL PASSED
root@xxx:/ssd1/hanhaowen/# python tests/bench_rmsnorm.py 
======================================================================
Occupancy Info (per SM)
======================================================================
  Baseline bs= 256: max active blocks/SM = 8, total blocks = 64
  Baseline bs= 512: max active blocks/SM = 4, total blocks = 64
  Baseline bs=1024: max active blocks/SM = 2, total blocks = 64
  Cluster  cs=2:  max active clusters = 444, max active blocks = 888, total blocks = 128
  Cluster  cs=4:  max active clusters = 213, max active blocks = 852, total blocks = 256
  Cluster  cs=8:  max active clusters = 104, max active blocks = 832, total blocks = 512

=== Benchmark: m=64, k=131072 ===
Kernel                      Time(ms)       GB/s  Speedup vs B256
-----------------------------------------------------------------
Baseline bs=256               0.0246    5447.93             1.00x
Baseline bs=512               0.0205    6541.90             1.20x
Baseline bs=1024              0.0205    6546.80             1.20x
Cluster cs=2                  0.0214    6263.99             1.15x
Cluster cs=4                  0.0144    9340.50             1.71x
Cluster cs=8                  0.0144    9350.49             1.72x

=== Benchmark: m=32, k=65536 ===
Kernel                      Time(ms)       GB/s  Speedup vs B256
-----------------------------------------------------------------
Baseline bs=256               0.0144    2336.95             1.00x
Baseline bs=512               0.0123    2724.95             1.17x
Baseline bs=1024              0.0123    2727.51             1.17x
Cluster cs=2                  0.0103    3268.58             1.40x
Cluster cs=4                  0.0082    4082.45             1.75x
Cluster cs=8                  0.0071    4712.60             2.02x
```