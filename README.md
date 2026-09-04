# GEMM CUDA Optimization

Investigating GPU microarchitecture and kernel design via CUDA GEMM optimization,
following Simon Boehm's kernel progression (https://siboehm.com/articles/22/CUDA-MMM),
across multiple NVIDIA architectures.

## Hardware
- RTX 6000 Ada (Turing cluster)
- L40S (Turing cluster) — primary dev/test node so far
- A100 (Turing cluster)
- (Targeting Colab T4/A100 as additional data points if cluster ncu access remains blocked)

## Status
- [x] Kernel 1: Naive
- [x] Kernel 2: Global memory coalescing
- [ ] Kernel 3: Shared memory tiling
- [ ] Kernel 4+: further optimizations (1D/2D blocktiling, vectorization, warptiling)
- [ ] Multi-architecture sweep
- [ ] ncu deep profiling (blocked cluster-wide by DCGM; escalated to HPC admin; Colab fallback in progress)

## Results so far (L40S)

| Kernel     | Size  | GFLOPS  |
|------------|-------|---------|
| Naive      | 1024³ | 617.62  |
| Naive      | 4092³ | 2056.39 |
| Naive      | 8192³ | 687.47  |
| Coalesced  | 1024³ | 5033.74 |
| Coalesced  | 4092³ | 5323.21 |

Correctness verified: `C[0][0]` matches exactly between naive and coalesced
kernels at every tested size (same math, only access pattern differs).

## Key finding: naive kernel is non-monotonic in matrix size

Naive performance peaks at 4092³ then collapses at 8192³ — inconsistent with
Boehm's fixed ~300 GFLOPS naive result on an A6000. Hypothesis: L40S's much
larger L2 cache (96MB vs A6000's 6MB) partially masks the naive kernel's
uncoalesced access pattern until the working set (A+B) exceeds L2 capacity,
at which point performance collapses back toward the DRAM-bound regime Boehm
describes. Pending `ncu` L2 hit-rate confirmation once profiling access is
available.

## Key finding: coalescing speedup is size-dependent, not fixed

| Size  | Naive → Coalesced speedup |
|-------|---------------------------|
| 1024³ | 8.15x                     |
| 4092³ | 2.59x                     |

Boehm reports a roughly fixed ~6.6x coalescing speedup (300 → 2000 GFLOPS).
On this hardware, speedup varies by 3x depending on matrix size — likely
because at 4092³ the naive kernel is already partially "rescued" by L2 cache
reuse (see above), leaving less headroom for coalescing to improve. Coalesced
performance itself is roughly flat (~5000-5300 GFLOPS) across sizes tested so
far, suggesting a different (likely compute or non-coalescing-related memory)
bottleneck now dominates — expected to improve with shared memory tiling
(kernel 3).
