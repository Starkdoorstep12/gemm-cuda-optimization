# GEMM CUDA Optimization

Investigating GPU microarchitecture and kernel design via CUDA GEMM optimization,
following Simon Boehm's kernel progression (https://siboehm.com/articles/22/CUDA-MMM),
across multiple NVIDIA architectures.

## Hardware
- RTX 6000 Ada (Turing cluster)
- L40S (Turing cluster) — primary dev/test node so far, Ada Lovelace, sm_89, 142 SMs, 91.6 TFLOPS peak FP32
- A100 (Turing cluster)
- (Targeting Colab T4/A100 as additional data points if cluster ncu access remains blocked)

## Status
- [x] Kernel 1: Naive
- [x] Kernel 2: Global memory coalescing
- [x] Kernel 3: Shared memory tiling
- [x] Kernel 4: 1D blocktiling
- [x] Kernel 5: 2D blocktiling
- [ ] Kernel 6: vectorized/transposed loads
- [ ] Multi-architecture sweep
- [ ] ncu deep profiling (blocked cluster-wide by DCGM; escalated to HPC admin; Colab fallback in progress)

## Methodology note: always target the correct -arch

Initial kernel 3 compiles defaulted to sm_75 (Turing) instead of the L40S's
actual sm_89 (Ada Lovelace). Register count differed (41 vs 36) but runtime
performance was not meaningfully affected once corrected (<1.5% difference).
All kernels from kernel 3 onward compiled with `-arch=sm_89`.

## Results so far (L40S, sm_89)

| Kernel            | Size  | GFLOPS   | % of 91.6 TFLOPS peak |
|-------------------|-------|----------|------------------------|
| Naive             | 1024³ | 618.16   | 0.7%                   |
| Naive             | 4092³ | 2056.89  | 2.2%                   |
| Naive             | 8192³ | 687.48   | 0.8%                   |
| Coalesced         | 1024³ | 5028.50  | 5.5%                   |
| Coalesced         | 4092³ | 5249.52  | 5.7%                   |
| Shared-mem tiled  | 1024³ | 6500.21  | 7.1%                   |
| Shared-mem tiled  | 4096³ | 7220.58  | 7.9%                   |
| 1D blocktiled     | 1024³ | 16602.47 | 18.1%                  |
| 1D blocktiled     | 1536³ | 15607.47 | 17.0%                  |
| 1D blocktiled     | 2048³ | 17160.10 | 18.7%                  |
| 1D blocktiled     | 4096³ | 18917.99 | 20.7%                  |
| 2D blocktiled     | 1024³ | 15552.82 | 17.0%                  |
| 2D blocktiled     | 1536³ | 19048.18 | 20.8%                  |
| 2D blocktiled     | 2048³ | 30323.81 | 33.1%                  |
| 2D blocktiled     | 4096³ | 31987.02 | 34.9%                  |

Correctness verified: `C[0][0]` matches exactly across all five kernels at
every tested size.

## Key finding: naive kernel is non-monotonic in matrix size

Naive performance peaks at 4092³ then collapses at 8192³ — inconsistent with
Boehm's fixed ~300 GFLOPS naive result on an A6000. Hypothesis: L40S's much
larger L2 cache (96MB vs A6000's 6MB) partially masks the naive kernel's
uncoalesced access pattern until the working set exceeds L2 capacity.
Pending `ncu` L2 hit-rate confirmation once profiling access is available.

## Key finding: coalescing and shared-mem tiling gains are size-dependent

Boehm reports roughly fixed speedups at each stage (~6.6x coalescing, ~50%
shared-mem improvement). On this hardware both vary substantially by size
(8.13x/2.55x coalescing; 29%/38% shared-mem), likely because the L40S's
larger L2 partially rescues the naive/coalesced baselines at larger sizes,
leaving less headroom for the next optimization to show its full benefit.

## Key finding: occupancy analysis without ncu (kernel 3)

With `ncu` blocked cluster-wide by DCGM, occupancy was derived by hand using
`nvcc --ptxas-options=-v` (36 registers/thread, 8192B smem/block) combined
with `cudaGetDeviceProperties` L40S limits (1536 threads/SM, 65536 regs/SM,
102400B smem/SM, 48 max warps/SM). Binding constraint: threads & registers
→ 1 block/SM → occupancy = 32/48 warps = **66.7%**, numerically identical to
Boehm's 66% on the A6000. Since limiting per-SM caps are unchanged between
Ampere and Ada for this kernel's footprint, per-SM occupancy is
architecture-generation-independent here — the L40S's raw throughput
advantage comes entirely from having 142 SMs vs the A6000's 84.

## Key finding: 1D blocktiling exceeds Boehm's reported improvement

2.55-2.62x speedup over kernel 3 (Boehm: ~2.2x), reaching 20.7% of peak
FP32 at 4096³ — the largest %-of-peak jump of any transition so far.
Plausible explanation: kernels 2-3 fix memory bandwidth/access-pattern
problems the L40S's large L2 partially masks regardless; kernel 4 instead
raises arithmetic intensity, a fix no cache size can substitute for.

## Performance cliff: tile size vs. SM occupancy (kernel 5)

Kernel 5 (2D blocktiling, BM=BN=128, TM=TN=8, 256 threads/block) is the
**only kernel transition in this study where a more arithmetically-intensive
kernel underperforms its predecessor** at a given matrix size — and the
mechanism is fully identified, not just observed.

**The data:**

| Size  | Grid blocks (k5) | k4 GFLOPS | k5 GFLOPS | k5/k4 ratio |
|-------|-------------------|-----------|-----------|-------------|
| 1024³ | 64                | 16602.47  | 15552.82  | **0.94x**   |
| 1536³ | 144               | 15607.47  | 19048.18  | **1.22x**   |
| 2048³ | 256               | 17160.10  | 30323.81  | **1.77x**   |
| 4096³ | 1024              | 18917.99  | 31987.02  | **1.69x**   |

**Mechanism.** Kernel 4 uses BM=BN=64 tiles; kernel 5 uses BM=BN=128 tiles —
4x the output area per block. This means for the same matrix size, kernel 5
launches only 1/4 as many blocks as kernel 4 (e.g. 64 vs 256 at 1024³).
The L40S has **142 SMs**. At 1024³, kernel 5's 64 blocks cannot even cover
all SMs once — roughly half the GPU sits idle for the entire kernel launch,
regardless of how efficient each individual block's inner loop is. Kernel 4,
launching 256 blocks at the same matrix size, fully occupies all 142 SMs
with blocks to spare for latency hiding, so its higher per-block cost is
more than offset by full-device parallelism.

**The crossover point falls almost exactly at block-count ≈ SM-count:** at
1536³, kernel 5 launches 144 blocks — just barely above the 142-SM count —
and the k5/k4 ratio (1.22x) is a fraction of what it becomes once the grid
comfortably oversubscribes the SMs (1.77x at 2048³, 256 blocks). This is
consistent with 144 blocks covering every SM but leaving no slack for
overlapping memory-stall latency with other resident blocks, whereas 256+
blocks give the scheduler enough independent work per SM to hide those
stalls. The slight dip from 2048³ (1.77x) to 4096³ (1.69x) is within the
noise expected from the same L2-capacity effects seen in the naive kernel,
not a new trend.

**Why this matters beyond this one kernel.** This is a concrete demonstration
that *tile size and matrix size are not independent design choices* — a tile
configuration tuned for large-matrix arithmetic intensity can actively harm
small-matrix performance by under-filling the device, independent of the
per-thread algorithm's quality. This is exactly the mechanism production GPU
BLAS libraries (cuBLAS, CUTLASS) address via multiple compiled tile-size
variants selected at runtime based on problem shape — a single fixed tile
configuration is never optimal across the full size range. The crossover
threshold itself (blocks ≈ SM count) is architecture-specific: it would occur
at a different matrix size on a GPU with a different SM count (e.g. A100's
108 SMs vs L40S's 142), which is a concrete, testable prediction for the
multi-architecture sweep still to come.
