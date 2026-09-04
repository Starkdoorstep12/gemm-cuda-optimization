# GEMM CUDA Optimization

Investigating GPU microarchitecture and kernel design via CUDA GEMM optimization,
following Simon Boehm's kernel progression (https://siboehm.com/articles/22/CUDA-MMM),
across multiple NVIDIA architectures.

## Hardware
- RTX 6000 Ada (Turing cluster)
- L40S (Turing cluster) — primary dev/test node so far, Ada Lovelace, sm_89, 142 SMs
- A100 (Turing cluster)
- (Targeting Colab T4/A100 as additional data points if cluster ncu access remains blocked)

## Status
- [x] Kernel 1: Naive
- [x] Kernel 2: Global memory coalescing
- [x] Kernel 3: Shared memory tiling
- [ ] Kernel 4+: further optimizations (1D/2D blocktiling, vectorization, warptiling)
- [ ] Multi-architecture sweep
- [ ] ncu deep profiling (blocked cluster-wide by DCGM; escalated to HPC admin; Colab fallback in progress)

## Methodology note: always target the correct -arch

Initial kernel 3 compiles defaulted to sm_75 (Turing) instead of the L40S's
actual sm_89 (Ada Lovelace) — `nvcc` silently picked a fallback target since
no `-arch` flag was passed. This changed the reported register count (41 vs
36) though runtime performance was not meaningfully affected once corrected
(<1.5% difference across all kernels/sizes re-tested). All kernels are now
compiled with `-arch=sm_89`. This is flagged as a reproducibility lesson:
always pass an explicit `-arch` matching the actual target GPU.

## Results so far (L40S, sm_89)

| Kernel            | Size  | GFLOPS  |
|-------------------|-------|---------|
| Naive             | 1024³ | 618.16  |
| Naive             | 4092³ | 2056.89 |
| Naive             | 8192³ | 687.48  |
| Coalesced         | 1024³ | 5028.50 |
| Coalesced         | 4092³ | 5249.52 |
| Shared-mem tiled  | 1024³ | 6500.21 |
| Shared-mem tiled  | 4096³ | 7220.58 |

Correctness verified: `C[0][0]` matches exactly across all three kernels at
every tested size (same math, only access pattern / memory hierarchy usage
differs).

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
| 1024³ | 8.13x                     |
| 4092³ | 2.55x                     |

Boehm reports a roughly fixed ~6.6x coalescing speedup (300 → 2000 GFLOPS).
On this hardware, speedup varies substantially by matrix size — likely
because at 4092³ the naive kernel is already partially "rescued" by L2 cache
reuse (see above), leaving less headroom for coalescing to improve.

## Key finding: shared-memory tiling gain, and occupancy analysis without ncu

Shared-mem tiling gives ~29% improvement over coalesced at 1024³ and ~38% at
~4096³ — smaller than Boehm's reported ~50%, plausibly because the coalesced
baseline already benefits more from the larger L2 on this hardware, leaving
less headroom (same pattern as the coalescing finding above).

With `ncu` blocked cluster-wide by DCGM, occupancy was instead derived by
hand using `nvcc --ptxas-options=-v` (register/shared-mem usage) combined
with `cudaGetDeviceProperties` (hardware limits), following the same manual
method Boehm describes:

**Kernel 3 resource usage (sm_89):** 36 registers/thread, 8192B shared mem/block, 1024 threads/block

**L40S hardware limits:** 1536 max threads/SM, 65536 max regs/SM, 102400B shared mem/SM, 142 SMs, 48 max warps/SM

- Shared memory: (8192+1024)B/block → 102400/9216 = 11.1 → 11 blocks/SM upper limit
- Threads: 1024/block, max 1536/SM → 1 block/SM upper limit
- Registers: 36×32=1152→1280 regs/warp (256-granularity rounding) × 32 warps/block = 40960 regs/block → 65536/40960 = 1.6 → 1 block/SM upper limit
- **Binding constraint: threads & registers, 1 block/SM → occupancy = 32/48 = 66.7%**

This is numerically identical to Boehm's 66% result on the A6000. Since the
limiting per-SM resource caps (max threads/SM, max regs/SM, warp size) are
unchanged between Ampere and Ada for this kernel's resource footprint,
per-SM occupancy is architecture-generation-independent here — all of the
L40S's raw throughput advantage over Boehm's numbers comes from having 142
SMs vs the A6000's 84 (1.7x more parallel occupancy-limited units), not from
better per-SM utilization.
