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
- [x] Kernel 4: 1D blocktiling (multiple results per thread)
- [ ] Kernel 5+: 2D blocktiling, vectorization, warptiling
- [ ] Multi-architecture sweep
- [ ] ncu deep profiling (blocked cluster-wide by DCGM; escalated to HPC admin; Colab fallback in progress)

## Methodology note: always target the correct -arch

Initial kernel 3 compiles defaulted to sm_75 (Turing) instead of the L40S's
actual sm_89 (Ada Lovelace) — `nvcc` silently picked a fallback target since
no `-arch` flag was passed. This changed the reported register count (41 vs
36) though runtime performance was not meaningfully affected once corrected
(<1.5% difference across all kernels/sizes re-tested). All kernels from
kernel 3 onward are compiled with `-arch=sm_89`.

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
| 1D blocktiled     | 4096³ | 18917.99 | 20.7%                  |

Correctness verified: `C[0][0]` matches exactly across all four kernels at
every tested size.

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

Boehm reports a roughly fixed ~6.6x coalescing speedup. On this hardware,
speedup varies substantially by matrix size — likely because at 4092³ the
naive kernel is already partially "rescued" by L2 cache reuse, leaving less
headroom for coalescing to improve.

## Key finding: shared-memory tiling gain, and occupancy analysis without ncu

Shared-mem tiling gives ~29% improvement over coalesced at 1024³ and ~38% at
~4096³ — smaller than Boehm's reported ~50%, plausibly for the same L2-masking
reason as above.

With `ncu` blocked cluster-wide by DCGM, occupancy was derived by hand using
`nvcc --ptxas-options=-v` (register/shared-mem usage) combined with
`cudaGetDeviceProperties` (hardware limits):

**Kernel 3 resource usage (sm_89):** 36 registers/thread, 8192B shared mem/block, 1024 threads/block
**L40S limits:** 1536 max threads/SM, 65536 max regs/SM, 102400B shared mem/SM, 142 SMs, 48 max warps/SM

- Shared memory: 9216B/block effective → 11 blocks/SM upper limit
- Threads: 1024/block, max 1536/SM → 1 block/SM upper limit
- Registers: 40960 regs/block → 1 block/SM upper limit
- **Binding constraint: threads & registers → occupancy = 32/48 warps = 66.7%**

Numerically identical to Boehm's 66% on the A6000 — the limiting per-SM
resource caps are unchanged between Ampere and Ada for this kernel's
footprint. All of the L40S's raw throughput advantage comes from having 142
SMs vs the A6000's 84, not from better per-SM utilization.

## Key finding: 1D blocktiling exceeds Boehm's reported improvement

| Size  | Shared-mem → 1D blocktiled speedup |
|-------|-------------------------------------|
| 1024³ | 2.55x                                |
| 4096³ | 2.62x                                |

Boehm reports ~2.2x. This is the first kernel where our hardware *outperforms*
his relative improvement, in contrast to kernels 2 and 3 where our gains were
smaller than his. Plausible explanation: kernels 2 and 3 primarily address
memory bandwidth/access-pattern problems that the L40S's much larger L2
cache partially masks regardless of the fix, capping the visible improvement.
Kernel 4's optimization instead raises arithmetic intensity (FLOPs per byte
moved between GMEM/SMEM and registers) — a more fundamental fix that a larger
cache cannot substitute for, so its full benefit shows through unmasked.

This kernel also reaches 20.7% of the L40S's 91.6 TFLOPS peak FP32 throughput
at 4096³, the largest jump in %-of-peak terms of any kernel transition so far
(7.9% → 20.7%), consistent with arithmetic intensity being the dominant
lever at this stage — matching Boehm's own framing that further gains should
come from continuing to raise arithmetic intensity (kernel 5: 2D blocktiling).
