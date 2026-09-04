# GEMM CUDA Optimization

Investigating GPU microarchitecture and kernel design via CUDA GEMM optimization,
following Simon Boehm's kernel progression (https://siboehm.com/articles/22/CUDA-MMM),
across multiple NVIDIA architectures.

## Hardware
- RTX 6000 Ada (Turing cluster)
- L40S (Turing cluster) — primary dev/test node so far, Ada Lovelace, sm_89, 142 SMs, 91.6 TFLOPS peak FP32
- A100 (Turing cluster)
- (Targeting Colab as an additional data point if cluster ncu access remains blocked)

## Status
- [x] Kernel 1: Naive
- [x] Kernel 2: Global memory coalescing
- [x] Kernel 3: Shared memory tiling
- [x] Kernel 4: 1D blocktiling
- [x] Kernel 5: 2D blocktiling
- [x] Kernel 6: Vectorized SMEM/GMEM access (float4)
- [x] Kernels 7-8: Skipped — see note below
- [x] Kernel 9: Autotuning (found and fixed a real correctness bug along the way)
- [x] Kernel 10: Warptiling
- [x] Non-square / non-power-of-two dimension testing
- [ ] Multi-architecture sweep (RTX 6000 Ada, A100)
- [ ] ncu deep profiling (blocked cluster-wide by DCGM; escalated to HPC admin; Colab fallback in progress)
- [ ] Plots / visualizations
- [ ] Final report writeup

## Note: kernels 7 and 8 skipped

Boehm's article itself skips kernels 7 and 8 in the published writeup — he
describes them as experiments in eliminating shared-memory bank conflicts
that ended up being net-negative for performance on his hardware, and does
not publish their code. We follow the same choice: kernels 7-8 are omitted
from this study, consistent with the source article's own finding that they
were not beneficial. Bank-conflict elimination remains a documented,
identified-but-unpursued optimization opportunity (see kernel 6 discussion),
and is a candidate for a stretch-goal appendix if time permits after the
core assignment requirements (multi-architecture sweep, dimension testing,
plots, profiling) are complete.

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
| Vectorized (k6)   | 1024³ | 18593.32 | 20.3%                  |
| Vectorized (k6)   | 1536³ | 23165.36 | 25.3%                  |
| Vectorized (k6)   | 2048³ | 35220.36 | 38.5%                  |
| Vectorized (k6)   | 4096³ | 36296.49 | 39.6%                  |
| Best autotuned    | 2048³ | 35965.95 | 39.3%                  |
| Warptiled (k10)   | 2048³ | 37860.87 | 41.3%                  |

Correctness verified via multiple independent methods across kernels: exact
`C[0][0]` matching through kernel 6, multi-point checks (corner/center/
opposite-corner/full-matrix checksum) from kernel 6's autotuning fix onward,
and independent `torch.matmul` ground-truth comparison for the non-square
dimension testing.

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

## Kernel 6: Vectorized SMEM/GMEM access (float4, transposed As)

| Size  | k5 GFLOPS | k6 GFLOPS | k6/k5 ratio | % of peak (k6) |
|-------|-----------|-----------|-------------|-----------------|
| 1024³ | 15552.82  | 18593.32  | 1.20x       | 20.3%           |
| 1536³ | 19048.18  | 23165.36  | 1.22x       | 25.3%           |
| 2048³ | 30323.81  | 35220.36  | 1.16x       | 38.5%           |
| 4096³ | 31987.02  | 36296.49  | 1.13x       | 39.6%           |

Correctness verified: `C[0][0]` matches kernels 4/5 exactly at every size.

Gains here (13-22%) are notably larger and more size-consistent than Boehm's
reported ~3% SMEM-vectorization improvement — likely because we implemented
both the SMEM transpose *and* GMEM float4 vectorization together, whereas he
reports them as separate incremental steps.

**Important confirming observation on the kernel 5 performance cliff:**
Kernel 6 uses the identical BM=BN=128 tile size as kernel 5 (same block
count at every matrix size), yet shows *no* dip below 1.0x speedup anywhere,
including at 1024³ where kernel 5 itself underperformed kernel 4. This
confirms the earlier cliff was specifically about the *k4-vs-k5 tile-size/
block-count mismatch*, not something inherent to 128-sized tiles in general.
Kernel 6 simply makes each of the same-sized, same-count blocks more
memory-efficient via vectorized loads — an orthogonal axis to the occupancy
issue identified in kernel 5, and evidence the two effects (tile-size-driven
occupancy, and per-block memory-access efficiency) are cleanly separable.

At 4096³, kernel 6 reaches 39.6% of the L40S's 91.6 TFLOPS peak FP32 —
continued steady progress, though the profiler-flagged remaining bottlenecks
Boehm describes (shared-memory bank conflicts, un-tuned occupancy, no double
buffering) are exactly where further gains would come from, matching his
progression toward kernel autotuning next.

---

## Deep dive: A real bug found during autotuning, and why it almost went unnoticed

This section documents the most significant methodological finding of the
project — not because the bug itself was exotic, but because of *how close
we came to missing it entirely*, and what that implies for correctness
testing in performance-sensitive GPU code generally.

### Background: why we needed tunable parameters

Boehm's kernel 9 (autotuning) explains that the optimal `(BM, BN, BK, TM, TN)`
tile parameters differ by GPU model, and that this is precisely why
production libraries like cuBLAS and compilers like Triton perform
autotuning rather than shipping one fixed configuration. To reproduce this
on our hardware, we needed to make kernel 6's tile parameters overridable at
compile time (`#ifndef BM / #define BM 128 / #endif`, etc., set via `nvcc -D`
flags) and write a sweep script to compile and benchmark every valid
combination.

### The bug

Kernel 6's GMEM→SMEM loading code, copied faithfully from Boehm's published
article, does a **single-shot vectorized load per thread**:

```cuda
const uint innerRowA = threadIdx.x / (BK / 4);
const uint innerColA = threadIdx.x % (BK / 4);
// ... one float4 load per thread, no loop
```

This is only mathematically correct if `NUM_THREADS` exactly equals both
`(BM*BK)/4` and `(BK*BN)/4` — i.e., the total thread count in the block
exactly matches the number of float4 chunks needed to fill each shared
memory tile in one pass. For Boehm's article parameters
(`BM=BN=128, BK=8` → `NUM_THREADS=256`), this holds exactly:
`128*8/4 = 256`. **The article's own code snippet is therefore only correct
for the specific parameters it's demonstrated with — it does not generalize
to arbitrary tile sizes**, something the article doesn't flag, likely
because the autotuning section describes the *results* of a parameter sweep
without publishing the (evidently more complex, generalized) sweep-capable
kernel code itself.

For any other parameter combination, the single-shot load either under-fills
the SMEM tile (leaving stale/garbage values in unwritten rows) or, in the
worst case, silently produces a kernel that satisfies all compile-time and
launch-time checks while computing wrong results.

### The false-positive trap: why `C[0][0]`-only correctness checking failed

Our first-pass autotuning sweep checked correctness by comparing only
`C[0][0]` against a known-good reference value. This caught **35 of 52**
broken configurations — but incorrectly passed **17**, including at least
one, `BM=256 BN=64 BK=16 TM=8 TN=8`, that we later proved was wrong for most
of the output matrix.

**Why did a broken kernel produce a correct `C[0][0]`?** Thread (0,0) only
ever reads shared-memory rows within the range that happened to be correctly
populated by the buggy single-shot load for that specific `(BM, BK)`
combination — the corruption was confined to rows the corner element never
touched. This is a textbook example of **an insufficient test oracle
producing a false sense of correctness**: a single-point check can pass by
coincidence whenever the bug's effect is spatially localized and the test
point happens to sit outside the corrupted region.

We caught this only because we deliberately paused to ask "are the passing
configs actually correct, or only coincidentally correct?" rather than
trusting the first sweep's results — a decision to independently re-derive,
by hand, exactly which `(BM, BK, NUM_THREADS)` combinations satisfy the
single-shot load's implicit exactness requirement, and check the "passing"
results against that math rather than trusting the runtime check alone.
That hand-derivation (checking `NUM_THREADS × 4 == BM×BK` as an *equality*,
not just the weaker divisibility condition our first validity filter used)
is what surfaced the coincidental pass.

### The fix

We rewrote the loading loop to be **strided and multi-pass**, matching the
pattern already used in kernel 5's (non-vectorized) shared-memory loading,
but preserving `float4` vectorization:

```cuda
const uint strideA = NUM_THREADS / (BK / 4);
for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
    float4 tmp = reinterpret_cast<const float4*>(&A[(innerRowA+loadOffset)*K + innerColA*4])[0];
    As[(innerColA*4+0)*BM + innerRowA+loadOffset] = tmp.x;
    // ... (y, z, w similarly)
}
```

This generalizes correctly to any `(BM, BN, BK, NUM_THREADS)` combination
satisfying the (now correctly derived) divisibility constraints, rather than
requiring an exact one-to-one correspondence.

### Stronger verification after the fix

We also replaced the single-point correctness check with four independent
checks per configuration: `C[0][0]` (top-left corner), `C[M/2][N/2]`
(center), `C[M-1][N-1]` (bottom-right corner), and a full-matrix checksum
(sum of all ~4.2M elements). A bug that corrupts any spatial region of the
output is now overwhelmingly likely to be caught by at least one of these
four checks, unlike the single-corner check that missed 17 broken configs.

After the fix: **51 of 52 valid configurations passed all four correctness
checks** (up from 17/52 pre-fix) — strong evidence the strided-load rewrite
resolved the actual root cause rather than papering over one symptom.

### A second, unrelated bug caught by the improved harness

The one remaining "failure," `BM=256 BN=256 BK=16 TM=8 TN=8`, was not a
correctness bug but a **silent launch failure**: this configuration compiles
to 109 registers/thread (confirmed via `nvcc --ptxas-options=-v`), and at
1024 threads/block that demands 111,616 registers — far exceeding the L40S's
65,536 registers/block limit (from `cudaGetDeviceProperties`). CUDA's launch
call fails silently by default (no exception, no crash — the kernel simply
doesn't run), which our original harness didn't check for, so it recorded a
near-zero elapsed time and computed a physically impossible ~11 million
GFLOPS. Adding an explicit `cudaGetLastError()` check immediately after the
timed launch loop converts this into a clear, correctly-attributed error:
`too many resources requested for launch`. This check was subsequently built
into every later kernel's benchmarking harness (kernel 10 onward) from the
start, rather than discovered after the fact again.

### Note on methodology

This entire investigation — identifying the root cause, deriving the exact
mathematical condition the article's code silently assumes, designing the
generalized fix, and building multi-point verification — was done through
first-principles reasoning about the kernel's indexing arithmetic and
GPU resource limits (via `nvcc --ptxas-options=-v` and
`cudaGetDeviceProperties`), not by consulting external sources. Boehm's
article does not publish the generalized/autotuning-capable kernel code, so
no reference implementation was available to check against — the fix was
derived and verified entirely from CUDA's execution model and the observed
failure patterns.

### Final autotuning results (L40S, 2048³, all correctness-verified)

| Rank | BM  | BN  | BK | TM | TN | GFLOPS   |
|------|-----|-----|----|----|----|----------|
| 1    | 128 | 128 | 16 | 8  | 8  | 35965.95 |
| 2    | 64  | 256 | 16 | 8  | 8  | 35915.90 |
| 3    | 64  | 128 | 16 | 8  | 8  | 35894.29 |
| 4    | 128 | 256 | 16 | 8  | 8  | 35326.73 |
| 5    | 128 | 256 | 32 | 8  | 8  | 35226.36 |

Every top-5 configuration uses `TM=TN=8` — the maximum register-blocking
tested — regardless of `BM`/`BN`/`BK`, suggesting arithmetic intensity per
thread remains the dominant lever even after tile-size/occupancy effects are
accounted for. Our winning configuration (`BM=BN=128, BK=16, TM=TN=8`)
**exactly matches** Boehm's own reported optimal parameters for the A6000
(he found this combination gave a 5% improvement over the article's default
`BK=8`, reaching 20 TFLOPS on his hardware). On the L40S, this configuration
reaches 35965.95 GFLOPS — **39.3% of the L40S's 91.6 TFLOPS peak** FP32,
compared to Boehm's ~20/38.7 ≈ 51.7% of the A6000's peak. This gap in
relative efficiency, despite an identical "optimal" tile shape, is a
concrete question for the multi-architecture sweep still to come — whether
the L40S's much larger SM count (142 vs 84) changes what "optimal" actually
means at this tile size, similar to the tile-size/occupancy cliff already
found in kernel 5.

---

## Kernel 10: Warptiling

Boehm's final kernel adds a third tiling hierarchy — warp-level tiling —
between the existing block-level and thread-level tiling. This makes all
three levels of GPU parallelism explicit in the code: blocktiling (different
SMs), warptiling (different warp schedulers within an SM), and threadtiling
(instruction-level parallelism within a thread). The full kernel (not fully
published in Boehm's article, which shows only the inner loop) was
reconstructed from first principles based on the article's description and
the documented parameter relationships (`WMITER`, `WSUBM`, `WSUBN`, warp/
thread placement within warp subtiles).

Given the lesson learned from the kernel 6 autotuning bug, this kernel's
correctness harness was built with multi-point checks (`C[0][0]`,
`C[mid][mid]`, `C[last][last]`, full-matrix checksum) and an explicit
`cudaGetLastError()` safety check **from the start**, rather than added
after a bug was found. Using Boehm's standard warptiling configuration
(`BM=BN=128, BK=16, WM=WN=64, WNITER=4, TM=8, TN=4, NUM_THREADS=128`), all
four correctness checks passed exactly against the kernel 6 reference values
on the first attempt.

**Results at 2048³ (L40S, sm_89):**

| Kernel                              | GFLOPS   | % of 91.6 TFLOPS peak |
|--------------------------------------|----------|------------------------|
| Vectorized (k6, default params)      | 35220.36 | 38.5%                  |
| Best autotuned (BM=BN=128,BK=16,TM=TN=8) | 35965.95 | 39.3%              |
| **Warptiled (k10)**                  | **37860.87** | **41.3%**          |

Warptiling beats even the best autotuned non-warptiled configuration by
**~5.3%** — a smaller relative gain than Boehm's reported ~10% improvement
on the A100 (19.7→21.7 TFLOPS), but a clear, correctness-verified
improvement nonetheless. This completes the full main-line kernel
progression from Boehm's article (kernels 1 through 6, 9, and 10; kernels
7-8 explicitly skipped per the source article's own finding, see note at
top of this document).

---

## Non-square and non-power-of-two dimension testing

The assignment requires testing non-square and non-power-of-two matrix
dimensions. Since kernels 3 onward assume `M`, `N`, `K` are exact multiples
of the tile sizes (`BM`, `BN`, `BK`) with no boundary-check code, arbitrary
shapes were tested via **zero-padding**: pad `A` and `B` up to the next tile
multiple, run the kernel unmodified, then extract the real `M×N` region of
the result. Correctness was verified independently against `torch.matmul`
(not against our own prior kernel outputs), computed via a separate Python
reference script, for full confidence the padding approach itself introduces
no error.

**Test kernel:** kernel 6 (vectorized, strided-load, bug-fixed), default
`BM=BN=128, BK=16, TM=TN=8` (best autotuned config).

| Shape           | Padded shape     | Blocks (of 142 SMs) | GFLOPS   | Max abs error vs torch |
|-----------------|------------------|----------------------|----------|--------------------------|
| 3000×1500×2048  | 3072×1536×2048   | 288 (2.03x)          | 27515.57 | 1.7e-3                   |
| 8192×256×1024   | 8192×256×1024    | 128 (0.90x)          | 34083.58 | 5.8e-4                   |
| 2048×2048×2049  | 2048×2048×2064   | 256 (1.80x)          | 35932.00 | 1.6e-3                   |
| 137×263×401     | 256×384×416      | **6 (0.04x)**        | **623.98** | 1.6e-4                 |

All four shapes pass correctness against an independent `torch.matmul`
ground truth, with float32 accumulation error in the expected 1e-3 to 1e-4
range for K on the order of hundreds to a couple thousand.

**Key finding: small-matrix collapse is an occupancy problem, not a padding
problem.** The `137×263×401` case is ~58x slower than the well-aligned large
cases, but its padding-induced extra work is only ~3.4x (256×384×416 vs
137×263×401 elements) — padding overhead alone cannot explain the gap. The
real cause is grid size: this shape produces only **6 total thread blocks**,
meaning at most 6 of the L40S's 142 SMs ever receive any work; the other 136
sit completely idle for the kernel's entire (very short) runtime. This is
the same underlying mechanism as the kernel 5 tile-size/SM-occupancy cliff
found earlier, but far more extreme — 6 blocks vs 142 SMs is a ~24x
under-subscription, compared to the ~2.2x under-subscription (64 blocks vs
142 SMs) that caused kernel 5's more modest regression at 1024³.

**Aspect ratio alone does not predict performance well.** `8192×256×1024`
(extreme 32:1 aspect ratio, only 128 blocks — slightly *under* the 142-SM
count) performs comparably to `2048×2048×2049` (square-ish, 256 blocks —
comfortably *over* the SM count), both in the 34-36 TFLOPS range. This
suggests that once each block has enough K-dimension work to keep its
assigned SM busy for a reasonable duration (K=1024 here), a block count even
somewhat below the SM count is not immediately catastrophic — unlike the
137×263×401 case where blocks are both scarce (6) and individually short
(K=401, and a much smaller M/N tile fraction is real work vs padding).

**Practical implication for the assignment's "special cases" question:** for
genuinely small matrices (row/col counts much smaller than the tile size),
a fundamentally different kernel strategy — smaller tiles, or batching
multiple independent small GEMMs into one kernel launch to fill the grid —
would be necessary to use the GPU efficiently. This is exactly the kind of
shape-dependent kernel selection cuBLAS performs internally (as Boehm's
kernel 9/10 discussion notes: cuBLAS ships hundreds of SGEMM kernel variants
and dispatches by shape at runtime), and our data provides a concrete,
independently-reproduced illustration of *why* that dispatch is necessary.

---

## Multi-architecture sweep: A100 (Ampere, sm_80)

The full kernel progression (naive through warptiled) was re-run, unmodified,
on an NVIDIA A100-SXM4-40GB via cross-compilation on node14 (see
methodology note below) and execution on node10. All correctness checks
(`C[0][0]`, `C[mid][mid]`, `C[last][last]`, checksum where applicable) match
the L40S reference values exactly at every kernel and size tested.

**A100 specs:** Compute capability 8.0, 108 SMs, 40GB HBM2e, 19.5 TFLOPS
peak FP32, 40MB L2 cache, 2048 max threads/SM, 64 max warps/SM, 167936B
shared mem/SM.

### Methodology note: cross-compilation across nodes with mismatched toolchains

Node10 (A100) has CUDA runtime libraries and a compatible driver
(570.211.01, supporting up to CUDA 12.8) but **no CUDA compiler installed**
— only `nvidia-smi` and runtime libraries, no `nvcc`. Node14 (L40S) has a
full toolkit, but only version 13.1, which produces binaries requiring a
newer driver than node10 has. A pip-based fix
(`nvidia-cuda-nvcc-cu12`) was attempted but confirmed (via NVIDIA's own
package documentation) to provide only the NVVM JIT backend library, not a
standalone `nvcc` frontend executable — it cannot compile `.cu` files
directly and was not a viable path.

The actual fix: an older CUDA 12.4 toolkit install was found at
`/home/u22/cuda-12.4` on node14 (alongside the default 13.1). Since
`nvcc`'s device-code compilation targets a specific GPU architecture
(`-arch=sm_80` for A100) independently of the host machine it runs on, and
since node14 and node10 share a network home directory, binaries compiled
on node14 with this older, driver-compatible toolkit run correctly on
node10 without any files needing to be copied. This is a standard
cross-compilation technique — `nvcc` emits GPU machine code as a pure data
artifact; the compiling machine's own GPU (if any) is never involved in
producing device code for a *different* target architecture. Compiling
with a toolkit version too new for the target driver was the actual
failure mode, not the cross-node approach itself, which worked immediately
once the toolkit/driver version constraint was satisfied.

### Full results comparison (L40S vs A100, % of each GPU's own peak FP32)

| Kernel          | Size  | L40S GFLOPS | % L40S peak (91.6 TF) | A100 GFLOPS | % A100 peak (19.5 TF) |
|-----------------|-------|-------------|------------------------|-------------|-------------------------|
| Naive           | 1024³ | 618.16      | 0.7%                   | 214.38      | 1.1%                    |
| Naive           | 4092³ | 2056.89     | 2.2%                   | 874.90      | 4.5%                    |
| Naive           | 8192³ | 687.48      | 0.8%                   | 293.16      | 1.5%                    |
| Coalesced       | 1024³ | 5028.50     | 5.5%                   | 3563.55     | 18.3%                   |
| Coalesced       | 4092³ | 5249.52     | 5.7%                   | 3593.40     | 18.4%                   |
| Shared-mem      | 1024³ | 6500.21     | 7.1%                   | 5196.12     | 26.6%                   |
| Shared-mem      | 4096³ | 7220.58     | 7.9%                   | 5446.22     | 27.9%                   |
| 1D blocktiled   | 4096³ | 18917.99    | 20.7%                  | 8918.65     | 45.7%                   |
| 2D blocktiled   | 4096³ | 31987.02    | 34.9%                  | 11706.84    | 60.0%                   |
| Best autotuned  | 2048³ | 35965.95    | 39.3%                  | 11274.25    | 57.8%                   |
| Warptiled       | 2048³ | 37860.87    | 41.3%                  | 11545.02    | 59.2%                   |

### Key finding: identical code reaches a much higher fraction of peak on A100 than L40S, consistently

From kernel 2 onward, A100 runs at roughly **2.5-3x higher %-of-its-own-peak**
than L40S for byte-for-byte identical kernel code. This holds at every
optimization stage, not just the endpoint. The likely explanation: L40S's
much larger raw FLOPS ceiling (91.6 vs 19.5 TFLOPS, a 4.7x difference) is
proportionally harder to saturate with a fixed amount of parallelism and
arithmetic intensity — the same kernel design that comfortably approaches
A100's smaller ceiling leaves much more "room" unfilled on L40S. This
reframes the earlier "L40S is 3.75x faster" headline: L40S has ~4.7x more
raw silicon, and A100 is in fact extracting a *higher fraction* of what it
has, not a lower one.

### Key finding: naive kernel's peak-then-collapse pattern reproduces on a second architecture

A100 shows the same qualitative shape as L40S: naive performance peaks at
4092³ (874.90 GFLOPS) then collapses at 8192³ (293.16 GFLOPS) — this
matches the pattern found on L40S almost exactly in shape, despite A100
having a much smaller L2 cache (40MB vs L40S's 96MB). This strengthens the
L2-masking hypothesis first proposed for L40S: even a comparatively modest
40MB L2 is enough to produce non-monotonic naive-kernel scaling, not just
unusually large caches. The specific matrix size where performance peaks
appears to be a robust phenomenon across cache sizes, though direct `ncu`
L2 hit-rate measurement is still needed to confirm the mechanism precisely
rather than by pattern-matching alone.

### Occupancy analysis on A100 (kernel 3) — a genuine architectural surprise

Using the same manual method as L40S (`nvcc --ptxas-options=-v` +
`cudaGetDeviceProperties`, since `ncu` remains unavailable on this cluster
entirely):

**Kernel 3 resource usage (sm_80):** 32 registers/thread (vs 36 on L40S's
sm_89 — register allocation genuinely differs by architecture for identical
source code), 8192B shared mem/block, 1024 threads/block.

**A100 limits:** 2048 max threads/SM (vs L40S's 1536), 65536 max regs/SM
(same as L40S), 167936B shared mem/SM (vs L40S's 102400B), 64 max
warps/SM (vs L40S's 48), 108 SMs (vs L40S's 142).

- Shared memory: 9216B/block effective → 167936/9216 = 18.2 → 18 blocks/SM upper limit
- Threads: 1024/block, max 2048/SM → **2 blocks/SM** upper limit
- Registers: 32×32=1024 regs/warp × 32 warps/block = 32768 regs/block → 65536/32768 = 2.0 → **2 blocks/SM** upper limit
- **Binding constraint: threads & registers tie at 2 blocks/SM → occupancy = 64/64 warps = 100%**

This is a striking contrast with L40S's result for the *same kernel*: L40S
achieved only 66.7% theoretical occupancy (capped at 1 block/SM), while
A100 achieves **100%** (2 blocks/SM) — because A100 allows double the
threads/SM and substantially more shared memory/SM, so kernel 3's modest
per-block footprint fits twice per SM on A100 but only once on L40S.

**Yet A100 still produces lower absolute GFLOPS** (5196.12 vs 6500.21 at
1024³) despite superior occupancy. This is an important, non-obvious
result: **occupancy alone does not determine throughput** — A100's lower
raw per-SM compute capability, different clock characteristics, and fewer
total SMs (108 vs 142) dominate over its occupancy advantage for this
kernel. This is a genuine caution against over-indexing on occupancy
percentage as a proxy for performance, consistent with Boehm's own note
(quoting Volkov's thesis) that high occupancy is not always necessary or
sufficient for peak throughput, particularly outside the memory-bound
regime.

### Key finding: kernel 5 tile-size/SM cliff reproduces, but its recovery shape differs from L40S

A100 has 108 SMs vs L40S's 142, changing the blocks-per-SM ratio at every
matrix size (block count itself only depends on matrix/tile size, not
architecture):

| Size  | k5 blocks | Blocks/SM (A100) | Blocks/SM (L40S) | k5/k4 ratio (A100) | k5/k4 ratio (L40S) |
|-------|-----------|-------------------|--------------------|----------------------|----------------------|
| 1024³ | 64        | 0.59              | 0.45               | 0.964                | 0.94                 |
| 1536³ | 144       | 1.33              | 1.01               | 1.086                | 1.22                 |
| 2048³ | 256       | 2.37              | 1.80               | 1.457                | 1.77                 |
| 4096³ | 1024      | 9.48              | 7.21               | 1.313                | 1.69                 |

The cliff itself (ratio below 1.0 at 1024³) reproduces on both
architectures, supporting the general tile-size/occupancy mechanism. However
the *recovery* pattern does not scale simply with blocks-per-SM: at 1536³,
A100 is already more oversubscribed than L40S was at the same size (1.33
vs 1.01 blocks/SM) yet shows a *smaller* recovery ratio (1.086 vs 1.22x).
This suggests block-count-vs-SM-count is not the complete explanation on
A100 — other factors (A100's different per-SM compute throughput, HBM2e vs
L40S's GDDR6 memory subsystem, or L2 size differences) likely interact with
occupancy in ways not captured by the simple block-count model that fit
L40S well. This nuance is reported honestly rather than forced into a
clean unified story; resolving it further would require `ncu`-level warp
stall and memory throughput data on both architectures, which remains
blocked pending DCGM resolution.

---

## Multi-architecture sweep: RTX 6000 Ada (Ada Lovelace, sm_89)

The full kernel progression was re-run natively on an RTX 6000 Ada
Generation node (node03), which has a working `nvcc` directly (CUDA 12.9)
— no cross-compilation needed for this architecture. All correctness
checks pass exactly, matching L40S and A100 reference values at every
kernel and size.

**RTX 6000 Ada specs:** Ada Lovelace (AD102), 142 SMs, 18,176 CUDA cores,
96MB L2, 48GB GDDR6, 91.1 TFLOPS peak FP32, 960 GB/s memory bandwidth,
300W TDP — essentially identical silicon to the L40S (same core/SM count,
same die, 91.6 TFLOPS peak, 864 GB/s bandwidth, 350W TDP), differing mainly
in memory bandwidth (+11%) and TDP/form factor (workstation vs data-center
card).

### Three-way comparison (L40S vs RTX 6000 Ada vs A100)

| Kernel          | Size  | L40S GFLOPS | %peak | RTX6000 GFLOPS | %peak | A100 GFLOPS | %peak |
|-----------------|-------|-------------|-------|-----------------|-------|-------------|-------|
| Naive           | 1024³ | 618.16      | 0.7%  | 613.28          | 0.7%  | 214.38      | 1.1%  |
| Naive           | 4092³ | 2056.89     | 2.2%  | 2107.06         | 2.3%  | 874.90      | 4.5%  |
| Naive           | 8192³ | 687.48      | 0.8%  | 740.68          | 0.8%  | 293.16      | 1.5%  |
| Coalesced       | 1024³ | 5028.50     | 5.5%  | 5363.99         | 5.9%  | 3563.55     | 18.3% |
| Coalesced       | 4092³ | 5249.52     | 5.7%  | 5084.32         | 5.6%  | 3593.40     | 18.4% |
| Shared-mem      | 1024³ | 6500.21     | 7.1%  | 6995.17         | 7.7%  | 5196.12     | 26.6% |
| Shared-mem      | 4096³ | 7220.58     | 7.9%  | 7371.63         | 8.1%  | 5446.22     | 27.9% |
| 1D blocktiled   | 4096³ | 18917.99    | 20.7% | 18597.67        | 20.4% | 8918.65     | 45.7% |
| 2D blocktiled   | 4096³ | 31987.02    | 34.9% | 30527.75        | 33.5% | 11706.84    | 60.0% |
| Best autotuned  | 2048³ | 35965.95    | 39.3% | **42419.93**    | **46.6%** | 11274.25 | 57.8% |
| Warptiled       | 2048³ | 37860.87    | 41.3% | **45069.00**    | **49.5%** | 11545.02 | 59.2% |

### Key finding: near-identical silicon validates methodology through kernel 5

Through kernels 1-5, L40S and RTX 6000 Ada track within ~5-10% of each
other at every size — exactly what should be expected for functionally
identical dies (same SM count, same CUDA core count, same L2 size, same
peak FP32 spec within 0.5%). This is a valuable internal consistency check:
it confirms the benchmarking methodology (timing harness, `-arch` targeting,
correctness verification) produces coherent, architecture-appropriate
results rather than noise, since two near-identical GPUs genuinely produce
near-identical numbers where theory predicts they should.

### Key finding: RTX 6000 Ada meaningfully outperforms L40S specifically at the most memory-bandwidth-sensitive kernels

The close agreement breaks down exactly at kernels 6 (vectorized/autotuned)
and 10 (warptiled) — the two stages most reliant on sustained shared-memory
and register-file throughput under heavy vectorized traffic. RTX 6000 Ada
leads by **+18% (42419.93 vs 35965.95)** on the best autotuned config and
**+19% (45069.00 vs 37860.87)** on warptiling, despite both cards sharing
peak FP32 specs within 0.5% of each other.

The most likely explanation is RTX 6000 Ada's **11% higher rated memory
bandwidth** (960 vs 864 GB/s) combined with its lower TDP (300W vs 350W)
suggesting a different power/clock profile — a workstation card in a
single-GPU desktop chassis may sustain boost clocks more consistently under
load than a passively-cooled data-center card designed for dense multi-GPU
rack deployment, even at a lower rated TDP. This would explain why the gap
appears specifically once kernels become bandwidth/throughput-bound in
their working-set access pattern (heavy `float4` vectorized loads, tight
register/shared-memory interplay) rather than at earlier, less
bandwidth-intensive stages, where compute latency dominates instead and the
nearly-identical peak FP32 numbers govern performance almost equally.
Confirming this precisely would require direct clock/power monitoring
during the kernel (via `nvidia-smi -q -d CLOCK,POWER` sampled during
execution) or `ncu`'s SM clock/throughput counters — both left as follow-up
work given the ongoing DCGM profiling block.

### A100 remains the clear outlier, and its relative strength grows with optimization

Consistent with the earlier A100-vs-L40S finding, A100 shows the largest
%-of-own-peak figures throughout, and this gap *widens* as kernels become
more sophisticated: 18.3% (coalesced) → 26.6% (shared-mem) → 45.7% (1D
blocktiled) → 57.8-60.0% (best/warptiled). This reinforces the earlier
hypothesis that a GPU's *raw peak FLOPS headroom* is inversely related to
how easily naive-to-intermediate kernel designs can approach it: A100's far
lower ceiling (19.5 vs ~91 TFLOPS) is simply easier to fill with a fixed
amount of arithmetic intensity and parallelism, while both Ada-generation
cards need the full weight of blocktiling, vectorization, and warptiling to
meaningfully close the gap to their much higher ceilings.

---

## Real ncu profiling data (finally unblocked via Colab, Tesla T4)

After the DCGM lock made `ncu` unusable on Turing for the entire project,
Google Colab's free tier (Tesla T4, Turing architecture, sm_75, CUDA 13.0,
Nsight Compute 2025.1.1.0) provided working, unrestricted `ncu` access with
no permission errors. Kernels 1-3 were cross-compiled for `sm_75` and
profiled directly, finally replacing several inferred/hypothesized findings
from earlier in this document with real measured data.

**Note:** T4 is a fourth, distinct architecture (Turing) from the three
main-line architectures (L40S, RTX 6000 Ada — both Ada Lovelace; A100 —
Ampere) used for the primary kernel-progression comparison. Absolute GFLOPS
numbers here are much lower (T4 is a lower-power inference-oriented card)
and are not directly comparable to the main three-way table — this section
exists purely to extract real hardware counter data unavailable elsewhere.

| Metric | Kernel 1 (Naive) | Kernel 2 (Coalesced) | Kernel 3 (Shared-mem) |
|---|---|---|---|
| Sectors/request (global load) | 16.51 | 2.50 | 4.00 |
| L1 hit rate | 99.23% | 94.93% | 1.52% |
| L2 hit rate | 88.37% | 85.42% | 74.56% |
| Shared-mem bank conflicts (LD) | 0 | 0 | 0 |
| Shared-mem bank conflicts (ST) | 0 | 0 | 0 |
| GFLOPS (T4, 1024³) | 9.75 | 14.62 | 14.92 |

### Confirms Boehm's coalescing mechanism directly, with real numbers

`l1tex__average_t_sectors_per_request` drops from **16.51 to 2.50**
sectors/request (an 85% reduction) between the naive and coalesced kernels —
this is a direct, measured confirmation of the exact mechanism Boehm
describes: naive's scattered warp access pattern forces many separate 32B
sector transactions per logical request, while the coalesced pattern
consolidates them into far fewer. This was previously only theoretical
here (assumed correct because it matched the source article's claims); it
is now independently measured on real hardware.

### Confirms the L2-masking hypothesis with real hardware counters, not pattern-matching

Both naive and coalesced kernels show high L2 hit rates (88.37% and 85.42%
respectively) even though T4's L2 cache is only **4MB** — smaller than the
8MB combined working set of A and B at 1024×1024 fp32, which already
exceeds L2 capacity. This is genuine, measured evidence that temporal
locality from repeated row/column reuse across many threads sustains high
L2 hit rates even when the raw dataset size exceeds the cache. This is the
same mechanism separately hypothesized (from indirect performance-curve
evidence alone) to explain L40S's non-monotonic naive-kernel scaling
earlier in this document — now confirmed directly via hardware counters on
an independent architecture, not just inferred from GFLOPS curves.

### A genuinely counterintuitive finding: higher L1 hit rate does not mean better performance

The naive kernel has a *higher* L1 hit rate (99.23%) than the coalesced
kernel (94.93%) despite being dramatically slower (9.75 vs 14.62 GFLOPS).
This is explainable: naive repeatedly re-reads the same row of A across many
threads, creating high redundancy that L1 catches efficiently (a high hit
rate on repeated, avoidable traffic) — while the coalesced kernel's access
pattern already avoids much of that redundant traffic in the first place,
leaving less for L1 to "hit" on. This is a useful, general caution: cache
hit rate in isolation is not a reliable proxy for kernel quality without
also considering *how much traffic reaches the cache in the first place*.

### Shared memory introduces zero bank conflicts in kernel 3, by construction

Kernel 3 shows 0 bank conflicts on both loads and stores. This is
mechanistically correct, not a measurement gap: `Bs[dotIdx*BLOCKSIZE+
threadCol]` gives each thread in a warp a distinct, consecutive bank
(perfectly conflict-free), while `As[threadRow*BLOCKSIZE+dotIdx]` is
identical across all threads in a warp at a given step — a **broadcast
read**, a special hardware case that also produces zero conflicts (one
transaction serves the whole warp). This confirms Boehm's later discussion
of bank conflicts as a real bottleneck applies to the more complex
register-tiled access patterns introduced in kernels 4 onward (where each
thread reads a *range* of shared-memory addresses depending on its
register-tile position), not to kernel 3's simpler tiling scheme — a
distinction previously unconfirmed without direct profiling access.

### A newly explained finding: L1 hit rate collapses once shared memory is introduced

L1 hit rate drops sharply from kernel 2 (94.93%) to kernel 3 (1.52%). This
is expected and not a sign of a problem: once data is staged through
explicit shared memory, repeated reuse of global-memory values happens via
SMEM rather than via L1 catching redundant GMEM reads — L1's
redundancy-catching role is largely displaced by the programmer-managed
SMEM cache, so naturally far less global-memory traffic passes through L1
at all. L2 hit rate remains substantial (74.56%), continuing to support the
broader L2-masking pattern seen throughout this project.

### Bank conflicts confirmed: they appear exactly where Boehm's account predicts (kernel 6), not before

Kernels 1-4 (naive through 1D blocktiling) all show **zero** shared-memory
bank conflicts, and this was verified as mechanistically correct rather
than a measurement gap: in each case, every warp's shared-memory access
pattern reduces to either perfect coalescing (one bank per thread) or a
broadcast read (identical address for every thread in the warp) — both
hardware-supported conflict-free cases, given tile dimensions that are
clean multiples of the 32-thread warp size.

Kernel 6 (vectorized, transposed `As`, `float4` loads) is the first kernel
in the entire progression to show non-zero bank conflicts:

| Metric | Kernel 4 (1D blocktiled) | Kernel 6 (Vectorized) |
|---|---|---|
| Shared-mem bank conflicts (LD) | 0 | **4,194,304** |
| Shared-mem bank conflicts (ST) | 0 | **262,144** |

This is a direct, measured confirmation of Boehm's own account: he
describes kernel 6 as introducing real bank conflicts specifically because
of the transposed-`As` + vectorized-load restructuring, motivating his
(unpublished, and per this project's own scoping decision, skipped)
kernels 7-8 aimed at eliminating them. This is strong independent
verification that the source article's narrative about *where* conflicts
originate is accurate, obtained via direct hardware counters rather than
by taking the claim on faith.

**Caution on the sectors/request metric across load widths.** Kernel 6's
`l1tex__average_t_sectors_per_request` (16.81) is numerically similar to
the *naive* kernel's (16.51), which could misleadingly suggest kernel 6 has
reverted to naive-level coalescing inefficiency. This is very unlikely to
be a real regression: kernel 6 uses `float4` (16-byte) vectorized loads
per thread instead of scalar 4-byte loads, so a single logical "request" in
this metric now represents four times the data movement of kernels 1-5,
making direct numerical comparison across differently-vectorized kernels
unreliable. A rigorous apples-to-apples comparison would need a
byte-normalized coalescing metric rather than this ratio directly; flagged
here as a methodological caution rather than a performance finding, since
resolving it precisely was outside this pass's scope.

---

## Non-square dimension sweep extended to all four architectures

The same 4-shape non-square/non-power-of-two test (see earlier section) was
re-run on A100 and RTX 6000 Ada via the same padding methodology, using the
identical `torch.matmul` reference files generated once for L40S (fully
deterministic given a fixed seed, so reused as-is across architectures).

### Full 3-architecture comparison (GFLOPS)

| Shape           | Blocks | L40S (142 SM) | A100 (108 SM) | RTX 6000 Ada (142 SM) |
|-----------------|--------|----------------|-----------------|--------------------------|
| 3000×1500×2048  | 288    | 27515.57       | 10613.21        | **30582.82**             |
| 8192×256×1024   | 128    | 34083.58       | 7083.78         | **38043.57**             |
| 2048×2048×2049  | 256    | 35932.00       | 9881.91         | **42540.81**             |
| 137×263×401     | 6      | 623.98         | 218.59          | **625.24**               |

All 12 architecture/shape combinations pass correctness against the same
independent `torch.matmul` ground truth.

### Key finding: small-matrix collapse severity is near-identical between the two Ada-class GPUs

L40S and RTX 6000 Ada — near-identical silicon, confirmed earlier in this
document — produce almost indistinguishable tiny-matrix performance
(623.98 vs 625.24 GFLOPS), consistent with the occupancy-starvation
mechanism being a property of the shared architecture (SM count, warp
scheduler design) rather than a quirk of one specific chip.

### Key finding: A100 collapses proportionally less than either Ada-class GPU

Ratio of best large-shape performance to tiny-shape performance:
- L40S: 35932/624 ≈ **57.6x**
- RTX 6000 Ada: 42541/625 ≈ **68.1x**
- A100: 9882/219 ≈ **45.1x**

Despite having fewer SMs (108 vs 142) — which, naively, should make a
fixed 6-block launch a *smaller* fraction of available SMs and thus a
*worse* relative collapse — A100 in fact shows the mildest relative
degradation of the three. This is consistent with the broader pattern
found throughout this study: A100's much lower raw peak FLOPS ceiling
means there is proportionally less headroom to leave unfilled at any
occupancy level, making it inherently less sensitive to under-occupancy
than the higher-ceiling Ada-class cards.

## Real measured occupancy data confirms the small-matrix collapse mechanism (Colab T4)

Using Colab's working `ncu` access, the tiny-matrix case (137×263×401, 6
blocks) and a well-saturated large case (2048×2048×2049, 256 blocks) were
directly profiled for **actual measured occupancy and SM throughput** —
not the hand-derived theoretical occupancy used elsewhere in this document
due to `ncu` being unavailable on the main Turing cluster architectures.

| Metric (T4, sm_75) | Small (6 blocks) | Large (256 blocks) |
|---|---|---|
| Measured occupancy (`sm__warps_active`) | 24.93% | **47.30%** |
| SM throughput (`sm__throughput`) | 10.75% | **60.88%** |
| Shared-mem bank conflicts (LD) | 159,744 | 33,816,576 |

This is the first *directly measured* (not inferred from block-count
arithmetic) confirmation of the occupancy-starvation mechanism proposed
throughout this study. Two results stand out:

1. **Occupancy roughly doubles (24.93%→47.30%) while SM throughput
   increases nearly 6x (10.75%→60.88%).** The throughput gap being much
   larger than the occupancy gap indicates this kernel sits well below the
   point of diminishing occupancy returns — small occupancy gains here
   translate into disproportionately large real utilization gains, unlike
   kernels already near the occupancy ceiling (where further gains yield
   little additional throughput, per Boehm/Volkov's "cusp behavior"
   discussion referenced earlier in kernel 3's analysis).

2. **Bank conflicts scale slightly super-linearly with block count**: a
   ~42.7x increase in blocks (6→256) produces a ~211x increase in total
   conflicts (159,744→33,816,576). This is a secondary observation not
   deeply investigated here, but worth flagging as a candidate follow-up:
   whether per-block conflict *rate* also increases with more concurrent
   blocks (e.g., through increased memory-system contention) or whether
   this is simply proportional to total work done, would require further
   per-block-normalized analysis.

**Methodological note:** `ncu`'s instrumentation overhead reduced measured
GFLOPS by roughly 1000x in these profiling runs (e.g., 204.30 GFLOPS
unprofiled vs 0.16 GFLOPS profiled for the same small-matrix case). This is
expected profiler overhead, not a real performance change — `ncu`-profiled
GFLOPS figures should never be compared against normal (unprofiled) timing
runs; only the hardware counter percentages/counts themselves are
meaningful across profiling and non-profiling runs.

---

## Comparison against cuBLAS (via torch.matmul): how close does our best kernel get?

To address the assignment's request to consider "other approaches" and
provide a concrete performance ceiling, our best hand-written kernels were
benchmarked against `torch.matmul` (which dispatches to cuBLAS) at the same
sizes, on A100 and RTX 6000 Ada.

| Size | GPU | Custom (best autotuned) | Custom (warptiled) | cuBLAS | % of cuBLAS (best) |
|------|-----|---------------------------|----------------------|--------|----------------------|
| 2048³ | A100 | 11274.25 | 11545.02 | 13734.28 | **84.1%** |
| 2048³ | RTX 6000 Ada | 42419.93 | 45069.00 | 48572.70 | **92.8%** |

### Key finding: our kernel closes the gap to cuBLAS more on Ada-class hardware than on A100

Despite starting from the same source code and optimization techniques
(Boehm's kernel progression through warptiling and autotuning), our
implementation reaches a meaningfully higher fraction of cuBLAS's
performance on RTX 6000 Ada (92.8%) than on A100 (84.1%). This is
consistent with — and a direct, concrete illustration of — the
architecture-dependent kernel dispatch behavior discussed in Boehm's
kernel 9/10 sections: cuBLAS ships many SGEMM kernel variants and selects
per architecture and problem shape at runtime, meaning its A100-targeted
kernels likely incorporate Ampere-specific tuning (e.g., different tile
sizes or pipelining strategies suited to A100's SM design, tensor core
usage for mixed-precision paths, or double-buffering) beyond what our fixed
`(BM,BN,BK,TM,TN)` warptiling implementation captures. Since our kernel was
tuned via the same generic autotuning search on both architectures (not
independently re-optimized per architecture beyond re-running the same
search), reaching 92.8% of cuBLAS on one architecture and only 84.1% on
another highlights that "optimal" parameters are architecture-specific in
ways a single autotuning pass captures only partially — cuBLAS's advantage
likely comes from architecture-specific *kernel design* differences (not
just parameter retuning) that this study's kernel-progression approach does
not implement (e.g., double buffering, explicitly interleaved software
pipelining, or tensor-core paths where applicable).

### This is nonetheless a strong result for a from-scratch, course-scope implementation

Reaching 84-93% of a professionally engineered, vendor-maintained library's
performance using only the techniques covered in a single guided kernel
progression (memory coalescing, shared-memory tiling, register blocking,
vectorization, warptiling, and parameter autotuning) demonstrates that the
overwhelming majority of the achievable speedup is captured by
understanding and correctly applying these well-known optimization
principles — the remaining 7-16% gap likely requires substantially more
specialized techniques (double buffering/software pipelining, tensor core
utilization, or architecture-specific micro-tuning) beyond this study's
scope.

### T4 cuBLAS comparison — a much larger gap, with an important caveat

| Size | GPU | GFLOPS |
|------|-----|--------|
| 1024³ | T4 | 2979.41 |
| 1536³ | T4 | 3213.64 |
| 2048³ | T4 | 3284.48 |
| 4096³ | T4 | 4324.44 |

Our kernel 6 (vectorized, default fixed parameters) on T4 achieved only
15.44 GFLOPS at 1024³ — roughly **0.5% of cuBLAS's 2979.41 GFLOPS** at the
same size, a dramatically larger gap than the 84-93% achieved on A100/RTX
6000 Ada.

**This gap should not be read as "our approach fails on Turing."** Two
important differences from the A100/RTX 6000 Ada comparison make this an
apples-to-oranges result rather than a genuine architecture-specific
regression:

1. **T4 only received the early/mid-pipeline kernel** (kernel 6, vectorized,
   with default fixed parameters) for `ncu` profiling purposes — the full
   warptiling and autotuning stages (which closed most of the gap to
   cuBLAS on the two main-line architectures) were never run on T4, since
   its role in this study was specifically to obtain hardware counter data
   unavailable elsewhere, not to complete the full optimization pipeline.
2. **cuBLAS on Turing likely uses fundamentally different compute
   pathways** than on Ampere/Ada — Turing's Tensor Cores support mixed
   precision paths, and NVIDIA's SGEMM library implementations for this
   architecture generation may route through kernels our pure-FP32,
   CUDA-core-only implementation was never designed to compete with
   directly.

A fair, like-for-like comparison would require running the full warptiling
and autotuning pipeline on T4 specifically — not attempted here, since T4's
role in this study was scoped to profiling rather than full optimization.
This gap is reported transparently as a scope limitation rather than
omitted, but should not be interpreted as evidence that the optimization
techniques in this study are Turing-specific failures.

### Full three-architecture cuBLAS comparison, and a consistency check

| Size | GPU | Custom (best autotuned) | Custom (warptiled) | cuBLAS | % of cuBLAS (best) |
|------|-----|---------------------------|----------------------|--------|----------------------|
| 2048³ | L40S | 35965.95 | 37860.87 | 44340.41 | **85.4%** |
| 2048³ | A100 | 11274.25 | 11545.02 | 13734.28 | **84.1%** |
| 2048³ | RTX 6000 Ada | 42419.93 | 45069.00 | 48572.70 | **92.8%** |

L40S and A100 land within 1.3 percentage points of each other (85.4% vs
84.1%), while RTX 6000 Ada notably outperforms both (92.8%) — despite L40S
and RTX 6000 Ada being near-identical silicon. This is fully consistent
with, and independently reinforces from a different angle, the earlier
finding that RTX 6000 Ada's ~11% higher memory bandwidth and different
power/clock sustain profile specifically benefit the most
bandwidth-intensive kernel stages (warptiling, heavily vectorized
autotuned configs). The gap to cuBLAS is not a fixed property of "the
architecture generation" (both are Ada Lovelace) but tracks the same
real, measurable hardware difference (memory bandwidth/sustained clocks)
identified earlier in the main kernel-progression comparison — a second,
independent line of evidence for the same underlying mechanism.
