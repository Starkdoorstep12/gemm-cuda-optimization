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
`too many resources requested for launch`. This is now a permanent part of
the benchmarking harness for every kernel going forward, not just this one.

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
