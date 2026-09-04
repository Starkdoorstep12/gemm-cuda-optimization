"""
Autotuning sweep for kernel 6 (vectorized GEMM, strided-load version).
Enumerates valid (BM, BN, BK, TM, TN) combinations, compiles each,
benchmarks, verifies correctness at multiple output points, and reports the best.
"""
import subprocess
import itertools
import re
import json
import os

KERNEL_SRC = "kernels/06_vectorized_tunable.cu"
BINARY_DIR = "/tmp/autotune_bins"
os.makedirs(BINARY_DIR, exist_ok=True)

BM_options = [64, 128, 256]
BN_options = [64, 128, 256]
BK_options = [8, 16, 32]
TM_options = [4, 8]
TN_options = [4, 8]

TEST_SIZE = (2048, 2048, 2048)
REFERENCE = {
    "c00": 510.334869,
    "cmid": 495.830078,
    "clast": 516.925964,
    "checksum": 2147751304.860840,
}
TOLERANCE = 1.0  # loose float tolerance; checksum needs a bigger absolute tolerance
CHECKSUM_TOLERANCE = 5000.0  # sum over 4M floats accumulates more float error
MAX_SANE_GFLOPS = 100000

ARCH = "sm_89"
MAX_SMEM_BYTES = 48 * 1024

def is_valid(BM, BN, BK, TM, TN):
    if BM % TM != 0 or BN % TN != 0:
        return False
    num_threads = (BM * BN) // (TM * TN)
    if num_threads <= 0 or num_threads > 1024:
        return False
    if num_threads % 32 != 0:
        return False
    if BK % 4 != 0:
        return False
    # Strided-load divisibility: NUM_THREADS must divide evenly into
    # (BK/4) row-groups for A's load, and (BN/4) row-groups for B's load,
    # AND BM/strideA, BK/strideB must be whole numbers of passes.
    if num_threads % (BK // 4) != 0:
        return False
    if num_threads % (BN // 4) != 0:
        return False
    strideA = num_threads // (BK // 4)
    strideB = num_threads // (BN // 4)
    if BM % strideA != 0:
        return False
    if BK % strideB != 0:
        return False
    # Shared memory limit
    smem_bytes = (BK * BM + BK * BN) * 4
    if smem_bytes > MAX_SMEM_BYTES:
        return False
    return True

def compile_and_run(BM, BN, BK, TM, TN, M, N, K):
    binary = os.path.join(BINARY_DIR, f"gemm_{BM}_{BN}_{BK}_{TM}_{TN}")
    compile_cmd = [
        "nvcc", "-O3", "-DSTANDALONE_MAIN", f"-arch={ARCH}",
        f"-DBM={BM}", f"-DBN={BN}", f"-DBK={BK}", f"-DTM={TM}", f"-DTN={TN}",
        "-o", binary, KERNEL_SRC
    ]
    result = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        return None, {}, f"COMPILE_FAIL: {result.stderr[:200]}"

    run_result = subprocess.run(
        [binary, str(M), str(N), str(K)],
        capture_output=True, text=True, timeout=30
    )
    if run_result.returncode != 0:
        return None, {}, f"RUNTIME_FAIL: {run_result.stderr[:200]}"

    out = run_result.stdout
    gflops_match = re.search(r"Performance:\s*([\d.]+)\s*GFLOPS", out)
    c00_match = re.search(r"C\[0\]\[0\]\s*=\s*([\-\d.]+)", out)
    cmid_match = re.search(r"C\[mid\]\[mid\]\s*=\s*([\-\d.]+)", out)
    clast_match = re.search(r"C\[last\]\[last\]\s*=\s*([\-\d.]+)", out)
    checksum_match = re.search(r"CHECKSUM\s*=\s*([\-\d.]+)", out)

    if not all([gflops_match, c00_match, cmid_match, clast_match, checksum_match]):
        return None, {}, "PARSE_FAIL"

    gflops = float(gflops_match.group(1))
    vals = {
        "c00": float(c00_match.group(1)),
        "cmid": float(cmid_match.group(1)),
        "clast": float(clast_match.group(1)),
        "checksum": float(checksum_match.group(1)),
    }

    if gflops > MAX_SANE_GFLOPS:
        return gflops, vals, f"IMPLAUSIBLE_GFLOPS (>{MAX_SANE_GFLOPS})"

    for key in ["c00", "cmid", "clast"]:
        if abs(vals[key] - REFERENCE[key]) > TOLERANCE:
            return gflops, vals, f"CORRECTNESS_FAIL ({key}: got {vals[key]}, expected {REFERENCE[key]})"

    if abs(vals["checksum"] - REFERENCE["checksum"]) > CHECKSUM_TOLERANCE:
        return gflops, vals, f"CORRECTNESS_FAIL (checksum: got {vals['checksum']}, expected {REFERENCE['checksum']})"

    return gflops, vals, "OK"


def main():
    M, N, K = TEST_SIZE
    results = []
    combos = list(itertools.product(BM_options, BN_options, BK_options, TM_options, TN_options))
    valid_combos = [c for c in combos if is_valid(*c)]

    print(f"Total combos: {len(combos)}, valid (pre-filter): {len(valid_combos)}")
    print(f"Testing at M=N=K={M}\n")

    for i, (BM, BN, BK, TM, TN) in enumerate(valid_combos):
        gflops, vals, status = compile_and_run(BM, BN, BK, TM, TN, M, N, K)
        record = {"BM": BM, "BN": BN, "BK": BK, "TM": TM, "TN": TN,
                   "gflops": gflops, **vals, "status": status}
        results.append(record)
        gflops_str = f"{gflops:.2f}" if gflops else "N/A"
        print(f"[{i+1}/{len(valid_combos)}] BM={BM} BN={BN} BK={BK} TM={TM} TN={TN} "
              f"-> {gflops_str} GFLOPS | {status}")

    with open("profiling/autotune_results.json", "w") as f:
        json.dump(results, f, indent=2)

    correct_results = [r for r in results if r["status"] == "OK"]
    correct_results.sort(key=lambda r: r["gflops"], reverse=True)

    print(f"\n=== {len(correct_results)}/{len(valid_combos)} configs passed full correctness check ===")
    print("=== TOP 10 (correctness-verified) ===")
    for r in correct_results[:10]:
        print(f"BM={r['BM']} BN={r['BN']} BK={r['BK']} TM={r['TM']} TN={r['TN']} -> {r['gflops']:.2f} GFLOPS")

    failed = [r for r in results if r["status"] not in ("OK",) and "COMPILE_FAIL" not in str(r["status"])]
    if failed:
        print(f"\n=== {len(failed)} configs FAILED (excluded from ranking) ===")
        for r in failed:
            print(f"BM={r['BM']} BN={r['BN']} BK={r['BK']} TM={r['TM']} TN={r['TN']} -> {r['status']}")

if __name__ == "__main__":
    main()
