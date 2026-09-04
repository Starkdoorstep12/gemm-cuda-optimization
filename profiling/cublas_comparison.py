"""
Benchmark torch.matmul (cuBLAS) against our best hand-written kernels,
at the same matrix sizes used throughout this study.
"""
import torch
import time
import json

def bench_torch_matmul(M, N, K, n_iters=20, seed=42):
    torch.manual_seed(seed)
    A = torch.rand(M, K, device='cuda', dtype=torch.float32)
    B = torch.rand(K, N, device='cuda', dtype=torch.float32)

    # warmup
    for _ in range(3):
        C = A @ B
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(n_iters):
        C = A @ B
    torch.cuda.synchronize()
    elapsed_ms = (time.time() - start) / n_iters * 1000

    flops = 2.0 * M * N * K
    gflops = flops / (elapsed_ms / 1000) / 1e9
    return elapsed_ms, gflops, C[0][0].item()

def main():
    sizes = [
        (1024, 1024, 1024),
        (1536, 1536, 1536),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
    ]

    device_name = torch.cuda.get_device_name(0)
    print(f"Device: {device_name}")
    print(f"{'Size':<20}{'Time (ms)':<15}{'GFLOPS':<15}{'C[0][0]'}")

    results = []
    for M, N, K in sizes:
        ms, gflops, c00 = bench_torch_matmul(M, N, K)
        print(f"{M}x{N}x{K:<12}{ms:<15.4f}{gflops:<15.2f}{c00:.6f}")
        results.append({"M": M, "N": N, "K": K, "ms": ms, "gflops": gflops, "c00": c00, "device": device_name})

    with open("profiling/cublas_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
