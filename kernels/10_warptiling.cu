#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

#define WARPSIZE 32

#ifndef BM
#define BM 128
#endif
#ifndef BN
#define BN 128
#endif
#ifndef BK
#define BK 16
#endif
#ifndef WM
#define WM 64
#endif
#ifndef WN
#define WN 64
#endif
#ifndef WNITER
#define WNITER 4
#endif
#ifndef TM
#define TM 8
#endif
#ifndef TN
#define TN 4
#endif
#ifndef NUM_THREADS
#define NUM_THREADS 128
#endif

#define WMITER ((WM * WN) / (WARPSIZE * TM * TN * WNITER))
#define WSUBM (WM / WMITER)
#define WSUBN (WN / WNITER)

__global__ void __launch_bounds__(NUM_THREADS) sgemm_warptiling(
    int M, int N, int K, float alpha,
    const float *A, const float *B, float beta, float *C) {

    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    // Warp placement within the threadblock tile
    const uint warpIdx = threadIdx.x / WARPSIZE;
    const uint warpCol = warpIdx % (BN / WN);
    const uint warpRow = warpIdx / (BN / WN);

    // Thread placement within its warp's subtile
    const uint threadIdxInWarp = threadIdx.x % WARPSIZE;
    const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN);
    const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN);

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    A += cRow * BM * K;
    B += cCol * BN;
    C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

    const uint innerRowA = threadIdx.x / (BK / 4);
    const uint innerColA = threadIdx.x % (BK / 4);
    const uint rowStrideA = (NUM_THREADS * 4) / BK;
    const uint innerRowB = threadIdx.x / (BN / 4);
    const uint innerColB = threadIdx.x % (BN / 4);
    const uint rowStrideB = NUM_THREADS / (BN / 4);

    float threadResults[WMITER * TM * WNITER * TN] = {0.0f};
    float regM[WMITER * TM] = {0.0f};
    float regN[WNITER * TN] = {0.0f};

    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        for (uint offset = 0; offset < BM; offset += rowStrideA) {
            float4 tmp = reinterpret_cast<const float4 *>(
                &A[(innerRowA + offset) * K + innerColA * 4])[0];
            As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
            As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
            As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
            As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
        }
        for (uint offset = 0; offset < BK; offset += rowStrideB) {
            reinterpret_cast<float4 *>(
                &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
                reinterpret_cast<const float4 *>(
                    &B[(innerRowB + offset) * N + innerColB * 4])[0];
        }
        __syncthreads();

        A += BK;
        B += BK * N;

        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
                for (uint i = 0; i < TM; ++i) {
                    regM[wSubRowIdx * TM + i] =
                        As[(dotIdx * BM) + warpRow * WM + wSubRowIdx * WSUBM +
                           threadRowInWarp * TM + i];
                }
            }
            for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                for (uint i = 0; i < TN; ++i) {
                    regN[wSubColIdx * TN + i] =
                        Bs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN +
                           threadColInWarp * TN + i];
                }
            }
            for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
                for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                            threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                                          (wSubColIdx * TN) + resIdxN] +=
                                regM[wSubRowIdx * TM + resIdxM] *
                                regN[wSubColIdx * TN + resIdxN];
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
            float *C_interim = C + wSubRowIdx * WSUBM * N + wSubColIdx * WSUBN;
            for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
                    float4 tmp = reinterpret_cast<float4 *>(
                        &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                                   threadColInWarp * TN + resIdxN])[0];
                    const int i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                                  wSubColIdx * TN + resIdxN;
                    tmp.x = alpha * threadResults[i + 0] + beta * tmp.x;
                    tmp.y = alpha * threadResults[i + 1] + beta * tmp.y;
                    tmp.z = alpha * threadResults[i + 2] + beta * tmp.z;
                    tmp.w = alpha * threadResults[i + 3] + beta * tmp.w;
                    reinterpret_cast<float4 *>(
                        &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                                   threadColInWarp * TN + resIdxN])[0] = tmp;
                }
            }
        }
    }
}

void run_sgemm_warptiling(int M, int N, int K, float alpha,
                           const float *A, const float *B,
                           float beta, float *C) {
    dim3 gridDim(N / BN, M / BM);
    dim3 blockDim(NUM_THREADS);
    sgemm_warptiling<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

#ifdef STANDALONE_MAIN
#include <vector>

void fill_random(std::vector<float> &v) {
    for (auto &x : v) x = static_cast<float>(rand()) / RAND_MAX;
}

int main(int argc, char **argv) {
    int M = argc > 1 ? atoi(argv[1]) : 2048;
    int N = argc > 2 ? atoi(argv[2]) : 2048;
    int K = argc > 3 ? atoi(argv[3]) : 2048;
    float alpha = 1.0f, beta = 0.0f;

    if (M % BM != 0 || N % BN != 0 || K % BK != 0) {
        fprintf(stderr, "WARNING: M multiple of %d, N multiple of %d, K multiple of %d required.\n", BM, BN, BK);
    }

    printf("Warptiled GEMM: M=%d N=%d K=%d (BM=%d BN=%d BK=%d WM=%d WN=%d WNITER=%d TM=%d TN=%d NUM_THREADS=%d)\n",
           M, N, K, BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS);

    std::vector<float> hA(M * K), hB(K * N), hC(M * N, 0.0f);
    fill_random(hA);
    fill_random(hB);

    float *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dB, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dC, M * N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(dA, hA.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dC, hC.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));

    run_sgemm_warptiling(M, N, K, alpha, dA, dB, beta, dC);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());

    const int n_iters = 10;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < n_iters; ++i) {
        run_sgemm_warptiling(M, N, K, alpha, dA, dB, beta, dC);
    }
    CHECK_CUDA(cudaGetLastError());
    cudaEventRecord(stop);
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= n_iters;

    double flops = 2.0 * M * N * K;
    double gflops = (flops / (ms / 1000.0)) / 1e9;

    printf("Avg time: %.4f ms | Performance: %.2f GFLOPS\n", ms, gflops);

    CHECK_CUDA(cudaMemcpy(hC.data(), dC, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    double checksum = 0.0;
    for (size_t i = 0; i < hC.size(); ++i) checksum += hC[i];

    printf("C[0][0] = %f\n", hC[0]);
    printf("C[mid][mid] = %f\n", hC[(M/2) * N + (N/2)]);
    printf("C[last][last] = %f\n", hC[(M-1) * N + (N-1)]);
    printf("CHECKSUM = %f\n", checksum);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
#endif
