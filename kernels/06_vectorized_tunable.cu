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

#ifndef BM
#define BM 128
#endif
#ifndef BN
#define BN 128
#endif
#ifndef BK
#define BK 8
#endif
#ifndef TM
#define TM 8
#endif
#ifndef TN
#define TN 8
#endif
#define NUM_THREADS ((BM * BN) / (TM * TN))

__global__ void sgemm_vectorized(int M, int N, int K, float alpha,
                                  const float *A, const float *B,
                                  float beta, float *C) {
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);

    __shared__ float As[BK * BM];  // transposed: As[k * BM + m]
    __shared__ float Bs[BK * BN];

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    // Strided loading indices - generalizes to any NUM_THREADS/BM/BN/BK
    // combination, as long as the divisibility constraints in is_valid()
    // hold (each thread issues float4-wide loads, possibly multiple
    // passes if the tile is larger than one pass can cover).
    const uint innerRowA = threadIdx.x / (BK / 4);
    const uint innerColA = threadIdx.x % (BK / 4);
    const uint strideA = NUM_THREADS / (BK / 4);

    const uint innerRowB = threadIdx.x / (BN / 4);
    const uint innerColB = threadIdx.x % (BN / 4);
    const uint strideB = NUM_THREADS / (BN / 4);

    float threadResults[TM * TN] = {0.0f};
    float regM[TM] = {0.0f};
    float regN[TN] = {0.0f};

    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // Strided multi-pass load of A, transposing into As as we go
        for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
            float4 tmp = reinterpret_cast<const float4 *>(
                &A[(innerRowA + loadOffset) * K + innerColA * 4])[0];
            As[(innerColA * 4 + 0) * BM + innerRowA + loadOffset] = tmp.x;
            As[(innerColA * 4 + 1) * BM + innerRowA + loadOffset] = tmp.y;
            As[(innerColA * 4 + 2) * BM + innerRowA + loadOffset] = tmp.z;
            As[(innerColA * 4 + 3) * BM + innerRowA + loadOffset] = tmp.w;
        }

        // Strided multi-pass load of B (no transpose needed)
        for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
            reinterpret_cast<float4 *>(
                &Bs[(innerRowB + loadOffset) * BN + innerColB * 4])[0] =
                reinterpret_cast<const float4 *>(
                    &B[(innerRowB + loadOffset) * N + innerColB * 4])[0];
        }

        __syncthreads();

        A += BK;
        B += BK * N;

        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            for (uint i = 0; i < TM; i += 4) {
                reinterpret_cast<float4 *>(&regM[i])[0] =
                    reinterpret_cast<float4 *>(
                        &As[dotIdx * BM + threadRow * TM + i])[0];
            }
            for (uint i = 0; i < TN; i += 4) {
                reinterpret_cast<float4 *>(&regN[i])[0] =
                    reinterpret_cast<float4 *>(
                        &Bs[dotIdx * BN + threadCol * TN + i])[0];
            }
            for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                    threadResults[resIdxM * TN + resIdxN] +=
                        regM[resIdxM] * regN[resIdxN];
                }
            }
        }
        __syncthreads();
    }

    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
            float4 c_tmp = reinterpret_cast<float4 *>(
                &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0];
            c_tmp.x = alpha * threadResults[resIdxM * TN + resIdxN + 0] + beta * c_tmp.x;
            c_tmp.y = alpha * threadResults[resIdxM * TN + resIdxN + 1] + beta * c_tmp.y;
            c_tmp.z = alpha * threadResults[resIdxM * TN + resIdxN + 2] + beta * c_tmp.z;
            c_tmp.w = alpha * threadResults[resIdxM * TN + resIdxN + 3] + beta * c_tmp.w;
            reinterpret_cast<float4 *>(
                &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0] = c_tmp;
        }
    }
}

void run_sgemm_vectorized(int M, int N, int K, float alpha,
                           const float *A, const float *B,
                           float beta, float *C) {
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 blockDim(NUM_THREADS);
    sgemm_vectorized<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

#ifdef STANDALONE_MAIN
#include <vector>

void fill_random(std::vector<float> &v) {
    for (auto &x : v) x = static_cast<float>(rand()) / RAND_MAX;
}

int main(int argc, char **argv) {
    int M = argc > 1 ? atoi(argv[1]) : 1024;
    int N = argc > 2 ? atoi(argv[2]) : 1024;
    int K = argc > 3 ? atoi(argv[3]) : 1024;
    float alpha = 1.0f, beta = 0.0f;

    if (M % BM != 0 || N % BN != 0 || K % BK != 0) {
        fprintf(stderr, "WARNING: M multiple of %d, N multiple of %d, K multiple of %d required.\n", BM, BN, BK);
    }

    printf("Vectorized GEMM: M=%d N=%d K=%d (BM=%d BN=%d BK=%d TM=%d TN=%d)\n",
           M, N, K, BM, BN, BK, TM, TN);

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

    run_sgemm_vectorized(M, N, K, alpha, dA, dB, beta, dC);
    CHECK_CUDA(cudaDeviceSynchronize());

    const int n_iters = 10;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < n_iters; ++i) {
        run_sgemm_vectorized(M, N, K, alpha, dA, dB, beta, dC);
    }
    CHECK_CUDA(cudaGetLastError());  // catch silent launch failures (e.g. register/resource limits exceeded)
    cudaEventRecord(stop);
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= n_iters;

    double flops = 2.0 * M * N * K;
    double gflops = (flops / (ms / 1000.0)) / 1e9;

    printf("Avg time: %.4f ms | Performance: %.2f GFLOPS\n", ms, gflops);

    CHECK_CUDA(cudaMemcpy(hC.data(), dC, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Multi-point correctness check: corner, center, opposite corner, checksum
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
