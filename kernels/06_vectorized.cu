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

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8
#define NUM_THREADS ((BM * BN) / (TM * TN))  // = 256

__global__ void sgemm_vectorized(int M, int N, int K, float alpha,
                                  const float *A, const float *B,
                                  float beta, float *C) {
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);

    // As is stored TRANSPOSED: As[k * BM + m] instead of As[m * BK + k]
    __shared__ float As[BK * BM];
    __shared__ float Bs[BK * BN];

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    // Loading indices - operate in float4 (4-element) chunks
    const uint innerRowA = threadIdx.x / (BK / 4);
    const uint innerColA = threadIdx.x % (BK / 4);
    const uint innerRowB = threadIdx.x / (BN / 4);
    const uint innerColB = threadIdx.x % (BN / 4);

    float threadResults[TM * TN] = {0.0f};
    float regM[TM] = {0.0f};
    float regN[TN] = {0.0f};

    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // Load A as float4, transpose into As during the SMEM write
        float4 tmp = reinterpret_cast<const float4 *>(
            &A[innerRowA * K + innerColA * 4])[0];
        As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
        As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
        As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
        As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;

        // Load B as float4, straight copy (no transpose needed)
        reinterpret_cast<float4 *>(
            &Bs[innerRowB * BN + innerColB * 4])[0] =
            reinterpret_cast<const float4 *>(
                &B[innerRowB * N + innerColB * 4])[0];

        __syncthreads();

        A += BK;
        B += BK * N;

        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // Vectorized SMEM loads (As now transposed -> contiguous reads)
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

    // Vectorized C write-back
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
    cudaEventRecord(stop);
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= n_iters;

    double flops = 2.0 * M * N * K;
    double gflops = (flops / (ms / 1000.0)) / 1e9;

    printf("Avg time: %.4f ms | Performance: %.2f GFLOPS\n", ms, gflops);

    CHECK_CUDA(cudaMemcpy(hC.data(), dC, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    printf("C[0][0] = %f (sanity check)\n", hC[0]);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
#endif
