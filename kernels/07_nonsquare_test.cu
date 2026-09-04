#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>
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
#define BK 16
#endif
#ifndef TM
#define TM 8
#endif
#ifndef TN
#define TN 8
#endif
#define NUM_THREADS ((BM * BN) / (TM * TN))

// Kernel 6 (vectorized, strided-load, bug-fixed version) - declared extern
// since it's identical to what's in 06_vectorized_tunable.cu.
// We redeclare it here to keep this test file self-contained for clarity.
__global__ void sgemm_vectorized(int M, int N, int K, float alpha,
                                  const float *A, const float *B,
                                  float beta, float *C) {
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;
    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);

    __shared__ float As[BK * BM];
    __shared__ float Bs[BK * BN];

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

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
        for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
            float4 tmp = reinterpret_cast<const float4 *>(
                &A[(innerRowA + loadOffset) * K + innerColA * 4])[0];
            As[(innerColA * 4 + 0) * BM + innerRowA + loadOffset] = tmp.x;
            As[(innerColA * 4 + 1) * BM + innerRowA + loadOffset] = tmp.y;
            As[(innerColA * 4 + 2) * BM + innerRowA + loadOffset] = tmp.z;
            As[(innerColA * 4 + 3) * BM + innerRowA + loadOffset] = tmp.w;
        }
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
                    reinterpret_cast<float4 *>(&As[dotIdx * BM + threadRow * TM + i])[0];
            }
            for (uint i = 0; i < TN; i += 4) {
                reinterpret_cast<float4 *>(&regN[i])[0] =
                    reinterpret_cast<float4 *>(&Bs[dotIdx * BN + threadCol * TN + i])[0];
            }
            for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                    threadResults[resIdxM * TN + resIdxN] += regM[resIdxM] * regN[resIdxN];
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

int ceil_to_multiple(int x, int m) { return ((x + m - 1) / m) * m; }

int main(int argc, char **argv) {
    int M = argc > 1 ? atoi(argv[1]) : 3000;
    int N = argc > 2 ? atoi(argv[2]) : 1500;
    int K = argc > 3 ? atoi(argv[3]) : 2048;
    float alpha = 1.0f, beta = 0.0f;

    // Pad M and N up to tile-size multiples (K only needs to be a multiple of BK)
    int Mp = ceil_to_multiple(M, BM);
    int Np = ceil_to_multiple(N, BN);
    int Kp = ceil_to_multiple(K, BK);

    printf("Non-square test: M=%d N=%d K=%d -> padded Mp=%d Np=%d Kp=%d\n", M, N, K, Mp, Np, Kp);

    // Load A, B from disk (unpadded), reference C
    std::vector<float> hA_orig(M * K), hB_orig(K * N), hC_ref(M * N);
    char fname[256];
    snprintf(fname, sizeof(fname), "profiling/nonsquare_test/A_%d_%d_%d.bin", M, N, K);
    FILE *fA = fopen(fname, "rb");
    if (!fA) { fprintf(stderr, "Cannot open %s\n", fname); return 1; }
    fread(hA_orig.data(), sizeof(float), M * K, fA);
    fclose(fA);

    snprintf(fname, sizeof(fname), "profiling/nonsquare_test/B_%d_%d_%d.bin", M, N, K);
    FILE *fB = fopen(fname, "rb");
    if (!fB) { fprintf(stderr, "Cannot open %s\n", fname); return 1; }
    fread(hB_orig.data(), sizeof(float), K * N, fB);
    fclose(fB);

    snprintf(fname, sizeof(fname), "profiling/nonsquare_test/C_ref_%d_%d_%d.bin", M, N, K);
    FILE *fC = fopen(fname, "rb");
    if (!fC) { fprintf(stderr, "Cannot open %s\n", fname); return 1; }
    fread(hC_ref.data(), sizeof(float), M * N, fC);
    fclose(fC);

    // Build padded host buffers (zero-padded)
    std::vector<float> hA_pad(Mp * Kp, 0.0f), hB_pad(Kp * Np, 0.0f), hC_pad(Mp * Np, 0.0f);
    for (int i = 0; i < M; ++i)
        memcpy(&hA_pad[i * Kp], &hA_orig[i * K], K * sizeof(float));
    for (int i = 0; i < K; ++i)
        memcpy(&hB_pad[i * Np], &hB_orig[i * N], N * sizeof(float));

    float *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, Mp * Kp * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dB, Kp * Np * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dC, Mp * Np * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dA, hA_pad.data(), Mp * Kp * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB_pad.data(), Kp * Np * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dC, hC_pad.data(), Mp * Np * sizeof(float), cudaMemcpyHostToDevice));

    dim3 gridDim(Np / BN, Mp / BM);
    dim3 blockDim(NUM_THREADS);

    sgemm_vectorized<<<gridDim, blockDim>>>(Mp, Np, Kp, alpha, dA, dB, beta, dC);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());

    const int n_iters = 10;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < n_iters; ++i) {
        sgemm_vectorized<<<gridDim, blockDim>>>(Mp, Np, Kp, alpha, dA, dB, beta, dC);
    }
    CHECK_CUDA(cudaGetLastError());
    cudaEventRecord(stop);
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= n_iters;
    double flops = 2.0 * M * N * K;  // real work, not padded
    double gflops = (flops / (ms / 1000.0)) / 1e9;
    printf("Avg time: %.4f ms | Performance (real MNK): %.2f GFLOPS\n", ms, gflops);

    CHECK_CUDA(cudaMemcpy(hC_pad.data(), dC, Mp * Np * sizeof(float), cudaMemcpyDeviceToHost));

    // Extract the real M x N region and compare against reference
    double max_abs_err = 0.0, sum_abs_err = 0.0;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float got = hC_pad[i * Np + j];
            float expected = hC_ref[i * N + j];
            double err = fabs((double)got - (double)expected);
            max_abs_err = fmax(max_abs_err, err);
            sum_abs_err += err;
        }
    }
    double mean_abs_err = sum_abs_err / (M * N);

    printf("C[0][0] = %f (expected %f)\n", hC_pad[0], hC_ref[0]);
    printf("C[last][last] = %f (expected %f)\n", hC_pad[(M-1)*Np + (N-1)], hC_ref[(M-1)*N + (N-1)]);
    printf("Max abs error: %e | Mean abs error: %e\n", max_abs_err, mean_abs_err);
    printf("%s\n", max_abs_err < 1e-1 ? "CORRECTNESS: PASS" : "CORRECTNESS: FAIL");

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
