#include <cstdio>
#include <cuda_runtime.h>

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Name: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Max threads per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Warp size: %d\n", prop.warpSize);
    printf("Max regs per block: %d\n", prop.regsPerBlock);
    printf("Max regs per multiprocessor: %d\n", prop.regsPerMultiprocessor);
    printf("Total global mem: %.0f MB\n", prop.totalGlobalMem / 1e6);
    printf("Max shared mem per block: %.0f KB\n", prop.sharedMemPerBlock / 1024.0);
    printf("Shared mem per multiprocessor: %.0f B\n", (double)prop.sharedMemPerMultiprocessor);
    printf("Multiprocessor count: %d\n", prop.multiProcessorCount);
    printf("Max warps per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor / prop.warpSize);
    return 0;
}
