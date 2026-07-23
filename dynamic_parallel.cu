#include <cuda_runtime.h>
#include <cstdio>

__global__ void nestHelloWorld(const int size, int depth) {
    const unsigned int tid = threadIdx.x;
    printf("depth: %d blockIdx: %d, threadIdx: %d\n", depth, blockIdx.x, tid);
    if (size == 1) return;
    int threads = (size >> 1);
    if (tid == 0 && threads > 0) {
        nestHelloWorld<<<1, threads>>>(threads, ++depth);
        printf("-------------------> nested execution depth: %d\n", depth);
    }
}

int main() {
    constexpr int size = 64;
    constexpr int block_x = 2;
    dim3 block(block_x, 1);
    dim3 grid((size - 1) / block.x + 1, 1);
    nestHelloWorld<<<grid, block>>>(size, 0);
    cudaGetLastError();
    cudaDeviceSynchronize();

    return 0;
}