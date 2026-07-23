#include <cstdio>
__global__ void reduceUnroll2(int *g_idata, int *g_odata, const int size) {
    const unsigned int tid = threadIdx.x;
    int *idata = g_idata + blockDim.x * blockIdx.x * 2;
    if (tid + blockDim.x < size) {
        idata[tid] += idata[tid + blockDim.x];   // idata 已含块偏移，用块内下标 tid
    }
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        g_odata[blockIdx.x] = idata[0];
    }
}

__global__ void reduceUnroll4(int *g_idata, int *g_odata, const int size) {
    const unsigned int tid = threadIdx.x;
    int *idata = g_idata + blockDim.x * blockIdx.x * 4;
    if (tid + blockDim.x * 3 < size) {
        idata[tid] += idata[tid + blockDim.x];
        idata[tid] += idata[tid + blockDim.x * 2];
        idata[tid] += idata[tid + blockDim.x * 3];
    }
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        g_odata[blockIdx.x] = idata[0];
    }
}


__global__ void reduceUnroll8(int *g_idata, int *g_odata, const int size) {
    const unsigned int tid = threadIdx.x;
    int *idata = g_idata + blockDim.x * blockIdx.x * 8;
    if (tid + blockDim.x * 7 < size) {
        int a1 = idata[tid];
        int a2 = idata[tid + blockDim.x];
        int a3 = idata[tid + blockDim.x * 2];
        int a4 = idata[tid + blockDim.x * 3];
        int a5 = idata[tid + blockDim.x * 4];
        int a6 = idata[tid + blockDim.x * 5];
        int a7 = idata[tid + blockDim.x * 6];
        int a8 = idata[tid + blockDim.x * 7];
        idata[tid] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
    }
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        g_odata[blockIdx.x] = idata[0];
    }
}

__global__ void reduceUnrollWarp8(int *g_idata, int *g_odata, const int size) {
    const unsigned int tid = threadIdx.x;
    int *idata = g_idata + blockDim.x * blockIdx.x * 8;
    if (tid + blockDim.x * 7 < size) {
        int a1 = idata[tid];
        int a2 = idata[tid + blockDim.x];
        int a3 = idata[tid + blockDim.x * 2];
        int a4 = idata[tid + blockDim.x * 3];
        int a5 = idata[tid + blockDim.x * 4];
        int a6 = idata[tid + blockDim.x * 5];
        int a7 = idata[tid + blockDim.x * 6];
        int a8 = idata[tid + blockDim.x * 7];
        idata[tid] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
    }
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }
    if (tid < 32) {
        volatile int *vmem = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }

    if (tid == 0) {
        g_odata[blockIdx.x] = idata[0];
    }

}

__global__ void reduceCompleteUnrollWarp8(int *g_idata, int *g_odata, const int size) {
    const unsigned int tid = threadIdx.x;
    int *idata = blockDim.x * blockIdx.x * 8 + g_idata;
    if (tid + blockDim.x * 7 < size) {
        int a1 = idata[tid];
        int a2 = idata[tid + blockDim.x];
        int a3 = idata[tid + blockDim.x * 2];
        int a4 = idata[tid + blockDim.x * 3];
        int a5 = idata[tid + blockDim.x * 4];
        int a6 = idata[tid + blockDim.x * 5];
        int a7 = idata[tid + blockDim.x * 6];
        int a8 = idata[tid + blockDim.x * 7];
        idata[tid] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
    }
    if (blockDim.x >= 1024 && tid < 512) {
        idata[tid] += idata[tid + 512];
    }
    __syncthreads();
    if (blockDim.x >= 512 && tid < 256) {
        idata[tid] += idata[tid + 256];
    }
    __syncthreads();
    if (blockDim.x >= 256 && tid < 128) {
        idata[tid] += idata[tid + 128];
    }
    __syncthreads();
    if (blockDim.x >= 128 && tid < 64) {
        idata[tid] += idata[tid + 64];
    }
    __syncthreads();
    if (tid < 32) {
        volatile int* vmem = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }

    if (tid == 0) {
        g_odata[blockIdx.x] = idata[0];
    }

}

int main() {
    constexpr int size = 16 * 1024 * 1024;
    int nBytes = size * sizeof(int);
    constexpr int blockSize = 1024;
    constexpr int rawGridSize = (size + blockSize - 1) / blockSize;
    dim3 block(blockSize, 1);
    dim3 grid2(rawGridSize / 2, 1);

    int *h_idata = static_cast<int *>(malloc(nBytes));
    int *h_odata = static_cast<int *>(malloc(grid2.x * sizeof(int)));

    for (int i = 0; i < size; ++i) {
        h_idata[i] = 1;
    }

    int *d_idata, *d_odata;
    cudaMalloc(reinterpret_cast<void**>(&d_idata), nBytes);
    cudaMalloc(reinterpret_cast<void**>(&d_odata), grid2.x * sizeof(int));
    cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice);

    reduceUnroll2<<<grid2, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    cudaMemcpy(h_odata, d_odata, grid2.x * sizeof(int), cudaMemcpyDeviceToHost);
    printf("the first num of h_odata is: %d\n", h_odata[0]);

    dim3 grid4(rawGridSize / 4, 1);
    cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice);
    reduceUnroll4<<<grid4, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    cudaMemcpy(h_odata, d_odata, grid4.x * sizeof(int), cudaMemcpyDeviceToHost);
    printf("the first num of h_odata is: %d\n", h_odata[0]);

    dim3 grid8(rawGridSize / 8, 1);
    cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice);
    reduceUnroll8<<<grid8, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    cudaMemcpy(h_odata, d_odata, grid4.x * sizeof(int), cudaMemcpyDeviceToHost);
    printf("the first num of h_odata is: %d\n", h_odata[0]);

    cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice);
    reduceUnrollWarp8<<<grid8, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    cudaMemcpy(h_odata, d_odata, grid4.x * sizeof(int), cudaMemcpyDeviceToHost);
    printf("the first num of h_odata is: %d\n", h_odata[0]);

    // 终极优化版：模板参数传入编译期已知的 blockSize
    cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice);
    reduceComplete<blockSize><<<grid8, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    cudaMemcpy(h_odata, d_odata, grid8.x * sizeof(int), cudaMemcpyDeviceToHost);
    printf("the first num of h_odata is: %d\n", h_odata[0]);

    cudaFree(d_idata);
    cudaFree(d_odata);
    free(h_idata);
    free(h_odata);

    return 0;
}