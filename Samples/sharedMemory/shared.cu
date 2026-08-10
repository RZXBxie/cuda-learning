#include <cstdio>
#include <cuda_utils.cuh>
#define BDIMX 32
#define BDIMY 32
#define PAD 1
__global__ void setRowReadRow(int *out) {
    __shared__ int tile[BDIMY][BDIMX];
    const unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
    tile[threadIdx.y][threadIdx.x] = idx;
    out[idx] = tile[threadIdx.y][threadIdx.x];
}

__global__ void setColumnReadColumn(int *out) {
    __shared__ int tile[BDIMY][BDIMX];
    const unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
    tile[threadIdx.x][threadIdx.y] = idx;
    out[idx] = tile[threadIdx.x][threadIdx.y];
}

__global__ void setRowReadColumn(int *out) {
    __shared__ int tile[BDIMY][BDIMX];
    const unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
    tile[threadIdx.y][threadIdx.x] = idx;
    __syncthreads();                    // 读的是别的线程写进去的格子,跨 warp 必须同步。
    out[idx] = tile[threadIdx.x][threadIdx.y];
}

__global__ void setRowReadColDynamic(int *out) {
    extern __shared__ int tile[];
    const unsigned int row_idx = threadIdx.y * blockDim.x + threadIdx.x;
    const unsigned int col_idx = threadIdx.x * blockDim.y + threadIdx.y;
    tile[row_idx]  = row_idx;
    __syncthreads();                    // 同上:col_idx 那格由别的线程负责写。
    out[row_idx] = tile[col_idx];

}

__global__ void setRowReadColumnPad(int *out) {
    __shared__ int tile[BDIMY][BDIMX + PAD];
    const unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
    tile[threadIdx.y][threadIdx.x] = idx;
    __syncthreads();
    out[idx] = tile[threadIdx.x][threadIdx.y];
}

__global__ void setRowReadColDynamicPad(int * out)
{
    extern __shared__ int tile[];
    // blockDim.x + 1是行宽，无论按行读还是按列度，下一行的同一列的位置永远是要加上行宽的大小
    const unsigned int row_idx = threadIdx.y * (blockDim.x + 1) + threadIdx.x;
    const unsigned int col_idx = threadIdx.x * (blockDim.x + 1) + threadIdx.y;
    tile[row_idx] = row_idx;
    __syncthreads();
    out[row_idx] = tile[col_idx];
}

int main(const int argc, char **argv) {
    constexpr int nElem = BDIMY * BDIMX;
    constexpr int nByte = nElem * sizeof(int);
    int kernel = 0;
    if (argc >= 2) {
        kernel = atoi(argv[1]);
    }
    int *out;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&out), nByte));

    dim3 block(BDIMX, BDIMY);
    dim3 grid(1,1);
    float ms = 0.0f;

    switch (kernel) {
        case 0:
            ms = timeKernel([&] {setRowReadRow<<<grid, block>>>(out);}, 100);
            break;
        case 1:
            setColumnReadColumn<<<grid, block>>>(out);
            break;
        case 2:
            setRowReadColumn<<<grid, block>>>(out);
            break;
        case 3:
            // 动态 shared:大小在启动时由第三个参数给出,kernel 里只能声明成 unsized 一维。
            setRowReadColDynamic<<<grid, block, nByte>>>(out);
            break;
        default:
            printf("unknown kernel %d (expect 0..3)\n", kernel);
            CUDA_CHECK(cudaFree(out));
            return EXIT_FAILURE;
    }

    printf("ms: %f\n", ms);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaFree(out));
    return 0;
}
