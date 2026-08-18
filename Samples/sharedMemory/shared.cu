#include <cstdio>
#include <cuda_utils.cuh>
#define BDIMX 32
#define BDIMY 32
#define BDIMX_RECT 32
#define BDIMY_RECT 16
#define PAD 1
#define PAD_RECT 2
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
    // 三个下标各管一件事,别混用:
    //   g_idx   —— 线程在 block 里的线性编号,也是 out 的下标,范围 0..nElem-1。
    //   row_idx —— 本线程在"带 pad 的 tile"里负责写的位置,行宽是 blockDim.x + 1。
    //   col_idx —— 按列读时要取的位置,同样按 blockDim.x + 1 换行。
    // row_idx / col_idx 最大能到 31*33+31 = 1054,已经超出 out 的 1024 个元素了,
    // 所以 out 只能用 g_idx 索引 —— 拿 row_idx 去写 out 就是越界。
    const unsigned int g_idx   = threadIdx.y * blockDim.x + threadIdx.x;
    // blockDim.x + 1是行宽，无论按行读还是按列度，下一行的同一列的位置永远是要加上行宽的大小
    const unsigned int row_idx = threadIdx.y * (blockDim.x + 1) + threadIdx.x;
    const unsigned int col_idx = threadIdx.x * (blockDim.x + 1) + threadIdx.y;
    tile[row_idx] = g_idx;
    __syncthreads();
    out[g_idx] = tile[col_idx];
}

// 矩形 tile:block 是 32x16,tile 也是 16 行 x 32 列。
// 不能像方形那样直接写 tile[threadIdx.x][threadIdx.y] —— threadIdx.x 能到 31,
// 而 tile 只有 16 行,一交换第一维就越界。办法是"先压成一维,再按目标形状升回二维":
//   idx     —— 线程线性编号,同时也是输出矩阵的线性下标,0..511
//   转置结果是 32 行 x 16 列,行宽 = 16 = blockDim.y,于是按行主序拆 idx:
//   row_idx = idx / blockDim.y  → 0..31   转置矩阵的第几行
//   col_idx = idx % blockDim.y  → 0..15   转置矩阵的第几列
// 最后按 T[r][c] = A[c][r] 回填,tile[col_idx][row_idx] 正好贴合 tile[16][32]。
// 注意除的是 16(转置后的行宽)、而编译器给 tile 寻址时乘的是 32(tile 自己的行宽),
// 两个数不一样不是笔误 —— 一进一出用了两套坐标系,这个错位本身就是转置。
//
// bank conflict:读地址 = col_idx * 32 + row_idx,步长 32 ≡ 0 (mod 32) → col 变了也回到同一 bank。
// warp 0 的 idx 只跨 0..31,row_idx 只能取 0/1,于是 32 个线程劈成两组 16:
//   idx  0..15: bank 0(地址 0,32,...,480)
//   idx 16..31: bank 1(地址 1,33,...,481)
// 每个 bank 上 16 个互不相同的地址 → 16 路冲突,同时 30 个 bank 闲着。
__global__ void setRowReadColRect(int *out) {
    __shared__  int tile[BDIMY_RECT][BDIMX_RECT];
    const unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
    const unsigned int col_idx = idx % blockDim.y; // 转置矩阵的行宽是blockDim.y=16, 0 - 15
    const unsigned int row_idx = idx / blockDim.y; // 0 - 31
    tile[threadIdx.y][threadIdx.x] = idx;
    __syncthreads();
    out[idx] = tile[col_idx][row_idx];
}

// 矩形 + padding。这里 pad 必须是 2,pad 1 是不够的:
//   行宽 33 → bank = (col*33 + row) % 32 = (col + row) % 32
//   row=0 那 16 个线程占 bank 0..15,row=1 那 16 个占 bank 1..16 → 重叠 15 个,仍有 2 路冲突。
// 行宽取 34 就错开了:
//   bank = (col*34 + row) % 32 = (2*col + row) % 32
//   row=0 → 0,2,4,...,30 (16 个偶数 bank);row=1 → 1,3,5,...,31 (16 个奇数 bank)
// 32 个线程正好铺满 32 个 bank,冲突归零。
//
__global__ void setRowReadColRectPad(int *out) {
    __shared__ int tile[BDIMY_RECT][BDIMX_RECT + PAD_RECT];
    const unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
    const unsigned int col_idx = idx % blockDim.y;
    const unsigned int row_idx = idx / blockDim.y;
    tile[threadIdx.y][threadIdx.x] = idx;
    __syncthreads();
    out[idx] = tile[col_idx][row_idx];
}

// 期望结果。tile[r][c] 里存的是自己的线性编号 r * dimx + c,所以按列读出来的是:
//   out[idx] = tile[col_idx][row_idx] = col_idx * dimx + row_idx
// 其中 row_idx = idx / dimy、col_idx = idx % dimy。方形(dimx=dimy=32)代进去就是
// out[y*32+x] = x*32+y,即普通转置;矩形 dimy=16 时同一个式子也成立,不用分开写。
static void expectTranspose(int *ref, const int dimx, const int dimy) {
    for (int idx = 0; idx < dimx * dimy; ++idx) {
        ref[idx] = (idx % dimy) * dimx + (idx / dimy);
    }
}

int main(const int argc, char **argv) {
    // 按最大的方形 tile 分配,矩形变体只用前 BDIMX_RECT * BDIMY_RECT 个元素。
    constexpr int nElem = BDIMY * BDIMX;
    constexpr int nByte = nElem * sizeof(int);
    int kernel = 0;
    int iters = 100;
    if (argc >= 2) {
        kernel = atoi(argv[1]);
    }
    if (argc >= 3) {
        iters = atoi(argv[2]);
    }

    int *out = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&out), nByte));
    int *h_out = nullptr, *h_ref = nullptr;
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&h_out), nByte));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&h_ref), nByte));

    dim3 block(BDIMX, BDIMY);                   // 方形变体的默认 block;矩形变体在 case 里改。
    const dim3 grid(1, 1);
    float ms = 0.0f;
    const char *name = nullptr;
    int nCheck = nElem;                         // 实际参与校验的元素个数,矩形变体只有一半。

    switch (kernel) {
        case 0:
            name = "setRowReadRow";
            // 行写行读,自己写自己读,无冲突 —— 作为基准。
            ms = timeKernel([&] { setRowReadRow<<<grid, block>>>(out); }, iters);
            initDataIota(h_ref, nCheck);
            break;
        case 1:
            name = "setColumnReadColumn";
            // 列写列读:写和读都按列,同样是自己写自己读,所以结果仍是 iota,
            // 但读写两边都撞 bank(32 路),用来和 case 0 对比耗时。
            ms = timeKernel([&] { setColumnReadColumn<<<grid, block>>>(out); }, iters);
            initDataIota(h_ref, nCheck);
            break;
        case 2:
            name = "setRowReadColumn";
            // 行写列读:真正做了一次转置,读侧 32 路冲突。
            ms = timeKernel([&] { setRowReadColumn<<<grid, block>>>(out); }, iters);
            expectTranspose(h_ref, BDIMX, BDIMY);
            break;
        case 3:
            name = "setRowReadColDynamic";
            // 动态 shared:大小在启动时由第三个参数给出,kernel 里只能声明成 unsized 一维。
            ms = timeKernel([&] { setRowReadColDynamic<<<grid, block, nByte>>>(out); }, iters);
            expectTranspose(h_ref, BDIMX, BDIMY);
            break;
        case 4:
            name = "setRowReadColumnPad";
            // 静态 shared + padding:行宽 33,读侧冲突归零。
            ms = timeKernel([&] { setRowReadColumnPad<<<grid, block>>>(out); }, iters);
            expectTranspose(h_ref, BDIMX, BDIMY);
            break;
        case 5:
            name = "setRowReadColDynamicPad";
            // 动态 shared + padding:行宽也是 blockDim.x + 1,所以要多申请 BDIMY 个 int。
            ms = timeKernel([&] {
                setRowReadColDynamicPad<<<grid, block, BDIMY * (BDIMX + PAD) * sizeof(int)>>>(out);
            }, iters);
            expectTranspose(h_ref, BDIMX, BDIMY);
            break;
        case 6:
            name = "setRowReadColRect";
            // 矩形 tile 必须换 block 维度:沿用 (32,32) 会让 threadIdx.y 到 31,
            // 而 tile 只有 BDIMY_RECT = 16 行 —— 直接越界写。
            block = dim3(BDIMX_RECT, BDIMY_RECT);
            nCheck = BDIMX_RECT * BDIMY_RECT;
            ms = timeKernel([&] { setRowReadColRect<<<grid, block>>>(out); }, iters);
            expectTranspose(h_ref, BDIMX_RECT, BDIMY_RECT);
            break;
        case 7:
            name = "setRowReadColRectPad";
            block = dim3(BDIMX_RECT, BDIMY_RECT);
            nCheck = BDIMX_RECT * BDIMY_RECT;
            ms = timeKernel([&] { setRowReadColRectPad<<<grid, block>>>(out); }, iters);
            expectTranspose(h_ref, BDIMX_RECT, BDIMY_RECT);
            break;
        default:
            printf("unknown kernel %d (expect 0..7)\n", kernel);
            CUDA_CHECK(cudaFree(out));
            CUDA_CHECK(cudaFreeHost(h_out));
            CUDA_CHECK(cudaFreeHost(h_ref));
            return EXIT_FAILURE;
    }

    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(h_out, out, nCheck * sizeof(int), cudaMemcpyDeviceToHost));

    printf("kernel %d (%s): block (%u,%u), %.6f ms/iter\n",
           kernel, name, block.x, block.y, ms);
    checkResult(h_ref, h_out, nCheck);

    CUDA_CHECK(cudaFree(out));
    CUDA_CHECK(cudaFreeHost(h_out));
    CUDA_CHECK(cudaFreeHost(h_ref));
    return 0;
}
