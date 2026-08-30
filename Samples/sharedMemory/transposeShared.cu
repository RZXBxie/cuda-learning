#include <cuda_utils.cuh>

#define BDIMX 32
#define BDIMY 16
#define IPAD  2

// 合并读 + 分散写。读 A 的一行(地址连续)，写 B 的一列(步长 ny 个 float)。
__global__ void transposeNaiveRow(const float *matrixA, float *matrixB, const int nx, const int ny) {
    const unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;   // A 的列号
    const unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;   // A 的行号
    if (ix < nx && iy < ny) {
        matrixB[ix * ny + iy] = matrixA[iy * nx + ix];
    }
}

// 分散读 + 合并写。和上面是同一次转置，只是把"分散"从写端挪到了读端。
//
// 这里的坑：线程网格是贴着 B 铺的，所以 ix/iy 按 B 的坐标命名 ——
// 而同一对下标拿去索引 A 时，身份正好互换：
//        在 B 里      在 A 里     取值范围
//   ix    列号        行号        0..ny-1
//   iy    行号        列号        0..nx-1
// 于是 ix * nx + iy 读的是 A 的第 ix 行第 iy 列(乘 A 的宽度 nx)，
// 而 iy * ny + ix 写的是 B 的第 iy 行第 ix 列(乘 B 的宽度 ny)。
// B(iy,ix) = A(ix,iy) 正是转置。口诀没变 —— 乘的数永远是"当前这张表的宽度"，
// 变的只是同一个变量名在两张表里代表行还是代表列。
//
// 顺带看清"分散"落在哪一端：warp 内 threadIdx.x 连续，即 ix 连续，
//   读 A：地址 ix*nx + iy，相邻线程差 nx 个 float → 32 条不同 cache line，分散
//   写 B：地址 iy*ny + ix，相邻线程差 1 个 float → 合并
//
// 网格也要按 B 的形状算：B 是 nx 行 x ny 列，故 grid = (divUp(ny,bx), divUp(nx,by))。
// 本 sample 是方阵 nx == ny，和上一个 kernel 共用一个 grid 无妨；
// 长方形矩阵时必须单独算，否则覆盖不全(已在 512x128 上验证过下标本身是对的)。
__global__ void transposeNaiveCol(const float *matrixA, float *matrixB, const int nx, const int ny) {
    const unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;   // B 的列号 = A 的行号
    const unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;   // B 的行号 = A 的列号
    if (ix < ny && iy < nx) {
        matrixB[iy * ny + ix] = matrixA[ix * nx + iy];
    }
}


// 使用共享内存的矩阵转置，步骤：
// 1.将一个block中的数据拷贝到shared memory
// 2.计算一个block内某个位置相对于该block的线性地址bidx，为block转置下标计算埋伏笔
// 3.计算shared memory转置后的下标irow, icol
// 4.计算转置后的全局行纵坐标oy, ox
// 5.将shared memory内容(来源于tile[icol][irow])写入到转置后的全局线性地址中
__global__ void transposeSmem(const float *matrixA, float *matrixB, const int nx, const int ny) {
    __shared__ float tile[BDIMY][BDIMX];

    const unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;

    const unsigned int bidx = threadIdx.y * blockDim.x + threadIdx.x;
    const unsigned int irow = bidx / blockDim.y;
    const unsigned int icol = bidx % blockDim.y;

    const unsigned int ox = blockIdx.y * blockDim.y + icol;
    const unsigned int oy = blockIdx.x * blockDim.x + irow;

    if (ix < nx && iy < ny) {
        tile[threadIdx.y][threadIdx.x] = matrixA[iy * nx + ix];
    }
    __syncthreads();
    if (ox < ny && oy < nx) {
        matrixB[oy * ny + ox] = tile[icol][irow];
    }
}

// ---------------------------------------------------------------------------
// bank-conflict-free：加 padding
// ---------------------------------------------------------------------------

// 冲突的根源是"tile 行宽 32 和 bank 数 32 相等"，列方向的步长成了 32 的整数倍。
// 把行宽改成和 32 互质/错位的数，同一列的相邻元素就会滑到不同 bank 上。
//
// 这里 pad 必须是 2，pad 1 不够 —— shared.cu 里已经算过一遍，再抄一次结论：
//   行宽 33：bank = (icol*33 + irow) % 32 = (icol + irow) % 32
//            irow=0 那 16 个线程占 bank 0..15，irow=1 那 16 个占 bank 1..16，
//            重叠 15 个 → 还剩 2 路冲突。
//   行宽 34：bank = (icol*34 + irow) % 32 = (2*icol + irow) % 32
//            irow=0 → 0,2,4,...,30(16 个偶数 bank)
//            irow=1 → 1,3,5,...,31(16 个奇数 bank)
//            32 个线程铺满 32 个 bank，冲突归零。
// 一般规律：矩形 tile 一个 warp 会横跨 32/blockDim.y = 2 个 tile 行，
// 需要 pad 让这 2 行错开，所以 pad 取 2。方形 tile(blockDim.y=32)一个 warp 只落在
// 1 行上，pad 1 就够。
//
// 代价：每块多用 BDIMY * IPAD * 4 = 128 字节 smem，换掉 16 倍的列读事务，非常划算。
__global__ void transposeSmemPad(const float *matrixA, float *matrixB, const int nx, const int ny) {
    __shared__ float tile[BDIMY][BDIMX + IPAD];

    const unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;

    const unsigned int bidx = threadIdx.y * blockDim.x + threadIdx.x;
    const unsigned int irow = bidx / blockDim.y;
    const unsigned int icol = bidx % blockDim.y;

    const unsigned int ox = blockIdx.y * blockDim.y + icol;
    const unsigned int oy = blockIdx.x * blockDim.x + irow;

    if (ix < nx && iy < ny) {
        tile[threadIdx.y][threadIdx.x] = matrixA[iy * nx + ix];
    }
    __syncthreads();
    if (ox < ny && oy < nx) {
        matrixB[oy * ny + ox] = tile[icol][irow];
    }
}

// 动态共享内存版。动态 smem 只能声明成 unsized 一维数组，
// 二维下标得自己乘行宽算出来 —— 而"行宽"正是带 pad 的 BDIMX + IPAD。
// 注意这里两个下标乘的都是同一个行宽(不是一个乘 32 一个乘 34)，
// 行宽是 tile 自己的属性，跟你是按行访问还是按列访问无关。
__global__ void transposeSmemDynPad(const float *matrixA, float *matrixB, const int nx, const int ny) {
    extern __shared__ float tile[];
    const unsigned int pitch = blockDim.x + IPAD;         // tile 一行有多少个 float

    const unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;

    const unsigned int bidx = threadIdx.y * blockDim.x + threadIdx.x;
    const unsigned int irow = bidx / blockDim.y;
    const unsigned int icol = bidx % blockDim.y;

    const unsigned int ox = blockIdx.y * blockDim.y + icol;
    const unsigned int oy = blockIdx.x * blockDim.x + irow;

    if (ix < nx && iy < ny) {
        tile[threadIdx.y * pitch + threadIdx.x] = matrixA[iy * nx + ix];
    }
    __syncthreads();
    if (ox < ny && oy < nx) {
        matrixB[oy * ny + ox] = tile[icol * pitch + irow];
    }
}

// ---------------------------------------------------------------------------
// 展开：一个 block 处理两个 tile
// ---------------------------------------------------------------------------

// 展开在这里解决的不是"算得不够快"，而是访存并行度不够：
// 转置几乎没有计算，性能完全由"同时有多少个访存请求在飞"决定(latency hiding)。
// 一个线程负责两个元素，两条独立的 load 可以背靠背发出、一起等，
// 同时 grid 少一半，索引计算和 block 启动的开销也摊薄了。
//
// 布局：block 仍是 (32,16)，但沿 x 方向吃两个 tile，
// 于是 smem 里放的是 A 的一块 16 行 x 64 列，行宽 = 2*BDIMX + IPAD = 66。
//
//   smem tile (16 行 x 64 列 + 2 列 pad)
//        列 0..31            列 32..63         pad
//   行0 [ 左 tile        ][ 右 tile        ][ x x ]
//   ...
//   行15[                ][                ][ x x ]
//
// 左半块写回 B 的行 2*bx*32 + irow，右半块的 A 列号大 32，
// 转置后就是 B 的行号大 32 → 输出下标相差 32 * ny，即 ny * BDIMX。
//
// pad 仍然取 2 就够：bank = (icol*66 + irow) % 32 = (2*icol + irow) % 32，
// 和行宽 34 的情形完全一样(66 ≡ 34 ≡ 2 mod 32)，偶/奇 bank 各占一半，无冲突。
__global__ void transposeSmemUnrollPad(const float *matrixA, float *matrixB, const int nx, const int ny) {
    __shared__ float tile[BDIMY][BDIMX * 2 + IPAD];

    // 本线程在 A 里负责的第一个元素；第二个在它右边 blockDim.x 列。
    const unsigned int ix = 2 * blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;

    const unsigned int bidx = threadIdx.y * blockDim.x + threadIdx.x;
    const unsigned int irow = bidx / blockDim.y;
    const unsigned int icol = bidx % blockDim.y;

    const unsigned int ox = blockIdx.y * blockDim.y + icol;
    const unsigned int oy = 2 * blockIdx.x * blockDim.x + irow;

    if (iy < ny) {
        if (ix < nx) {
            tile[threadIdx.y][threadIdx.x] = matrixA[iy * nx + ix];
        }
        if (ix + blockDim.x < nx) {
            tile[threadIdx.y][threadIdx.x + blockDim.x] = matrixA[iy * nx + ix + blockDim.x];
        }
    }
    __syncthreads();
    if (ox < ny) {
        if (oy < nx) {
            matrixB[oy * ny + ox] = tile[icol][irow];
        }
        if (oy + blockDim.x < nx) {
            matrixB[(oy + blockDim.x) * ny + ox] = tile[icol][irow + blockDim.x];
        }
    }
}

static void transposeHost(const float *in, float *out, const int nx, const int ny) {
    for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix) {
            out[ix * ny + iy] = in[iy * nx + ix];
        }
    }
}

int main(const int argc, char **argv) {
    constexpr int nx = 1 << 10;
    constexpr int ny = 1 << 10;
    constexpr int nxy = nx * ny;
    constexpr int bytes = nxy * sizeof(float);

    int kernel = 0;
    int iters = 100;
    if (argc >= 2) {
        kernel = atoi(argv[1]);
    }
    if (argc >= 3) {
        iters = atoi(argv[2]);
    }

    float *h_A = nullptr, *h_B = nullptr, *h_ref = nullptr;
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&h_A), bytes));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&h_B), bytes));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&h_ref), bytes));
    initDataIota(h_A, nxy);                       // 1<<20 以内的整数 float 能精确表示，比较可以用 eps=0
    transposeHost(h_A, h_ref, nx, ny);

    float *d_A = nullptr, *d_B = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));

    const dim3 block(BDIMX, BDIMY);
    const dim3 grid(divUp(nx, block.x), divUp(ny, block.y));
    const dim3 grid2(divUp(nx, block.x * 2), divUp(ny, block.y));   // 展开版：x 方向少一半 block

    // 先跑一次验证，再计时 iters 次。所有 kernel 都不改写输入，
    // 所以不像 reduction 那样需要每轮重新 H2D。
    auto runOne = [&](const char *name, auto launch, const dim3 g, const size_t smem) {
        CUDA_CHECK(cudaMemset(d_B, 0, bytes));    // 清零，漏写的位置会在校验时暴露
        launch();
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaMemcpy(h_B, d_B, bytes, cudaMemcpyDeviceToHost));

        const float ms = timeKernel(launch, iters);

        // 有效带宽：读一遍 + 写一遍 => 2 * bytes。
        // smem 里的往返不计入分子 —— 它是手段不是目的，
        // 所以 bank conflict 变多只会表现为 GB/s 变低，不需要单独记账。
        printf("kernel %d (%-22s) grid (%u,%u) block (%u,%u) smem %4zuB  %.4f ms/iter  %.1f GB/s\n",
               kernel, name, g.x, g.y, block.x, block.y, smem, ms, effectiveGBps(2.0 * bytes, ms));
        checkResult(h_ref, h_B, nxy);
    };

    switch (kernel) {
        case 0:
            runOne("transposeNaiveRow",
                   [&] { transposeNaiveRow<<<grid, block>>>(d_A, d_B, nx, ny); }, grid, 0);
            break;
        case 1:
            runOne("transposeNaiveCol",
                   [&] { transposeNaiveCol<<<grid, block>>>(d_A, d_B, nx, ny); }, grid, 0);
            break;
        case 2:
            runOne("transposeSmem",
                   [&] { transposeSmem<<<grid, block>>>(d_A, d_B, nx, ny); },
                   grid, sizeof(float) * BDIMY * BDIMX);
            break;
        case 3:
            runOne("transposeSmemPad",
                   [&] { transposeSmemPad<<<grid, block>>>(d_A, d_B, nx, ny); },
                   grid, sizeof(float) * BDIMY * (BDIMX + IPAD));
            break;
        case 4: {
            // 动态 smem：第三个执行配置参数就是每块要分配的字节数，
            // 别忘了把 pad 那一列也算进去，否则最后一行会越界。
            constexpr size_t smem = sizeof(float) * BDIMY * (BDIMX + IPAD);
            runOne("transposeSmemDynPad",
                   [&] { transposeSmemDynPad<<<grid, block, smem>>>(d_A, d_B, nx, ny); },
                   grid, smem);
            break;
        }
        case 5:
            runOne("transposeSmemUnrollPad",
                   [&] { transposeSmemUnrollPad<<<grid2, block>>>(d_A, d_B, nx, ny); },
                   grid2, sizeof(float) * BDIMY * (BDIMX * 2 + IPAD));
            break;
        default:
            printf("unknown kernel %d (expect 0..5)\n", kernel);
            break;
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFreeHost(h_A));
    CUDA_CHECK(cudaFreeHost(h_B));
    CUDA_CHECK(cudaFreeHost(h_ref));
    return 0;
}
