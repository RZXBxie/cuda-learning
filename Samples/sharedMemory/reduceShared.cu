// 归约的落脚点之争:同一棵归约树,数据放在全局内存 vs 共享内存。
//
// 前面 reduction / reductionUnroll 两个 sample 里,树形归约的每一级都是
//     idata[tid] += idata[tid + stride];
// 而 idata 指向全局内存 —— 一次全局读 + 一次全局写,而且刚写下去的值
// 下一级马上又要读回来。这是典型的"短命数据放错了地方":
// 它的生命周期只有一级归约那么长,却每次都要跑一趟片外显存。
//
// 共享内存正是为这种数据准备的:片上、延迟约为全局内存的 1/20~1/30、
// 块内可见。于是正确的做法是"进块时把自己那份搬进 smem,整棵树都在 smem 上塌缩,
// 只有最后的 smem[0] 写回全局"。全局内存于是变成读一遍、写 grid.x 个数,
// 中间所有的往返都消失了。
//
// 附带的好处:输入数组不再被原地改写。reductionUnroll.cu 里每换一个 kernel
// 都得重新 H2D 拷一遍输入,这里的 smem 版本天生可重复运行。
#include <cuda_utils.cuh>

// 静态共享内存的大小必须在编译期已知,所以 block 尺寸只能是常量,
// 不能像别的 sample 那样从 argv 里读进来。
// 顺带让下面 blockReduce 里的 if (DIM >= N) 也在编译期就被裁掉。
constexpr int DIM = 1024;

// 块内树形归约,完全展开版。mem 指向哪里都行 —— 全局内存、静态 smem、动态 smem,
// 这正是本 sample 要对比的唯一变量。要求 mem 至少有 DIM 个元素可用。
//

__device__ inline void blockReduce(int *mem, const unsigned int tid) {
    if (DIM >= 1024 && tid < 512) {
        mem[tid] += mem[tid + 512];
    }
    __syncthreads();
    if (DIM >= 512 && tid < 256) {
        mem[tid] += mem[tid + 256];
    }
    __syncthreads();
    if (DIM >= 256 && tid < 128) {
        mem[tid] += mem[tid + 128];
    }
    __syncthreads();
    if (DIM >= 128 && tid < 64) {
        mem[tid] += mem[tid + 64];
    }
    __syncthreads();
    if (tid < 32) {
        volatile int *vmem = mem;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }
}

__global__ void reduceGmem(int *g_idata, int *g_odata, const int size) {
    const unsigned int tid = threadIdx.x;
    int *idata = g_idata + blockIdx.x * blockDim.x;   // 本块负责的那 blockDim.x 个元素
    // size 能被 blockDim.x 整除,所以这个 return 要么整块都走、要么整块都不走,
    // 不会出现"一部分线程提前退出、剩下的卡在 __syncthreads() 上"的死锁。
    if (blockIdx.x * blockDim.x + tid >= size) return;

    blockReduce(idata, tid);

    if (tid == 0) {
        g_odata[blockIdx.x] = idata[0];
    }
}

// 静态共享内存版。和 reduceGmem 的差别只有两行:多一个 smem 声明、
// 多一次 "把自己那格从全局搬进 smem"。之后归约树读写的全是片上内存。
__global__ void reduceSmem(int *g_idata, int *g_odata, const int size) {
    __shared__ int smem[DIM];
    const unsigned int tid = threadIdx.x;
    const unsigned int idx = blockIdx.x * blockDim.x + tid;

    // 越界的位置填 0 —— 加法的单位元,参与归约也不影响结果。
    smem[tid] = (idx < size) ? g_idata[idx] : 0;
    __syncthreads();                     // 下一级要读别人写的格子,跨 warp 必须同步。

    blockReduce(smem, tid);

    if (tid == 0) {
        g_odata[blockIdx.x] = smem[0];
    }
}

__global__ void reduceSmemDyn(int *g_idata, int *g_odata, const int size) {
    extern __shared__ int smem[];
    const unsigned int tid = threadIdx.x;
    const unsigned int idx = blockIdx.x * blockDim.x + tid;

    smem[tid] = (idx < size) ? g_idata[idx] : 0;
    __syncthreads();

    blockReduce(smem, tid);

    if (tid == 0) {
        g_odata[blockIdx.x] = smem[0];
    }
}

__global__ void reduceSmemUnroll4(int *g_idata, int *g_odata, const int size) {
    __shared__ int smem[DIM];
    const unsigned int tid = threadIdx.x;
    const unsigned int idx = blockIdx.x * blockDim.x * 4 + tid;   // 本线程在段 0 的全局下标

    int sum = 0;
    if (idx + blockDim.x * 3 < size) {
        sum = g_idata[idx]
            + g_idata[idx + blockDim.x]
            + g_idata[idx + blockDim.x * 2]
            + g_idata[idx + blockDim.x * 3];
    }
    smem[tid] = sum;
    __syncthreads();

    blockReduce(smem, tid);

    if (tid == 0) {
        g_odata[blockIdx.x] = smem[0];
    }
}

int main(const int argc, char **argv) {
    constexpr int size = 16 * 1024 * 1024;      // 能被 DIM 和 4 * DIM 整除,省掉尾块的特殊情况
    constexpr int nBytes = size * sizeof(int);

    int kernel = 0;
    int iters = 100;
    if (argc >= 2) {
        kernel = atoi(argv[1]);
    }
    if (argc >= 3) {
        iters = atoi(argv[2]);
    }

    constexpr dim3 block(DIM, 1);
    const dim3 grid(divUp(size, DIM), 1);            // 一线程一元素
    const dim3 grid4(divUp(size, DIM * 4), 1);       // 展开版:一线程四元素

    int *h_idata = nullptr, *h_odata = nullptr;
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&h_idata), nBytes));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&h_odata), grid.x * sizeof(int)));
    initData(h_idata, size);                    // 全填 1,归约结果应等于 size。

    int *d_idata = nullptr, *d_odata = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_idata), nBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_odata), grid.x * sizeof(int)));

    // 校验一次 + 计时 iters 次。每块只归约出一个部分和,剩下 grid.x 个数在 host 上累加。
    auto runOne = [&](const char *name, auto launch, const dim3 g) {
        // 校验:先把输入复位 —— reduceGmem 会原地改写 d_idata。
        CUDA_CHECK(cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice));
        launch();
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaMemcpy(h_odata, d_odata, g.x * sizeof(int), cudaMemcpyDeviceToHost));
        long long sum = 0;
        for (unsigned int i = 0; i < g.x; ++i) {
            sum += h_odata[i];
        }

        // 计时:循环里不重新拷输入,否则 H2D 的时间会混进 kernel 耗时。
        // 代价是 reduceGmem 从第二次迭代起读到的是上一轮的残骸,数值没意义;
        // 但访存量和指令流一模一样,所以耗时依然可比。smem 三个版本不改输入,无此问题。
        const float ms = timeKernel(launch, iters);

        // 有效带宽只算"真正有用的字节":读一遍输入。写回的 grid.x 个部分和可忽略。
        // reduceGmem 在归约树里额外来回搬的那些数据不计入分子,于是直接表现为更低的 GB/s。
        printf("%-18s grid %7u block %4u  sum %10lld (expected %d)  %.4f ms  %.2f GB/s\n",
               name, g.x, block.x, sum, size, ms,
               effectiveGBps(nBytes, ms));
    };

    switch (kernel) {
        case 0:
            runOne("reduceGmem", [&] { reduceGmem<<<grid, block>>>(d_idata, d_odata, size); }, grid);
            break;
        case 1:
            runOne("reduceSmem", [&] { reduceSmem<<<grid, block>>>(d_idata, d_odata, size); }, grid);
            break;
        case 2:
            // 动态 smem:第三个执行配置参数就是每块要分配的字节数。
            runOne("reduceSmemDyn",
                   [&] { reduceSmemDyn<<<grid, block, DIM * sizeof(int)>>>(d_idata, d_odata, size); },
                   grid);
            break;
        case 3:
            runOne("reduceSmemUnroll4",
                   [&] { reduceSmemUnroll4<<<grid4, block>>>(d_idata, d_odata, size); },
                   grid4);
            break;
        default:
            printf("unknown kernel %d (expect 0..3)\n", kernel);
            CUDA_CHECK(cudaFree(d_idata));
            CUDA_CHECK(cudaFree(d_odata));
            CUDA_CHECK(cudaFreeHost(h_idata));
            CUDA_CHECK(cudaFreeHost(h_odata));
            return EXIT_FAILURE;
    }

    CUDA_CHECK(cudaFree(d_idata));
    CUDA_CHECK(cudaFree(d_odata));
    CUDA_CHECK(cudaFreeHost(h_idata));
    CUDA_CHECK(cudaFreeHost(h_odata));
    return 0;
}
