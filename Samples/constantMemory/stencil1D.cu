 #include <cuda_utils.cuh>

#define RADIUS 4              // 单侧半径:左右各看 4 个点
#define BDIM   128            // 默认 block 大小

// 常量内存里的系数。__constant__ 变量的名字本身就是 symbol,
// cudaMemcpyToSymbol 直接拿它当第一个参数(不是取地址,也不是设备指针)。
__constant__ float c_coef[RADIUS];

// ---------------------------------------------------------------------------
// 变体 0:系数放全局内存(对照组)
// ---------------------------------------------------------------------------
//
// 除了系数来源不同,和变体 1 一模一样。每个线程都要把 4 个系数从全局内存读进来,
// 走的是 L1/L2 那条普通数据通路:数据量小、命中率高,所以不会慢到离谱,
// 但它挤占的是 L1 里本该留给 in[] 的容量,并且每次都要发访存指令。
__global__ void stencil1DGlobal(const float *in, float *out, const float *coef, const int n) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= n) return;

    // in 已经被 host 端偏移过了(见 main),所以 in[idx-RADIUS] 落在 halo 里,合法。
    float sum = 0.0f;
#pragma unroll
    for (int r = 1; r <= RADIUS; ++r) {
        sum += coef[r - 1] * (in[idx + r] - in[idx - r]);
    }
    out[idx] = sum;
}

// ---------------------------------------------------------------------------
// 变体 1:系数放常量内存
// ---------------------------------------------------------------------------
//
// 循环被 #pragma unroll 完全展开后,c_coef[0..3] 都变成编译期已知的常量地址,
// 于是每一条乘加取的都是"整个 warp 同一个地址" —— 广播命中,不占访存带宽。
// 输入 in[] 仍然走普通全局内存:相邻线程读相邻地址,合并访问;
// 而 idx±r 这 8 次读里有 7/8 的数据是邻居线程也要读的,靠 L1 复用。
__global__ void stencil1DConst(const float *in, float *out, const int n) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= n) return;

    float sum = 0.0f;
#pragma unroll
    for (int r = 1; r <= RADIUS; ++r) {
        sum += c_coef[r - 1] * (in[idx + r] - in[idx - r]);
    }
    out[idx] = sum;
}

// ---------------------------------------------------------------------------
// 变体 2:常量内存系数 + 共享内存缓存输入
// ---------------------------------------------------------------------------
//
// 上面那版每个点被读了 8 遍(自己不读,左右各 4 个邻居来读它),重复量全靠 L1 兜。
// 把 block 负责的那一段搬进共享内存,重复读就变成读 smem,确定性地只读一次全局。
//
// smem 布局:中间是本 block 的 blockDim.x 个点,两端各挂 RADIUS 个 halo(晕区),
// 因为块边缘的线程要看到隔壁块的点:
//
//   smem[]  |<-R->|<--------- blockDim.x --------->|<-R->|
//           [ 左halo ][ t0 t1 t2 ...        t127 ][ 右halo ]
//              ↑                                      ↑
//        前 R 个线程搬                          前 R 个线程搬
//
// sidx = threadIdx.x + RADIUS 是本线程数据在 smem 里的位置;
// 左 halo 取 in[idx-RADIUS],右 halo 取 in[idx+blockDim.x] —— 注意右侧要用
// blockDim.x 而不是 RADIUS 去偏移,它对应的是"本块最后一个点再往右数 threadIdx.x+1 个"。
//
// halo 让 smem 的行宽变成 blockDim.x + 2*RADIUS = 136 个 float。
// 这里不用担心 bank conflict:一维模板每个线程读的是 sidx±r,warp 内地址连续,
// 32 个线程正好铺满 32 个 bank,无论 r 取几都只是整体平移。
//
// 越界处理这里和前几个变体不一样,不能一上来就 `if (idx >= n) return;` ——
// 提前 return 的线程不会参与后面的 __syncthreads(),同一个 block 里有人到、有人
// 永远不到,行为未定义(实际表现常常是挂死)。所以把 return 挪到 __syncthreads() 之后,
// 保证所有线程都先走过那道栅栏。
__global__ void stencil1DConstSmem(const float *in, float *out, const int n) {
    __shared__ float smem[BDIM + 2 * RADIUS];

    const int idx  = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int sidx = static_cast<int>(threadIdx.x) + RADIUS;

    if (idx < n) {
        smem[sidx] = in[idx];
        if (threadIdx.x < RADIUS) {
            smem[sidx - RADIUS]     = in[idx - RADIUS];        // 左 halo
            smem[sidx + blockDim.x] = in[idx + blockDim.x];    // 右 halo
        }
    }
    __syncthreads();                 // halo 是别的线程写的,必须等齐了再读。
    if (idx >= n) return;

    float sum = 0.0f;
#pragma unroll
    for (int r = 1; r <= RADIUS; ++r) {
        sum += c_coef[r - 1] * (smem[sidx + r] - smem[sidx - r]);
    }
    out[idx] = sum;
}

// ---------------------------------------------------------------------------
// 变体 3:常量内存系数 + 只读缓存读输入
// ---------------------------------------------------------------------------
//
// 只读数据缓存(read-only / texture cache,Kepler 起每个 SM 48KB)是常量内存的
// "兄弟通路",区别正好补上常量内存的短板:
//                 常量内存                只读缓存
//   容量          64KB(全设备)           48KB(每 SM)
//   最佳访问      warp 内同一地址,广播    warp 内不同地址,合并
//   怎么用        __constant__ + MemcpyToSymbol   const + __restrict__(或 __ldg)
// 所以这里两个一起用:系数走常量内存(同地址),输入走只读缓存(连续地址)。
//
// 触发方式是给指针同时加 const 和 __restrict__ —— const 说明不写,__restrict__
// 说明没有别名(不会和 out 指向同一块),编译器据此才敢生成 LDG 指令。
// 只要有一个指针漏了 __restrict__,编译器就无法排除 out 的写会改到 in,只能退回普通 load。
__global__ void stencil1DConstReadOnly(const float *__restrict__ in,
                                       float *__restrict__ out,
                                       const int n) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= n) return;

    float sum = 0.0f;
#pragma unroll
    for (int r = 1; r <= RADIUS; ++r) {
        sum += c_coef[r - 1] * (in[idx + r] - in[idx - r]);
    }
    out[idx] = sum;
}

// host 参考实现。in 同样是已经偏移过 RADIUS 的指针。
static void stencil1DHost(const float *in, float *out, const float *coef, const int n) {
    for (int i = 0; i < n; ++i) {
        float sum = 0.0f;
        for (int r = 1; r <= RADIUS; ++r) {
            sum += coef[r - 1] * (in[i + r] - in[i - r]);
        }
        out[i] = sum;
    }
}

int main(const int argc, char **argv) {
    constexpr int n = 1 << 22;                          // 有效点数
    constexpr int nPadded = n + 2 * RADIUS;             // 两端各留 RADIUS 个 halo
    constexpr size_t bytes = nPadded * sizeof(float);
    constexpr size_t outBytes = n * sizeof(float);

    int kernel = 0;
    int iters = 100;
    if (argc >= 2) {
        kernel = atoi(argv[1]);
    }
    if (argc >= 3) {
        iters = atoi(argv[2]);
    }

    // 八阶精度中心差分的标准系数(h = 1)。由泰勒展开配平得来:
    //   in[i+r] - in[i-r] = 2*( r*f' + r^3/3!*f''' + r^5/5!*f5 + r^7/7!*f7 + ... )
    // 四个未知数配四个方程 —— Σ2*r*c_r = 1 让 f' 的系数归一,
    // Σr^3*c_r = Σr^5*c_r = Σr^7*c_r = 0 把 f'''/f5/f7 三个误差项杀掉,
    // 解唯一。第一个杀不掉的是 f9,故误差 O(h^8)。
    // 规律:RADIUS = R 的中心差分给 2R 阶精度(R=1 是二阶的 (f[i+1]-f[i-1])/2)。
    const float h_coef[RADIUS] = {4.0f / 5.0f, -1.0f / 5.0f, 4.0f / 105.0f, -1.0f / 280.0f};

    float *h_in = nullptr, *h_out = nullptr, *h_ref = nullptr;
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&h_in), bytes));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&h_out), outBytes));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&h_ref), outBytes));

    // 不能用 initData 的常量填充:常数的导数恒为 0,下标算错了也看不出来。
    // 填一段光滑的正弦,每个点的值都不同,任何错位都会被 checkResult 抓到。
    for (int i = 0; i < nPadded; ++i) {
        h_in[i] = sinf(static_cast<float>(i - RADIUS) * 0.01f);
    }

    // 主机端和设备端都用"偏移后的指针"看数组,下标 0 就是第一个有效点,
    // 于是 kernel 里写 in[idx - RADIUS] 不用再额外加偏移,索引算术干净很多。
    stencil1DHost(h_in + RADIUS, h_ref, h_coef, n);

    float *d_in = nullptr, *d_out = nullptr, *d_coef = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_in), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_out), outBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_coef), RADIUS * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_coef, h_coef, RADIUS * sizeof(float), cudaMemcpyHostToDevice));

    // 往常量内存里写。第一个参数直接写变量名 c_coef:它是个 symbol,
    // 不是 host 指针也不是 device 指针,取地址反而编译不过。
    CUDA_CHECK(cudaMemcpyToSymbol(c_coef, h_coef, RADIUS * sizeof(float)));

    const dim3 block(BDIM);
    const dim3 grid(divUp(n, block.x));

    float *d_inOffset = d_in + RADIUS;      // 同 host:让 kernel 看到的 [0, n) 是有效区

    auto runOne = [&](const char *name, auto launch) {
        CUDA_CHECK(cudaMemset(d_out, 0, outBytes));
        launch();
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaMemcpy(h_out, d_out, outBytes, cudaMemcpyDeviceToHost));

        const float ms = timeKernel(launch, iters);

        // 有效带宽只算"真正要搬的字节":读一遍 + 写一遍 = 2 * n * sizeof(float)。
        // halo 那 8 倍的重复读不计入分子 —— 它是实现手段,缓存/共享内存复用得好不好,
        // 直接体现为 GB/s 的高低,不需要单独记账。
        printf("kernel %d (%-22s) grid %u block %u  %.4f ms/iter  %.1f GB/s\n",
               kernel, name, grid.x, block.x, ms,
               effectiveGBps(2.0 * outBytes, ms));
        checkResult(h_ref, h_out, n, 1e-6);
    };

    switch (kernel) {
        case 0:
            runOne("stencil1DGlobal",
                   [&] { stencil1DGlobal<<<grid, block>>>(d_inOffset, d_out, d_coef, n); });
            break;
        case 1:
            runOne("stencil1DConst",
                   [&] { stencil1DConst<<<grid, block>>>(d_inOffset, d_out, n); });
            break;
        case 2:
            // smem 是静态声明的 BDIM + 2*RADIUS,所以 block 大小不能从命令行改,
            // 只能改这里的宏 —— 静态共享内存的尺寸必须是编译期常量。
            runOne("stencil1DConstSmem",
                   [&] { stencil1DConstSmem<<<grid, block>>>(d_inOffset, d_out, n); });
            break;
        case 3:
            runOne("stencil1DConstReadOnly",
                   [&] { stencil1DConstReadOnly<<<grid, block>>>(d_inOffset, d_out, n); });
            break;
        default:
            printf("unknown kernel %d (expect 0..3)\n", kernel);
            break;
    }

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_coef));      // 注意:c_coef 是常量内存,没有对应的 free
    CUDA_CHECK(cudaFreeHost(h_in));
    CUDA_CHECK(cudaFreeHost(h_out));
    CUDA_CHECK(cudaFreeHost(h_ref));
    return 0;
}
