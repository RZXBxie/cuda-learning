#include <cuda_utils.cuh>

// ---------------------------------------------------------------------------
// 数据布局
//
// 每个粒子有 x/y/z 三个字段。本例只对 x 做一次读改写（把所有粒子的 x 各 +1），
// 这是"按字段批量处理"的典型 GPU 访问模式，用来放大 AoS / SoA 的差异。
// ---------------------------------------------------------------------------

// AoS：Array of Structures —— 同一粒子的 x/y/z 在内存里挨在一起。
struct Particle {
    float x, y, z;
};

// SoA：Structure of Arrays —— 所有粒子的 x 挨在一起，y、z 各自成段。
struct ParticlesSoA {
    float* x;
    float* y;
    float* z;
};

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

// AoS：warp 内相邻线程访问的 x 之间隔着 y、z（步长 sizeof(Particle)=12B），
// 访存无法合并，一个 128B 事务里 2/3 是用不上的 y、z，有效带宽 ~1/3。
__global__ void updateAoS(Particle* p, const int n) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        p[i].x += 1.0f;
    }
}

// SoA：warp 内相邻线程访问 x[i] 天然连续，32 个线程正好落在连续的 128B 内，
// 硬件用一次事务全部取回，带宽利用率接近 100%。
__global__ void updateSoA(float* x, const int n) {
    if (const int i = blockDim.x * blockIdx.x + threadIdx.x; i < n) {
        x[i] += 1.0f;
    }
}

// ---------------------------------------------------------------------------
// 计时辅助由 cuda_utils.cuh 的 timeKernel 提供。
// ---------------------------------------------------------------------------

int main() {
    constexpr int n     = 1 << 24;         // ~1600 万个粒子。
    constexpr int iters = 100;             // 多跑几次取平均，读数更稳。

    dim3 block(256);
    dim3 grid(divUp(n, block.x));

    // --- AoS：一块连续显存，元素是 Particle ---
    Particle* aos_d = nullptr;
    CUDA_CHECK(cudaMalloc(&aos_d, n * sizeof(Particle)));
    CUDA_CHECK(cudaMemset(aos_d, 0, n * sizeof(Particle)));

    // --- SoA：三块独立的连续数组 ---
    ParticlesSoA soa_d{};
    CUDA_CHECK(cudaMalloc(&soa_d.x, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&soa_d.y, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&soa_d.z, n * sizeof(float)));
    CUDA_CHECK(cudaMemset(soa_d.x, 0, n * sizeof(float)));

    const float aosMs = timeKernel([&] { updateAoS<<<grid, block>>>(aos_d, n); }, iters);
    CUDA_CHECK(cudaGetLastError());
    const float soaMs = timeKernel([&] { updateSoA<<<grid, block>>>(soa_d.x, n); }, iters);
    CUDA_CHECK(cudaGetLastError());

    // 有效带宽：只统计我们真正想要的数据(x)，读+写各一遍 => 2 * n * 4B。
    // AoS 实际还搬运了没用的 y/z，所以同样的"有效字节数"下耗时更长、有效带宽更低。
    auto gbps = [](float ms) { return effectiveGBps(2.0 * n * sizeof(float), ms); };

    printf("n = %d particles, iters = %d\n\n", n, iters);
    printf("AoS: %7.4f ms/iter,  effective BW = %6.1f GB/s\n", aosMs, gbps(aosMs));
    printf("SoA: %7.4f ms/iter,  effective BW = %6.1f GB/s\n", soaMs, gbps(soaMs));
    printf("\nSoA speedup over AoS: %.2fx\n", aosMs / soaMs);

    CUDA_CHECK(cudaFree(aos_d));
    CUDA_CHECK(cudaFree(soa_d.x));
    CUDA_CHECK(cudaFree(soa_d.y));
    CUDA_CHECK(cudaFree(soa_d.z));

    return 0;
}
