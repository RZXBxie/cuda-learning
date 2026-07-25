// cuda_utils.cuh —— 学习 CUDA 时反复用到的辅助工具集中放这里。
//
// 头文件形式(全是宏 / 模板 / inline),各 sample 直接 #include <cuda_utils.cuh> 即可,
// 不用在每个 .cu 里重抄一遍错误检查、初始化、验证、计时。
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// 错误检查
// ---------------------------------------------------------------------------

// 包裹任意返回 cudaError_t 的调用,出错就打印文件/行号并退出。
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(err_));            \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// kernel 启动是异步的、本身不返回错误码。启动后紧跟这一句:
// cudaGetLastError() 捕获"启动参数非法"之类的同步错误,
// cudaDeviceSynchronize() 等待执行完并捕获运行期错误(越界访问等)。
#define CUDA_CHECK_KERNEL()                                                   \
    do {                                                                      \
        CUDA_CHECK(cudaGetLastError());                                       \
        CUDA_CHECK(cudaDeviceSynchronize());                                  \
    } while (0)

// ---------------------------------------------------------------------------
// 网格尺寸
// ---------------------------------------------------------------------------

// 向上取整的除法:算 grid 维度时用 divUp(n, block) 代替 (n - 1) / block + 1。
__host__ __device__ inline unsigned int divUp(unsigned int n, unsigned int d) {
    return (n + d - 1) / d;
}

// ---------------------------------------------------------------------------
// 初始化
// ---------------------------------------------------------------------------

// 把整个数组填成同一个常量(默认 1),验证时结果好预测。
template<typename T>
void initData(T* data, const int size, const T value = static_cast<T>(1)) {
    for (int i = 0; i < size; ++i) {
        data[i] = value;
    }
}

// 填成递增序列 0,1,2,...,方便肉眼核对下标 / 转置之类的位置关系。
template<typename T>
void initDataIota(T* data, const int size) {
    for (int i = 0; i < size; ++i) {
        data[i] = static_cast<T>(i);
    }
}

// ---------------------------------------------------------------------------
// 结果验证
// ---------------------------------------------------------------------------

// 逐元素比较两个数组,返回是否全部一致并打印结论。
// eps 是允许的绝对误差:浮点比较传个小量(如 1e-5),整数用默认的 0 即可。
template<typename T>
bool checkResult(const T* a, const T* b, const int size, const double eps = 0.0) {
    for (int i = 0; i < size; ++i) {
        if (std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i])) > eps) {
            printf("result check FAIL at %d: %g vs %g\n",
                   i, static_cast<double>(a[i]), static_cast<double>(b[i]));
            return false;
        }
    }
    printf("result check SUCCESS\n");
    return true;
}

// ---------------------------------------------------------------------------
// 计时 / 带宽
// ---------------------------------------------------------------------------

// 用 CUDA event 给一段 GPU 操作计时,返回单次平均耗时(ms)。
// launch 是个可调用对象(lambda),里面写 kernel 启动或 cudaMemcpy 都行:
//     float ms = timeKernel([&]{ myKernel<<<grid, block>>>(...); }, 100);
template<typename Launch>
float timeKernel(Launch launch, const int iters) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    launch();                              // 预热一次:排除首次启动/JIT 的开销。
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int it = 0; it < iters; ++it) {
        launch();
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / iters;                     // 返回单次平均耗时。
}

// 有效带宽(GB/s):bytes 是这次操作真正搬运的字节数,ms 是耗时。
// 例:读一遍+写一遍 n 个 float => bytes = 2 * n * sizeof(float)。
inline double effectiveGBps(const double bytes, const float ms) {
    return bytes / (ms / 1e3) / 1e9;
}
