#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(err_));            \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

__global__ void sumArray(const float* a, const float* b, float* res, const int offset, const int size) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (const int k = i + offset; k < size) {
        res[i] = a[k] + b[k];
    }
}

template<typename T>
void initArray(T* arr, const int n, T value) {
    for (int i = 0; i < n; ++i) {
        arr[i] = value;
    }
}

int main(int argc, char** argv) {
    constexpr int size = 1 << 20;
    int offset = 0;

    if (argc > 1) {
        offset = atoi(argv[1]);
    }

    constexpr int bytes = size * sizeof(float);

    // Host 端用 pinned（页锁定）内存：H2D/D2H 拷贝走 DMA，更快，也可配合异步拷贝。
    float *a_h = nullptr, *b_h = nullptr, *res_h = nullptr;
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&a_h),   bytes));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&b_h),   bytes));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&res_h), bytes));

    memset(res_h, 0, bytes);
    initArray(a_h, size, 1.0f);
    initArray(b_h, size, 2.0f);

    // Device 端用 cudaMalloc：kernel 真正读写的显存。
    float *a_d = nullptr, *b_d = nullptr, *res_d = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&a_d),   bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&b_d),   bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&res_d), bytes));

    CUDA_CHECK(cudaMemcpy(a_d, a_h, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b_d, b_h, bytes, cudaMemcpyHostToDevice));

    dim3 block(1024);
    dim3 grid((size - 1) / block.x + 1);

    sumArray<<<grid, block>>>(a_d, b_d, res_d, offset, size);
    CUDA_CHECK(cudaGetLastError());        // 捕获 kernel 启动错误
    CUDA_CHECK(cudaDeviceSynchronize());   // 捕获 kernel 执行错误

    // 把结果拷回 host。
    CUDA_CHECK(cudaMemcpy(res_h, res_d, bytes, cudaMemcpyDeviceToHost));

    // 正确性校验：res[i] 应等于 a[i+offset] + b[i+offset] = 3.0f（在有效范围内）。
    int errors = 0;
    for (int i = 0; i + offset < size; ++i) {
        if (fabsf(res_h[i] - 3.0f) > 1e-5f) {
            if (errors < 5) {
                fprintf(stderr, "mismatch at %d: %f\n", i, res_h[i]);
            }
            ++errors;
        }
    }
    printf("offset=%d, errors=%d\n", offset, errors);

    CUDA_CHECK(cudaFree(a_d));
    CUDA_CHECK(cudaFree(b_d));
    CUDA_CHECK(cudaFree(res_d));
    CUDA_CHECK(cudaFreeHost(a_h));
    CUDA_CHECK(cudaFreeHost(b_h));
    CUDA_CHECK(cudaFreeHost(res_h));

    return 0;
}
