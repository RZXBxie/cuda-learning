#include <cuda_utils.cuh>

__global__ void gpuSumArrays(float* a, float* b, float* res) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	res[idx] = a[idx] + b[idx];
}

// 多次拷贝取平均,测 host->device 传输带宽 (GB/s)
float benchH2D(float* d_dst, float* h_src, int bytes, int iters) {
	// timeKernel 已含预热 + 单次平均,直接把 memcpy 包进去计时。
	float ms = timeKernel([&] {
		cudaMemcpy(d_dst, h_src, bytes, cudaMemcpyHostToDevice);
	}, iters);
	return static_cast<float>(effectiveGBps(bytes, ms));  // 每次搬运 bytes 字节
}

int main() {
	// 带宽测试用大 buffer 才有意义(64MB)
	constexpr int size = 1 << 24;
	constexpr int bytes = size * sizeof(float);
	constexpr int iters = 20;

	float* d_buf = nullptr;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_buf), bytes));

	// ---------- pageable 内存 (malloc) ----------
	float* pageable = static_cast<float *>(malloc(bytes));
	initData(pageable, size);
	float bwPageable = benchH2D(d_buf, pageable, bytes, iters);

	// ---------- pinned 内存 (cudaMallocHost) ----------
	float* pinned = nullptr;
	CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&pinned), bytes));
	initData(pinned, size);
	float bwPinned = benchH2D(d_buf, pinned, bytes, iters);

	printf("transfer size    : %d MB\n", bytes / (1 << 20));
	printf("pageable H2D bw  : %.2f GB/s\n", bwPageable);
	printf("pinned   H2D bw  : %.2f GB/s\n", bwPinned);
	printf("speedup          : %.2fx\n", bwPinned / bwPageable);

	free(pageable);
	CUDA_CHECK(cudaFreeHost(pinned));
	CUDA_CHECK(cudaFree(d_buf));

	// ---------- 顺便跑一遍标准 vector-add ----------
	constexpr int n = 1 << 24;
	constexpr int nbytes = n * sizeof(float);

	float *h_a = nullptr, *h_b = nullptr, *h_res = nullptr;
	CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&h_a), nbytes));
	CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&h_b), nbytes));
	CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&h_res), nbytes));
	float* h_expected = static_cast<float *>(malloc(nbytes));
	initData(h_a, n);
	initData(h_b, n);
	initData(h_expected, n, 2.0f);         // a+b = 1+1 = 2,作为验证基准。

	float *d_a = nullptr, *d_b = nullptr, *d_res = nullptr;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a), nbytes));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b), nbytes));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_res), nbytes));

	CUDA_CHECK(cudaMemcpy(d_a, h_a, nbytes, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_b, h_b, nbytes, cudaMemcpyHostToDevice));

	dim3 block(1024);
	dim3 grid(divUp(n, block.x));
	gpuSumArrays<<<grid, block>>>(d_a, d_b, d_res);
	CUDA_CHECK_KERNEL();
	printf("execution configuration<<<%d, %d>>>\n", grid.x, block.x);

	CUDA_CHECK(cudaMemcpy(h_res, d_res, nbytes, cudaMemcpyDeviceToHost));
	checkResult(h_res, h_expected, n);

	CUDA_CHECK(cudaFree(d_a));
	CUDA_CHECK(cudaFree(d_b));
	CUDA_CHECK(cudaFree(d_res));
	CUDA_CHECK(cudaFreeHost(h_a));
	CUDA_CHECK(cudaFreeHost(h_b));
	CUDA_CHECK(cudaFreeHost(h_res));
	free(h_expected);

	// ---------- 零拷贝内存 (zero-copy / mapped pinned) ----------
	// 思路:主机内存分配成"可映射的固定内存",kernel 直接通过 PCIe 访问,
	//       不需要 cudaMalloc + cudaMemcpy。省显存、省显式拷贝,
	//       但每次访问都走 PCIe,适合小数据 / 只读一次的场景,大数据反而慢。

	// 先确认设备支持映射主机内存
	cudaDeviceProp prop;
	CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
	if (!prop.canMapHostMemory) {
		printf("device does not support mapped host memory, skip zero-copy\n");
		return 0;
	}
	// 必须在任何 CUDA 上下文操作前后开启映射支持
	CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));

	constexpr int zn = 1 << 20;  // 零拷贝用小一点的数据(1M)
	constexpr int znbytes = zn * sizeof(float);

	// cudaHostAllocMapped:分配的主机内存可被设备映射
	float *z_a = nullptr, *z_b = nullptr, *z_res = nullptr;
	CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void **>(&z_a), znbytes, cudaHostAllocMapped));
	CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void **>(&z_b), znbytes, cudaHostAllocMapped));
	CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void **>(&z_res), znbytes, cudaHostAllocMapped));
	initData(z_a, zn);
	initData(z_b, zn);

	// 拿到这些主机指针对应的"设备端地址",kernel 里用它来访问
	float *zd_a = nullptr, *zd_b = nullptr, *zd_res = nullptr;
	CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void **>(&zd_a), z_a, 0));
	CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void **>(&zd_b), z_b, 0));
	CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void **>(&zd_res), z_res, 0));

	dim3 zblock(1024);
	dim3 zgrid(divUp(zn, zblock.x));
	// 注意:传给 kernel 的是设备端指针 zd_*,没有任何 cudaMemcpy
	gpuSumArrays<<<zgrid, zblock>>>(zd_a, zd_b, zd_res);
	CUDA_CHECK_KERNEL();          // kernel 写的是主机内存,同步后 host 可直接读 z_res

	float* z_expected = static_cast<float *>(malloc(znbytes));
	initData(z_expected, zn, 2.0f);
	printf("zero-copy ");
	checkResult(z_res, z_expected, zn);
	free(z_expected);

	CUDA_CHECK(cudaFreeHost(z_a));
	CUDA_CHECK(cudaFreeHost(z_b));
	CUDA_CHECK(cudaFreeHost(z_res));

	return 0;
}
