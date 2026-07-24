#include <cstdio>

__global__ void gpuSumArrays(float* a, float* b, float* res) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	res[idx] = a[idx] + b[idx];
}

void initialData(float* array, const int size) {
	for (int i = 0; i < size; ++i) {
		array[i] = 1.0;
	}
}

// 多次拷贝取平均,测 host->device 传输带宽 (GB/s)
float benchH2D(float* d_dst, float* h_src, int bytes, int iters) {
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// warmup,避免首次调用的初始化开销计入
	cudaMemcpy(d_dst, h_src, bytes, cudaMemcpyHostToDevice);

	cudaEventRecord(start);
	for (int i = 0; i < iters; ++i) {
		cudaMemcpy(d_dst, h_src, bytes, cudaMemcpyHostToDevice);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float ms = 0.0f;
	cudaEventElapsedTime(&ms, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	double seconds = ms / 1000.0;
	double totalBytes = static_cast<double>(bytes) * iters;
	return static_cast<float>(totalBytes / seconds / 1e9);  // GB/s
}

int main() {
	// 带宽测试用大 buffer 才有意义(64MB)
	constexpr int size = 1 << 24;
	constexpr int bytes = size * sizeof(float);
	constexpr int iters = 20;

	float* d_buf = nullptr;
	cudaMalloc(reinterpret_cast<void **>(&d_buf), bytes);

	// ---------- pageable 内存 (malloc) ----------
	float* pageable = static_cast<float *>(malloc(bytes));
	initialData(pageable, size);
	float bwPageable = benchH2D(d_buf, pageable, bytes, iters);

	// ---------- pinned 内存 (cudaMallocHost) ----------
	float* pinned = nullptr;
	cudaMallocHost(reinterpret_cast<void **>(&pinned), bytes);
	initialData(pinned, size);
	float bwPinned = benchH2D(d_buf, pinned, bytes, iters);

	printf("transfer size    : %d MB\n", bytes / (1 << 20));
	printf("pageable H2D bw  : %.2f GB/s\n", bwPageable);
	printf("pinned   H2D bw  : %.2f GB/s\n", bwPinned);
	printf("speedup          : %.2fx\n", bwPinned / bwPageable);

	free(pageable);
	cudaFreeHost(pinned);
	cudaFree(d_buf);

	// ---------- 顺便跑一遍标准 vector-add ----------
	constexpr int n = 1 << 24;
	constexpr int nbytes = n * sizeof(float);

	float *h_a = nullptr, *h_b = nullptr, *h_res = nullptr;
	cudaMallocHost(reinterpret_cast<void **>(&h_a), nbytes);
	cudaMallocHost(reinterpret_cast<void **>(&h_b), nbytes);
	cudaMallocHost(reinterpret_cast<void **>(&h_res), nbytes);
	memset(h_res, 0, nbytes);
	initialData(h_a, n);
	initialData(h_b, n);

	float *d_a = nullptr, *d_b = nullptr, *d_res = nullptr;
	cudaMalloc(reinterpret_cast<void **>(&d_a), nbytes);
	cudaMalloc(reinterpret_cast<void **>(&d_b), nbytes);
	cudaMalloc(reinterpret_cast<void **>(&d_res), nbytes);

	cudaMemcpy(d_a, h_a, nbytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, nbytes, cudaMemcpyHostToDevice);

	dim3 block(1024);
	dim3 grid((n - 1) / block.x + 1);
	gpuSumArrays<<<grid, block>>>(d_a, d_b, d_res);
	printf("execution configuration<<<%d, %d>>>\n", grid.x, block.x);

	cudaMemcpy(h_res, d_res, nbytes, cudaMemcpyDeviceToHost);

	bool ok = true;
	for (int i = 0; i < n; ++i) {
		if (h_res[i] != 2.0f) {
			ok = false;
			printf("mismatch at %d: %f\n", i, h_res[i]);
			break;
		}
	}
	printf("result: %s\n", ok ? "OK" : "FAILED");

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_res);
	cudaFreeHost(h_a);
	cudaFreeHost(h_b);
	cudaFreeHost(h_res);

	// ---------- 零拷贝内存 (zero-copy / mapped pinned) ----------
	// 思路:主机内存分配成"可映射的固定内存",kernel 直接通过 PCIe 访问,
	//       不需要 cudaMalloc + cudaMemcpy。省显存、省显式拷贝,
	//       但每次访问都走 PCIe,适合小数据 / 只读一次的场景,大数据反而慢。

	// 先确认设备支持映射主机内存
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	if (!prop.canMapHostMemory) {
		printf("device does not support mapped host memory, skip zero-copy\n");
		return 0;
	}
	// 必须在任何 CUDA 上下文操作前后开启映射支持
	cudaSetDeviceFlags(cudaDeviceMapHost);

	constexpr int zn = 1 << 20;  // 零拷贝用小一点的数据(1M)
	constexpr int znbytes = zn * sizeof(float);

	// cudaHostAllocMapped:分配的主机内存可被设备映射
	float *z_a = nullptr, *z_b = nullptr, *z_res = nullptr;
	cudaHostAlloc(reinterpret_cast<void **>(&z_a), znbytes, cudaHostAllocMapped);
	cudaHostAlloc(reinterpret_cast<void **>(&z_b), znbytes, cudaHostAllocMapped);
	cudaHostAlloc(reinterpret_cast<void **>(&z_res), znbytes, cudaHostAllocMapped);
	initialData(z_a, zn);
	initialData(z_b, zn);

	// 拿到这些主机指针对应的"设备端地址",kernel 里用它来访问
	float *zd_a = nullptr, *zd_b = nullptr, *zd_res = nullptr;
	cudaHostGetDevicePointer(reinterpret_cast<void **>(&zd_a), z_a, 0);
	cudaHostGetDevicePointer(reinterpret_cast<void **>(&zd_b), z_b, 0);
	cudaHostGetDevicePointer(reinterpret_cast<void **>(&zd_res), z_res, 0);

	dim3 zblock(1024);
	dim3 zgrid((zn - 1) / zblock.x + 1);
	// 注意:传给 kernel 的是设备端指针 zd_*,没有任何 cudaMemcpy
	gpuSumArrays<<<zgrid, zblock>>>(zd_a, zd_b, zd_res);
	cudaDeviceSynchronize();  // kernel 写的是主机内存,同步后 host 可直接读 z_res

	bool zok = true;
	for (int i = 0; i < zn; ++i) {
		if (z_res[i] != 2.0f) {
			zok = false;
			printf("zero-copy mismatch at %d: %f\n", i, z_res[i]);
			break;
		}
	}
	printf("zero-copy result: %s\n", zok ? "OK" : "FAILED");

	cudaFreeHost(z_a);
	cudaFreeHost(z_b);
	cudaFreeHost(z_res);

	return 0;
}
