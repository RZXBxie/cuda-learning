#include <cuda_utils.cuh>

__global__ void reduceNeighbored(int *g_idata, int *g_odata, const int size) {
	const unsigned int idx = threadIdx.x;
	if (idx >= size) return;

	// 每个线程负责的块
	int *idata = g_idata + blockIdx.x * blockDim.x;
	for (int stride = 1; stride < blockDim.x; stride *= 2) {
		if (idx % (2 * stride) == 0) {
			idata[idx] += idata[idx + stride];
		}
		__syncthreads();
	}

	if (idx == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceNeighboredLess(int *g_idata, int *g_odata, const int size) {
	const unsigned int idx = threadIdx.x;
	if (idx >= size) return;

	int *idata = g_idata +blockIdx.x * blockDim.x;
	for (int stride = 1; stride < blockDim.x; stride *= 2) {
		if (int index = 2 * stride * idx; index < blockDim.x) {
			idata[index] += idata[index +stride];
		}

		__syncthreads();
	}

	if (idx == 0) {
		g_odata[blockIdx.x] = idata[0];
	}
}

__global__ void reduceNeighboredInterval(int *g_idata, int *g_odata, const int size) {
	const unsigned int idx = threadIdx.x;
	if (idx >= size) return;
	int *idata = g_idata + blockIdx.x * blockDim.x;
	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		if (idx < stride) {
			idata[idx] += idata[idx + stride];
		}
		__syncthreads();
	}

	if (idx == 0) {
		g_odata[blockIdx.x] = idata[0];
	}
}

int main(int argc, char** argv) {
	constexpr int size = 4 * 1024 * 1024;
	constexpr int nBytes = size * sizeof(int);

	constexpr int blockSize = 1024;
	dim3 block(blockSize, 1);
	dim3 grid(divUp(size, blockSize), 1);

	int* h_idata = static_cast<int *>(malloc(nBytes));
	int* h_odata = static_cast<int *>(malloc(grid.x * sizeof(int)));
	initData(h_idata, size);               // 全填 1,归约结果应等于 size。

	int *d_idata, *d_odata;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_idata), nBytes));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_odata), grid.x * sizeof(int)));

	// 每个 kernel 都是"每块归约出一个部分和,再把 grid.x 个部分和在 host 累加"。
	// 用 lambda 把这套流程收敛起来,顺便每次重置输入(kernel 会原地改写 d_idata)。
	auto runReduce = [&](const char* name, auto kernel) {
		CUDA_CHECK(cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice));
		kernel<<<grid, block>>>(d_idata, d_odata, size);
		CUDA_CHECK_KERNEL();
		CUDA_CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
		int sum = 0;
		for (int i = 0; i < grid.x; ++i) {
			sum += h_odata[i];
		}
		printf("%-24s sum: %d (expected %d)\n", name, sum, size);
	};

	runReduce("reduceNeighbored", reduceNeighbored);
	runReduce("reduceNeighboredLess", reduceNeighboredLess);
	runReduce("reduceNeighboredInterval", reduceNeighboredInterval);

	CUDA_CHECK(cudaFree(d_idata));
	CUDA_CHECK(cudaFree(d_odata));
	free(h_idata);
	free(h_odata);

	return 0;
}
