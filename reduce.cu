#include <cstdio>

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
	dim3 grid((size + blockSize - 1) / blockSize, 1);

	int* h_idata = static_cast<int *>(malloc(nBytes));
	int* h_odata = static_cast<int *>(malloc(grid.x * sizeof(int)));

	for (int i = 0; i < size; ++i) {
		h_idata[i] = 1;
	}
	int *d_idata, *d_odata;
	cudaError_t err = cudaMalloc(reinterpret_cast<void **>(&d_idata), nBytes);
	if (err != cudaSuccess) {
		printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
	}
	err = cudaMalloc(reinterpret_cast<void **>(&d_odata), grid.x * sizeof(int));
	if (err != cudaSuccess) {
		printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
	}

	cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice);

	reduceNeighbored<<<grid, block>>>(d_idata, d_odata, size);
	cudaDeviceSynchronize();
	cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
	int sum = 0;
	for (int i = 0; i < grid.x; ++i) {
		sum += h_odata[i];
	}
	printf("sum: %d\n", sum);

	reduceNeighboredLess<<<grid, block>>>(d_idata, d_odata, size);
	cudaDeviceSynchronize();
	cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
	sum = 0;
	for (int i = 0; i < grid.x; ++i) {
		sum += h_odata[i];
	}
	printf("less sum: %d\n", sum);

	reduceNeighboredInterval<<<grid, block>>>(d_idata, d_odata, size);
	cudaDeviceSynchronize();
	cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
	sum = 0;
	for (int i = 0; i < grid.x; ++i) {
		sum += h_odata[i];
	}
	printf("interval sum: %d\n", sum);


	cudaFree(d_idata);
	cudaFree(d_odata);
	free(h_idata);
	free(h_odata);


	return 0;
}
