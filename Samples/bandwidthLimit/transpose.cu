#include <cuda_utils.cuh>

__global__ void copyRow(const float *matrixA, float *matrixB, const int nx, const int ny) {
	const int col = threadIdx.x + blockDim.x * blockIdx.x;
	const int row = threadIdx.y + blockDim.y * blockIdx.y;
	const int idx = row * nx + col;
	if (col < nx && row < ny) {
		matrixB[idx] = matrixA[idx];
	}
}

__global__ void copyColumn(const float *matrixA, float *matrixB, const int nx, const int ny) {
	const int col = threadIdx.x + blockDim.x * blockIdx.x;
	const int row = threadIdx.y + blockDim.y * blockIdx.y;
	const int idx = col * ny + row;
	if (col < nx && row < ny) {
		matrixB[idx] = matrixA[idx];
	}
}

// 记忆示例：block(5,3) 即 blockDim.x=5, blockDim.y=3；输入 A 宽 nx=5、高 ny=3。
// 让矩阵大小刚好等于一个 block，则 blockIdx=0，故 ix=threadIdx.x(=col), iy=threadIdx.y(=row)。
//
// 线程网格 (x 横向增长=列, y 纵向增长=行):
//            x=0   x=1   x=2   x=3   x=4
//   y=0    (0,0) (1,0) (2,0) (3,0) (4,0)
//   y=1    (0,1) (1,1) (2,1) (3,1) (4,1)
//   y=2    (0,2) (1,2) (2,2) (3,2) (4,2)
//
// 输入 A(3行5列, 行主序, 格内=线性下标 idx=iy*nx+ix):
//          col0 col1 col2 col3 col4
//   row0     0    1    2    3    4
//   row1     5    6    7    8    9
//   row2    10   11   12   13   14
//
// 输出 B(转置后 5行3列, 宽 nx'=ny=3, 格内=线性下标):
//          col0 col1 col2
//   row0     0    1    2
//   row1     3    4    5
//   row2     6    7    8
//   row3     9   10   11
//   row4    12   13   14
//
// 以线程 (x=3,y=1) 为例: ix=3(col), iy=1(row)
//   idx = ix + iy*nx = 3 + 1*5 = 8    // 读 A 的 (row1,col3)
//   idy = iy + ix*ny = 1 + 3*3 = 10   // 写 B 的 (row3,col1)
//   => matrixB[10] = matrixA[8]，即 A(1,3) -> B(3,1)，行列互换即转置。
//
// 口诀：乘的那个数永远是"当前这张表的宽度"——读 A 乘 nx，写 B 乘 ny；
//       且 ix/iy 在读写两端位置对调，这正是转置。
// 按行读取，按列写入
__global__ void transposeRow(const float *matrixA, float *matrixB, const int nx, const int ny) {
	const int ix = threadIdx.x + blockDim.x * blockIdx.x;
	const int iy = threadIdx.y + blockDim.y * blockIdx.y;

	if (ix < nx && iy < ny) {
		const int idx = ix + iy * nx;
		const int idy = iy + ix * ny;
		matrixB[idy] = matrixA[idx];
	}
}

// 按列读取，按行写入
__global__ void transposeColumn(const float *matrixA, float *matrixB, const int nx, const int ny) {
	const int ix = threadIdx.x + blockDim.x * blockIdx.x;
	const int iy = threadIdx.y + blockDim.y * blockIdx.y;
	if (ix < nx && iy < ny) {
		const int idx = iy + ix * ny;
		const int idy = ix + iy * nx;
		matrixB[idy] = matrixA[idx];
	}
}

__global__ void transposeRowUnroll(const float *matrixA, float *matrixB, const int nx, const int ny) {
	const int ix = threadIdx.x + blockDim.x * blockIdx.x * 4;
	const int iy = threadIdx.y + blockDim.y * blockIdx.y;
	if (ix < nx && iy < ny) {
		const int idx = ix + iy * nx;
		const int idy = iy + ix * ny;
		matrixB[idy] = matrixA[idx];
		matrixB[idy + ny * 1 * blockDim.x] = matrixA[idx + 1 * blockDim.x];
		matrixB[idy + ny * 2 * blockDim.x] = matrixA[idx + 2 * blockDim.x];
		matrixB[idy + ny * 3 * blockDim.x] = matrixA[idx + 3 * blockDim.x];
	}
}

__global__ void transposeColUnroll(const float *matrixA, float *matrixB, const int nx, const int ny) {
	const int ix = threadIdx.x + blockDim.x * blockIdx.x * 4;
	const int iy = threadIdx.y + blockDim.y * blockIdx.y;
	if (ix < nx && iy < ny) {
		const int idx = ix + iy * nx;
		const int idy = iy + ix * ny;
		matrixB[idx] = matrixA[idy];
		matrixB[idx + 1 * blockDim.x] = matrixA[idy + ny * 1 * blockDim.x];
		matrixB[idx + 2 * blockDim.x] = matrixA[idy + ny * 2 * blockDim.x];
		matrixB[idx + 3 * blockDim.x] = matrixA[idy + ny * 3 * blockDim.x];
	}
}

int main(const int argc, char **argv) {
	constexpr int nx = 1 << 10;
	constexpr int ny = 1 << 10;
	int dimx = 32;
	int dimy = 32;
	constexpr int nxy = nx * ny;
	constexpr int bytes = nxy * sizeof(float);
	constexpr int iters = 100;
	int transformKernel = 0;
	if (argc >= 2) {
		transformKernel = atoi(argv[1]);
	}
	if (argc >= 3) {
		dimx = atoi(argv[2]);
	}
	if (argc >= 4) {
		dimy = atoi(argv[3]);
	}

	float *A_host = nullptr, *B_host = nullptr;
	CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&A_host), bytes));
	CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&B_host), bytes));
	initData(A_host, nxy);

	float *A_dev = nullptr, *B_dev = nullptr;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&A_dev), bytes));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&B_dev), bytes));

	CUDA_CHECK(cudaMemcpy(A_dev, A_host, bytes, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemset(B_dev, 0, bytes));

	dim3 block(dimx, dimy);
	dim3 grid(divUp(nx, block.x), divUp(ny, block.y));

	float* test = nullptr;
	cudaMallocManaged(reinterpret_cast<void **>(&test), bytes);

	float ms = 0.0f;

	switch (transformKernel) {
		case 0:
			ms = timeKernel([&] { copyRow<<<grid, block>>>(A_dev, B_dev, nx, ny); }, iters);
			break;

		case 1:
			ms = timeKernel([&] { copyColumn<<<grid, block>>>(A_dev, B_dev, nx, ny); }, iters);
			break;

		case 2:
			ms = timeKernel([&] { transposeRow<<<grid, block>>>(A_dev, B_dev, nx, ny); }, iters);
			break;
		case 3:
			ms = timeKernel([&] { transposeColumn<<<grid, block>>>(A_dev, B_dev, nx, ny); }, iters);
			break;
		case 4: {
			dim3 grid_r((grid.x) / 4, grid.y);
			ms = timeKernel([&]{ transposeRowUnroll<<<grid_r, block>>>(A_dev, B_dev, nx, ny); }, iters);
			break;
		}
		case 5: {
			dim3 grid_c((grid.x) / 4, grid.y);
			ms = timeKernel([&]{ transposeColUnroll<<<grid_c, block>>>(A_dev, B_dev, nx, ny); }, iters);
			break;
		}

		default:
			break;
	}
	CUDA_CHECK_KERNEL();

	CUDA_CHECK(cudaMemcpy(B_host, B_dev, bytes, cudaMemcpyDeviceToHost));

	// 有效带宽:读一遍 + 写一遍 => 2 * bytes。
	printf("kernel %d: %.4f ms/iter, effective BW = %.1f GB/s\n",
				 transformKernel, ms, effectiveGBps(2.0 * bytes, ms));
	checkResult(A_host, B_host, nxy);

	CUDA_CHECK(cudaFree(A_dev));
	CUDA_CHECK(cudaFree(B_dev));
	CUDA_CHECK(cudaFreeHost(A_host));
	CUDA_CHECK(cudaFreeHost(B_host));

	return 0;
}
