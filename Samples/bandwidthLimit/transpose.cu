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
//
// 和 transposeRow 做的是同一次转置,只是把"分散访问"从写端挪到读端。
// 关键在于 ix / iy 的身份跟着换了 —— 这里线程网格是贴着 B 铺的:
//        在 B 里    在 A 里    取值范围
//   ix    列号      行号       0..ny-1
//   iy    行号      列号       0..nx-1
// 所以 guard 也要跟着换成 ix < ny && iy < nx,不能照抄 transposeRow 的。
// 注意这个 guard 要配合 main 里的 grid_t 看:grid本身就按 B 的形状铺
// (x 方向 divUp(ny,bx)),ix 的上界天生就是 ny 附近,guard 只是收掉不整除的尾块,
//
// 口诀依旧:乘的那个数永远是"当前这张表的宽度" —— 读 A 乘 nx,写 B 乘 ny。
__global__ void transposeColumn(const float *matrixA, float *matrixB, const int nx, const int ny) {
	const int ix = threadIdx.x + blockDim.x * blockIdx.x;
	const int iy = threadIdx.y + blockDim.y * blockIdx.y;
	if (ix < ny && iy < nx) {
		const int idx = iy + ix * nx;   // 读 A: 行 ix, 列 iy —— 乘 A 的宽度 nx
		const int idy = ix + iy * ny;   // 写 B: 行 iy, 列 ix —— 乘 B 的宽度 ny
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

// 按列读、按行写 + 展开 4 倍。
//
// 犯过和 transposeColumn 一模一样的错:ix/iy 的身份已经贴着 B 了,
// guard 却照抄了 transposeRow 的 ix < nx && iy < ny,两个宽度也乘反。
// 方阵下数值恰好相同所以一直是 SUCCESS,1024x512 上立刻 FAIL。
//
// 展开的步长同样要认准是哪张表:相邻的 4 段在 B 里沿列方向(x)错开 blockDim.x 个元素,
// 是同一行内的偏移,所以 B 侧直接 +blockDim.x;而它们在 A 里是错开 blockDim.x 个"行",
// 所以 A 侧要 + nx * blockDim.x —— 乘的是 A 的宽度 nx,不是 ny。
__global__ void transposeColUnroll(const float *matrixA, float *matrixB, const int nx, const int ny) {
	const int ix = threadIdx.x + blockDim.x * blockIdx.x * 4;   // B 的列 = A 的行, 0..ny-1
	const int iy = threadIdx.y + blockDim.y * blockIdx.y;       // B 的行 = A 的列, 0..nx-1
	if (ix < ny && iy < nx) {
		const int idx = ix + iy * ny;   // 写 B: 行 iy, 列 ix —— 乘 B 的宽度 ny
		const int idy = iy + ix * nx;   // 读 A: 行 ix, 列 iy —— 乘 A 的宽度 nx
		matrixB[idx] = matrixA[idy];
		matrixB[idx + 1 * blockDim.x] = matrixA[idy + nx * 1 * blockDim.x];
		matrixB[idx + 2 * blockDim.x] = matrixA[idy + nx * 2 * blockDim.x];
		matrixB[idx + 3 * blockDim.x] = matrixA[idy + nx * 3 * blockDim.x];
	}
}

static void transposeHost(const float *in, float *out, const int nx, const int ny) {
	for (int iy = 0; iy < ny; ++iy) {
		for (int ix = 0; ix < nx; ++ix) {
			out[ix * ny + iy] = in[iy * nx + ix];   // B(ix,iy) = A(iy,ix),B 宽 ny
		}
	}
}

int main(const int argc, char **argv) {
	// 想验证转置下标的话把 nx / ny 改成不相等的值(例如 nx = 1<<10, ny = 1<<9):
	// 方阵会让"乘 nx"和"乘 ny"数值相同,错误无法暴露 ——
	// transposeColumn 和 transposeColUnroll 的宽度乘反 bug 就是这么藏了很久的。
	// 注意展开版仍要求展开方向能被 4*blockDim 整除(默认尺寸都是 2 的幂,满足)。
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

	float *A_host = nullptr, *B_host = nullptr, *ref_host = nullptr;
	CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&A_host), bytes));
	CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&B_host), bytes));
	CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&ref_host), bytes));
	initDataIota(A_host, nxy);                  // 1<<20 以内的整数 float 可精确表示,eps 用 0 即可
	transposeHost(A_host, ref_host, nx, ny);

	float *A_dev = nullptr, *B_dev = nullptr;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&A_dev), bytes));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&B_dev), bytes));

	CUDA_CHECK(cudaMemcpy(A_dev, A_host, bytes, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemset(B_dev, 0, bytes));

	dim3 block(dimx, dimy);
	dim3 grid(divUp(nx, block.x), divUp(ny, block.y));
	// transposeColumn 的线程网格贴着输出矩阵 B 铺(一个线程负责 B 的一个元素),
	// 而 B 是 nx 行 x ny 列,所以 x 方向按 ny 算、y 方向按 nx 算 —— 和 grid 正好对调。
	//
	// grid 与 kernel 里的 guard 必须成对理解,单看 guard 会觉得 "ix < ny 把线程拦掉了一半"。
	// 以 nx=1024, ny=512, block(32,16) 为例:
	//   grid   = (divUp(1024,32), divUp(512,16)) = (32,32) → ix 最大 1023
	//            此时 ix < ny=512 确实砍掉一半线程,B 有一半格子没人写 —— 这才是 bug。
	//   grid_t = (divUp(512,32),  divUp(1024,16)) = (16,64) → ix 最大 511
	//            ix 压根长不到 512,guard 一个线程都没拦,B 的 1024*512 格全覆盖。
	// 换句话说"覆盖多少"由 grid 决定,guard 只负责 nx / ny 不能被 block 整除时的尾块。
	dim3 grid_t(divUp(ny, block.x), divUp(nx, block.y));

	float ms = 0.0f;
	// copyRow / copyColumn 只是拷贝,期望结果是 A 本身;其余 kernel 做转置,期望是 ref_host。
	const float *expected = ref_host;

	switch (transformKernel) {
		case 0:
			expected = A_host;
			ms = timeKernel([&] { copyRow<<<grid, block>>>(A_dev, B_dev, nx, ny); }, iters);
			break;

		case 1:
			expected = A_host;
			ms = timeKernel([&] { copyColumn<<<grid, block>>>(A_dev, B_dev, nx, ny); }, iters);
			break;

		case 2:
			ms = timeKernel([&] { transposeRow<<<grid, block>>>(A_dev, B_dev, nx, ny); }, iters);
			break;
		case 3:
			ms = timeKernel([&] { transposeColumn<<<grid_t, block>>>(A_dev, B_dev, nx, ny); }, iters);
			break;
		// 展开版一个线程干 4 个元素,所以 x 方向的 block 数除以 4。
		// 用 divUp 而不是 grid.x / 4:后者在 grid.x 不是 4 的倍数时会少铺一列 block。
		// transposeRowUnroll 贴 A 铺,transposeColUnroll 贴 B 铺,和非展开版同理。
		case 4: {
			dim3 grid_r(divUp(nx, block.x * 4), divUp(ny, block.y));
			ms = timeKernel([&]{ transposeRowUnroll<<<grid_r, block>>>(A_dev, B_dev, nx, ny); }, iters);
			break;
		}
		case 5: {
			dim3 grid_c(divUp(ny, block.x * 4), divUp(nx, block.y));
			ms = timeKernel([&]{ transposeColUnroll<<<grid_c, block>>>(A_dev, B_dev, nx, ny); }, iters);
			break;
		}

		default:
			printf("unknown kernel %d (expect 0..5)\n", transformKernel);
			return EXIT_FAILURE;
	}
	CUDA_CHECK_KERNEL();

	CUDA_CHECK(cudaMemcpy(B_host, B_dev, bytes, cudaMemcpyDeviceToHost));

	// 有效带宽:读一遍 + 写一遍 => 2 * bytes。
	printf("kernel %d: %.4f ms/iter, effective BW = %.1f GB/s\n",
				 transformKernel, ms, effectiveGBps(2.0 * bytes, ms));
	checkResult(expected, B_host, nxy);

	CUDA_CHECK(cudaFree(A_dev));
	CUDA_CHECK(cudaFree(B_dev));
	CUDA_CHECK(cudaFreeHost(A_host));
	CUDA_CHECK(cudaFreeHost(B_host));
	CUDA_CHECK(cudaFreeHost(ref_host));

	return 0;
}
