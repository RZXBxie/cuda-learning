# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A personal CUDA learning repo. Each directory under `Samples/` is a self-contained
experiment in one GPU performance topic (memory bandwidth, coalesced access,
reduction, shared-memory bank conflicts, dynamic parallelism). There is no library
and no test suite — every sample is a standalone executable whose *printed output*
(timings, effective bandwidth, correctness check) is the result being studied.

Code comments are written in Chinese and are deliberately verbose/tutorial-style —
they explain *why* a memory pattern is fast or slow, sometimes with ASCII diagrams of
thread grids and index arithmetic (see `Samples/bandwidthLimit/transpose.cu:21-50`).
Preserve that style when editing; the comments are the point of the repo, not noise.
Commit messages follow Conventional Commits with Chinese subjects (`feat: 学习动态并行`).

## Build and run

Windows(当前开发机:RTX 3060 Laptop = sm_86,CUDA Toolkit 13.3,VS 2026 MSVC 14.51):

```bat
build.cmd                        :: 一键配置 + 构建全部 sample(Release,Ninja)
build.cmd Release reduction      :: 只构建单个 target
bin\transpose.exe 2 32 32        :: 跑 sample;可执行文件全部落在 bin\
```

`build.cmd` 自己用 `vswhere` 找 MSVC 并 `call vcvars64.bat`(nvcc 需要 `cl.exe` 作
host 编译器),PATH 里没有 cmake/ninja 时退回 Visual Studio 自带的那份,再按
`CUDA_PATH` 找 nvcc。手动在 *x64 Native Tools 命令提示符* 里也可以:

```bat
cmake --preset ninja-release
cmake --build --preset ninja-release
```

Linux / 容器:

```bash
cmake -S . -B build && cmake --build build
```

`cmake-build-*/` 是 CLion 自己的构建树 —— 别去动,命令行统一用 `build/`。三者都被
gitignore,`bin/` 也是。

There is no lint step and no tests. To "test" a change, build it and run the affected
binary; samples self-check with `checkResult` and print `result check SUCCESS/FAIL`.

**注意 Windows 上 CUDA Toolkit 装在非默认目录时**(本机是 `F:\software\NVIDIA`),
安装器不会写 `CUDA_PATH`,CMake 与 `build.cmd` 都会找不到 nvcc。已设好用户环境变量
`CUDA_PATH`;换机器时同样先设它,或配置时传
`-DCMAKE_CUDA_COMPILER=<toolkit>/bin/nvcc.exe`。

## Build-system structure

The root `CMakeLists.txt` does all the real configuration; per-sample `CMakeLists.txt`
files are usually a single `add_executable` line.

Two settings must stay **before** `project()` or they silently don't apply:
- `CMAKE_CUDA_ARCHITECTURES`(默认 86,`-DCMAKE_CUDA_ARCHITECTURES=75` 之类可覆盖)
  — 显式给死还能省掉 CMake 探测显卡的一步。
- `CMAKE_CUDA_COMPILER` — 由一段 `find_program` 兜底查找:先看
  `CUDAToolkit_ROOT` / `CUDA_PATH` / `CUDA_HOME`,再看 PATH 和 Windows/Linux 的默认
  安装位置;找不到就直接 `FATAL_ERROR` 提示怎么修。CLion 和刚开的终端常常没把
  toolkit 的 bin 放进 PATH,靠的就是这段。

MSVC 下还额外加了 `/utf-8`(CUDA 侧用 `-Xcompiler=/utf-8`):源码里全是 UTF-8 中文
注释,不加 cl 会按 GBK 猜、刷满 C4819 警告。多配置生成器的
`CMAKE_RUNTIME_OUTPUT_DIRECTORY_<CONFIG>` 也逐个覆盖过,免得产物跑到 `bin/Release/`。

`CMAKE_RUNTIME_OUTPUT_DIRECTORY` (→ `bin/`) and `include_directories(common)` are set
before `add_subdirectory` so every sample inherits them; that inherited include path is
why sources use the angle-bracket form `#include <cuda_utils.cuh>`.

### Adding a sample

1. Create `Samples/<name>/<name>.cu` plus a `CMakeLists.txt` with `add_executable(<name> <name>.cu)`.
2. **Add `add_subdirectory(Samples/<name>)` to the root `CMakeLists.txt`** — this is easy
   to forget, and the sample just never builds.

Samples needing device-side kernel launches must opt into separable compilation, which
makes CMake do the device link and pull in `cudadevrt`:

```cmake
set_target_properties(dynamicParallelism PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
```

## common/cuda_utils.cuh

Header-only (macros/templates/inline), included by every sample. Use these instead of
re-rolling boilerplate:

- `CUDA_CHECK(call)` — wrap any `cudaError_t`-returning call; prints file/line and exits.
- `CUDA_CHECK_KERNEL()` — put immediately after a launch. Kernel launches are async and
  return no error, so this pairs `cudaGetLastError()` (bad launch config) with
  `cudaDeviceSynchronize()` (runtime faults such as out-of-bounds).
- `divUp(n, d)` — ceiling division for grid dims.
- `initData(p, size, value=1)` / `initDataIota(p, size)` — constant fill (makes expected
  results trivial: reducing all-ones gives `size`) or `0,1,2,...` for checking index math.
- `checkResult(a, b, size, eps=0)` — element-wise compare and print verdict; pass an
  `eps` like `1e-5` for floats.
- `timeKernel(lambda, iters)` — CUDA-event timing; **already warms up once** and returns
  ms *per iteration*. Works for `cudaMemcpy` too, not just launches.
- `effectiveGBps(bytes, ms)` — `bytes` is what the operation genuinely moves. Convention
  in this repo: a read-once + write-once pass counts `2 * n * sizeof(T)`.

## Sample conventions

A typical sample declares several kernel variants of the same computation, then times
them against each other to expose one hardware effect. Common idioms:

- Variant selection and block dims come from `argv` with defaults, so one binary sweeps a
  parameter space: `./bin/transpose <kernel> [dimx] [dimy]`, `./bin/memoryUnAligned <offset>`.
- Kernels that overwrite their input in place (all the reduction ones) need the input
  re-copied H2D before each variant runs — `reduction.cu:71` wraps that in a `runReduce`
  lambda; `reductionUnroll.cu` repeats it inline.
- Effective bandwidth is deliberately computed from the *useful* bytes only, so a layout
  that also drags along unwanted data (AoS hauling `y`/`z` to touch `x`) shows up directly
  as lower GB/s rather than needing separate accounting.
- Host buffers are usually pinned via `cudaMallocHost` (freed with `cudaFreeHost`), with
  plain `malloc` kept around where the sample is specifically contrasting pageable vs
  pinned transfer rates.
- Compile-time-known block sizes are passed as template parameters so size checks fold
  away at compile time (`reduceComplete<blockSize>` in `reductionUnroll.cu:106`).
