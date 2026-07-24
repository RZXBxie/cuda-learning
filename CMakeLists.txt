cmake_minimum_required(VERSION 3.18)

# 必须在 project() 之前设置,否则对 target 不生效。
# 没有本地 GPU 时,显式指定架构可避免 CMake 尝试自动探测显卡。
set(CMAKE_CUDA_ARCHITECTURES 75)

# CLion 的环境里 PATH 常常没有 /usr/local/cuda/bin,导致找不到 nvcc。
# 在 project() 之前显式指定编译器路径,最稳妥。
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# 所有 target 的可执行文件统一输出到 <repo>/bin。
# 在 add_subdirectory 之前设置,各子目录自动继承。
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

add_subdirectory(Samples/memoryBandwidth)
add_subdirectory(Samples/reduction)
add_subdirectory(Samples/reductionUnroll)
add_subdirectory(Samples/dynamicParallelism)
