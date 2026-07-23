cmake_minimum_required(VERSION 3.18)

# 必须在 project() 之前设置，否则对 target 不生效。
# 没有本地 GPU 时，显式指定架构可避免 CMake 尝试自动探测显卡。
set(CMAKE_CUDA_ARCHITECTURES 75)

# CLion 的环境里 PATH 常常没有 /usr/local/cuda/bin，导致找不到 nvcc。
# 在 project() 之前显式指定编译器路径，最稳妥。
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# 如果 reduce.cu 里有 main()，用这行（可执行文件）：
add_executable(reduce reduce.cu
        reduceRoll.cu
        dynamic_parallel.cu)
# 如果 reduce.cu 只是 kernel、没有 main()，改用这行，
# 这样没显卡、没 main 也能正常构建，CLion 补全不受影响：
# add_library(reduce STATIC reduce.cu)
