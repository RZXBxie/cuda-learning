# 动态并行:设备端嵌套 kernel 启动需要可分离编译(-rdc=true),
# CMake 会自动完成设备链接并链上 cudadevrt。
add_executable(dynamicParallelism dynamicParallelism.cu)
set_target_properties(dynamicParallelism PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
