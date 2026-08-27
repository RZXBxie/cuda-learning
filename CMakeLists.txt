cmake_minimum_required(VERSION 3.21)

# ---------------------------------------------------------------------------
# 以下两项必须在 project() 之前设置,否则对 target 不生效。
# ---------------------------------------------------------------------------

# 目标 GPU 架构。可用 -DCMAKE_CUDA_ARCHITECTURES=... 覆盖:
#   75 = Turing(RTX 20 / GTX 16),86 = Ampere(RTX 30),89 = Ada(RTX 40)
# 显式指定还有一个好处:没有本地 GPU 时 CMake 不会去探测显卡。
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
	set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

# 查找 nvcc。CUDA Toolkit 未装在默认位置、或 bin 目录不在 PATH 上时(CLion、
# 未重启的终端都常见),CMake 自己找不到,这里按常见线索兜底:
#   1. 命令行 -DCMAKE_CUDA_COMPILER=... 显式指定(最高优先级,跳过查找)
#   2. 环境变量 CUDAToolkit_ROOT / CUDA_PATH / CUDA_HOME
#   3. PATH,以及 Windows/Linux 的默认安装路径
if(NOT DEFINED CMAKE_CUDA_COMPILER)
	file(GLOB _cuda_default_roots
			"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*"
			"/usr/local/cuda"
			"/usr/local/cuda-*")
	find_program(CUDA_NVCC_EXECUTABLE
			NAMES nvcc
			HINTS
			ENV CUDAToolkit_ROOT
			ENV CUDA_PATH
			ENV CUDA_HOME
			${_cuda_default_roots}
			PATH_SUFFIXES bin)
	if(CUDA_NVCC_EXECUTABLE)
		set(CMAKE_CUDA_COMPILER "${CUDA_NVCC_EXECUTABLE}")
	else()
		message(FATAL_ERROR
				"找不到 nvcc。请安装 CUDA Toolkit,并设置环境变量 CUDA_PATH,"
				"或配置时传 -DCMAKE_CUDA_COMPILER=<toolkit>/bin/nvcc(.exe)。")
	endif()
endif()

project(cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 单配置生成器(Ninja / Makefiles)下默认给个 Release,免得跑出来的
# 计时数据是未优化的 Debug 版本。
get_property(_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT _multi_config AND NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

if(MSVC)
	# nvcc 把 host 编译器选项透传给 cl,MSVC 对源码字符集的猜测会把
	# UTF-8 中文注释当成 GBK,于是 warning C4819 满屏刷。统一按 UTF-8 处理。
	add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
	add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
endif()

# 所有 target 的可执行文件统一输出到 <repo>/bin。
# 在 add_subdirectory 之前设置,各子目录自动继承。
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
# 多配置生成器(Visual Studio / Ninja Multi-Config)默认还会往 bin 下再套一层
# Release/ Debug/ 子目录,这里按配置逐个覆盖,保证产物都直接落在 bin/。
foreach(_cfg IN ITEMS DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${_cfg} ${CMAKE_SOURCE_DIR}/bin)
endforeach()

# 公共头文件(cuda_utils.cuh 等)所在目录,各子目录自动继承,
# 于是任意 .cu 里都能 #include <cuda_utils.cuh>。
include_directories(${CMAKE_SOURCE_DIR}/common)

add_subdirectory(Samples/memoryBandwidth)
add_subdirectory(Samples/reduction)
add_subdirectory(Samples/reductionUnroll)
add_subdirectory(Samples/dynamicParallelism)
add_subdirectory(Samples/memoryAccess)
add_subdirectory(Samples/bandwidthLimit)
add_subdirectory(Samples/sharedMemory)
