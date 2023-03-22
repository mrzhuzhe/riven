#https://developer.nvidia.com/zh-cn/blog/building-cuda-applications-cmake/
cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CUDA_ARCHITECTURES 80)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

project(hellocuda LANGUAGES CXX CUDA)


find_package(OpenMP REQUIRED)

add_executable(sm sm.cu)
add_executable(pip pip.cu)
add_executable(cb cb.cu)
add_executable(prior prior.cu)
add_executable(event event.cu)
add_executable(sm_event sm_event.cu)
#   calling a __host__ function("cudaDeviceSynchronize") from a __global__ function("parent_kernel") is not allowed
#add_executable(dynamic_parenet dynamic_parenet.cu)
#add_executable(dynamic_recursion dynamic_recursion.cu)
add_executable(cg cg.cu)  
add_executable(overhead overhead.cu)


target_include_directories(pip PUBLIC . ../../include)
target_include_directories(cb PUBLIC . ../../include)
target_include_directories(prior PUBLIC . ../../include)
target_include_directories(event PUBLIC . ../../include)
target_include_directories(sm_event PUBLIC . ../../include)
target_include_directories(cg PUBLIC . ../../include)

# https://zhuanlan.zhihu.com/p/411118177 maybe use this
target_compile_options (pip PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(pip PRIVATE ${OpenMP_CXX_LIBRARIES})

target_compile_options (cb PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(cb PRIVATE ${OpenMP_CXX_LIBRARIES})