cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CUDA_ARCHITECTURES 80)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

project(hellocuda LANGUAGES CXX CUDA)

add_executable(warp warp.cu)
add_executable(reduction reduction.cu)
add_executable(reduction_shared reduction_shared.cu)
add_executable(divergence divergence.cu)


target_include_directories(reduction PUBLIC . ../../include)
target_include_directories(reduction_shared PUBLIC . ../../include)
target_include_directories(divergence PUBLIC . ../../include)