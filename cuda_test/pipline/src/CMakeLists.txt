cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CUDA_ARCHITECTURES 80)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

project(hellocuda LANGUAGES CXX CUDA)

add_executable(sm sm.cu)
add_executable(pip pip.cu)


target_include_directories(pip PUBLIC . ../../include)