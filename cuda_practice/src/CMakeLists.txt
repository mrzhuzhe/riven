cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CUDA_ARCHITECTURES 80)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

project(hellocuda LANGUAGES CXX CUDA)
add_executable(main main.cu)
add_executable(memmory memmory.cu)
add_executable(thrust thrust.cu)
add_executable(sharedMM sharedMM.cu)

#set_property(TARGET main PROPERTY_CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(memmory PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_compile_options(thrust PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_compile_options(sharedMM PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)