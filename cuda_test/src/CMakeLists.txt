cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CUDA_ARCHITECTURES 80)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

project(hellocuda LANGUAGES CXX CUDA)

add_executable(add add.cu)
add_executable(mul mul.cu)
add_executable(trans trans.cu)
add_executable(aos aos.cu)
add_executable(soa soa.cu)
add_executable(image_scaling image_scaling.cu srcImagePgmPpmPackage.cpp)

target_include_directories(mul PUBLIC . ../include)
target_include_directories(image_scaling PUBLIC . ../include)


#target_compile_options(main PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
#target_compile_options(main PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

#set_property(TARGET main PROPERTY_CUDA_SEPARABLE_COMPILATION ON)

