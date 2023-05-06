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
add_executable(balance balance.cu)
add_executable(balance_opt balance_opt.cu)
add_executable(wp wp.cu)
add_executable(cg cg.cu)
add_executable(cg_shift cg_shift.cu)
add_executable(atomic atomic.cu)
add_executable(atomic_wp atomic_wp.cu)
add_executable(atomic_blk atomic_blk.cu)

add_executable(mix mix.cu)
add_executable(dp4a dp4a.cu)
add_executable(mix_half mix_half.cu)

target_include_directories(reduction PUBLIC . ../../include)
target_include_directories(reduction_shared PUBLIC . ../../include)
target_include_directories(divergence PUBLIC . ../../include)
target_include_directories(balance PUBLIC . ../../include)
target_include_directories(balance_opt PUBLIC . ../../include)
target_include_directories(wp PUBLIC . ../../include)
target_include_directories(cg PUBLIC . ../../include)
target_include_directories(cg_shift PUBLIC . ../../include)
target_include_directories(atomic PUBLIC . ../../include)
target_include_directories(atomic_wp PUBLIC . ../../include)
target_include_directories(atomic_blk PUBLIC . ../../include)

target_include_directories(mix PUBLIC . ../../include)
target_include_directories(dp4a PUBLIC . ../../include)
target_include_directories(mix_half PUBLIC . ../../include)


add_executable(test test.cu)
