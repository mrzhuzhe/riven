cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

project(helloavx)

add_executable(avx_pd_test avx_pd_test.cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")