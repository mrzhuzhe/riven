cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

project(helloopenmp LANGUAGES CXX)


find_package(OpenMP REQUIRED)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE OpenMP::OpenMP_CXX)
