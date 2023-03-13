cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CUDA_ARCHITECTURES 80)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

project(hellocuda LANGUAGES CXX CUDA)
add_executable(main main.cu)
target_compile_options(main PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_compile_options(main PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

#set_property(TARGET main PROPERTY_CUDA_SEPARABLE_COMPILATION ON)

#https://github.com/AcademySoftwareFoundation/openvdb/issues/412
list(APPEND CMAKE_MODULE_PATH "/usr/local/lib/cmake/OpenVDB")
message(STATUS "CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}")

find_package(OpenVDB REQUIRED)
message(STATUS "Found OpenVDB ${OpenVDB_VERSION} at ${OpenVDB_LIBRARIES}")
target_link_libraries(main PUBLIC OpenVDB::openvdb)
target_sources(main PUBLIC writevdb.cpp)

