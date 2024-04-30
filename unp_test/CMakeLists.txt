cmake_minimum_required (VERSION 3.0)
project(Cpp_unp_test)

set( target_name main)
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin) 
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib )

add_executable(${target_name} main.cpp)