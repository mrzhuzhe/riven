cmake_minimum_required (VERSION 3.0)
project(Cpp_template_test)

include(ENV.cmake)

set( target_name main)
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin) 
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib )

link_directories(${opencv4_libary_path})

add_executable(${target_name} main.cpp)