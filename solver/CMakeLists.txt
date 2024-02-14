cmake_minimum_required (VERSION 3.0)
project(solver_test)

include(ENV.cmake)

set( target_name main)
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin) 
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib )

include_directories(${Eigen3_include_path})

add_executable(test_simp test_simp.cpp)
add_executable(test_lu test_lu.cpp)
add_executable(test_eigen test_eigen.cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")