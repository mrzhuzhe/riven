cmake_minimum_required (VERSION 3.0)
project(Cpp_template_test)

include(ENV.cmake)

set( target_name main)
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin) 
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib )

include_directories(${Eigen3_include_path} ${opencv4_include_path})
link_directories(${opencv4_libary_path})


add_library(libtest SHARED libtest.cpp)
link_libraries(opencv_core opencv_imgproc opencv_highgui)

add_executable(${target_name} main.cpp)

target_link_libraries(${target_name} libtest)