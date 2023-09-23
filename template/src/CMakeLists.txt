cmake_minimum_required (VERSION 3.0)
project(Cpp_template_test)

set( RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin) 
set( LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib )

add_executable(main main.cpp)
add_library(libtest libtest.cpp)

target_link_libraries(main libtest)