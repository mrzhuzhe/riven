cmake_minimum_required (VERSION 3.0)
project(Intel_test C CXX ASM)

set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin) 
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib )

add_executable(main_test main_test.cpp test.s)
add_executable(main_testf main_testf.cpp testf.s)
add_executable(main_bandwidth main_bandwidth.cpp test_bandwidth.s test_bandwidth_2p.s)
add_executable(main_bankconflict main_bankconflict.cpp test_bankconflict.s test_bankconflict_b.s)
add_executable(main_speculation main_speculation.cpp test_speculation.s)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")