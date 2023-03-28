cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CUDA_ARCHITECTURES 80)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

project(hellocuda LANGUAGES CXX CUDA)

#Modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")


add_executable(train train.cu layer.cu loss.cu mnist.cu network.cu)
target_include_directories(train PUBLIC . ../../include)

message(STATUS "11111111111 ${CMAKE_CURRENT_SOURCE_DIR}")

target_include_directories(train PUBLIC . ${CUDNN_LIBRARIES})
#target_include_directories(train PUBLIC . ${CUDNN_LIBRARIES})