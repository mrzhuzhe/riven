cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CUDA_ARCHITECTURES 80)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

project(hellocuda LANGUAGES CXX CUDA)



#Modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")


#set(CUDNN_LIBRARY /usr/lib/x86_64-linux-gnu/  CACHE STRING "")
#set(CUDNN_ROOT_DIR /usr/local/cudnn CACHE STRING "")
set(CUDNN_STATIC OFF )
set(CUDNN_LIBRARY /usr/lib/x86_64-linux-gnu/)
find_package(CUDNN REQUIRED)

add_executable(train train.cu layer.cu loss.cu mnist.cu network.cu)
target_include_directories(train PUBLIC . ../../include)

message(STATUS "CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CUDNN_LIBRARIES} ${CUDNN_INCLUDE_DIRS}")

target_include_directories(train PUBLIC . ${CUDNN_INCLUDE_DIRS})
target_link_libraries(train PUBLIC . ${CUDNN_LIBRARIES})
#target_include_directories(train PUBLIC . ${CUDNN_LIBRARIES})