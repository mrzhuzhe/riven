cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_C_COMPILER "/usr/bin/clang")
set(CMAKE_CXX_COMPILER "/usr/bin/clang++")

project(helloopenmp)

#set(CMAKE_C_FLAGS "-mfloat-abi=hard -mfpu=neon")  
#set(CMAKE_CXX_FLAGS "-mfloat-abi=hard -mfpu=neon")  

find_package(OpenMP REQUIRED)

add_executable(main main.cpp)
target_link_libraries(main PUBLIC OpenMP::OpenMP_CXX)

#   this is wrong
if (APPLE) 
    #set(CMAKE_C_FLAGS "-mfloat-abi=softfp -mfpu=neon")  
    #set(CMAKE_CXX_FLAGS "-mfloat-abi=softfp -mfpu=neon")  
    #add_compile_options(-mfloat-abi=softfp -mfpu=neon)  
    MESSAGE(STATUS "this is APPLE")
    add_executable(neon neon.cpp)
    #target_compile_options(neon PUBLIC -mfloat-abi=softfp -mfpu=neon)

elseif (UNIX) 
    MESSAGE(STATUS "this is UNIX")
    add_executable(sse sse.cpp)
endif()