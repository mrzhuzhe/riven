cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

project(helloopenmp LANGUAGES CXX)

find_package(OpenMP REQUIRED)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE OpenMP::OpenMP_CXX)

#   this is wrong
if (APPLE) 
    set(CMAKE_C_FLAGS "-mfloat-abi=softfp -mfpu=neon")  
    set(CMAKE_CXX_FLAGS "-mfloat-abi=softfp -mfpu=neon")  
    #add_compile_options(-mfloat-abi=softfp -mfpu=neon)  
    MESSAGE(STATUS "this is APPLE")
    add_executable(neon neon.cpp)
elseif (UNIX) 
    MESSAGE(STATUS "this is UNIX")
    add_executable(sse sse.cpp)
endif()