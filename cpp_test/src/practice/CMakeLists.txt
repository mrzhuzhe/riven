project(CPP_PRACTICE)

find_package(OpenMP REQUIRED)

add_executable(pointer.o pointer.cpp)
add_executable(condition.o condition.cpp)

add_executable(loopop.o loopop.cpp)
target_link_libraries(loopop.o PUBLIC OpenMP::OpenMP_CXX)

add_executable(multiprocess.o multiprocess.cpp)
add_executable(mutex.o mutex.cpp)

add_executable(raii.o raii.cpp)

# https://github.com/google/benchmark and tbb