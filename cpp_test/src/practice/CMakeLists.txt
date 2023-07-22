project(CPP_PRACTICE)

find_package(OpenMP REQUIRED)

add_executable(pointer.o pointer.cpp)
add_executable(condition.o condition.cpp)

add_executable(loopop.o loopop.cpp)
target_link_libraries(loopop.o PUBLIC OpenMP::OpenMP_CXX)

add_executable(multiprocess.o multiprocess.cpp)
add_executable(mutex.o mutex.cpp)

add_executable(raii.o raii.cpp)

find_package(benchmark REQUIRED)
add_executable(stencil.o stencil.cpp)
target_link_libraries(stencil.o PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(stencil.o benchmark::benchmark)

# add_executable(tbb2.o tbb2.cpp)
# find_package(TBB)
# target_link_libraries(tbb2.o TBB::tbb)