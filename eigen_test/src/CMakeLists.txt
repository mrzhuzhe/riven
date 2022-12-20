cmake_minimum_required (VERSION 3.0)
project(Eigen_Test)

# cmake -S src -B build
# cmake --build build
 
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(test01.o test01.cpp)

target_link_libraries (test01.o Eigen3::Eigen)