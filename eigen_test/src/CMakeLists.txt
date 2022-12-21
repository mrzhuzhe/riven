cmake_minimum_required (VERSION 3.0)
project(Eigen_Test)

# cmake -S src -B build
# cmake --build build

# https://eigen.tuxfamily.org/dox/TopicCMakeGuide.html
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(test01.o test01.cpp)

add_executable(test02.o test02.cpp)

target_link_libraries (test01.o Eigen3::Eigen)
target_link_libraries (test02.o Eigen3::Eigen)