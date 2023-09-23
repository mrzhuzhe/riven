#include <iostream>
#include "libtest.h"
#include <Eigen/Dense>

template <typename T>
void test_lib(T val){
    printf("I'm lib val is %f \n", val);
    
    int rows = 2, cols = 2;
    Eigen::MatrixXd mat01(rows, cols);
    mat01 = Eigen::MatrixXd::Random(rows, cols);
    std::cout << mat01 << std::endl;

}

template void test_lib(float val);