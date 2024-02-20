#include <iostream>
#include <Eigen/Dense>
#include <limits.h>
#include "jacobian.h"

int main() {
    int rows = 16, cols = 16;
    Eigen::MatrixXf mat01(rows, cols);
    Eigen::MatrixXf b(rows, 1);
    Eigen::MatrixXf x(rows, 1);
    mat01 = Eigen::MatrixXf::Random(rows, cols);
    mat01 = mat01.transpose() * mat01;  // easy to solve positive sysmetric
    // mat01 << 10, -1 , 2, 0, 
    // -1, 11, -1,3,
    // 2, -1, 10,-1,
    // 0, -3, -1, 8;
    // b << 6, 25, -11, 15;
    b = Eigen::MatrixXf::Random(rows, 1);
    jacobian_solver(mat01, x, b, rows, cols);
    
    Eigen::MatrixXf x1(rows, 1);
    gs_solver(mat01, x1, b, rows, cols);

    Eigen::MatrixXf x2(rows, 1);
    x2 = mat01.colPivHouseholderQr().solve(b);
    std::cout << "eigen solution\n" << x2 << std::endl;

    return 0;
}