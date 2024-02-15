#include <iostream>
#include <Eigen/Dense>
#include <limits.h>
#include "jacobian.h"

int main() {
    int rows = 6, cols = 6;
    Eigen::MatrixXf mat01(rows, cols);
    Eigen::MatrixXf b(rows, 1);
    Eigen::MatrixXf x(rows, 1);
    mat01 = Eigen::MatrixXf::Random(rows, cols);
    b = Eigen::MatrixXf::Random(rows, 1);
    jacobian_solver(mat01, x, b, rows, cols);

    Eigen::MatrixXf x2(rows, 1);
    x2 = mat01.colPivHouseholderQr().solve(b);
    std::cout << "eigen solution\n" << x2 << std::endl;

    return 0;
}