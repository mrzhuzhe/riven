#include <iostream>
#include <Eigen/Dense>
#include "multigrid.h"
#include "cg.h"

int main(int argc, char *argv[]) {
    int size = 64;
    if (argc > 1) {
        size = atoi(argv[1]);
    }
    int rows = size, cols = size;
    Eigen::MatrixXf mat01(rows, cols);
    Eigen::MatrixXf b(rows, 1);
    Eigen::MatrixXf x(rows, 1);
    b = Eigen::MatrixXf::Random(rows, 1);

    int test_type = 0;
    if (argc > 2) {
        test_type = atoi(argv[2]);
    }
    mat01 = Eigen::MatrixXf::Random(rows, cols);
    switch (test_type)
    {
    case 1:
        Tridiagonal_mat(mat01, rows, cols);
        break;
    case 2:
        mat01 = mat01.transpose() * mat01;
        break;
    default:
        break;
    }   

    Eigen::MatrixXf x2(rows, 1);
    x2 = mat01.colPivHouseholderQr().solve(b);
    //std::cout << "eigen solution\n" << x2 << std::endl;

    std::cout << "\n ------------------------------ test cg \n" << std::endl;
    //  CG 
    Eigen::MatrixXf cg_x(rows, 1);
    cg_x.setZero(rows, 1);
    cg(mat01, rows, cols, cg_x, b);
    //std::cout << cg_x << std::endl;
    std::cout << "cg error " << (cg_x - x2).maxCoeff() << " " << (cg_x - x2).minCoeff() << std::endl;

    //  PCG

    //  BICG
    Eigen::MatrixXf bicg_x(rows, 1);
    bicg_x.setZero(rows, 1);
    bicg(mat01, rows, cols, bicg_x, b);
    //std::cout << cg_x << std::endl;
    std::cout << "bicg error " << (bicg_x - x2).maxCoeff() << " " << (bicg_x - x2).minCoeff() << std::endl;

    //  BICGSTAB

    //  GMRES

    //  GMRES-LU0

    //  GMRES-LUT

    return 0;
}