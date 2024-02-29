#include <iostream>
#include <Eigen/Dense>
#include "multigrid.h"
#include "cg.h"
#include "bicg.h"

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

    x = mat01.colPivHouseholderQr().solve(b);
    //std::cout << "eigen solution\n" << x << std::endl;

    std::cout << "\n ------------------------------ test cg \n" << std::endl;
    //  CG 
    Eigen::MatrixXf cg_x(rows, 1);
    cg_x.setZero(rows, 1);
    cg(mat01, rows, cols, cg_x, b);
    //std::cout << cg_x << std::endl;
    std::cout << "cg error " << (cg_x - x).maxCoeff() << " " << (cg_x - x).minCoeff() << std::endl;

    //  PCG
    Eigen::MatrixXf pcg_x(rows, 1);
    pcg_x.setZero(rows, 1);
    Eigen::MatrixXf jacobian_M(rows, cols);
    // jacobian precondition
    for (int j=0;j<cols;j++){
        for (int i=0;i<rows;i++){
            if (i==j) {
                jacobian_M(i,j) = mat01(i,j);
            } else {
                jacobian_M(i,j) = 0;
            }
        }
    }
    pcg(mat01, rows, cols, pcg_x, b, jacobian_M);
    //std::cout << pcg_x << std::endl;
    std::cout << "pcg jacobian_M error " << (pcg_x - x).maxCoeff() << " " << (pcg_x - x).minCoeff() << std::endl;
    
    //  BICG
    Eigen::MatrixXf bicg_x(rows, 1);
    bicg_x.setZero(rows, 1);
    bicg(mat01, rows, cols, bicg_x, b);
    //std::cout << bicg_x << std::endl;
    std::cout << "bicg error " << (bicg_x - x).maxCoeff() << " " << (bicg_x - x).minCoeff() << std::endl;

    //  BICGSTAB

    //  GMRES

    //  GMRES-LU0

    //  GMRES-LUT

    return 0;
}