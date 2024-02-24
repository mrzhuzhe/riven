#include <iostream>
#include <Eigen/Dense>
#include <limits.h>
#include "jacobian.h"
#include "multigrid.h"


void test_case(Eigen::MatrixXf mat01, int rows, int cols){
    Eigen::MatrixXf b(rows, 1);
    Eigen::MatrixXf x(rows, 1);
    b = Eigen::MatrixXf::Random(rows, 1);
    jacobian_solver(mat01, x, b, rows, cols);
    
    Eigen::MatrixXf x1(rows, 1);
    gs_solver(mat01, x1, b, rows, cols);

    Eigen::MatrixXf x2(rows, 1);
    x2 = mat01.colPivHouseholderQr().solve(b);
    //std::cout << "eigen solution\n" << x2 << std::endl;
    
    std::cout << "jacobian error " << (x - x2).maxCoeff() << " " << (x - x2).minCoeff() << std::endl;
    std::cout << "gs error " << (x1 - x2).maxCoeff() << " " << (x1 - x2).minCoeff() << std::endl;

    Eigen::MatrixXf mat01_16(rows, cols);
    Eigen::MatrixXf mat01_8(rows/2, cols/2);
    Eigen::MatrixXf mat01_4(rows/4, cols/4); 
    
    Fmat2Cmat(mat01, mat01_8, rows, cols);  // initial mat8 and mat4
    Fmat2Cmat(mat01_8, mat01_4, rows/2, cols/2);    
    //std::cout << "\n mat01_8 \n" << mat01_8 << std::endl;
    
    Cmat2Fmat(mat01_8, mat01_16, rows/2, cols/2);    // this interpolate is not good
    //std::cout << "\n mat01_16 \n" << mat01_16 << std::endl;
    
    std::cout << "\n V multi grid" << std::endl;
    //  16 -> 8 -> 4 -> 8 -> 16 
    Eigen::MatrixXf x_16(rows, 1);
    Eigen::MatrixXf x_8(rows/2, 1);
    Eigen::MatrixXf x_4(rows/4, 1);   
    Eigen::MatrixXf b_8(rows/2, 1);
    Eigen::MatrixXf b_4(rows/4, 1);
    Fmat2Cmat(b, b_8, rows, 1);  // initial mat8 and mat4
    Fmat2Cmat(b_8, b_4, rows/2, 1);    

    // 16 - 8
    jacobian_solver(mat01, x_16, b, rows, cols, 1e-2f);    
    Fmat2Cmat(x_16, x_8, rows, 1);

    // 8-4
    jacobian_solver(mat01_8, x_8, b_8, rows/2, cols/2, 1e-3f);
    Fmat2Cmat(x_8, x_4, rows/2, 1);
    
    // 4
    jacobian_solver(mat01_4, x_4, b_4, rows/4, cols/4, 1e-4f);
    
    // 4-8
    Cmat2Fmat(x_4, x_8, rows/4, 1);
    jacobian_solver(mat01_8, x_8, b_8, rows/2, cols/2, 1e-5f);

    // 8 - 16
    Cmat2Fmat(x_8, x_16, rows/2, 1);
    jacobian_solver(mat01, x_16, b, rows, cols);

    std::cout << "multi grid error " << (x_16 - x2).maxCoeff() << " " << (x_16 - x2).minCoeff() << std::endl;
}

int main() {
    int rows = 96, cols = 96;
    Eigen::MatrixXf mat01(rows, cols);
    Tridiagonal_mat(mat01, rows, cols);
    std::cout << "\n ------------------------------ test case 1 mat01.block(0, 0, 8, 8) \n" 
    << mat01.block(0, 0, 8, 8) 
    << " \n rows cols " << rows << " " << cols << std::endl;

    // mat01 << 10, -1 , 2, 0, 
    // -1, 11, -1,3,
    // 2, -1, 10,-1,
    // 0, -3, -1, 8;
    // b << 6, 25, -11, 15;
    test_case(mat01, rows, cols);

    Eigen::MatrixXf mat02(rows, cols);
    mat02 = Eigen::MatrixXf::Random(rows, cols);
    mat02 = mat02.transpose() * mat02;  // easy to solve positive sysmetric
    std::cout << "\n ------------------------------ test case2 mat02.block(0, 0, 8, 8) \n" 
    << mat02.block(0, 0, 8, 8) 
    << " \n rows cols " << rows << " " << cols << std::endl;

    test_case(mat02, rows, cols);

    return 0;
}