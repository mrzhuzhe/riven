//  https://johnfoster.pge.utexas.edu/numerical-methods-book/LinearAlgebra_LU.html lu solver
#include <iostream>
#include <Eigen/Dense>
#include "lu_common.h"


int main(){
    int rows = 3, cols = 3;
    Eigen::MatrixXf mat01(rows, cols);
    //mat01 = Eigen::MatrixXf::Random(rows, cols);
    //std::cout << mat01 << std::endl;

    mat01 << 1,1,0,  2,1,-1, 3,-1,-1;
    // Eigen::Vector3f b;
    // b << 3, 3, 4;

    std::cout << mat01 << std::endl;
    lu_factor(mat01, rows, cols);
    
    // Eigen::Vector3f x = mat01.colPivHouseholderQr().solve(b);
    // std::cout << x << std::endl;
    
    // int comcol = cols+1;
    // Eigen::MatrixXf mat02(rows, comcol);

    // mat02 << mat01 , b;
    // std::cout << mat02 << std::endl;
    
    // std::cout << "random matrix test" << std::endl;

    //solve a random matrix
    Eigen::MatrixXf mat03(2*rows, 2*cols);
    mat03 = Eigen::MatrixXf::Random(2*rows, 2*cols);
    
    // this will cause a nan
    for (int i =0; i < 2*rows-1; i++) {
        mat03(1, i) = 0;
    }
    // this case can be a showoff for low pivot
    // for (int i =0; i < 2*rows; i++) {
    //     mat03(3, i) = 0;
    // }
    for (int i =0; i < 2*rows-2; i++) {
        mat03(3, i) = 0;
    }
    plu_factor(mat03, 2*rows, 2*cols);

    return 0;
}