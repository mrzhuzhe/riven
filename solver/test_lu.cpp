//  https://johnfoster.pge.utexas.edu/numerical-methods-book/LinearAlgebra_LU.html lu solver
#include <iostream>
#include <Eigen/Dense>
#include "lu.h"
#include "plu.h"

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
    int dbrow = 2*rows;
    int dbcols = 2*cols;
    Eigen::MatrixXf mat03(dbrow, dbcols);
    mat03 = Eigen::MatrixXf::Random(dbrow, dbcols);
    Eigen::MatrixXf U_mat(dbrow, dbcols);
    Eigen::MatrixXf L_mat(dbrow, dbcols);
    Eigen::MatrixXf P_mat(dbrow, dbcols);  // P matrix and be store as spatial

    Eigen::MatrixXf x2(dbrow, 1);
    Eigen::MatrixXf y_hat(dbrow, 1);
    Eigen::MatrixXf b2(dbrow, 1);
    b2 = Eigen::MatrixXf::Random(dbrow, 1);

    // this will cause a nan
    for (int i =0; i < dbrow-1; i++) {
        mat03(1, i) = 0;
    }
    // this case can be a showoff for low pivot
    // for (int i =0; i < dbrow; i++) {
    //     mat03(3, i) = 0;
    // }
    for (int i =0; i < dbrow-2; i++) {
        mat03(3, i) = 0;
    }
    plu_factor(mat03, dbrow, dbcols, U_mat, L_mat, P_mat);

    std::cout << "\n U:\n" << U_mat << std::endl;
    std::cout << "\n L:\n" << L_mat << std::endl;
    std::cout << "\n P:\n" << P_mat << std::endl;   
    std::cout << "\n origin:\n" <<  P_mat * mat03 << std::endl; 
    std::cout << "\n solution :\n" <<  L_mat * U_mat << std::endl; 

    Eigen::MatrixXf pb2(dbrow, 1);
    pb2 = P_mat * b2;
    solve_l(
        L_mat, 
        y_hat,
        pb2,
        dbrow,
        dbcols
    );
    
    solve_u(U_mat, 
        x2,
        y_hat,
        dbrow,
        dbcols);

    Eigen::MatrixXf x3(dbrow, 1);
    x3 = mat03.colPivHouseholderQr().solve(b2);
    std::cout << "x3 \n" << x3 << std::endl;

    std::cout << "x2 \n" << x2 << std::endl;

    return 0;
}