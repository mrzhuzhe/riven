//  https://johnfoster.pge.utexas.edu/numerical-methods-book/LinearAlgebra_IterativeSolvers.html
#pragma once
#include <float.h>

void jacobian_solver(const Eigen::MatrixXf& A, Eigen::MatrixXf& x,const Eigen::MatrixXf& b, int rows, int cols, float tol=FLT_EPSILON ){
    std::cout << "jacobian_solver" << std::endl;
    std::cout << "A\n" << A << std::endl;    
    std::cout << "b\n" << b << std::endl;

    Eigen::MatrixXf T(rows, cols);
    Eigen::MatrixXf old(rows, cols);

    for (int j=0;j<cols;j++){
        for (int i=0;i<rows;i++){
            if (i==j) {
                T(i, j) = 0.0f;
            } else {
                T(i, j) = A(i, j);
            }
        }
    }
    std::cout << "T\n" << T << std::endl;
    const int max_iter = 10000;
    float temp = 0, max_diff_norm=0, max_norm=0;
    for (int iter=0; iter<max_iter; iter++){
        old = x;    // this is deep copy in eigen
        for (int i=0; i < rows; i++){
            temp = 0;
            for (int j=0;j<cols;j++){
                temp += T(i, j) * x(j);
            }
            x(i) = b(i) - temp;
            x(i) /= A(i, i);        
        }
        max_diff_norm=0;
        max_norm=0;
        for (int i=0;i<rows;i++){
            temp = std::abs(x(i));
            max_norm = temp > max_norm ? temp : max_norm;
            temp = std::abs(x(i) - old(i));
            max_diff_norm = temp > max_diff_norm ? temp : max_diff_norm;            
        }
        if ( max_diff_norm / max_norm < tol ) {
            std::cout << "break " << iter << std::endl;
            break;
        }
    }
    std::cout << "x\n" << x << std::endl;
}

void gs_solver(const Eigen::MatrixXf& A, Eigen::MatrixXf& x,const Eigen::MatrixXf& b, int rows, int cols, float tol=FLT_EPSILON){
    std::cout << "gs_solver" << std::endl;
    std::cout << "A\n" << A << std::endl;
    std::cout << "b\n" << b << std::endl;

    Eigen::MatrixXf old(rows, cols);

    const int max_iter = 10000;
    float temp = 0, max_diff_norm=0, max_norm=0;
    for (int iter=0; iter<max_iter; iter++){
        old = x;    // this is deep copy in eigen
        for (int i=0; i < rows; i++){
            temp = 0;
            for (int j=0;j<i;j++){
                temp += A(i, j) * x(j);
            }
            for (int j=i+1;j<cols;j++){
                temp += A(i, j) * old(j);
            }
            x(i) = b(i) - temp;
            x(i) /= A(i, i);        
        }
        max_diff_norm=0;
        max_norm=0;
        for (int i=0;i<rows;i++){
            temp = std::abs(x(i));
            max_norm = temp > max_norm ? temp : max_norm;
            temp = std::abs(x(i) - old(i));
            max_diff_norm = temp > max_diff_norm ? temp : max_diff_norm;            
        }
        if ( max_diff_norm / max_norm < tol ) {
            std::cout << "break " << iter << std::endl;
            break;
        }
    }

    std::cout << "x\n" << x << std::endl;
}