#include <iostream>
#include <Eigen/Dense>
#include <limits.h>
#include "householder.h"
#include "qr.h"

int find_maxnorm_index(Eigen::MatrixXf x, int rows) {
    float max = 0;
    float abs = 0;
    int p = 0; 
    for (int i=0; i< rows;i++){
        abs = std::abs(x(i)); 
        max = abs > max ? abs : max; 
        //std::cout << x(i) << " " << max << " " ;
    }
    for (int i=0; i< rows;i++){
        p = i;
        if (std::abs(std::abs(x(i)) - max) < __FLT_EPSILON__) {
            break;
        }
    }
    return p;
}

void power_method(Eigen::MatrixXf& A, int cols, int rows) {
    Eigen::MatrixXf x(rows, 1);
    Eigen::MatrixXf y(rows, 1);
    float u=0;
    int p=0;
    float error=1;
    for (int i=0;i<cols;i++){
        x(i) = 1.0;
    };
    int iteration=1e4;
    float tolerance=1e-10;
    float temp = 0, max = 0;
    for(int i=0;i<iteration;i++){
        if (error < tolerance){
            break;
            std::cout << "early stoped " << "i " << i << "error " << error << "u " << u << std::endl;
            std::cout << "x " << x << std::endl;
        }
        y = A * x;
        u = y(p);
        p = find_maxnorm_index(y, cols);
        for (int j=0; j< cols;j++){
            temp = std::abs(x(j) - y(j) / y(p)); 
            max = temp > max ? temp : max;
            error = max;
            x(j) = y(j) / y(p);
        }        
    }
    std::cout << "error " << error << " u " << u  << " \n y \n " << y << std::endl;
    std::cout << "x \n" << x << std::endl;

    std::cout << "A * x \n" << A * x << std::endl;
    std::cout << "u * x \n" << u * x << std::endl;
}

int main(){
    int rows = 6, cols = 6;
    Eigen::MatrixXf mat01(rows, 1);
    mat01 = Eigen::MatrixXf::Random(rows, 1);
    int p = find_maxnorm_index(mat01, rows);    
    std::cout << "p "  << p << std::endl;
    
    int dbrows=3, dbcols=3;
    Eigen::MatrixXf mat02(dbrows, dbcols);
    mat02 = Eigen::MatrixXf::Random(dbrows, dbcols);
    //mat02 << 1, 2, 3, 2, 4, 5, 3, 5, 6; // This method has the disadvantage that it will not work if the matrix does not have a single dominant eigenvalue.
    power_method(mat02, dbcols, dbrows);
    std::cout << "mat02.eigenvalues() \n"<< mat02.eigenvalues() << std::endl;

    dbrows=4, dbcols=4;
    Eigen::MatrixXf mat03(dbrows, dbcols);
    Eigen::MatrixXf mat04(dbrows, dbcols);
    Eigen::MatrixXf mat05(dbrows, dbcols);
    mat03 = Eigen::MatrixXf::Random(dbrows, dbcols);
    //mat03 = mat03.transpose() * mat03;
    mat04 = mat03;
    // for (int i=0;i<dbcols;i++){
    //     for (int j=0;j<dbrows;j++){
    //         mat03(j,i) = mat04(j, i) = std::abs(static_cast <float> (rand()) / static_cast <float> (RAND_MAX));
    //     }
    // }
    //mat04 << 4, 1, -2, 2,    1, 2, 0, 1,    -2, 0, 3, -2,   2, 1, -2, -1;
    //mat03 = mat04;
    std::cout << "origin matrix \n " << mat04 << std::endl;
    householder(mat03, dbrows, dbcols);
    std::cout << "householder triangular matrix \n" << mat03 << std::endl;

    std::cout << "mat03 eigen value \n" << mat03.eigenvalues() << std::endl;

    std::cout << "mat04 eigen value \n" << mat04.eigenvalues() << std::endl;
    
    // mat05 << 4.0f, -3.0f, 0.0f, 0.0f,
    // -3.0f, (10.f/3.f), (-5.f/3.f), 0.0f,    
    // 0.0f, (-5.f/3.f), (-33.f/25.f), (68.f/75.f),   
    // 0.0f, 0.0f, (68.f/75.f), (149.f/75.f);
    mat05 = Eigen::MatrixXf::Random(dbrows, dbcols);
    mat05 = mat05.transpose() * mat05;
    
    std::cout << "mat05 \n" << mat05 << std::endl;
    std::cout << "mat05 eigen value \n" << mat05.eigenvalues() << std::endl;


    Eigen::MatrixXf mat06(dbrows, dbcols);
    mat06 = mat05;
    Eigen::MatrixXf Qmat06(dbrows, dbcols);
    gram_schmidt(mat06, Qmat06, dbrows, dbcols);
    std::cout << "Qmat06\n" << Qmat06  << std::endl;
    std::cout << "R\n" << Qmat06.transpose() * mat06  << std::endl;

    std::cout << "A = Qmat06 * (Qmat06.transpose() * mat06)\n" << Qmat06 * (Qmat06.transpose() * mat06)  << std::endl;
    std::cout << "mat06 \n" << mat06 << std::endl;

    qr_eigen(mat06, dbrows, dbcols);
    // std::cout << "Q * Q \n" << Qmat06 * Qmat06 << std::endl;
    return 0;
}