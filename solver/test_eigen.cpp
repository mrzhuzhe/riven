#include <iostream>
#include <Eigen/Dense>
#include <limits.h>
#include "householder.h"

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
    std::cout << "error " << error << " u " << u  << " y \n " << y << std::endl;
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
    //mat02 = Eigen::MatrixXf::Random(dbrows, dbcols);
    mat02 << 1, 2, 3, 2, 4, 5, 3, 5, 6; // This method has the disadvantage that it will not work if the matrix does not have a single dominant eigenvalue.
    power_method(mat02, dbcols, dbrows);

    Eigen::MatrixXf mat03(dbrows, dbcols);
    mat03 = Eigen::MatrixXf::Random(dbrows, dbcols);
    householder(mat03, rows, cols);

    return 0;
}