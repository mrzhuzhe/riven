#pragma once
#include "limits.h"

//Continuing in this manner, the tridiagonal and symmetric matrix  A(n−1) is formed
void householder(Eigen::MatrixXf A, int rows, int cols) {
    Eigen::MatrixXf U(rows, 1);
    Eigen::MatrixXf V(rows, 1);
    Eigen::MatrixXf Z(rows, 1);
    float alpha = 0;
    float two_r_squared =0;
    float UV_dot, AV_dot;
    for (int i=0; i < rows-2; i++){
        for (int j = i+1;j < rows; j++){
            alpha += A(j, i) * A(j, i);
        }
        if (std::abs(A(i+1, i) < __FLT_EPSILON__ )) {            
            alpha = -1.0f * std::sqrt(alpha);
        } else {           
            alpha = -1.0f * std::sqrt(alpha);
            alpha *= A(i+1, i) >= 0 ? 1 : -1;
        }
        two_r_squared = alpha * alpha - alpha * A(i+1, i);
        V(i) = 0.0f;
        V(i+1) = A(i+1, i) - alpha;
        for (int j=i+2;j<rows-2;j++) {
            V(j) = A(j, i);
        }
        for (int j=i;j<rows;j++){
            AV_dot = 0;
            for (int k=i; i < rows-1; i++){
                AV_dot += A(j,k+1) * V(k+1);
            }            
            U(i) = 1 / two_r_squared * AV_dot;            
        }
        //  Todo uv dot
        UV_dot = 0;
        for (int j=0;j<rows;j++){
             UV_dot += U(j) * V(j);
        }       
        for (int j=i;j<rows;j++){
            Z(i) = U(i) - UV_dot / 2.0f * two_r_squared * V(i);
        }
        for (int j = i+1; j < rows -1;j++){
            for (int k = j+1; k< rows-2; k++){
                A(k, j) -= V(j) * Z(k) - V(k) * Z(j); 
                A(j, k) = A(k, j);
            }
            A(j, j) -= 2 * V(j) * Z(j);
        }
        A(-1, -1) -= 2 * V(-1) * Z(-1);
        for (int j = i+2; j < rows-2; j++){
            A(i, j) =0;
            A(j, i) =0;
        }
        A(i+1, i) -= V(i+1) * Z(i);
        A(i, i+1) = A(i+1, i);
    }
    std::cout << "householder" << std::endl;
} 