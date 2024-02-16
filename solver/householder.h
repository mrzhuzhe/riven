//  https://johnfoster.pge.utexas.edu/numerical-methods-book/LinearAlgebra_EigenProblem2.html
#pragma once
#include "limits.h"
//  tridiagonalize a symmetric matrix without changing its eignenvalues. 
//  I just translate this code from python and totally dont understand
void householder(Eigen::MatrixXf& A, int rows, int cols) {
    Eigen::MatrixXf U(rows, 1);
    Eigen::MatrixXf V(rows, 1);
    Eigen::MatrixXf Z(rows, 1);
    Eigen::MatrixXf eyes(rows, cols);
    Eigen::MatrixXf Q(rows, cols);

    for (int j=0; j < cols; j++){
        for (int i=0; i < rows; i++){
            if (i == j){
                eyes(i, j) =1.0;
            } else {
                eyes(i, j) =0.0;
            }
        }
    }

    float alpha = 0;
    float two_r_squared =0;
    float UV_dot, AV_dot;
    for (int i=0; i < rows-2; i++){
        alpha=0;
        for (int j = i+1;j < rows; j++){
            alpha += A(j, i) * A(j, i);
        }
        alpha = -1.0f * std::sqrt(alpha);
        if (std::abs(A(i+1, i)) > __FLT_EPSILON__ ) {                     
            alpha *= (A(i+1, i) >= 0 ? 1.0f : -1.0f);
        }
        // std::cout << " i:"<< i << " alpha:" << alpha << " ";
        two_r_squared = alpha * alpha - alpha * A(i+1, i);
        V(i) = 0.0f;
        V(i+1) = A(i+1, i) - alpha;
        for (int j=i+2;j<rows;j++) {
            V(j) = A(j, i);
        }
        Q = (eyes - (V * V.transpose()) / (two_r_squared));
        // Q and V matrix is done
        // reffer to" q1 q2 https://en.wikipedia.org/wiki/Householder_transformation
        std::cout << "Q" << i << " V*VT \n" <<  Q << std::endl;
    
        A = Q * A * Q;
        
        std::cout << "Q*Q \n" << Q * Q << std::endl;

        // triangular A matrix start this part is equal to A = Q * A * Q
        // for (int j=i;j<rows;j++){
        //     AV_dot = 0;
        //     for (int k=i+1; k < cols; k++){
        //         AV_dot += A(j,k) * V(k);
        //     }            
        //     U(j) = 1 / two_r_squared * AV_dot;   
        // }
        // UV_dot = 0;
        // for (int j=0;j<rows;j++){
        //      UV_dot += U(j) * V(j);
        // }       
        // for (int j=i;j<rows;j++){
        //     Z(j) = U(j) - UV_dot / (2.0f * two_r_squared) * V(j);
        // }
        // for (int j = i+1; j < rows-1;j++){
        //     for (int k = j+1; k< rows; k++){
        //         A(k, j) = A(k, j) - V(j) * Z(k) - V(k) * Z(j); 
        //         A(j, k) = A(k, j);
        //     }
        //     A(j, j) -= 2 * V(j) * Z(j);
        // }
        // A(rows-1, rows-1) -= 2 * V(rows-1) * Z(rows-1);
        // for (int j = i+2; j < rows; j++){
        //     A(i, j) =0;
        //     A(j, i) =0;
        // }
        // A(i+1, i) -= V(i+1) * Z(i);
        // A(i, i+1) = A(i+1, i);   
        //  triangular A matrix end    
    }
    std::cout << "householder \n" << A << std::endl;    
} 