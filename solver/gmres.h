/*
function [h, q] = arnoldi(A, Q, k)
  q = A*Q(:,k);   % Krylov Vector
  for i = 1:k     % Modified Gram-Schmidt, keeping the Hessenberg matrix
    h(i) = q' * Q(:, i);
    q = q - h(i) * Q(:, i);
  end
  h(k + 1) = norm(q);
  q = q / h(k + 1);
end
*/
#pragma once
#include "cg.h"

//  https://en.wikipedia.org/wiki/Arnoldi_iteration
void arnoldi(const Eigen::MatrixXf& A, Eigen::MatrixXf& b, Eigen::MatrixXf& Q , int rows, int cols, int k, Eigen::MatrixXf& H, float tol=FLT_EPSILON){
    int bcols = b.cols();
    for (int i=0;i<bcols;i++){
        Q.col(i) = b / getNorm(b.col(i));
    }
    Q.col(k+1) = A * Q.col(k);
    for (int i=0;i<=k;i++){
        H(i, k) = Q.col(k+1).transpose() * Q.col(i);
        Q.col(k+1) -= H(i, k) * Q.col(i);
    }
    H(k+1, k) = getNorm(Q.col(k+1));
    std::cout << Q.col(k+1) << std::endl;
    Q.col(k+1) /= H(k+1, k);
}

void test_arnoldi(const Eigen::MatrixXf& A, Eigen::MatrixXf& b, int rows, int cols, float tol=FLT_EPSILON){
    int max_iteration = 32;
    int bcols = b.cols();
    Eigen::MatrixXf Q(rows, (max_iteration+1)*bcols);
    Eigen::MatrixXf H(max_iteration+1, max_iteration+1);
    for (int iter=0;iter<max_iteration;iter++){   // in out side
        arnoldi(A, b, Q, rows, cols, iter, H);
        if (H(iter+1, iter) <= tol ){
            std::cout << "test_arnoldi break iter:" << iter << std::endl;
            break;
        }
    }
    std::cout << "H \n" << H << "\n Q \n" << Q << std::endl;
}