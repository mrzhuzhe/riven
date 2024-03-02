#pragma once
#include "cg.h"

//  https://en.wikipedia.org/wiki/Arnoldi_iteration
void arnoldi(const Eigen::MatrixXf& A, Eigen::MatrixXf& b, Eigen::MatrixXf& Q , int rows, int cols, int k, Eigen::MatrixXf& H, float tol=FLT_EPSILON){
    int bcols = b.cols();
    for (int i=0;i<bcols;i++){
        Q.col(i) = b / getNorm(b.col(i));
    }
    Q.col(k+1) = A * Q.col(k);
    for (int i=0;i<=k;i++){ // Notice this is a eq
        H(i, k) = Q.col(k+1).transpose() * Q.col(i);
        Q.col(k+1) -= H(i, k) * Q.col(i);
    }
    H(k+1, k) = getNorm(Q.col(k+1));
    Q.col(k+1) /= H(k+1, k);
}

std::tuple<float, float> given_rotation(float v1, float v2) {
    float t = std::sqrt(v1*v1+v2*v2);
    float cs = v1 / t;
    float sn = v2 / t;
    return std::tuple<float, float>{cs, sn};
}

void apply_givens_rotation(Eigen::MatrixXf& H, Eigen::VectorXf& cs, Eigen::VectorXf& sn, int k) {
    float temp = 0;
    for (int i = 0;i<k-1;i++){
        temp = cs(i) * H(i, k) + sn(i) *H(i+1, k);
        H(i+1,k) = -sn(i) * H(i, k) + cs(i) * H(i+1, k);
        H(i, k) = temp;
    }
    std::tuple<float, float> temp_tuple = given_rotation(H(k , k), H(k + 1, k));
    float cs_k = std::get<0>(temp_tuple);
    float sn_k = std::get<1>(temp_tuple);
    cs(k) = cs_k;
    sn(k) = sn_k;
    H(k,k) = cs_k * H(k,k) + sn_k * H(k+1,k);
    H(k+1,k) = 0;
}

void test_arnoldi(const Eigen::MatrixXf& A, Eigen::MatrixXf& b, int rows, int cols, float tol=FLT_EPSILON){
    int max_iteration = 8;
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

    std::tuple<float, float> res = given_rotation(1.0, 2.0);
    std::cout << std::get<0>(res) << " " << std::get<1>(res) << std::endl;
}

// todo test bcols > 1
void gmres(const Eigen::MatrixXf& A, Eigen::MatrixXf& x, Eigen::MatrixXf& b, int rows, int cols, float tol=FLT_EPSILON) {
    int max_iteration = 1000;
    int bcols = b.cols();
    
    Eigen::MatrixXf residual(rows, bcols);
    residual = b - A * x;
    float bnorm = getNorm(b);
    float rnorm = getNorm(residual);
    float error = rnorm / bnorm;

    Eigen::VectorXf sn(max_iteration);
    Eigen::VectorXf cs(max_iteration);
    Eigen::VectorXf e1(max_iteration+1);
    e1(0) = 1.0f;
    Eigen::VectorXf beta(max_iteration+1);
    Eigen::MatrixXf Q(rows, (max_iteration+1)*bcols);
    Eigen::MatrixXf H(max_iteration+1, max_iteration+1);
    for (int i=0;i<bcols;i++){
        Q.col(i) = residual / rnorm;
    }
    beta(0) = rnorm * e1(0);
    int iter_count = 0;
    while(iter_count < max_iteration){
        arnoldi(A, b, Q, rows, cols, iter_count, H);
        // eliminate the last element in H ith row and update the rotation matrix
        apply_givens_rotation(H, cs, sn, iter_count);

        beta(iter_count + 1) = -sn(iter_count) * beta(iter_count);
        beta(iter_count)     = cs(iter_count) * beta(iter_count);
        error = std::abs(beta(iter_count + 1)) / bnorm;
        iter_count++;
        if (error <= tol) {
            std::cout << "gmres break iter_count:" << iter_count << std::endl;
            break;
        }
    }
    // Todo ugly implement
    Eigen::MatrixXf Y(iter_count, iter_count);
    Y = H.block(0, 0, iter_count, iter_count).inverse() * beta.head(iter_count);
    x = x + Q.block(0, 0, rows, iter_count) * Y;
}