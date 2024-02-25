#include <float.h>
#pragma once

void cg(const Eigen::MatrixXf& mat, int rows, int cols, Eigen::MatrixXf& x, const Eigen::MatrixXf& b, float tol=FLT_EPSILON ){
    int brows = b.rows();
    int bcols = b.cols();
    Eigen::MatrixXf search_direction(brows, bcols); 
    Eigen::MatrixXf residual(brows, bcols); 
    Eigen::MatrixXf A_search_direction(brows, bcols); // brows == A.rows()
    Eigen::MatrixXf step_size(bcols, bcols);
    Eigen::MatrixXf old_sqr_resid_norm(bcols, bcols);
    Eigen::MatrixXf new_sqr_resid_norm(bcols, bcols);

    search_direction = residual=  b - mat * x;
    old_sqr_resid_norm = residual.transpose() * residual;
    int iter_count = 0;
    while (old_sqr_resid_norm.maxCoeff() > tol) {
        iter_count++;
        A_search_direction = mat * search_direction;
        step_size =  (old_sqr_resid_norm * old_sqr_resid_norm) / (search_direction.transpose() * A_search_direction);
        x += A_search_direction * step_size;
        residual -= A_search_direction * step_size;
        new_sqr_resid_norm = residual.transpose() * residual;
        search_direction = residual + A_search_direction * std::pow(new_sqr_resid_norm / old_sqr_resid_norm, 2) ;
        old_sqr_resid_norm = new_sqr_resid_norm;
    }
    std::cout << "cg break! iter_count: " << iter_count << std::endl;

}