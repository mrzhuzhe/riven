#pragma once
#include <float.h>


float getNorm(const Eigen::MatrixXf& mat){
    int rows = mat.rows();
    int cols = mat.cols();
    float sum = 0;
    for (int j=0;j<cols;j++){
        for (int i=0;i<rows;i++){
            sum += mat(i,j) * mat(i,j);
        }
    }
    return std::sqrt(sum);
}

void cg(const Eigen::MatrixXf& mat, int rows, int cols, Eigen::MatrixXf& x, const Eigen::MatrixXf& b, float tol=FLT_EPSILON ){

    // std::cout << " \n A \n " << mat << std::endl;
    // std::cout << " \n x \n " << x << std::endl;
    // std::cout << " \n b \n " << b << std::endl;

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
    //while ((old_sqr_resid_norm.maxCoeff() > tol) && (iter_count < 100)) {
    //while ((old_sqr_resid_norm.maxCoeff() > tol)) {
    while ((std::sqrt(old_sqr_resid_norm.sum()) > tol*getNorm(b)) && (iter_count < 10000)) {
        iter_count++;
        A_search_direction = mat * search_direction;
        step_size = (search_direction.transpose() * A_search_direction);
        for (int j=0;j<bcols;j++) {
            for (int i=0;i<bcols;i++) {
                //step_size(i, j) =  (old_sqr_resid_norm(i, j) * old_sqr_resid_norm(i, j)) / step_size(i, j);
                step_size(i, j) =  (old_sqr_resid_norm(i, j)) / step_size(i, j);
            }
        }        
        x += search_direction * step_size;
        residual -= A_search_direction * step_size;
        new_sqr_resid_norm = residual.transpose() * residual;
        for (int j=0;j<bcols;j++) {
            for (int i=0;i<bcols;i++) {                
                //step_size(i, j) = (new_sqr_resid_norm(i, j) / old_sqr_resid_norm(i, j)) * (new_sqr_resid_norm(i, j) / old_sqr_resid_norm(i, j));
                step_size(i, j) = (new_sqr_resid_norm(i, j) / old_sqr_resid_norm(i, j));
            }
        }
        search_direction = residual + search_direction * step_size;
        old_sqr_resid_norm = new_sqr_resid_norm;
    }
    std::cout << "cg break! iter_count: " << iter_count << std::endl;

}

/*
https://mathworld.wolfram.com/BiconjugateGradientMethod.html
wikipedia is wrong
https://www.cfd-online.com/Wiki/Biconjugate_gradient_method
*/
void bicg(const Eigen::MatrixXf& mat, int rows, int cols, Eigen::MatrixXf& x, const Eigen::MatrixXf& b, float tol=FLT_EPSILON ){

    // std::cout << " \n A \n " << mat << std::endl;
    // std::cout << " \n x \n " << x << std::endl;
    // std::cout << " \n b \n " << b << std::endl;

    int brows = b.rows();
    int bcols = b.cols();
    Eigen::MatrixXf search_direction(brows, bcols); 
    Eigen::MatrixXf residual(brows, bcols); 
    Eigen::MatrixXf A_search_direction(brows, bcols); // brows == A.rows()

    Eigen::MatrixXf search_direction_2(brows, bcols); 
    Eigen::MatrixXf residual_2(brows, bcols); 
    Eigen::MatrixXf A_search_direction_2(brows, bcols); 

    Eigen::MatrixXf step_size(bcols, bcols);
    Eigen::MatrixXf old_sqr_resid_norm(bcols, bcols);
    Eigen::MatrixXf new_sqr_resid_norm(bcols, bcols);

    search_direction_2 = search_direction = residual_2 = residual =  b - mat * x;
    old_sqr_resid_norm = residual_2.transpose() * residual;
    int iter_count = 0;
    //while ((old_sqr_resid_norm.maxCoeff() > tol) && (iter_count < 100)) {
    while ((getNorm(residual) > tol*getNorm(b)) && (iter_count < 10000)) {
        iter_count++;
        A_search_direction = mat * search_direction;
        A_search_direction_2 = mat.transpose() * search_direction_2;
        step_size = (search_direction_2.transpose() * A_search_direction);
        for (int j=0;j<bcols;j++) {
            for (int i=0;i<bcols;i++) {
                //step_size(i, j) =  (old_sqr_resid_norm(i, j) * old_sqr_resid_norm(i, j)) / step_size(i, j);
                step_size(i, j) =  (old_sqr_resid_norm(i, j)) / step_size(i, j);
            }
        }        
        x += search_direction * step_size;
        residual -= A_search_direction * step_size;
        residual_2 -= A_search_direction_2 * step_size;
        new_sqr_resid_norm = residual_2.transpose() * residual;
        for (int j=0;j<bcols;j++) {
            for (int i=0;i<bcols;i++) {                
                //step_size(i, j) = (new_sqr_resid_norm(i, j) / old_sqr_resid_norm(i, j)) * (new_sqr_resid_norm(i, j) / old_sqr_resid_norm(i, j));
                step_size(i, j) = (new_sqr_resid_norm(i, j) / old_sqr_resid_norm(i, j));
            }
        }
        search_direction = residual + search_direction * step_size;
        search_direction_2 = residual_2 + search_direction_2 * step_size;
        old_sqr_resid_norm = new_sqr_resid_norm;
    }
    std::cout << "bicg break! iter_count: " << iter_count << std::endl;

}