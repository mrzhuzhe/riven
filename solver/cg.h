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

void ichl(Eigen::MatrixXf& mat, int rows, int cols){
    for (int k=0;k<cols;k++){
        mat(k, k) = std::sqrt(mat(k, k));
        for (int i=k+1;i<rows;i++){
            if (mat(i, k)!=0) {
                mat(i, k) /= mat(k, k);
            }            
        }
        for (int j=k+1;j<cols;j++){
            for (int i=j;i<rows;i++){
                if (mat(i, j)!=0) {
                    mat(i, j) -= mat(i, k) * mat(j,k);
                }
            }
        }
        for (int j=0;j<cols;j++){
            for (int i=0;i<j;i++){
                mat(i, j) = 0;
            }
        }
    }
}

void cg(const Eigen::MatrixXf& mat, int rows, int cols, Eigen::MatrixXf& x, const Eigen::MatrixXf& b, float tol=FLT_EPSILON){

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
    float bnorm_tol = tol*getNorm(b);
    //while ((old_sqr_resid_norm.maxCoeff() > tol) && (iter_count < 100)) {
    //while ((old_sqr_resid_norm.maxCoeff() > tol)) {
    while ((std::sqrt(old_sqr_resid_norm.sum()) > bnorm_tol) && (iter_count < 10000)) {
        iter_count++;
        A_search_direction = mat * search_direction;
        step_size = (search_direction.transpose() * A_search_direction);
        //std::cout << (old_sqr_resid_norm) << "  \n\n" <<  step_size << std::endl; //  must be othogonal to preview 
        for (int j=0;j<bcols;j++) {
            for (int i=0;i<bcols;i++) {
                //step_size(i, j) =  (old_sqr_resid_norm(i, j) * old_sqr_resid_norm(i, j)) / step_size(i, j);
                step_size(i, j) = step_size(i, j)!=0 ? (old_sqr_resid_norm(i, j)) / step_size(i, j) : (old_sqr_resid_norm(i, j)) ;
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


//  https://www.cse.psu.edu/~b58/cse456/lecture20.pdf
void pcg(const Eigen::MatrixXf& mat, int rows, int cols, Eigen::MatrixXf& x, const Eigen::MatrixXf& b, const Eigen::MatrixXf&  M, float tol=FLT_EPSILON ){

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

    Eigen::MatrixXf M_inv(rows, cols);
    //M_inv = M.inverse(); 
    cg(M, rows, cols, M_inv, Eigen::MatrixXf::Identity(rows, cols));
    std::cout << "diff " << M_inv - M.inverse()<< std::endl;
    Eigen::MatrixXf Z(brows, bcols); 

    residual =  b - mat * x;
    Z = M_inv * residual; // Notice this is only good for jacobian can be further optimized
    search_direction = Z;
    old_sqr_resid_norm = residual.transpose() * Z;
    int iter_count = 0;
    float bnorm_tol = tol*getNorm(b);
    while ((std::sqrt(old_sqr_resid_norm.sum()) > bnorm_tol) && (iter_count < 10000)) {
        iter_count++;
        A_search_direction = mat * search_direction;
        step_size = (search_direction.transpose() * A_search_direction);
        for (int j=0;j<bcols;j++) {
            for (int i=0;i<bcols;i++) {
                step_size(i, j) =  (old_sqr_resid_norm(i, j)) / step_size(i, j);
            }
        }        
        x += search_direction * step_size;
        residual -= A_search_direction * step_size;
        Z = M_inv * residual;
        new_sqr_resid_norm = residual.transpose() * Z;
        for (int j=0;j<bcols;j++) {
            for (int i=0;i<bcols;i++) {                
                step_size(i, j) = old_sqr_resid_norm(i, j)!=0?(new_sqr_resid_norm(i, j) / old_sqr_resid_norm(i, j)):new_sqr_resid_norm(i, j);
            }
        }
        search_direction = Z + search_direction * step_size;
        old_sqr_resid_norm = new_sqr_resid_norm;
    }
    std::cout << "pcg break! iter_count: " << iter_count << std::endl;

}