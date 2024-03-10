//  https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method
#pragma once
#include <float.h>
#include "cg.h"

void bicg_stab(const Eigen::MatrixXf& mat, int rows, int cols, Eigen::MatrixXf& x, const Eigen::MatrixXf& b, float tol=FLT_EPSILON ){

    // std::cout << " \n A \n " << mat << std::endl;
    // std::cout << " \n x \n " << x << std::endl;
    // std::cout << " \n b \n " << b << std::endl;
    tol = 1e-4;    
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
    
    Eigen::MatrixXf h(x.rows(), x.cols());
    Eigen::MatrixXf s(brows, bcols);
    Eigen::MatrixXf s2(brows, bcols);
    Eigen::MatrixXf t(brows, bcols);
    Eigen::MatrixXf t2(brows, bcols);
    Eigen::MatrixXf w(bcols, bcols);
    Eigen::MatrixXf w2(bcols, bcols);
    Eigen::MatrixXf alpha_div_w(bcols, bcols);
    Eigen::MatrixXf alpha_div_w2(bcols, bcols);

    search_direction_2 = search_direction = residual_2 = residual =  b - mat * x;
    old_sqr_resid_norm = residual_2.transpose() * residual;
    int iter_count = 0;
    float temp0 = 0, temp1 = 0, temp2 = 0, temp3 = 0;
    float bnorm = getNorm(b);
    //std::cout << tol*bnorm << std::endl;

    while ((iter_count < 10000)) {
    //while ((getNorm(residual) > tol*bnorm) && (iter_count < 10000)) {
        iter_count++;
        A_search_direction = mat * search_direction;
        A_search_direction_2 = mat.transpose() * search_direction_2;
        step_size = (residual_2.transpose() * A_search_direction);
        for (int j=0;j<bcols;j++) {
            for (int i=0;i<bcols;i++) {
                step_size(i, j) =  (old_sqr_resid_norm(i, j)) / step_size(i, j);    // alpha
            }
        }
        h = x + search_direction * step_size;  
        s = residual - A_search_direction * step_size; 
        s2 = residual_2 - A_search_direction_2 * step_size;
        std::cout << "s " << getNorm(s) << std::endl;
        if (getNorm(s) <= tol*bnorm) {
            x = h;
            std::cout << "s earlier break" << std::endl;
            break;
        }
        t = mat * s;
        t2 = mat * s2;
        for (int j=0;j<bcols;j++) {
            for (int i=0;i<bcols;i++) {
                temp0 = 0;
                temp1 = 0;
                temp2 = 0;
                temp3 = 0;
                for (int k=0;k<brows;k++){
                    temp0 += t(k, i) * s(k, j);
                    temp1 += t(k, i) * t(k, j); 
                    temp2 += t2(k, i) * s2(k, j);
                    temp3 += t2(k, i) * t2(k, j); 
                }
                w(i, j) = temp0 / temp1;    // beta
                w2(i, j) = temp2 / temp3;    // beta
            }
        }  
        x = h + s * w;  
        residual = s - t * w; 
        std::cout << getNorm(residual) << std::endl;
        if (getNorm(residual) <= tol*bnorm) {
            std::cout << "residual earlier break" << std::endl;
            break;
        }
        residual_2 = s2 - t2 * w2;
        new_sqr_resid_norm = residual_2.transpose() * residual;
        for (int j=0;j<bcols;j++) {
            for (int i=0;i<bcols;i++) {   
                alpha_div_w(i, j) = step_size(i, j) / w(i, j);      
                alpha_div_w2(i, j) = step_size(i, j) / w2(i, j);             
                step_size(i, j) = (new_sqr_resid_norm(i, j) / old_sqr_resid_norm(i, j)); // beta
            }
        }
        search_direction = residual + (search_direction - A_search_direction * w ) * (step_size * alpha_div_w);
        search_direction_2 = residual_2 + (search_direction_2 - A_search_direction_2 * w2) * (step_size * alpha_div_w);
        old_sqr_resid_norm = new_sqr_resid_norm;
    }
    std::cout << "bicg stablized break! iter_count: " << iter_count << std::endl;

}