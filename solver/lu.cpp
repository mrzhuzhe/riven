//  https://johnfoster.pge.utexas.edu/numerical-methods-book/LinearAlgebra_LU.html lu solver
#include <iostream>
#include <Eigen/Dense>

void lu_factor(Eigen::MatrixXf mat, int rows, int cols){
    int i=0, j=0, iter=0;
    float pivot_first, pivot_current, factor;
    Eigen::MatrixXf U_mat(rows, cols);
    Eigen::MatrixXf L_mat(rows, cols);
    
    //  this is for eliminate branch
    // i = 0 do copy
    i = 0;  
    pivot_first = mat(i, i);
    // first row
    for (j = 0; j < cols; j++) {
        U_mat(0, j) = mat(0, j);
    }
    L_mat(0, 0) = 1;
    //  top 2 down
    for (iter = i+1; iter < rows; iter++) {
        L_mat(iter, iter) = 1;   
        pivot_current = mat(iter, i);
        factor = pivot_current / pivot_first;
        for (j = i; j < cols; j++) {
            U_mat(iter, j) = mat(iter, j) - factor * mat(i, j);
            L_mat(iter, j) += factor * L_mat(i, j);
        }  
    }
    std::cout << "\n U:\n" << U_mat << std::endl;
    std::cout << "\n L:\n" << L_mat << std::endl;     

    // i > 0 do eliminate
    for (i = 1; i < rows-1; i++) {        
        pivot_first = U_mat(i, i);
        // first row                
        //  top 2 down
        for (iter = i+1; iter < rows; iter++) {
            pivot_current = U_mat(iter, i);
            //std::cout << pivot_current << " " << pivot_first << std::endl;
            factor = pivot_current / pivot_first;
            for (j = i; j < cols; j++) {
                U_mat(iter, j) -= factor * U_mat(i, j);
                L_mat(iter, j) += factor * L_mat(i, j);
            }          
        }
        // show progress 
        std::cout << "\n U:\n" << U_mat << std::endl;
        std::cout << "\n L:\n" << L_mat << std::endl;   
    }
    std::cout << "\n origin:\n" << L_mat * U_mat << std::endl; 
}

int main(){
    int rows = 3, cols = 3;
    Eigen::MatrixXf mat01(rows, cols);
    //mat01 = Eigen::MatrixXf::Random(rows, cols);
    //std::cout << mat01 << std::endl;

    mat01 << 1,1,0,  2,1,-1, 3,-1,-1;
    // Eigen::Vector3f b;
    // b << 3, 3, 4;

    std::cout << mat01 << std::endl;
    lu_factor(mat01, rows, cols);
    
    // Eigen::Vector3f x = mat01.colPivHouseholderQr().solve(b);
    // std::cout << x << std::endl;
    
    // int comcol = cols+1;
    // Eigen::MatrixXf mat02(rows, comcol);

    // mat02 << mat01 , b;
    // std::cout << mat02 << std::endl;
    
    // std::cout << "random matrix test" << std::endl;

    //solve a random matrix
    Eigen::MatrixXf mat03(2*rows, 2*cols);
    mat03 = Eigen::MatrixXf::Random(2*rows, 2*cols);
    
    lu_factor(mat03, 2*rows, 2*cols);
    std::cout << "mat3\n" << mat03 << std::endl;

    // Eigen::MatrixXf b2(2, 2*rows);
    // b2 = Eigen::MatrixXf::Random(2*rows, 2);
    // Eigen::MatrixXf mat04(2*rows, 2*cols+2);
    // mat04 << mat03 , b2;
    // std::cout << "random matrix A, b: \n" << mat04 << std::endl;    

    // Eigen::MatrixXf x2(2*rows, 2);
    // x2 = mat03.colPivHouseholderQr().solve(b2);
    // std::cout << "\n random matrix ans: \n" << x2 << std::endl;

    return 0;
}