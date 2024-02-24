#include <iostream>
#include <Eigen/Dense>
#include <limits.h>
#include "jacobian.h"
#include "multigrid.h"

int main() {
    int rows = 16, cols = 16;
    Eigen::MatrixXf mat01(rows, cols);
    Eigen::MatrixXf b(rows, 1);
    Eigen::MatrixXf x(rows, 1);
    mat01 = Eigen::MatrixXf::Random(rows, cols);
    mat01 = mat01.transpose() * mat01;  // easy to solve positive sysmetric
    // mat01 << 10, -1 , 2, 0, 
    // -1, 11, -1,3,
    // 2, -1, 10,-1,
    // 0, -3, -1, 8;
    // b << 6, 25, -11, 15;
    b = Eigen::MatrixXf::Random(rows, 1);
    jacobian_solver(mat01, x, b, rows, cols);
    
    Eigen::MatrixXf x1(rows, 1);
    gs_solver(mat01, x1, b, rows, cols);

    Eigen::MatrixXf x2(rows, 1);
    x2 = mat01.colPivHouseholderQr().solve(b);
    std::cout << "eigen solution\n" << x2 << std::endl;
  
    std::cout << "\n------------------------------ 1d possion intepolate" << std::endl;
    int nc = 8;
    int nf = 16;
    double uc[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    double uf[] = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.1, 11.11, 12.12, 13.13, 14.14, 15.15 };
    double rf[] = {11.1, 12.2, 13.3, 14.4, 15.5, 16.6, 17.7, 18.8, 19.9, 110.1, 111.11, 112.12, 113.13, 114.14, 115.15 };
    double uf2c[nf];
    double uc2f[nf];
    double rc[nf];
    ftoc (nf+1, uf, rf, nc+1, uf2c, rc);

    // for (int i=0; i < nf; i++){
    //      std::cout <<  uf2c[i] << " ";
    // }
    std::cout <<  std::endl;
    for (int i=0; i < nf; i++){
         std::cout <<  rc[i] << " ";
    }
    std::cout <<  std::endl;

    ctof( nc+1, uc, nf, uc2f);
    std::cout <<  std::endl;
    for (int i=0; i < nf; i++){
        std::cout <<  uc2f[i] << " ";
    }
        
    std::cout << "\n------------------------------ V multi grid" << std::endl;
    Eigen::MatrixXf mat01_16(rows, cols);
    Eigen::MatrixXf mat01_8(rows/2, cols/2);
    Eigen::MatrixXf mat01_4(rows/4, cols/4); 
    
    Fmat2Cmat(mat01, mat01_8, rows, cols);  // initial mat8 and mat4
    Fmat2Cmat(mat01_8, mat01_4, rows/2, cols/2);    
    std::cout << "\n mat01_8 \n" << mat01_8 << std::endl;
    
    Cmat2Fmat(mat01_8, mat01_16, rows/2, cols/2);    
    std::cout << "\n mat01_16 \n" << mat01_16 << std::endl;

    //  16 -> 8 -> 4 -> 8 -> 16 
    Eigen::MatrixXf x_16(rows, 1);
    Eigen::MatrixXf x_8(rows/2, 1);
    Eigen::MatrixXf x_4(rows/4, 1);   
    Eigen::MatrixXf b_8(rows/2, 1);
    Eigen::MatrixXf b_4(rows/4, 1);
    Fmat2Cmat(b, b_8, rows, 1);  // initial mat8 and mat4
    Fmat2Cmat(b_8, b_4, rows/2, 1);    

    // // 16 - 8
    // jacobian_solver(mat01, x_16, b, rows, cols, 1e-2f);    
    // Fmat2Cmat(x_16, x_8, rows, 1);

    // // 8-4
    // jacobian_solver(mat01_8, x_8, b_8, rows/2, cols/2, 1e-3f);
    // Fmat2Cmat(x_8, x_4, rows/2, 1);
    
    // // 4 -8
    // jacobian_solver(mat01_4, x_4, b_4, rows/4, cols/4, 1e-3f);

    return 0;
}