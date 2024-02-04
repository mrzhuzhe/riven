//  https://johnfoster.pge.utexas.edu/numerical-methods-book/LinearAlgebra_LU.html lu solver
#include <iostream>
#include <Eigen/Dense>

void lu_factor(Eigen::MatrixXf& mat02, int rows, int comcol){
    
}

int main(){
    int rows = 3, cols = 3;
    Eigen::MatrixXf mat01(rows, cols);
    //mat01 = Eigen::MatrixXf::Random(rows, cols);
    //std::cout << mat01 << std::endl;

    mat01 << 1,2,3,  4,5,6,  7,8,10;  
    Eigen::Vector3f b;
    b << 3, 3, 4;
    
    Eigen::Vector3f x = mat01.colPivHouseholderQr().solve(b);
    std::cout << x << std::endl;
    
    int comcol = cols+1;
    Eigen::MatrixXf mat02(rows, comcol);

    mat02 << mat01 , b;
    std::cout << mat02 << std::endl;
    
    std::cout << "random matrix test" << std::endl;

    // solve a random matrix
    Eigen::MatrixXf mat03(2*rows, 2*cols);
    mat03 = Eigen::MatrixXf::Random(2*rows, 2*cols);
    Eigen::MatrixXf b2(2, 2*rows);
    b2 = Eigen::MatrixXf::Random(2*rows, 2);
    Eigen::MatrixXf mat04(2*rows, 2*cols+2);
    mat04 << mat03 , b2;
    std::cout << "random matrix A, b: \n" << mat04 << std::endl;    

    Eigen::MatrixXf x2(2*rows, 2);
    x2 = mat03.colPivHouseholderQr().solve(b2);
    std::cout << "\n random matrix ans: \n" << x2 << std::endl;

    return 0;
}