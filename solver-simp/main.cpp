//  https://eigen.tuxfamily.org/dox/group__TutorialLinearAlgebra.html
#include <iostream>
#include <Eigen/Dense>

void simp_solver(Eigen::MatrixXf& mat02, int rows, int comcol){
    // pivot up
    int i = 0, j = 0, iter = 0;
    float pivot = 0;
    // so time complex is x^2 
    // but each row rely on prev row result this algorithem cannot be parallize
    for (i = 0; i < rows; i++) {        
        pivot = mat02(i, i);
        // first row
        for (j = i; j < comcol; j++) {                
            mat02(i, j) /= pivot;
        }
        //  top 2 down
        for (iter = i+1; iter < rows; iter++) {
            pivot = mat02(iter, i);
            for (j = i; j < comcol; j++) {                
                mat02(iter, j) /= pivot;
                mat02(iter, j) -= mat02(i, j);
            }          
        }
        // down 2 top
        for (iter = 0; iter < i; iter++) {
            pivot = mat02(iter, i) / mat02(i, i) ;
            for (j = i; j < comcol; j++) {
                mat02(iter, j) -= pivot * mat02(i, j);
            }          
        }
        // show progress 
        std::cout << "\n" << mat02 << std::endl;
    }
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

    simp_solver(mat02, rows, comcol);
    
    std::cout << "random matrix test" << std::endl;

    // solve a random matrix
    Eigen::MatrixXf mat03(2*rows, 2*cols);
    mat03 = Eigen::MatrixXf::Random(2*rows, 2*cols);
    Eigen::VectorXf b2(2*rows);
    b2 = Eigen::VectorXf::Random(2*rows);
    Eigen::MatrixXf mat04(2*rows, 2*cols+1);
    mat04 << mat03 , b2;
    std::cout << "random matrix A, b: \n" << mat04 << std::endl;    

    Eigen::VectorXf x2(2*rows);
    x2 = mat03.colPivHouseholderQr().solve(b2);
    std::cout << "\n random matrix ans: \n" << x2 << std::endl;

    std::cout << "\n my ans \n" << std::endl;
    simp_solver(mat04, mat04.rows(), mat04.cols());

    return 0;
}