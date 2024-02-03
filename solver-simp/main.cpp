#include <iostream>
#include <Eigen/Dense>


int main(){
    int rows = 3, cols = 3;
    Eigen::MatrixXd mat01(rows, cols);
    mat01 = Eigen::MatrixXd::Random(rows, cols);
    std::cout << mat01 << std::endl;

    return 0;
}