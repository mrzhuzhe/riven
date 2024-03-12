#include <iostream>
#include <chrono>
#include <Eigen/Dense>
#include <limits.h>
#include "jacobian.h"
#include "multigrid.h"
#include "cg.h"

void test_case(const Eigen::MatrixXf& mat01, int rows, int cols, Eigen::MatrixXf& x, const Eigen::MatrixXf& b, const Eigen::MatrixXf& eigen_ans){
    
    auto start{std::chrono::steady_clock::now()};
    auto end{std::chrono::steady_clock::now()};
    std::chrono::duration<double> elapsed_seconds{end - start};
    
    start = std::chrono::steady_clock::now();
    jacobian_solver(mat01, x, b, rows, cols);
    end = std::chrono::steady_clock::now();
    elapsed_seconds = end - start;
    std::cout << "jacobian_solver: " << elapsed_seconds.count() << " Elapsed time " << std::endl;


    
    Eigen::MatrixXf x1(rows, 1);
    x1.setZero(rows, 1);

    start = std::chrono::steady_clock::now();
    gs_solver(mat01, x1, b, rows, cols);
    end = std::chrono::steady_clock::now();
    elapsed_seconds = end - start;
    std::cout << "gs_solver: " << elapsed_seconds.count() << " Elapsed time " << std::endl;

    std::cout << "jacobian error " << (x - eigen_ans).maxCoeff() << " " << (x - eigen_ans).minCoeff() << std::endl;
    std::cout << "gs error " << (x1 - eigen_ans).maxCoeff() << " " << (x1 - eigen_ans).minCoeff() << std::endl;

    Eigen::MatrixXf mat01_16(rows, cols);
    Eigen::MatrixXf mat01_8(rows/2, cols/2);
    Eigen::MatrixXf mat01_4(rows/4, cols/4); 
    
    Fmat2Cmat(mat01, mat01_8, rows, cols);  // initial mat8 and mat4
    Fmat2Cmat(mat01_8, mat01_4, rows/2, cols/2);    
    //std::cout << "\n mat01_8 \n" << mat01_8 << std::endl;
    
    Cmat2Fmat(mat01_8, mat01_16, rows/2, cols/2);    // this interpolate is not good
    //std::cout << "\n mat01_16 \n" << mat01_16 << std::endl;
    
    std::cout << "\n V multi grid" << std::endl;
    //  16 -> 8 -> 4 -> 8 -> 16 
    Eigen::MatrixXf x_16(rows, 1);
    Eigen::MatrixXf x_8(rows/2, 1);
    Eigen::MatrixXf x_4(rows/4, 1);   
    Eigen::MatrixXf b_8(rows/2, 1);
    Eigen::MatrixXf b_4(rows/4, 1);
    Fmat2Cmat(b, b_8, rows, 1);  // initial mat8 and mat4
    Fmat2Cmat(b_8, b_4, rows/2, 1);    

    // 16 - 8
    jacobian_solver(mat01, x_16, b, rows, cols, 1e-2f);    
    Fmat2Cmat(x_16, x_8, rows, 1);

    // 8-4
    jacobian_solver(mat01_8, x_8, b_8, rows/2, cols/2, 1e-3f);
    Fmat2Cmat(x_8, x_4, rows/2, 1);
    
    // 4
    jacobian_solver(mat01_4, x_4, b_4, rows/4, cols/4, 1e-4f);
    
    // 4-8
    Cmat2Fmat(x_4, x_8, rows/4, 1);
    jacobian_solver(mat01_8, x_8, b_8, rows/2, cols/2, 1e-5f);

    // 8 - 16
    Cmat2Fmat(x_8, x_16, rows/2, 1);
    jacobian_solver(mat01, x_16, b, rows, cols);

    std::cout << "multi grid error " << (x_16 - eigen_ans).maxCoeff() << " " << (x_16 - eigen_ans).minCoeff() << std::endl;
}

int main(int argc, char *argv[]) {
    int size = 64;
    if (argc > 1) {
        size = atoi(argv[1]);
    }
    int rows = size, cols = size;
    Eigen::MatrixXf mat01(rows, cols);
    Eigen::MatrixXf b(rows, 1);
    Eigen::MatrixXf x(rows, 1);
    b = Eigen::MatrixXf::Random(rows, 1);

    int test_type = 0;
    if (argc > 2) {
        test_type = atoi(argv[2]);
    }
    mat01 = Eigen::MatrixXf::Random(rows, cols);
    switch (test_type)
    {
    case 1:
        Tridiagonal_mat(mat01, rows, cols);
        break;
    case 2:
        mat01 = mat01.transpose() * mat01;
        break;
    default:
        break;
    }   

    Eigen::MatrixXf x2(rows, 1);

    auto start{std::chrono::steady_clock::now()};
    auto end{std::chrono::steady_clock::now()};
    std::chrono::duration<double> elapsed_seconds{end - start};
    
    start = std::chrono::steady_clock::now();
    x2 = mat01.colPivHouseholderQr().solve(b);
    end = std::chrono::steady_clock::now();
    elapsed_seconds = end - start;
    std::cout << "eigen solver: " << elapsed_seconds.count() << " Elapsed time " << std::endl;

    
    //std::cout << "eigen solution\n" << x2 << std::endl;

    std::cout << "\n ------------------------------ test case 1 mat01.block(0, 0, 8, 8) \n" 
    << mat01.block(0, 0, 8, 8) 
    << " \n rows cols " << rows << " " << cols << std::endl;

    // mat01 << 10, -1 , 2, 0, 
    // -1, 11, -1,3,
    // 2, -1, 10,-1,
    // 0, -3, -1, 8;
    // b << 6, 25, -11, 15;
    x.setZero(rows, 1);
    test_case(mat01, rows, cols, x, b, x2);

    std::cout << "\n ------------------------------ test cg \n" << std::endl;
    //  CG 
    Eigen::MatrixXf cg_x(rows, 1);
    cg_x.setZero(rows, 1);

    start = std::chrono::steady_clock::now();
    cg(mat01, rows, cols, cg_x, b);
    end = std::chrono::steady_clock::now();
    elapsed_seconds = end - start;
    std::cout << "cg_solver: " << elapsed_seconds.count() << " Elapsed time " << std::endl;

    //std::cout << cg_x << std::endl;
    std::cout << "cg error " << (cg_x - x2).maxCoeff() << " " << (cg_x - x2).minCoeff() << std::endl;

    return 0;
}