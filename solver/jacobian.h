#pragma once

void jacobian_solver(Eigen::MatrixXf A, Eigen::MatrixXf& x, Eigen::MatrixXf b, int rows, int cols){
    std::cout << "jacobian_solver" << std::endl;
    std::cout << "A\n" << A << std::endl;    
    std::cout << "b\n" << b << std::endl;

    Eigen::MatrixXf T(rows, cols);
    Eigen::MatrixXf old(rows, cols);

    for (int j=0;j<cols;j++){
        for (int i=0;i<rows;i++){
            if (i==j) {
                continue;
            }
            T(i, j) = A(i, j);
        }
    }
    const int max_iter = 100;
    float temp = 0, max_diff_norm=0, max_norm=0;
    for (int iter=0; iter<max_iter; iter++){
        old = x;    // this is deep copy in eigen
        for (int i=0; i < rows; i++){
            temp = 0;
            for (int j=0;j<cols;j++){
                temp += T(i, j) * x(j);
            }
            x(i) = b(i) - temp;
            x(i) /= A(i, i);        
        }
        max_diff_norm=0;
        max_norm=0;
        for (int i=0;i<rows;i++){
            temp = std::abs(x(i));
            max_norm = temp > max_norm ? temp : max_norm;
            temp = std::abs(x(i) - old(i));
            max_diff_norm = temp > max_diff_norm ? temp : max_diff_norm;            
        }
        if ( max_diff_norm / max_norm < __FLT_EPSILON__ ) {
            break;
        }
    }
    /*    

        if np.linalg.norm(x - x_old, ord=np.inf) / np.linalg.norm(x, ord=np.inf) < tolerance:
            break
            
    */
    std::cout << "x\n" << x << std::endl;
}

void gs_solver(Eigen::MatrixXf A, Eigen::MatrixXf& x, Eigen::MatrixXf b, int rows, int cols){
    std::cout << "gs_solver" << std::endl;
    std::cout << "A\n" << A << std::endl;
    std::cout << "b\n" << b << std::endl;

    std::cout << "x\n" << x << std::endl;
}