//  https://johnfoster.pge.utexas.edu/numerical-methods-book/LinearAlgebra_EigenProblem2.html
#pragma once 

/*
    persedu code is wrong this is from latex description
*/

void gram_schmidt(Eigen::MatrixXf A, Eigen::MatrixXf& Q, int rows, int cols){
    Eigen::MatrixXf temp_vec(rows, 1);
    float norm = 0;
    float temp = 0;
    for (int i=0;i<cols;i++){        
        for (int k =0;k < i;k++){
            temp_vec(k) = 0;
            for (int j=0;j<rows;j++){
                temp_vec(k) += Q(j, k) *  A(j, i);
            }   
        }
        for (int j=0;j<rows;j++){
            temp = 0;
            for (int k =0;k < i;k++){
                temp += temp_vec(k) * Q(j, k);
            }
            Q(j, i) = A(j, i) - temp;
        } 
        norm = 0;
        for (int j=0;j<rows;j++){
            norm += (Q(j, i) * Q(j, i));
        }
        norm = std::sqrt(norm);
        for (int j=0;j<rows;j++){
            Q(j, i) = Q(j, i) / norm;
        }
    }
}

void qr_eigen(Eigen::MatrixXf& A, int rows, int cols) {
    Eigen::MatrixXf Qmat(rows, rows);
    gram_schmidt(A, Qmat, rows, cols);
    std::cout << "qr eigen:\n" << Qmat << std::endl;    
    std::cout << "qr eigen Qmat inverse:\n" << Qmat.inverse() << std::endl;
    std::cout << "qr eigen Qmat transpose * Qmat:\n" << Qmat.transpose() * Qmat << std::endl;

    std::cout << "eigenvalues \n" << A.eigenvalues() << std::endl;
    int iter=0;
    for (iter=0;iter<1000;iter++){
        A = Qmat.transpose() * A * Qmat;
        gram_schmidt(A, Qmat, rows, cols);
        if (iter % 100 == 0) {
            std::cout << "iter " << iter << "\n" << A << std::endl;
        }
    }
    
}
