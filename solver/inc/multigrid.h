/*
0. a very good code https://people.sc.fsu.edu/~jburkardt/cpp_src/poisson_1d_multigrid/poisson_1d_multigrid.html
1. https://www10.cs.fau.de/publications/reports/TechRep_2008-03.pdf
2. https://github.com/ddemidov/amgcl
*/

void Tridiagonal_mat(Eigen::MatrixXf& mat, int rows, int cols){
    for (int j=0; j<cols; j++){
        for (int i=0; i<rows; i++){  
            if (i==j-1) {
                mat(i, j) = -1;
            } else if (i==j) {
                mat(i, j) = 2;
            } else if (i==j+1) {
                mat(i, j) = -1;
            } else {
                mat(i, j) = 0;
            }
        }
    }
}


void Fmat2Cmat(const Eigen::MatrixXf& F_mat, Eigen::MatrixXf& C_mat, int rows, int cols){
    for (int j=0; j<cols/2; j++){
        for (int i=0; i<rows/2; i++){            
            C_mat(i, j) = 0.25 * (F_mat(2*i, 2*j) + F_mat(2*i, 2*j+1) + F_mat(2*i+1, 2*j) + F_mat(2*i+1, 2*j+1));
        }
    }
    for (int j=0; j<cols%2; j++){
        for (int i=0; i<rows/2; i++){            
            C_mat(i, j) = 0.5 * (F_mat(2*i, 2*j) + F_mat(2*i+1, 2*j));
        }
    }
}

void Cmat2Fmat(const Eigen::MatrixXf& C_mat, Eigen::MatrixXf& F_mat, int rows, int cols){
    if (cols>1) {
        for (int j=0; j<cols; j++){
            for (int i=0; i<rows; i++){            
                F_mat(2*i, 2*j+1) = F_mat(2*i+1, 2*j+1) = C_mat(i, j);
                F_mat(2*i, 2*j) = F_mat(2*i+1, 2*j) = C_mat(i, j);
            }
        }
    } else {
        for (int j=0; j<cols; j++){
            for (int i=0; i<rows; i++){            
                F_mat(2*i, 2*j) = F_mat(2*i+1, 2*j) = C_mat(i, j);
            }
        }
    }
}