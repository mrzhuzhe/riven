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
            L_mat(iter, i) = factor;
            for (j = i; j < cols; j++) {
                U_mat(iter, j) -= factor * U_mat(i, j);
                // L_mat(iter, j) += factor * L_mat(i, j);
            }          
        }
        // show progress 
        std::cout << "\n U:\n" << U_mat << std::endl;
        std::cout << "\n L:\n" << L_mat << std::endl;   
    }
    std::cout << "\n origin:\n" << L_mat * U_mat << std::endl; 
}