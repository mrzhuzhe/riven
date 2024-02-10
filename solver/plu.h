void plu_factor(
    Eigen::MatrixXf mat, int rows, int cols, 
    Eigen::MatrixXf& U_mat, Eigen::MatrixXf& L_mat,  Eigen::MatrixXf& P_mat
    ){
    int i=0, j=0, iter=0, k=0;
    float pivot_first, pivot_current, factor, temp;
    // initial P and L
    for (i = 0; i < rows; i++) {
        P_mat(i, i) = 1;
    }
    // initial U
    for (j = 0; j < cols; j++) {
        for (i=0;i<rows;i++) {
            U_mat(j, i) = mat(j, i);
        }        
    }
    // i > 0 do eliminate
    for (i = 0; i < rows-1; i++) {                
        // first row
        // check and swap pivot  
        k = i;
        while (U_mat(i, i) == 0 && k < rows)
        {
            for (j = 0; j < cols; j++) {
                temp = U_mat(i, j);
                U_mat(i, j) = U_mat(k, j);
                U_mat(k, j) = temp; 

                temp = P_mat(i, j);
                P_mat(i, j) = P_mat(k, j);
                P_mat(k, j) = temp; 
            }
            k++;
        }
        pivot_first = U_mat(i, i);
        //  top 2 down
        for (iter = i+1; iter < rows; iter++) {
            pivot_current = U_mat(iter, i);
            //std::cout << pivot_current << " " << pivot_first << std::endl;
            factor = pivot_current / pivot_first;
            L_mat(iter, i) = factor;
            for (j = i; j < cols; j++) {
                U_mat(iter, j) -= factor * U_mat(i, j);
            }          
        }
        // show progress 
        // std::cout << "\n U:\n" << U_mat << std::endl;
        // std::cout << "\n L:\n" << L_mat << std::endl;
        // std::cout << "\n P:\n" << P_mat << std::endl;      
    }
    L_mat = P_mat * L_mat;  // Todo this permutate matrix
    for (i = 0; i < rows; i++) {
        L_mat(i, i) = 1;
    }    
    // std::cout << "\n origin:\n" <<  P_mat * mat << std::endl; 
    // std::cout << "\n solution :\n" <<  L_mat * U_mat << std::endl; 
}