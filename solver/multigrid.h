/*
0. a very good code https://people.sc.fsu.edu/~jburkardt/cpp_src/poisson_1d_multigrid/poisson_1d_multigrid.html
1. https://www10.cs.fau.de/publications/reports/TechRep_2008-03.pdf
2. https://github.com/ddemidov/amgcl
*/
/*
void ctof ( int nc, double uc[], int nf, double uf[] )
{
  int ic;
  int iff;

  for ( ic = 0; ic < nc; ic++ )
  {
    iff = 2 * ic;
    uf[iff] = uf[iff] + uc[ic];
  }

  for ( ic = 0; ic < nc - 1; ic++ )
  {
    iff = 2 * ic + 1;
    uf[iff] = uf[iff] + 0.5 * ( uc[ic] + uc[ic+1] );
  }

  return;
}

void ftoc ( int nf, double uf[], double rf[], int nc, double uc[], 
  double rc[] )
{
  int ic;
  int iff;

  for ( ic = 0; ic < nc; ic++ )
  {
    uc[ic] = 0.0;
  }

  rc[0] = 0.0;
  for ( ic = 1; ic < nc - 1; ic++ )
  {
    iff = 2 * ic;
    rc[ic] = 4.0 * ( rf[iff] + uf[iff-1] - 2.0 * uf[iff] + uf[iff+1] );
  }
  rc[nc-1] = 0.0;

  return;
}
*/
void Fmat2Cmat(const Eigen::MatrixXf& F_mat, Eigen::MatrixXf& C_mat, int rows, int cols){
    for (int j=0; j<cols/2; j++){
        for (int i=0; i<rows/2; i++){            
            C_mat(i, j) += 0.25 * (F_mat(2*i, 2*j) + F_mat(2*i, 2*j+1) + F_mat(2*i+1, 2*j) + F_mat(2*i+1, 2*j+1));
        }
    }
    for (int j=0; j<cols%2; j++){
        for (int i=0; i<rows/2; i++){            
            C_mat(i, j) += 0.5 * (F_mat(2*i, 2*j) + F_mat(2*i+1, 2*j));
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