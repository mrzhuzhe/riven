
#define A(i,j) a[ (j)*lda + (i) ]
#define B(i,j) b[ (j)*ldb + (i) ]
#define C(i,j) c[ (j)*ldc + (i) ]

#include <cblas.h>
/* Routine for computing C = A * B + C */

/*
export OMP_NUM_THREADS=1
*/
void MY_MMult( int m, int n, int k, double *a, int lda, 
                                    double *b, int ldb,
                                    double *c, int ldc ){

  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, a, lda,
              b, ldb, 0, c, ldc);
}