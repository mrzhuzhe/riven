#include <stdlib.h>

#define A(i, j) a[(j)*lda + (i)]

double drand48();
void random_matrix( int m, int n, float *a, int lda, int pad )
{
  int i,j;
#pragma omp parallel for 
  for ( j=0; j<n; j++ )
    for ( i=0; i<m; i++ )
      if (i < pad || j < pad || i > m - pad - 1 || j > n - pad - 1 ){
        A( i,j ) = 0;
      } else {
        A( i,j ) = 2.0 * drand48( ) - 1.0;
 
      }
}