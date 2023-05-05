#include <stdlib.h>

#define A( i,j ) a[ (j)*lda + (i) ]

void random_matrix( int m, int n, double *a, int lda, int pad )
{
  double drand48();
  int i,j;
  for ( j=0; j<n; j++ )
    for ( i=0; i<m; i++ )
      if (i < pad || j < pad || i > m - pad - 1 || j > n - pad - 1 ){
        A( i,j ) = 0;
      } else {
        A( i,j ) = 2.0 * drand48( ) - 1.0;
      }
}
