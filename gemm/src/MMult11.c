// 4 X 4 tiles

/* Create macros so that the matrices are stored in column-major order */

#define A(i,j) a[ (j)*lda + (i) ]
#define B(i,j) b[ (j)*ldb + (i) ]
#define C(i,j) c[ (j)*ldc + (i) ]

/* Routine for computing C = A * B + C */

void AddDot( int, double *, int, double *, double * );

void AddDot4x4(int k, double *a, int lda, double *b, int ldb, double *c, int ldc){
      
    int p;
    for ( p=0; p<k; p++ ){
      C( 0, 0 ) += A( 0, p ) * B( p, 0 );     
    }
    for ( p=0; p<k; p++ ){
      C( 0, 1 ) += A( 0, p ) * B( p, 1 );     
    }
    for ( p=0; p<k; p++ ){
      C( 0, 2 ) += A( 0, p ) * B( p, 2 );     
    }
    for ( p=0; p<k; p++ ){
      C( 0, 3 ) += A( 0, p ) * B( p, 3 );     
    }

    for ( p=0; p<k; p++ ){
      C( 1, 0 ) += A( 1, p ) * B( p, 0 );     
    }
    for ( p=0; p<k; p++ ){
      C( 1, 1 ) += A( 1, p ) * B( p, 1 );     
    }
    for ( p=0; p<k; p++ ){
      C( 1, 2 ) += A( 1, p ) * B( p, 2 );     
    }
    for ( p=0; p<k; p++ ){
      C( 1, 3 ) += A( 1, p ) * B( p, 3 );     
    }

    for ( p=0; p<k; p++ ){
      C( 2, 0 ) += A( 2, p ) * B( p, 0 );     
    }
    for ( p=0; p<k; p++ ){
      C( 2, 1 ) += A( 2, p ) * B( p, 1 );     
    }
    for ( p=0; p<k; p++ ){
      C( 2, 2 ) += A( 2, p ) * B( p, 2 );     
    }
    for ( p=0; p<k; p++ ){
      C( 2, 3 ) += A( 2, p ) * B( p, 3 );     
    }

    for ( p=0; p<k; p++ ){
      C( 3, 0 ) += A( 3, p ) * B( p, 0 );     
    }
    for ( p=0; p<k; p++ ){
      C( 3, 1 ) += A( 3, p ) * B( p, 1 );     
    }
    for ( p=0; p<k; p++ ){
      C( 3, 2 ) += A( 3, p ) * B( p, 2 );     
    }
    for ( p=0; p<k; p++ ){
      C( 3, 3 ) += A( 3, p ) * B( p, 3 );     
    }
    
};

void MY_MMult( int m, int n, int k, double *a, int lda, 
                                    double *b, int ldb,
                                    double *c, int ldc )
{
  int i, j;

  for ( j=0; j<n; j+=4 ){        /* Loop over the columns of C */
    for ( i=0; i<m; i+=4 ){        /* Loop over the rows of C */
      /* Update the C( i,j ) with the inner product of the ith row of A
	 and the jth column of B */

      AddDot4x4(k, &A( i,0 ), lda, &B( 0,j ), lda , &C( i,j ), ldc);
            
    }
  }
}


/* Create macro to let X( i ) equal the ith element of x */

#define X(i) x[ (i)*incx ]

void AddDot( int k, double *x, int incx,  double *y, double *gamma )
{
  /* compute gamma := x' * y + gamma with vectors x and y of length n.

     Here x starts at location x with increment (stride) incx and y starts at location y and has (implicit) stride of 1.
  */
 
  int p;

  for ( p=0; p<k; p++ ){
    *gamma += X( p ) * y[ p ];     
  }
}
