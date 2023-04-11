/* Create macros so that the matrices are stored in column-major order */

#define A(i,j) a[ (j)*lda + (i) ]
#define B(i,j) b[ (j)*ldb + (i) ]
#define C(i,j) c[ (j)*ldc + (i) ]

/* Routine for computing C = A * B + C */

void AddDot( int, double *, int, double *, double * );


void AddDot1x4( int k, double *a, int lda, double *b, int ldb, double *c, int ldc)
{
   for (int p = 0; p < k ; p++){
      C(0, 0) += A(0, p) * B(p , 0);
      //C(0, 1) += A(0, p) * B(p , 1);
      //C(0, 2) += A(0, p) * B(p , 2);
      //C(0, 3) += A(0, p) * B(p , 3);
   }
   for (int p = 0; p < k ; p++){
      
      C(0, 1) += A(0, p) * B(p , 1);
      
   }
   for (int p = 0; p < k ; p++){
      
      C(0, 2) += A(0, p) * B(p , 2);
     
   }
   for (int p = 0; p < k ; p++){
     
      C(0, 3) += A(0, p) * B(p , 3);
   }
}

void MY_MMult( int m, int n, int k, double *a, int lda, 
                                    double *b, int ldb,
                                    double *c, int ldc )
{
  int i, j;

  for ( j=0; j<n; j+=4 ){        /* Loop over the columns of C */
    for ( i=0; i<m; i+=1 ){        /* Loop over the rows of C */
      /* Update the C( i,j ) with the inner product of the ith row of A
	 and the jth column of B */

      
      AddDot1x4(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc);
        
    }
  }
}


