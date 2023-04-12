// 4 X 4 tiles

/* Create macros so that the matrices are stored in column-major order */

#define A(i,j) a[ (j)*lda + (i) ]
#define B(i,j) b[ (j)*ldb + (i) ]
#define C(i,j) c[ (j)*ldc + (i) ]

/* Routine for computing C = A * B + C */

void AddDot( int, double *, int, double *, double * );

void AddDot4x4(int k, double *a, int lda, double *b, int ldb, double *c, int ldc){
      
    int p;
    double c00_reg = 0.0, c01_reg = 0.0, c02_reg = 0.0, c03_reg = 0.0,
    c10_reg = 0.0, c11_reg = 0.0, c12_reg = 0.0, c13_reg = 0.0,
    c20_reg = 0.0, c21_reg = 0.0, c22_reg = 0.0, c23_reg = 0.0,
    c30_reg = 0.0, c31_reg = 0.0, c32_reg = 0.0, c33_reg = 0.0;
    
    double a0p_reg, a1p_reg, a2p_reg, a3p_reg;
    double *b0p, *b1p, *b2p, *b3p;
    
    b0p = &B( 0, 0 );
    b1p = &B( 0, 1 );
    b2p = &B( 0, 2 );
    b3p = &B( 0, 3 );

    double b0r, b1r, b2r, b3r;
    for ( p=0; p<k; p++ ){
      a0p_reg = A( 0, p );
      a1p_reg = A( 1, p );
      a2p_reg = A( 2, p );
      a3p_reg = A( 3, p );
      
      /*
      b0r = *(b0p++);
      b1r = *(b1p++);
      b2r = *(b2p++);
      b3r = *(b3p++);
      */
      /*
      C( 0, 0 ) += a0p_reg * b0r;         
      C( 0, 1 ) += a0p_reg * b1r;         
      C( 0, 2 ) += a0p_reg * b2r;         
      C( 0, 3 ) += a0p_reg * b3r;     
    

    
      C( 1, 0 ) += a1p_reg * b0r;         
      C( 1, 1 ) += a1p_reg * b1r;         
      C( 1, 2 ) += a1p_reg * b2r;         
      C( 1, 3 ) += a1p_reg * b3r;     
    

    
      C( 2, 0 ) += a2p_reg * b0r;         
      C( 2, 1 ) += a2p_reg * b1r;         
      C( 2, 2 ) += a2p_reg * b2r;         
      C( 2, 3 ) += a2p_reg * b3r;     
    

    
      C( 3, 0 ) += a3p_reg * b0r;         
      C( 3, 1 ) += a3p_reg * b1r;         
      C( 3, 2 ) += a3p_reg * b2r;         
      C( 3, 3 ) += a3p_reg * b3r;  
      */
      
      C( 0, 0 ) += a0p_reg * *b0p;         
      C( 1, 0 ) += a1p_reg * *b0p;   
            
      C( 0, 1 ) += a0p_reg * *b1p;         
      C( 1, 1 ) += a1p_reg * *b1p;    
      
      
      C( 0, 2 ) += a0p_reg * *b2p;         
      C( 1, 2 ) += a1p_reg * *b2p;
      
      C( 0, 3 ) += a0p_reg * *b3p;     
      C( 1, 3 ) += a1p_reg * *b3p;   
    
      C( 2, 0 ) += a2p_reg * *b0p;  
      C( 3, 0 ) += a3p_reg * *b0p; 

      C( 2, 1 ) += a2p_reg * *b1p;         
      C( 3, 1 ) += a3p_reg * *b1p;    

      C( 2, 2 ) += a2p_reg * *b2p;         
      C( 3, 2 ) += a3p_reg * *b2p;   

      C( 2, 3 ) += a2p_reg * *b3p;           
      C( 3, 3 ) += a3p_reg * *b3p;   
      
      b0p += 1;
      b1p += 1;
      b2p += 1;
      b3p += 1;

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