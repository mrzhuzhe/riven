#define A(i,j) a[ (j)*lda + (i) ]
#define C(i,j) c[ (j)*ldc + (i) ]
#define KERNEL(i,j) kernel[ (j)*kw + (i) ]

void REF_MMult( int m,  int k,  double *a, int lda, 
                                    int kw, int kh, double *kernel,                                    
                                    double *c, int ldc )
{
  int i, j, w, h;

  /*
  for ( i=0; i<m; i++ ){
    for ( j=0; j<n; j++ ){
      for ( p=0; p<k; p++ ){
	C( i,j ) = C( i,j ) +  A( i,p ) * B( p,j );
      }
    }
  }
  */
 int Wo = m - kw + 1;
 int Ho = k - kh + 1;
 for ( i=0; i< Wo; i++ ){
    for ( j=0; j< Ho; j++ ){
      for (w = 0; w < kw; w++ ){
        for (h = 0; h < kh; h++){
          C( i,j ) = C( i,j ) + A( i + w, j + h) * KERNEL(w, h);          
        }
      }  
    }
  }
}


  
