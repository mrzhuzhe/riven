#define A(i,j) a[ (j)*lda + (i) ]
#define C(i,j) c[ (j)*lda + (i) ]
#define KERNEL(i,j) kernel[ (j)*kw + (i) ]

void MY_MMult( int m,  int k,  double *a, int lda, 
                                    int kw, int kh, double *kernel,                                    
                                    double *c, int ldc )
{
  int padding = 1;
  int stride = 1;
  int i, j, w, h;
  int Wo = m - kw + 1;
  int Ho = k - kh + 1;
  for ( i=0; i< Wo; i+=stride ){
      for ( j=0; j< Ho; j+=stride ){
        for (w = 0; w < kw; w++ ){
          for (h = 0; h < kh; h++){
            C( i,j ) = C( i,j ) + A( i + w, j + h) * KERNEL(w, h);          
          }
        }  
      }
    }
}


  
