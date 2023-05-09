/* Routine for computing C = A * B + C */

#define A(i,j) a[ (j)*lda + (i) ]
#define C(i,j) c[ (j)*lda + (i) ]
#define KERNEL(i,j) kernel[ (j)*kw + (i) ]

void REF_MMult( int m,  int k,  float *a, int lda, 
                                    int kw, int kh, float *kernel,                                    
                                    float *c, int ldc, int stride )
{  
  int i, j, w, h;
  int Wo = (m - kw) / stride + 1;
  int Ho = (k - kh) / stride + 1;
  for ( i=0; i< Wo; i+=1 ){
      for ( j=0; j< Ho; j+=1 ){
        for (w = 0; w < kw; w++ ){
          for (h = 0; h < kh; h++){
            C( i,j ) = C( i,j ) + A( i * stride + w, j * stride + h) * KERNEL(w, h);          
          }
        }  
      }
    }
}


