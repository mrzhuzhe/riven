#define A(i,j) a[ (j)*lda + (i) ]
#define C(i,j) c[ (j)*lda + (i) ]
#define KERNEL(i,j) kernel[ (j)*kw + (i) ]

#include <omp.h>

void MY_MMult( int m,  int k,  double *a, int lda, 
                                    int kw, int kh, double *kernel,                                    
                                    double *c, int ldc, int stride )
{
  //  multi channel ? multi batch ?
  //  img2col how to do img2features how to map result back
  int i, j, w, h;
  int Wo = (m - kw) / stride + 1;
  int Ho = (k - kh) / stride + 1;
#pragma omp parallel for private(i, j, w, h) shared(a, Wo, Ho, kw, kh)
  for ( i=0; i< Wo; i+=1 ){
      for ( j=0; j< Ho; j+=1 ){
        double sum = 0;
        for (w = 0; w < kw; w++ ){
          for (h = 0; h < kh; h++){
             sum += A( i * stride + w, j * stride + h) * KERNEL(w, h);          
          }
        }  
        C( i,j ) = sum;
      }
    }
}


  
