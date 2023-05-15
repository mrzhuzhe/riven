// 4 X 4 tiles
// SIMD
// still has variable bad case 
// add kernel pack
//  First, we pack the block of A so that we march through it contiguously.

/* Create macros so that the matrices are stored in column-major order */

#include <mmintrin.h>
#include <emmintrin.h>  // sse3
#include <xmmintrin.h>  // sse
#include <pmmintrin.h>  // sse2
#include <immintrin.h>  // avx

//#include <iostream>
#define min( i, j ) ( (i)<(j) ? (i): (j) )

#define mc 256
#define kc 128
#define nb 1000

typedef union {
  __m256d v;
  __m256d u;
  double d[4];
} v4df_t


#define A(i,j) a[ (j)*lda + (i) ]
#define B(i,j) b[ (j)*ldb + (i) ]
#define C(i,j) c[ (j)*ldc + (i) ]

/* Routine for computing C = A * B + C */

void AddDot8x4(int k, double *a, int lda, double *b, int ldb, double *c, int ldc){
      
    int p;
    v2df_t vc0010, vc0111, vc0212, vc0313, vc2030, vc2131, vc2232, vc2333;
    v2df_t va01, va23;
    v2df_t vb0p, vb1p, vb2p, vb3p; 
    //double *b0p, *b1p, *b2p, *b3p;
    
    /*
    b0p = &B( 0, 0 );
    b1p = &B( 0, 1 );
    b2p = &B( 0, 2 );
    b3p = &B( 0, 3 );
    */
    vc0010.v = _mm_setzero_pd(); 
    vc0111.v = _mm_setzero_pd(); 
    vc0212.v = _mm_setzero_pd();
    vc0313.v = _mm_setzero_pd();
    vc2030.v = _mm_setzero_pd();
    vc2131.v = _mm_setzero_pd();
    vc2232.v = _mm_setzero_pd();
    vc2333.v = _mm_setzero_pd();

    for ( p=0; p<k; p++ ){
      va01.v = _mm_load_pd((double *) a);      
      va23.v = _mm_load_pd((double *) a+2);
      
      a += 4;

      vb0p.v = _mm_loaddup_pd((double *) b);
      vb1p.v = _mm_loaddup_pd((double *) (b + 1));
      vb2p.v = _mm_loaddup_pd((double *) (b + 2));
      vb3p.v = _mm_loaddup_pd((double *) (b + 3));

      vc0010.v += va01.v * vb0p.v;                    
      vc0111.v += va01.v * vb1p.v;          
      vc0212.v += va01.v * vb2p.v; 
      vc0313.v += va01.v * vb3p.v;   
      //  _mm256_mul_pd _mm256_add_pd
      vc2030.v += va23.v * vb0p.v;   
      vc2131.v += va23.v * vb1p.v;   
      vc2232.v += va23.v * vb2p.v;   
      vc2333.v += va23.v * vb3p.v;        
      
      b +=4;

    }

    C(0, 0) += vc0010.d[0];
    C(0, 1) += vc0111.d[0];
    C(0, 2) += vc0212.d[0];
    C(0, 3) += vc0313.d[0];

    C(1, 0) += vc0010.d[1];
    C(1, 1) += vc0111.d[1];
    C(1, 2) += vc0212.d[1];
    C(1, 3) += vc0313.d[1];

    C(2, 0) += vc2030.d[0];
    C(2, 1) += vc2131.d[0];
    C(2, 2) += vc2232.d[0];
    C(2, 3) += vc2333.d[0];

    C(3, 0) += vc2030.d[1];
    C(3, 1) += vc2131.d[1];
    C(3, 2) += vc2232.d[1];
    C(3, 3) += vc2333.d[1];

    
};


void PackMatrixA(int k, double *a, int lda, double *a_to){
  int j;
  for (j=0; j <k ; j++){
    double *a_ij_pntr = &A(0, j);
    *a_to++ = *a_ij_pntr;
    *a_to++ = *(a_ij_pntr+1);
    *a_to++ = *(a_ij_pntr+2);
    *a_to++ = *(a_ij_pntr+3);
  }
}


void PackMatrixB(int k, double *b, int ldb, double *b_to){
  int i;
  double 
    *b0p = &B(0, 0), *b1p = &B(0,1),
    *b2p = &B(0, 2), *b3p = &B(0,3); 
  for (i=0; i < k ; i++){    
    *b_to++ = *(b0p++);
    *b_to++ = *(b1p++);
    *b_to++ = *(b2p++);
    *b_to++ = *(b3p++);
  }
}

void Innerkernel(int m, int n, int k, double *a, int lda, double *b, int ldb, double *c, int ldc, int first_time)
{
  int i, j;
  double packedA[m*k];
  static double packedB[kc*nb];
  for (j = 0; j < n; j+=4){
    if ( first_time ) {
      PackMatrixB(k, &B(0, j), ldb, &packedB[j*k]);
    }
    
    for (i = 0; i < m; i +=4){
      if ( j == 0 ) PackMatrixA(k, &A(i, 0), lda, &packedA[i*k]);
      AddDot8x4(k, &packedA[i*k], 4, &packedB[j*k], k , &C( i,j ), ldc);
    }
  }
}

void MY_MMult( int m, int n, int k, double *a, int lda, 
                                    double *b, int ldb,
                                    double *c, int ldc )
{
  int i, j, jb, ib;

  for ( j=0; j< k ; j+=kc ){        /* Loop over the columns of C */
    jb = min(k-j, kc);
    for ( i=0; i<m; i+=mc ){        /* Loop over the rows of C */
      /* Update the C( i,j ) with the inner product of the ith row of A
	 and the jth column of B */
      ib = min(m-i, mc);
      Innerkernel(ib, n, jb, &A(i, j), lda, &B(j,0), ldb, &C(i, 0), ldc, i==0);
            
    }
  }
}