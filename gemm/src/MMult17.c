// 4 X 4 tiles
// SIMD
// still has variable bad case 
// add kernel pack

/* Create macros so that the matrices are stored in column-major order */

#include <mmintrin.h>
#include <emmintrin.h>  // sse3
#include <xmmintrin.h>  // sse
#include <pmmintrin.h>  // sse2

//#include <iostream>
#define min( i, j ) ( (i)<(j) ? (i): (j) )

#define mc 256
#define kc 128

// 2 type use same memory
//  https://en.cppreference.com/w/cpp/language/union
typedef union 
{
  __m128d v;
  double d[2];
} v2df_t;


#define A(i,j) a[ (j)*lda + (i) ]
#define B(i,j) b[ (j)*ldb + (i) ]
#define C(i,j) c[ (j)*ldc + (i) ]

/* Routine for computing C = A * B + C */

void AddDot4x4(int k, double *a, int lda, double *b, int ldb, double *c, int ldc){
      
    int p;
    v2df_t vc0010, vc0111, vc0212, vc0313, vc2030, vc2131, vc2232, vc2333;
    v2df_t va01, va23;
    v2df_t vb0p, vb1p, vb2p, vb3p; 
    double *b0p, *b1p, *b2p, *b3p;
    
    b0p = &B( 0, 0 );
    b1p = &B( 0, 1 );
    b2p = &B( 0, 2 );
    b3p = &B( 0, 3 );

    vc0010.v = _mm_setzero_pd(); 
    vc0111.v = _mm_setzero_pd(); 
    vc0212.v = _mm_setzero_pd();
    vc0313.v = _mm_setzero_pd();
    vc2030.v = _mm_setzero_pd();
    vc2131.v = _mm_setzero_pd();
    vc2232.v = _mm_setzero_pd();
    vc2333.v = _mm_setzero_pd();

    for ( p=0; p<k; p++ ){
      va01.v = _mm_load_pd((double *) &A( 0, p ));      
      va23.v = _mm_load_pd((double *) &A( 2, p ));
      
      vb0p.v = _mm_loaddup_pd((double *) b0p);
      vb1p.v = _mm_loaddup_pd((double *) b1p);
      vb2p.v = _mm_loaddup_pd((double *) b2p);
      vb3p.v = _mm_loaddup_pd((double *) b3p);

      vc0010.v += va01.v * vb0p.v;                    
      vc0111.v += va01.v * vb1p.v;          
      vc0212.v += va01.v * vb2p.v; 
      vc0313.v += va01.v * vb3p.v;   
    
      vc2030.v += va23.v * vb0p.v;   
      vc2131.v += va23.v * vb1p.v;   
      vc2232.v += va23.v * vb2p.v;   
      vc2333.v += va23.v * vb3p.v;        
      
      b0p += 1;
      b1p += 1;
      b2p += 1;
      b3p += 1;

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


void Innerkernel(int m, int n, int k, double *a, int lda, double *b, int ldb, double *c, int ldc)
{
  int i, j;
  for (j = 0; j < n; j+=4){
    for (i = 0; i < m; i +=4){
      AddDot4x4(k, &A( i,0 ), lda, &B( 0,j ), ldb , &C( i,j ), ldc);
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
      Innerkernel(ib, n, jb, &A(i, j), lda, &B(j,0), ldb, &C(i, 0), ldc);
            
    }
  }
}