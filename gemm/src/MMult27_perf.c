// use perf tools to find bottle neck
// problem is in innerkernel 

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
  double d[4];
} v4df_t;


#define A(i,j) a[ (j)*lda + (i) ]
#define B(i,j) b[ (j)*ldb + (i) ]
#define C(i,j) c[ (j)*ldc + (i) ]

/* Routine for computing C = A * B + C */

void AddDot8x4(int k, double *a, int lda, double *b, int ldb, double *c, int ldc){
      
    int p;
    v4df_t vc00102030, vc01112131, vc02122232, vc03132333, vc40506070, vc41516171, vc42526272, vc43536373;
    v4df_t va0123, va4567;
    v4df_t vb0p, vb1p, vb2p, vb3p; 

    
   
    vc00102030.v = _mm256_setzero_pd(); 
    vc01112131.v = _mm256_setzero_pd(); 
    vc02122232.v = _mm256_setzero_pd();
    vc03132333.v = _mm256_setzero_pd();
    vc40506070.v = _mm256_setzero_pd();
    vc41516171.v = _mm256_setzero_pd();
    vc42526272.v = _mm256_setzero_pd();
    vc43536373.v = _mm256_setzero_pd();

    for ( p=0; p<k; p++ ){

      //  数据需要对其
      //  https://stackoverflow.com/questions/33373318/avx-segmentation-fault-on-linux
      // unalign will suddenly fail but align will be a little slower
      va0123.v = _mm256_load_pd((double *) a);      
      va4567.v = _mm256_load_pd((double *) a+4);

      //va0123.v = _mm256_loadu_pd((double *) a);      
      //va4567.v = _mm256_loadu_pd((double *) a+4);
      
      a += 8;

      vb0p.v = _mm256_broadcast_sd((double *) b);
      vb1p.v = _mm256_broadcast_sd((double *) (b + 1));
      vb2p.v = _mm256_broadcast_sd((double *) (b + 2));
      vb3p.v = _mm256_broadcast_sd((double *) (b + 3));

      //vc00102030.v += va0123.v * vb0p.v;                    
      vc00102030.v += (va0123.v * vb0p.v);     
      vc01112131.v += (va0123.v * vb1p.v);          
      vc02122232.v += (va0123.v * vb2p.v); 
      vc03132333.v += (va0123.v * vb3p.v);   
      //  _mm256_mul_pd _mm256_add_pd
      vc40506070.v += va4567.v * vb0p.v;   
      vc41516171.v += va4567.v * vb1p.v;   
      vc42526272.v += va4567.v * vb2p.v;   
      vc43536373.v += va4567.v * vb3p.v;      

      b += 4;

    }

    //
    _mm256_store_pd(c, _mm256_add_pd(_mm256_load_pd(c), vc00102030.v));
    _mm256_store_pd((c + ldc), _mm256_add_pd(_mm256_load_pd((c + ldc)), vc01112131.v));
    _mm256_store_pd((c + ldc * 2), _mm256_add_pd(_mm256_load_pd((c + ldc * 2)), vc02122232.v));
    _mm256_store_pd((c + ldc * 3), _mm256_add_pd(_mm256_load_pd((c + ldc * 3)), vc03132333.v));
    

    //
    _mm256_store_pd((c + 4), _mm256_add_pd(_mm256_load_pd((c + 4)), vc40506070.v));
    _mm256_store_pd((c + ldc + 4), _mm256_add_pd(_mm256_load_pd((c + ldc + 4)), vc41516171.v));
    _mm256_store_pd((c + ldc * 2 + 4), _mm256_add_pd(_mm256_load_pd((c + ldc * 2 + 4)), vc42526272.v));
    _mm256_store_pd((c + ldc * 3 + 4), _mm256_add_pd(_mm256_load_pd((c + ldc * 3 + 4)), vc43536373.v));

            
};


void PackMatrixA(int k, double *a, int lda, double *a_to){
  int j;
  for (j=0; j <k ; j++){
    double *a_ij_pntr = &A(0, j);
    *a_to++ = *a_ij_pntr;
    *a_to++ = *(a_ij_pntr+1);
    *a_to++ = *(a_ij_pntr+2);
    *a_to++ = *(a_ij_pntr+3);
    *a_to++ = *(a_ij_pntr+4);
    *a_to++ = *(a_ij_pntr+5);
    *a_to++ = *(a_ij_pntr+6);
    *a_to++ = *(a_ij_pntr+7);
  }
}


void PackMatrixB(int k, double *b, int ldb, double *b_to){
  int i;
  double 
    *b0p = &B(0, 0), *b1p = &B(0,1),
    *b2p = &B(0, 2), *b3p = &B(0,3)
    /*,
    *b4p = &B(0, 4), *b5p = &B(0,5),
    *b6p = &B(0, 6), *b7p = &B(0,7)
    */
   ; 

  for (i=0; i < k ; i++){    
    *b_to++ = *(b0p++);
    *b_to++ = *(b1p++);
    *b_to++ = *(b2p++);
    *b_to++ = *(b3p++);
    /*
    *b_to++ = *(b4p++);
    *b_to++ = *(b5p++);
    *b_to++ = *(b6p++);
    *b_to++ = *(b7p++);
    */
  }
}

void Innerkernel(int m, int n, int k
, double *a, int lda
, double *b, int ldb
, double *c, int ldc
, int first_time
, double *packA
, double *packB)
{
  int i, j;
  // todo this part need align and malloc
  for (j = 0; j < n; j+=4){
    if ( first_time ) {
      PackMatrixB(k, &B(0, j), ldb, &packB[j*k]);
    }
    
    for (i = 0; i < m; i +=8){
      //if ( j == 0 ) PackMatrixA(k, &A(i, 0), lda, ((packA + i*k))); // also ok
      if ( j == 0 ) PackMatrixA(k, &A(i, 0), lda, (&(packA[i*k])));
      AddDot8x4(k, &packA[i*k], 8, &packB[j*k], k , &C( i,j ), ldc);
    }
  }
}

void MY_MMult( int m, int n, int k, double *a, int lda, 
                                    double *b, int ldb,
                                    double *c, int ldc )
{
  int i, j, jb, ib;
  double *packA;
  double *packB;

  const int align = 32;
  packA = ( double * ) aligned_alloc(align, kc * mc * sizeof( double ) );
  packB = ( double * ) aligned_alloc(align, kc * nb * sizeof( double ) );

  for ( j=0; j< k ; j+=kc ){        /* Loop over the columns of C */
    jb = min(k-j, kc);
    for ( i=0; i<m; i+=mc ){        /* Loop over the rows of C */
      /* Update the C( i,j ) with the inner product of the ith row of A
	 and the jth column of B */
      ib = min(m-i, mc);
      Innerkernel(ib, n, jb, &A(i, j), lda, &B(j,0), ldb, &C(i, 0), ldc, i==0, packA, packB);
            
    }
  }

  free(packA);
  free(packB);

}