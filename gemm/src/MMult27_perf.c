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
#define nb 1024

typedef union {
  __m256d v;
  double d[4];
} v4df_t;


#define A(i,j) a[ (j)*lda + (i) ]
#define B(i,j) b[ (j)*ldb + (i) ]
#define C(i,j) c[ (j)*ldc + (i) ]

/* Routine for computing C = A * B + C */

void AddDot8x4(int k, double *a, int lda, double *b, int ldb, double *c, int ldc){
      
    const int cs_c = ldc;
    const int rs_c = 1;
    double alpha_val = 1.0, beta_val = 0;
    double *alpha, *beta;

    alpha = &alpha_val;
    beta  = &beta_val;

    ////void* a_next = bli_auxinfo_next_a( data );
    //void* b_next = bli_auxinfo_next_b( data );
    const int __stride__ = 1;

    int k_iter  = (unsigned long long)k / __stride__;
    //int k_left  = (unsigned long long)k % __stride__;

    int i;

    double *c00, *c01, *c02, *c03;
    double *c40, *c41, *c42, *c43;

    // Quad registers.
    __m256d va0_3, va4_7;
    //__m256d vA0_3, vA4_7;
    __m256d vb0, vb1, vb2, vb3;
    //__m256d vb;
    //__m256d vB0;

    __m256d va0_3b_0, va4_7b_0; 
    __m256d va0_3b_1, va4_7b_1; 
    __m256d va0_3b_2, va4_7b_2; 
    __m256d va0_3b_3, va4_7b_3; 

    __m256d va0_3b0, va4_7b0; 
    __m256d va0_3b1, va4_7b1; 
    __m256d va0_3b2, va4_7b2; 
    __m256d va0_3b3, va4_7b3; 


    __m256d valpha, vbeta, vtmp; 
    __m256d vc0_3_0, vc0_3_1, vc0_3_2, vc0_3_3;
    __m256d vc4_7_0, vc4_7_1, vc4_7_2, vc4_7_3;

    va0_3b0 = _mm256_setzero_pd();
    va0_3b1 = _mm256_setzero_pd();
    va0_3b2 = _mm256_setzero_pd();
    va0_3b3 = _mm256_setzero_pd();

    va4_7b0 = _mm256_setzero_pd();
    va4_7b1 = _mm256_setzero_pd();
    va4_7b2 = _mm256_setzero_pd();
    va4_7b3 = _mm256_setzero_pd();

    va0_3b_0 = _mm256_setzero_pd();
    va0_3b_1 = _mm256_setzero_pd();
    va0_3b_2 = _mm256_setzero_pd();
    va0_3b_3 = _mm256_setzero_pd();

    va4_7b_0 = _mm256_setzero_pd();
    va4_7b_1 = _mm256_setzero_pd();
    va4_7b_2 = _mm256_setzero_pd();
    va4_7b_3 = _mm256_setzero_pd();

    // Load va0_3
    va0_3 = _mm256_load_pd( a );
    // Load va4_7
    va4_7 = _mm256_load_pd( a + 4 );

    // Load vb (b0,b1,b2,b3) 
    vb0 = _mm256_load_pd( b );
    

    for( i = 0; i < k_iter; ++i )
	{
		//__asm__ volatile( "prefetcht0 192(%0)          \n\t" : :"r"(a)  );

		// Iteration 0.
		vtmp = _mm256_mul_pd( va0_3, vb0 );
		va0_3b_0 = _mm256_add_pd( va0_3b_0, vtmp );

		vtmp = _mm256_mul_pd( va4_7, vb0 );
		va4_7b_0 = _mm256_add_pd( va4_7b_0, vtmp );

		// Shuffle vb (b1,b0,b3,b2)
		//	https://www.intel.com/content/www/us/en/docs/cpp-compiler/developer-guide-reference/2021-8/mm256-shuffle-pd.html
 		vb1 = _mm256_shuffle_pd( vb0, vb0, 0x5 );

		vtmp = _mm256_mul_pd( va0_3, vb1 );
		va0_3b_1 = _mm256_add_pd( va0_3b_1, vtmp );

		vtmp = _mm256_mul_pd( va4_7, vb1 );
		va4_7b_1 = _mm256_add_pd( va4_7b_1, vtmp );

		// Permute vb (b3,b2,b1,b0)
		//	https://www.intel.com/content/www/us/en/docs/cpp-compiler/developer-guide-reference/2021-8/mm256-permute2f128-pd.html
 		vb2 = _mm256_permute2f128_pd( vb1, vb1, 0x1 );

		// Load vb (b0,b1,b2,b3) (Prefetch)
 		//vB0 = _mm256_load_pd( b + 4 ); 

		vtmp = _mm256_mul_pd( va0_3, vb2 );
		va0_3b_2 = _mm256_add_pd( va0_3b_2, vtmp );

		vtmp = _mm256_mul_pd( va4_7, vb2 );
		va4_7b_2 = _mm256_add_pd( va4_7b_2, vtmp );

		// Shuffle vb (b2,b3,b0,b1)
 		vb3 = _mm256_shuffle_pd( vb2, vb2, 0x5 );

		vtmp = _mm256_mul_pd( va0_3, vb3 );
		va0_3b_3 = _mm256_add_pd( va0_3b_3, vtmp );

		vtmp = _mm256_mul_pd( va4_7, vb3 );
		va4_7b_3 = _mm256_add_pd( va4_7b_3, vtmp );
		
		va0_3 = _mm256_load_pd( a + 8 );
		va4_7 = _mm256_load_pd( a + 12 );
		vb0 = _mm256_load_pd( b + 4 ); 
		a += 8;
		b += 4;
	}

  vbeta = _mm256_broadcast_sd( beta );

	//	https://www.intel.com/content/www/us/en/docs/cpp-compiler/developer-guide-reference/2021-8/mm256-blend-pd.html
	__m256d vtmpa_0_3b_0 = _mm256_blend_pd( va0_3b_0, va0_3b_1, 0x6 );
	__m256d vtmpa_0_3b_1 = _mm256_blend_pd( va0_3b_1, va0_3b_0, 0x6 );

	__m256d vtmpa_0_3b_2 = _mm256_blend_pd( va0_3b_2, va0_3b_3, 0x6 );
	__m256d vtmpa_0_3b_3 = _mm256_blend_pd( va0_3b_3, va0_3b_2, 0x6 );

	__m256d vtmpa_4_7b_0 = _mm256_blend_pd( va4_7b_0, va4_7b_1, 0x6 );
	__m256d vtmpa_4_7b_1 = _mm256_blend_pd( va4_7b_1, va4_7b_0, 0x6 );

	__m256d vtmpa_4_7b_2 = _mm256_blend_pd( va4_7b_2, va4_7b_3, 0x6 );
	__m256d vtmpa_4_7b_3 = _mm256_blend_pd( va4_7b_3, va4_7b_2, 0x6 );


	valpha = _mm256_broadcast_sd( alpha );


	va0_3b0 = _mm256_permute2f128_pd( vtmpa_0_3b_0, vtmpa_0_3b_2, 0x30 );
	va0_3b3 = _mm256_permute2f128_pd( vtmpa_0_3b_2, vtmpa_0_3b_0, 0x30 );

	va0_3b1 = _mm256_permute2f128_pd( vtmpa_0_3b_1, vtmpa_0_3b_3, 0x30 );
	va0_3b2 = _mm256_permute2f128_pd( vtmpa_0_3b_3, vtmpa_0_3b_1, 0x30 );

	va4_7b0 = _mm256_permute2f128_pd( vtmpa_4_7b_0, vtmpa_4_7b_2, 0x30 );
	va4_7b3 = _mm256_permute2f128_pd( vtmpa_4_7b_2, vtmpa_4_7b_0, 0x30 );

	va4_7b1 = _mm256_permute2f128_pd( vtmpa_4_7b_1, vtmpa_4_7b_3, 0x30 );
	va4_7b2 = _mm256_permute2f128_pd( vtmpa_4_7b_3, vtmpa_4_7b_1, 0x30 );

  // Calculate address
  c00 = ( c + 0*rs_c + 0*cs_c );
  // Load
  //vc0_3_0 = _mm256_load_pd( c + 0*rs_c + 0*cs_c  );
  vc0_3_0 = _mm256_load_pd( c00  );
  // Scale by alpha
  vtmp = _mm256_mul_pd( valpha, va0_3b0);
  // Scale by beta
  vc0_3_0 = _mm256_mul_pd( vbeta, vc0_3_0 );
  // Add gemm result
  vc0_3_0 = _mm256_add_pd( vc0_3_0, vtmp );
  // Store back to memory
  _mm256_store_pd( c00, vc0_3_0 );

  // Calculate address
  c40 = ( c + 4*rs_c + 0*cs_c );
  // Load
  //vc4_7_0 = _mm256_load_pd( c + 4*rs_c + 0*cs_c  );
  vc4_7_0 = _mm256_load_pd( c40  );
  // Scale by alpha
  vtmp = _mm256_mul_pd( valpha, va4_7b0);
  // Scale by beta
  vc4_7_0 = _mm256_mul_pd( vbeta, vc4_7_0 );
  // Add gemm result
  vc4_7_0 = _mm256_add_pd( vc4_7_0, vtmp );
  // Store back to memory
  _mm256_store_pd( c40, vc4_7_0 );

  // Calculate address
  c01 = ( c + 0*rs_c + 1*cs_c );
  // Load
  //vc0_3_1 = _mm256_load_pd( c + 0*rs_c + 1*cs_c  );
  vc0_3_1 = _mm256_load_pd( c01  );
  // Scale by alpha
  vtmp = _mm256_mul_pd( valpha, va0_3b1);
  // Scale by beta
  vc0_3_1 = _mm256_mul_pd( vbeta, vc0_3_1 );
  // Add gemm result
  vc0_3_1 = _mm256_add_pd( vc0_3_1, vtmp );
  // Store back to memory
  _mm256_store_pd( c01, vc0_3_1 );


  // Calculate address
  c41 = ( c + 4*rs_c + 1*cs_c );
  // Load
  //vc4_7_1 = _mm256_load_pd( c + 4*rs_c + 1*cs_c  );
  vc4_7_1 = _mm256_load_pd( c41  );
  // Scale by alpha
  vtmp = _mm256_mul_pd( valpha, va4_7b1);
  // Scale by beta
  vc4_7_1 = _mm256_mul_pd( vbeta, vc4_7_1 );
  // Add gemm result
  vc4_7_1 = _mm256_add_pd( vc4_7_1, vtmp );
  // Store back to memory
  _mm256_store_pd( c41, vc4_7_1 );

  // Calculate address
  c02 = ( c + 0*rs_c + 2*cs_c );
  // Load
  //vc0_3_2 = _mm256_load_pd( c + 0*rs_c + 2*cs_c  );
  vc0_3_2 = _mm256_load_pd( c02 );
  // Scale by alpha
  vtmp = _mm256_mul_pd( valpha, va0_3b2);
  // Scale by beta
  vc0_3_2 = _mm256_mul_pd( vbeta, vc0_3_2 );
  // Add gemm result
  vc0_3_2 = _mm256_add_pd( vc0_3_2, vtmp );
  // Store back to memory
  _mm256_store_pd( c02, vc0_3_2 );

  // Calculate address
  c42 = ( c + 4*rs_c + 2*cs_c );
  // Load
  //vc4_7_2 = _mm256_load_pd( c + 4*rs_c + 2*cs_c  );
  vc4_7_2 = _mm256_load_pd( c42 );
  // Scale by alpha
  vtmp = _mm256_mul_pd( valpha, va4_7b2);
  // Scale by beta
  vc4_7_2 = _mm256_mul_pd( vbeta, vc4_7_2 );
  // Add gemm result
  vc4_7_2 = _mm256_add_pd( vc4_7_2, vtmp );
  // Store back to memory
  _mm256_store_pd( c42, vc4_7_2 );
  
  // Calculate address
  c03 = ( c + 0*rs_c + 3*cs_c );
  // Load
  //vc0_3_3 = _mm256_load_pd( c + 0*rs_c + 3*cs_c  );
  vc0_3_3 = _mm256_load_pd( c03 );
  // Scale by alpha
  vtmp = _mm256_mul_pd( valpha, va0_3b3);
  // Scale by beta
  vc0_3_3 = _mm256_mul_pd( vbeta, vc0_3_3 );
  // Add gemm result
  vc0_3_3 = _mm256_add_pd( vc0_3_3, vtmp );
  // Store back to memory
  _mm256_store_pd( c03, vc0_3_3 );

  // Calculate address
  c43 = ( c + 4*rs_c + 3*cs_c );
  // Load
  //vc4_7_3 = _mm256_load_pd( c + 4*rs_c + 3*cs_c  );
  vc4_7_3 = _mm256_load_pd( c43 );
  // Scale by alpha
  vtmp = _mm256_mul_pd( valpha, va4_7b3);
  // Scale by beta
  vc4_7_3 = _mm256_mul_pd( vbeta, vc4_7_3 );
  // Add gemm result
  vc4_7_3 = _mm256_add_pd( vc4_7_3, vtmp );
  // Store back to memory
  _mm256_store_pd( c43, vc4_7_3 );
          
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