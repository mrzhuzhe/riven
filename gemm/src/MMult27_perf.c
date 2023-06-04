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

#define mc 96
#define kc 256
#define nc 4096

typedef union {
  __m256d v;
  double d[4];
} v4df_t;


#define A(i,j) a[ (j)*lda + (i) ]
#define B(i,j) b[ (j)*ldb + (i) ]
#define C(i,j) c[ (j)*ldc + (i) ]


void bl_dgemm_int_8x4(
                        int      k,
                        double*  a,
                        int lda,
                        double*  b,
                        int ldb,
                        double*  c,
                        int ldc
                      )
{
    const int cs_c = ldc;
    int rs_c = 1;
    double alpha_val = 1.0, beta_val = 1.0;
    double *alpha, *beta;

    alpha = &alpha_val;
    beta  = &beta_val;

    ////void* a_next = bli_auxinfo_next_a( data );
    //void* b_next = bli_auxinfo_next_b( data );
    const int __stride__ = 1;    

    int k_iter  = (unsigned long long)k / __stride__;
    int k_left  = (unsigned long long)k % __stride__;

    int i;

    double *c00, *c01, *c02, *c03;
    double *c40, *c41, *c42, *c43;

    // Quad registers.
    __m256d va0_3, va4_7;
    __m256d vA0_3, vA4_7;
    __m256d vb0, vb1, vb2, vb3;
    __m256d vb;
    __m256d vB0;

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

    __m128d aa, bb;
    

    //__asm__ volatile( "prefetcht0 0(%0)          \n\t" : :"r"(a)  );
    //__asm__ volatile( "prefetcht2 0(%0)          \n\t" : :"r"(b_next)  );
    //__asm__ volatile( "prefetcht0 0(%0)          \n\t" : :"r"(c)  );



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

      // Load va0_3 (Prefetch)
          // Prefetch A03
      //vA0_3 = _mm256_load_pd( a + 8 );

      // Iteration 0.
      vtmp = _mm256_mul_pd( va0_3, vb0 );
      va0_3b_0 = _mm256_add_pd( va0_3b_0, vtmp );

      vtmp = _mm256_mul_pd( va4_7, vb0 );
      va4_7b_0 = _mm256_add_pd( va4_7b_0, vtmp );

      // Load va4_7 (Prefetch)
          // Prefetch A47
      //vA4_7 = _mm256_load_pd( a + 12 );

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
}

/* Routine for computing C = A * B + C */

void AddDot8x4(int k, double *a, int lda, double *b, int ldb, double *c, int ldc){
      
    int p;
    register v4df_t vc00102030, vc01112131, vc02122232, vc03132333, vc40506070, vc41516171, vc42526272, vc43536373;
    register v4df_t va0123, va4567;
    register v4df_t vb0p, vb1p, vb2p, vb3p; 

    
   
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
      vc00102030.v += _mm256_mul_pd(va0123.v, vb0p.v);     
      vc01112131.v += _mm256_mul_pd(va0123.v, vb1p.v);          
      vc02122232.v += _mm256_mul_pd(va0123.v, vb2p.v); 
      vc03132333.v += _mm256_mul_pd(va0123.v, vb3p.v);   
      //  _mm256_mul_pd _mm256_add_pd
      vc40506070.v += _mm256_mul_pd(va4567.v, vb0p.v);  
      vc41516171.v += _mm256_mul_pd(va4567.v, vb1p.v);  
      vc42526272.v += _mm256_mul_pd(va4567.v, vb2p.v);  
      vc43536373.v += _mm256_mul_pd(va4567.v, vb3p.v);     

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


inline void packA_mcxkc_d(
        int    m,
        int    k,
        double *XA,
        int    ldXA,
        int    offseta,
        double *packA
        )
{
    int    i, p;
    double *a_pntr[ 8 ];

    for ( i = 0; i < m; i ++ ) {
        a_pntr[ i ] = XA + ( offseta + i );
    }

    for ( i = m; i < 8; i ++ ) {
        a_pntr[ i ] = XA + ( offseta + 0 );
    }

    for ( p = 0; p < k; p ++ ) {
        for ( i = 0; i < 8; i ++ ) {
            *packA = *a_pntr[ i ];
            packA ++;
            a_pntr[ i ] = a_pntr[ i ] + ldXA;
        }
    }
}


/*
 * --------------------------------------------------------------------------
 */

inline void packB_kcxnc_d(
        int    n,
        int    k,
        double *XB,
        int    ldXB, // ldXB is the original k
        int    offsetb,
        double *packB
        )
{
    int    j, p; 
    double *b_pntr[ 4 ];

    for ( j = 0; j < n; j ++ ) {
        b_pntr[ j ] = XB + ldXB * ( offsetb + j );
    }

    for ( j = n; j < 4; j ++ ) {
        b_pntr[ j ] = XB + ldXB * ( offsetb + 0 );
    }

    for ( p = 0; p < k; p ++ ) {
        for ( j = 0; j < 4; j ++ ) {
            *packB ++ = *b_pntr[ j ] ++;
        }
    }
}

/*
 * --------------------------------------------------------------------------
 */
void bl_macro_kernel(
        int    m,
        int    n,
        int    k,
        double *packA,
        double *packB,
        double *c,
        int    ldc
        )
{
    int    i, ii, j;
    //aux_t  aux;
    char *str;

    //aux.b_next = packB;

    for ( j = 0; j < n; j += 4 ) {                        // 2-th loop around micro-kernel
        //aux.n  = min( n - j, 4 );
        for ( i = 0; i < m; i += 8 ) {                    // 1-th loop around micro-kernel
            /*
            aux.m = min( m - i, 8 );
            if ( i + 8 >= m ) {
                aux.b_next += 4 * k;
            }

            ( *bl_micro_kernel ) (
                    k,
                    &packA[ i * k ],
                    &packB[ j * k ],
                    &C[ j * ldc + i ],
                    (unsigned long long) ldc,
                    &aux
                    );
            */
           
           //AddDot8x4(
           bl_dgemm_int_8x4(
            k,
            &packA[ i * k ],
            ldc,
            &packB[ j * k ],
            ldc,
            &C(i, j),
            ldc);
            
        }                                                        // 1-th loop around micro-kernel
    }                                                            // 2-th loop around micro-kernel
}

void MY_MMult( int m, int n, int k, double *a, int lda, 
                                    double *b, int ldb,
                                    double *c, int ldc )
{
  int    i, j, p;
  int    ic, ib, jc, jb, pc, pb;
  int    ir, jr;
  static double *packA, *packB;
  char   *str;

  const int align = 32;
  packA = ( double * ) aligned_alloc(align, kc * mc * sizeof( double ) );
  packB = ( double * ) aligned_alloc(align, kc * nc * sizeof( double ) );

  //packA  = bl_malloc_aligned( kc, ( mc + 1 ) , sizeof(double) );
  //packB  = bl_malloc_aligned( kc, ( nc + 1 ) , sizeof(double) );

  /*
  for ( j=0; j< k ; j+=kc ){        /* Loop over the columns of C * /
    jb = min(k-j, kc);
    for ( i=0; i<m; i+=mc ){        /* Loop over the rows of C * /
      /* Update the C( i,j ) with the inner product of the ith row of A
	 and the jth column of B * /
      ib = min(m-i, mc);
      Innerkernel(ib, n, jb, &A(i, j), lda, &B(j,0), ldb, &C(i, 0), ldc, i==0, packA, packB);
            
    }
  }
  */

  for ( jc = 0; jc < n; jc += nc ) {                                       // 5-th loop around micro-kernel
        jb = min( n - jc, nc );
        for ( pc = 0; pc < k; pc += kc ) {                                   // 4-th loop around micro-kernel
            pb = min( k - pc, kc );

            for ( j = 0; j < jb; j += 4 ) {
                packB_kcxnc_d(
                        min( jb - j, 4 ),
                        pb,
                        &B(pc, 0),
                        k, // should be ldXB instead
                        jc + j,
                        &packB[ j * pb ]
                        );
            }


            for ( ic = 0; ic < m; ic += mc ) {                               // 3-rd loop around micro-kernel

                ib = min( m - ic, mc );

                for ( i = 0; i < ib; i += 8 ) {
                    packA_mcxkc_d(
                            min( ib - i, 8 ),
                            pb,
                            &A(0, pc),
                            m,
                            ic + i,
                            &packA[ 0 * mc * pb + i * pb ]
                            );
                }

                bl_macro_kernel(
                        ib,
                        jb,
                        pb,
                        packA  + 0 * mc * pb,
                        packB,
                        &C(ic, jc), 
                        ldc
                        );
            }                                                                     // End 3.rd loop around micro-kernel
        }                                                                         // End 4.th loop around micro-kernel
    }         


  free(packA);
  free(packB);

}