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
           AddDot8x4(k, 
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