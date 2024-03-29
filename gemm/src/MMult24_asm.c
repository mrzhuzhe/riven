// https://mrzhuzhe.github.io/ulmBLAS-sites/page08/index.html

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

void AddDot8x4(const int k, const double *a, int lda, const double *b, int ldb, double *c, int ldc){    
    //  How to Use Inline Assembly Language in C Code
    //  https://gcc.gnu.org/onlinedocs/gcc/extensions-to-the-c-language-family/how-to-use-inline-assembly-language-in-c-code.html     
    __asm__ volatile
        (
        "movl      %0,      %%esi    \n\t"  // k (32 bit) stored in %esi
        "movq      %1,      %%rax    \n\t"  // Address of A stored in %rax
        "movq      %2,      %%rbx    \n\t"  // Address of B stored in %rbx
        "movq      %3,      %%rcx    \n\t"  // Address of C stored in %rcx

        "                            \n\t"
        "vxorpd    %%ymm8, %%ymm8,  %%ymm8   \n\t"  // vc00102030 = _mm256_setzero_pd()
        "vmovapd   %%ymm8,  %%ymm9   \n\t"  // vc01112131 = _mm256_setzero_pd()
        "vmovapd   %%ymm8, %%ymm10  \n\t"  // vc02122232 = _mm256_setzero_pd()
        "vmovapd   %%ymm8, %%ymm11  \n\t"  // vc03132333 = _mm256_setzero_pd()
        "vmovapd   %%ymm8, %%ymm12  \n\t"  // vc40506070 = _mm256_setzero_pd()
        "vmovapd   %%ymm8, %%ymm13  \n\t"  // vc41516171 = _mm256_setzero_pd()
        "vmovapd   %%ymm8, %%ymm14  \n\t"  // vc42526272 = _mm256_setzero_pd()
        "vmovapd   %%ymm8, %%ymm15  \n\t"  // vc43536373 = _mm256_setzero_pd()
        "                            \n\t"

        "testl     %%esi,   %%esi    \n\t"  // if p==k start writeback to C
        "je        .DWRITEBACK%=     \n\t"
        "                            \n\t"
        ".DLOOP%=:                   \n\t"  // for p = 0, k do
        "                            \n\t"
        "vmovapd    0(%%rax), %%ymm0   \n\t"  // va0123 = _mm256_load_pd(a)
        "vmovapd  32(%%rax), %%ymm1   \n\t"  // va4567 = _mm256_load_pd(a+4)
        "addq      $0x40,     %%rax    \n\t"  // a += 8;
        "                            \n\t"

        "vbroadcastsd    0(%%rbx),   %%ymm2    \n\t" // vb0p = _mm256_broadcast_sd(b);
        "vbroadcastsd    8(%%rbx),   %%ymm3    \n\t" // vb0p = _mm256_broadcast_sd(b+1);
        "vbroadcastsd    16(%%rbx),   %%ymm4    \n\t" // vb0p = _mm256_broadcast_sd(b+2);
        "vbroadcastsd    24(%%rbx),   %%ymm5    \n\t" // vb0p = _mm256_broadcast_sd(b+3);
        "                            \n\t"

        // this part only use 2 regs for temp 
        "vmulpd           %%ymm0,  %%ymm2, %%ymm6  \n\t"  //  vc00102030.v += _mm256_mul_pd(va0123.v, vb0p.v);  
        "vaddpd           %%ymm8,  %%ymm6, %%ymm8  \n\t" 
        "vmulpd           %%ymm0,  %%ymm3, %%ymm7  \n\t"  //  vc01112131.v += _mm256_mul_pd(va0123.v, vb1p.v); 
        "vaddpd           %%ymm9,  %%ymm7, %%ymm9  \n\t" 
        "vmulpd           %%ymm0,  %%ymm4, %%ymm6  \n\t"  //  vc02122232.v += _mm256_mul_pd(va0123.v, vb2p.v); 
        "vaddpd           %%ymm10,  %%ymm6, %%ymm10 \n\t" 
        "vmulpd           %%ymm0,  %%ymm5, %%ymm7 \n\t"  //  vc03132333.v += _mm256_mul_pd(va0123.v, vb3p.v);  
        "vaddpd           %%ymm11,  %%ymm7, %%ymm11 \n\t" 
        "                            \n\t"

        "vmulpd           %%ymm1,  %%ymm2, %%ymm6  \n\t"  //  vc40506070.v += _mm256_mul_pd(va4567.v, vb0p.v); 
        "vaddpd           %%ymm12,  %%ymm6, %%ymm12  \n\t" 
        "vmulpd           %%ymm1,  %%ymm3, %%ymm7  \n\t"  //  vc41516171.v += _mm256_mul_pd(va4567.v, vb1p.v); 
        "vaddpd           %%ymm13,  %%ymm7, %%ymm13  \n\t" 
        "vmulpd           %%ymm1,  %%ymm4, %%ymm6  \n\t"  //  vc42526272.v += _mm256_mul_pd(va4567.v, vb2p.v); 
        "vaddpd           %%ymm14,  %%ymm6, %%ymm14  \n\t" 
        "vmulpd           %%ymm1,  %%ymm5, %%ymm7  \n\t"  //  vc42526272.v += _mm256_mul_pd(va4567.v, vb2p.v);  
        "vaddpd           %%ymm15,  %%ymm7, %%ymm15  \n\t"
        
        "                            \n\t"
        "addq      $0x20,     %%rbx    \n\t"  // b += 4;
        "decl      %%esi             \n\t"  // p ++
        "jne       .DLOOP%=          \n\t"  // go back
        "                            \n\t"
        
        ".DWRITEBACK%=:              \n\t"  // Fill c with computed values
        "                            \n\t"
        
        "movq                   %4, %%rdi            \n\t" // load address of ldc
	      "leaq           (,%%rdi,8), %%rdi            \n\t" // rdi = ldc * sizeof(double)

        "vmovapd    0(%%rcx), %%ymm2   \n\t"             // _mm256_store_pd(c, _mm256_add_pd(_mm256_load_pd(c), vc00102030.v));
        "vaddpd   %%ymm8,    %%ymm2, %%ymm8 \n\t"  
        "vmovapd           %%ymm8,   0(%%rcx)         \n\t" 
        
        "vmovapd    32(%%rcx), %%ymm2   \n\t"      // _mm256_store_pd((c + 4), _mm256_add_pd(_mm256_load_pd((c + 4)), vc40506070.v));
        "vaddpd   %%ymm12,    %%ymm2, %%ymm12 \n\t"  
        "vmovapd           %%ymm12,   32(%%rcx)         \n\t" 

        "addq              %%rdi,   %%rcx   \t\n"  

        "vmovapd    (%%rcx), %%ymm3   \n\t"       // _mm256_store_pd((c + ldc), _mm256_add_pd(_mm256_load_pd((c + ldc)), vc01112131.v));
        "vaddpd   %%ymm9,  %%ymm3, %%ymm9 \n\t"  
        "vmovapd           %%ymm9,   (%%rcx)         \n\t" 


        "vmovapd    32(%%rcx), %%ymm3   \n\t"   // __mm256_store_pd((c + ldc + 4), _mm256_add_pd(_mm256_load_pd((c + ldc + 4)), vc41516171.v));
        "vaddpd   %%ymm13,    %%ymm3, %%ymm13 \n\t"  
        "vmovapd           %%ymm13,   32(%%rcx)         \n\t" 

        "addq              %%rdi,   %%rcx   \t\n"  

        "vmovapd    0(%%rcx), %%ymm4   \n\t"         // _mm256_store_pd((c + ldc * 2), _mm256_add_pd(_mm256_load_pd((c + ldc * 2)), vc02122232.v));
        "vaddpd   %%ymm10,   %%ymm4, %%ymm10 \n\t"  
        "vmovapd           %%ymm10,  0(%%rcx)         \n\t" 
                
        "vmovapd    32(%%rcx), %%ymm4   \n\t"   // _mm256_store_pd((c + ldc * 2 + 4), _mm256_add_pd(_mm256_load_pd((c + ldc * 2 + 4)), vc42526272.v));
        "vaddpd   %%ymm14,    %%ymm4, %%ymm14 \n\t"      
        "vmovapd           %%ymm14,   32(%%rcx)         \n\t" 
        
        "addq              %%rdi,   %%rcx   \t\n"                
        
        "vmovapd    (%%rcx), %%ymm5   \n\t"          // _mm256_store_pd((c + ldc * 3), _mm256_add_pd(_mm256_load_pd((c + ldc * 3)), vc03132333.v));
        "vaddpd   %%ymm11,   %%ymm5, %%ymm11 \n\t"  
        "vmovapd           %%ymm11,  (%%rcx)         \n\t"        
        "                            \n\t"

        "vmovapd    32(%%rcx), %%ymm5   \n\t"   // _mm256_store_pd((c + ldc * 3 + 4), _mm256_add_pd(_mm256_load_pd((c + ldc * 3 + 4)), vc43536373.v));
        "vaddpd   %%ymm15,    %%ymm5, %%ymm15 \n\t"  
        "vmovapd           %%ymm15,   32(%%rcx)         \n\t"         
        "                                            \n\t"
        : // output
        : // input
            "m" (k),      // 0
            "m" (a),      // 1
            "m" (b),      // 2
            "m" (c),      // 3
            "m" (ldc)     // 4
        : // register clobber list
            "rax", "rbx", "rcx", "esi", "rdi",
            "xmm0", "xmm1", "xmm2", "xmm3",
            "xmm4", "xmm5", "xmm6", "xmm7",
            "xmm8", "xmm9", "xmm10", "xmm11",
            "xmm12", "xmm13", "xmm14", "xmm15"
        );
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

void Innerkernel(int m, int n, int k, double *a, int lda, double *b, int ldb, double *c, int ldc, int first_time)
{
  int i, j;
  // todo this part need align and malloc
  double packedA[mc*kc] __attribute__ ((aligned (32)));
  static double packedB[kc*nb] __attribute__ ((aligned (32)));
  for (j = 0; j < n; j+=4){
    if ( first_time ) {
      PackMatrixB(k, &B(0, j), ldb, &packedB[j*k]);
    }
    
    for (i = 0; i < m; i +=8){
      if ( j == 0 ) PackMatrixA(k, &A(i, 0), lda, &packedA[i*k]);
      AddDot8x4(k, &packedA[i*k], 8, &packedB[j*k], k , &C( i,j ), ldc);
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
      //printf(" %d ", jb);
      Innerkernel(ib, n, jb, &A(i, j), lda, &B(j,0), ldb, &C(i, 0), ldc, i==0);
            
    }
  }
}