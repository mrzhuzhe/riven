#include <stdio.h>
// #include <malloc.h>
#include <stdlib.h>

#include "parameters.h"

void REF_MMult(int, int, double *, int, int, int, double*, double *, int, int );
void MY_MMult(int, int, double *, int, int, int, double*, double *, int, int );
void copy_matrix(int, int, double *, int, double *, int );
void random_matrix(int, int, double *, int, int);
double compare_matrices( int, int, double *, int, double *, int );
void print_matrix( int, int, double *, int );

double dclock();

int main()
{

  const int debug = 0;
  int 
    p, 
    m, k,
    kw, kh,
    lda, 
    rep;

  double
    dtime, dtime_best,        
    gflops, 
    diff;

  double 
    *a, *c, *cref, *cold, *kernel;    
  
  printf( "MY_MMult = [\n" );

  kw = 3;
  kh = 3;

  for ( p=PFIRST; p<=PLAST; p+=PINC ){
    
    if (debug){
      p = 5;
    }

    m = ( M == -1 ? p : M );
    k = ( K == -1 ? p : K );

    gflops = 2.0 * m * k * 1.0e-09;

    lda = ( LDA == -1 ? m : LDA );

    /* Allocate space for the matrices */
    /* Note: I create an extra column in A to make sure that
       prefetching beyond the matrix does not cause a segfault */
    a = ( double * ) malloc( lda * (k+1) * sizeof( double ) );     
    kernel = ( double * ) malloc( kw * kh * sizeof( double ) );      
    c = ( double * ) malloc( lda * (k+1) * sizeof( double ) );
    cold = ( double * ) malloc( lda * (k+1) * sizeof( double ) );
    cref = ( double * ) malloc( lda * (k+1) * sizeof( double ) );

    const int padding = 1;
    const int stride = 2;

    /* Generate random matrices A, B, Cold */
    random_matrix( m, k, a, lda, padding);
    random_matrix( kw, kh, kernel, kw, 0);

    /* Run the reference implementation so the answers can be compared */     
    if (debug){
      MY_MMult( m, k, a, lda, kw, kh, kernel, cref, lda, stride);
      print_matrix(m, k, a, lda);
      print_matrix(kw, kh, kernel, kw);
      print_matrix(m, k, cref, lda);

      free( a );
      free( c );
      free( cold );
      free( cref );

      return 0;
    }    

    REF_MMult( m, k, a, lda, kw, kh, kernel, cref, lda, stride );

    for ( rep=0; rep<NREPEATS; rep++ ){
      copy_matrix( m, k, cold, lda, c, lda );

      
      dtime = dclock();

      MY_MMult( m, k, a, lda, kw, kh, kernel, c, lda, stride );
      
      dtime = dclock() - dtime;

      if ( rep==0 )
	dtime_best = dtime;
      else
	dtime_best = ( dtime < dtime_best ? dtime : dtime_best );
    }

    diff = compare_matrices( m, k, c, lda, cref, lda );
    
    printf( "%d %le %le \n", p, gflops / dtime_best, diff );
    fflush( stdout );
    
    free( a );
    free( c );
    free( cold );
    free( cref );
  }

  printf( "];\n" );

  exit( 0 );
}

