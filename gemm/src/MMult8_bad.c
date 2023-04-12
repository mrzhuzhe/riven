/* Create macros so that the matrices are stored in column-major order */
//  cpu 缓存同步问题

#define A(i,j) a[ (j)*lda + (i) ]
#define B(i,j) b[ (j)*ldb + (i) ]
#define C(i,j) c[ (j)*ldc + (i) ]

/* Routine for computing C = A * B + C */

void AddDot( int, double *, int, double *, double * );


void AddDot1x4( int k, double *a, int lda, double *b, int ldb, double *c, int ldc)
{
    //register double reg_a = A(0, p);  //https://zh.wikipedia.org/zh-hk/%E5%AF%84%E5%AD%98%E5%99%A8  // has a benifit to use register    
    register double reg_a_0, reg_a_1, reg_a_2, reg_a_3;
    reg_a_0 = 0.0;
    reg_a_1 = 0.0;
    reg_a_2 = 0.0;
    reg_a_3 = 0.0;
    double reg_a;
    double 
    /* Point to the current elements in the four columns of B */
    *bp0_pntr, *bp1_pntr, *bp2_pntr, *bp3_pntr; 

  
    bp0_pntr = &B(0 , 0);
    bp1_pntr = &B(0 , 1);
    bp2_pntr = &B(0 , 2);
    bp3_pntr = &B(0 , 3);
   
   for (int p = 0; p < k ; p+=4){
      

      reg_a = A(0, p);      
      ///*
      reg_a_0 += reg_a * *bp0_pntr++;
      reg_a_1 += reg_a * *bp1_pntr++;
      reg_a_2 += reg_a * *bp2_pntr++;
      reg_a_3 += reg_a * *bp3_pntr++;  
      //*/
      
      reg_a = A(0, p+1);      
      ///*
      reg_a_0 += reg_a * *bp0_pntr++;
      reg_a_1 += reg_a * *bp1_pntr++;
      reg_a_2 += reg_a * *bp2_pntr++;
      reg_a_3 += reg_a * *bp3_pntr++;  
      //*/

      reg_a = A(0, p+2);      
      ///*
      reg_a_0 += reg_a * *bp0_pntr++;
      reg_a_1 += reg_a * *bp1_pntr++;
      reg_a_2 += reg_a * *bp2_pntr++;
      reg_a_3 += reg_a * *bp3_pntr++;  
      //*/

      reg_a = A(0, p+3);      
      ///*
      reg_a_0 += reg_a * *bp0_pntr++;
      reg_a_1 += reg_a * *bp1_pntr++;
      reg_a_2 += reg_a * *bp2_pntr++;
      reg_a_3 += reg_a * *bp3_pntr++;  
      //*/

     /*     
     // faster multi ？
     reg_a_0 += reg_a * B(p , 0);
     reg_a_1 += reg_a * B(p , 1);
     reg_a_2 += reg_a * B(p , 2);
     reg_a_3 += reg_a * B(p , 3);  
     */
   }
   
  C(0, 0) += reg_a_0;
  C(0, 1) += reg_a_1;
  C(0, 2) += reg_a_2;
  C(0, 3) += reg_a_3; 
  
}

void MY_MMult( int m, int n, int k, double *a, int lda, 
                                    double *b, int ldb,
                                    double *c, int ldc )
{
  int i, j;

  for ( j=0; j<n; j+=4 ){        /* Loop over the columns of C */
    for ( i=0; i<m; i+=1 ){        /* Loop over the rows of C */
      /* Update the C( i,j ) with the inner product of the ith row of A
	 and the jth column of B */

      
      AddDot1x4(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc);
        
    }
  }
}


