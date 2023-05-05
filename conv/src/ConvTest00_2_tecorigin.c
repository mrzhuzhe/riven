#define A(i,j) a[ (j)*lda + (i) ]
#define C(i,j) c[ (j)*lda + (i) ]
#define KERNEL(i,j) kernel[ (j)*kw + (i) ]

void MY_MMult( int m,  int k,  double *a, int lda, 
                                    int kw, int kh, double *kernel,                                    
                                    double *c, int ldc, int stride )
{  
  int i, j, w, h;
  int Wo = (m - kw) / stride + 1;
  int Ho = (k - kh) / stride + 1;
  const int BLOCK = 256;
  for ( i=0; i< Wo; i+= BLOCK ){
      for ( j=0; j< Ho; j+= BLOCK ){

        // 在 block 内部循环 这里的话 要把 a 矩阵 截取出 a[BLOCK][dim] pack  
        // 此处处理数据缩小到L1
        for (w = 0; w < kw; w += 4 ){
          for (h = 0; h < kh; h +=4 ){

            C( i,j ) = C( i,j ) + A( i * stride + w, j * stride + h) * KERNEL(w, h);      
            // 此处 C 的赋值 用 union 重新整理成向量的连续内存    
          }
        }

      }
    }
}


  
