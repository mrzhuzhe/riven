//  https://zhuanlan.zhihu.com/p/393636855
/*
  def conv2d(input_numpy, kernel_weight_numpy, kernel_bias_numpy, padding = 0):
    B, Ci, Hi, Wi = input_numpy.shape
    input_pad_numpy = torch.zeros(B, Ci, Hi+2*padding, Wi+2*padding)
    if padding > 0:
        input_pad_numpy[:, :, padding:-padding, padding:-padding] = input_numpy
    else:
        input_pad_numpy = input_numpy
    B, Ci, Hi, Wi = input_pad_numpy.shape
    Co, Ci, Hf, Wf = kernel_weight_numpy.shape
    Ho, Wo = Hi - Hf + 1, Wi - Wf + 1
    # conv2d weight 7 loop
    out = np.zeros((B,Co,Ho,Wo))
    for b in range(B):
        for i in range(Ho):
            for j in range(Wo):
                for k in range(Co):
                    for l in range(Hf):
                        for m in range(Wf):
                            for n in range(Ci):
                                out[b,k,i,j] += input_pad_numpy[b,n,i+l,j+m]*kernel_weight_numpy[k,n,l,m]
    for b in range(B):
        for i in range(Ho):
            for j in range(Wo):
                for k in range(Co):
                    out[b,k,i,j] += kernel_bias_numpy[k]
    return out
*/

#define A(i,j) a[ (j)*lda + (i) ]
#define C(i,j) c[ (j)*lda + (i) ]
#define KERNEL(i,j) kernel[ (j)*kw + (i) ]

/*
1. need padding 
2. need bais
3. stride
*/

void MY_MMult( int m,  int k,  double *a, int lda, 
                                    int kw, int kh, double *kernel,                                    
                                    double *c, int ldc, int stride  )
{
  int i, j, w, h;
  int Wo = m - kw + 1;
  int Ho = k - kh + 1;
  for ( i=0; i<Wo; i++ ){
      for ( j=0; j<Ho; j++ ){
        for (w = 0; w < kw; w++ ){
          for (h = 0; h < kh; h++){
            C( i,j ) = C( i,j ) + A( i + w, j + h) * KERNEL(w, h);          
          }
        }  
      }
    }
}


  
