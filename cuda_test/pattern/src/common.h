#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdarg.h>
#include <iostream>

void random_init(float *data, int length)
{
    for (int i = 0; i < length; i++) {
        //data[i] = (rand() & 0xFFFF) / (float)RAND_MAX;
        data[i] = 1.f;
    }
}

bool value_test(float *a, float *b, int length){
    float e = 0.000001;
    for (int i = 0; i< length; i++){
        if (abs(a[i]-b[i])>=e){
            printf("valid fail %d \n", i);
            return false;
        }
            
    }
    return true;
}

void generate_data(float *h_buffer, int num_row, int num_col){
    for (int row = 0; row < num_row; row ++){
        for (int col = 0; col < num_col; col++){
            h_buffer[row*num_col+col] = (rand() & 0xFFFF) / (float)RAND_MAX;
            //h_buffer[row*num_col+col] = 1.f;
        }
    }
}

void generate_filter(float *h_filter, int filter_size){
    float blur_kernel_sigma = 2;
    float sum_filter = 0.f;

    for (int row = -filter_size/2; row <= filter_size/2; row++){
        for (int col = -filter_size/2; col <= filter_size/2; col++){
            float filterValue = expf(-(float)(col*col+row*row)/(2.f * blur_kernel_sigma * blur_kernel_sigma));
            h_filter[(row + filter_size/2)*filter_size + col + filter_size/2] = filterValue;
            sum_filter += filterValue;
        }
    }
    float normalizeFactor = 1.f / sum_filter;
    for (int row = -filter_size/2; row <= filter_size/2; row++){
        for (int col = -filter_size/2; col <= filter_size/2; col++){
            h_filter[(row + filter_size/2)*filter_size + col + filter_size/2] *= normalizeFactor;
        }
    }
}

// cpp 11
const auto validation = value_test;



void print_val(float *h_list, int length, ...)
{
    va_list argptr;
    va_start(argptr, length);

    printf("%s\t", va_arg(argptr, char *));
    for (int i = 0; i < length; i++)
        printf("%7.4f\t", h_list[i]);
    printf("\n");
}



// Initialize data on the host.
void initialize_data(unsigned int *dst, unsigned int nitems)
{
  // Fixed seed for illustration
  srand(2047);

  // Fill dst with random values
  for (unsigned i = 0 ; i < nitems ; i++)
    dst[i] = rand() % nitems ;
}


// Verify the results.
void check_results( int n, unsigned int *results_d )
{
  unsigned int *results_h = new unsigned[n];
  cudaMemcpy( results_h, results_d, n*sizeof(unsigned), cudaMemcpyDeviceToHost );
  for( int i = 1 ; i < n ; ++i )
    if( results_h[i-1] > results_h[i] )
    {
      std::cout << "Invalid item[" << i-1 << "]: " << results_h[i-1] << " greater than " << results_h[i] << std::endl;
      exit(EXIT_FAILURE);
    }
  std::cout << "OK" << std::endl;
  delete[] results_h;
}

#endif  