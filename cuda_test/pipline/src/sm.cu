#include <iostream>

__global__ void single_kernel(int step){
    printf("loop %d\n", step);
}

int main(){
    int n_loop = 5;
    for (int i = 0; i < n_loop; i++){
        single_kernel<<<1, 1, 0, 0>>>(i);
    }
    cudaDeviceSynchronize();

    int n_stream = 5;
    cudaStream_t *ls_stream;
    ls_stream = (cudaStream_t*) new cudaStream_t[n_stream];

    for (int i = 0; i < n_stream; i++)
        cudaStreamCreate(&ls_stream[i]);

    for (int i = 0; i < n_stream; i++){
        //*
        if (i==3)
            single_kernel<<<1, 1, 0, 0>>>(i);
        else
            single_kernel<<<1, 1, 0, ls_stream[i]>>>(i);
        //*/
        //single_kernel<<<1, 1, 0, ls_stream[i]>>>(i);
        cudaStreamSynchronize(ls_stream[i]);  //  multi threads sync
    }
    cudaDeviceSynchronize();

    for (int i = 0; i < n_stream; i++)
        cudaStreamDestroy(ls_stream[i]);
    delete [] ls_stream;

    return 0;
}