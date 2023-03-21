
void init_buffer(float *data, const int size){
    for (int i =0 ;i < size; i++)
        //data[i] = rand() / (float)RAND_MAX;
        data[i] = 1.f;
}

__global__ void vecAdd_kernel(float *c, const float *a, const float *b){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i< 500; i++)
        c[idx] = a[idx] + b[idx];
    //printf("%f\n", c[idx]);
}


void init_input(float *data, int size){
    for (int i = 0; i< size; i++){
        //data[i] = (rand() & 0xFF) / (float)RAND_MAX;
        data[i] = 1.f;
    }
}

float get_cpu_result(float *data, int size){
    double result = 0.f;
    for (int i = 0; i< size; i++)
        result += data[i];
    return (float)result;
}
