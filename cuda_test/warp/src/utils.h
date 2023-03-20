
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
