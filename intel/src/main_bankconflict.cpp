#include <iostream>
#include "test_bankconflict.h"
#include "test_bankconflict_b.h"
int main(){


    printf("Intel ASM test go \n");  
    float sum = 0;
    int BUFF_SIZE = 3200 * 3200;
    int *A = (int*)malloc(BUFF_SIZE*sizeof(int));
    int *B = (int*)malloc(BUFF_SIZE*sizeof(int));
    int *C = (int*)malloc(BUFF_SIZE*sizeof(int));
    sum = 0;
    for (int i = 0; i < BUFF_SIZE; i++){
        A[i] = 1;
        B[i] = 1;
        C[i] = 1;
    }
    float start_time = clock();
    for (int i = 0; i < BUFF_SIZE; i+=4){
        C[i] = A[i] + B[i];
        C[i+1] = A[i+1] + B[i+1];
        C[i+2] = A[i+2] + B[i+2];
        C[i+3] = A[i+3] + B[i+3];
    }
    float end_time = clock();
    std::cout << "sum " << C[0] << std::endl;
    std::cout << "cost " << (end_time - start_time) << std::endl;

    start_time = clock();
    test_bankconflict(A, B, C, BUFF_SIZE/4);
    end_time = clock();
    std::cout << "sum 2 " << C[0] << std::endl;
    std::cout << "cost 2 " << (end_time - start_time) << std::endl;

    start_time = clock();
    test_bankconflict_b(A, B, C, BUFF_SIZE/4);
    end_time = clock();
    std::cout << "sum 2 " << C[0] << std::endl;
    std::cout << "cost 2 " << (end_time - start_time) << std::endl;

    free(A);
    A = nullptr;
    free(B);
    B = nullptr;
    free(C);
    C = nullptr;

    return 0;
}