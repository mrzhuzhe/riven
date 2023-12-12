#include <iostream>
int main(){
    //  valgrind --leak-check=yes ./build/bin/main
    int* x = (int*)malloc(10 * sizeof(int));
    x[10] = 0;        // problem 1: heap block overrun
                      // problem 2: memory leak -- x not freed
    return 0;
}