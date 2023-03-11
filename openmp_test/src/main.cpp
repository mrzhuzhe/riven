#include <iostream>
#include <omp.h>

float cpuPi(int num) {
    float sum = 0.0;
    float temp;
    for (int i = 0; i< num; i++ ){
        temp = (i + 0.5f)/num;
        sum += 4 / ( 1 + temp * temp);
    }
    return sum/num;
};

int main(){
    std::cout << cpuPi(10) << std::endl;
    return 0;
}