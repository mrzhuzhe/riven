//  clang++ test.cpp outputs/output.o -o outputs/test.o
//  clang++ test.cpp outputs/lib.o -o outputs/test.o

#include <iostream>

extern "C" {
    double average(double, double);
}

int main() {
    std::cout << "average of 3.0 and 4.0: " << average(3.0, 4.0) << std::endl;
}