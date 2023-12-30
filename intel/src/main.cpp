#include <iostream>
#include "test.h"

int main(){
    printf("Intel ASM test go \n");  
    int* a;
    a = new int{123};
    
    test(a);
    std::cout << "first param a: " << *a << std::endl;

    delete a;
    a = nullptr;
    return 0;
}