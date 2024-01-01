#include <iostream>
#include "test_speculation.h"

int main(){

    printf("Intel ASM test go \n");  
    int BUFF_SIZE = 10;
    AA* _array = (AA*)malloc(BUFF_SIZE*sizeof(AA));
    AA _aa;
    _aa.array = &_array;

    for (int i = BUFF_SIZE - 1 ; i>=0; i--){
        //_aa[i] = new 
        std::cout << &_array[i]  << std::endl;
    }
    std::cout << " "  << sizeof(_array[3]) << std::endl;
    //nullifyy_array(&_aa, BUFF_SIZE, &_array[3]);

    test_speculation(&_aa, BUFF_SIZE*4, &_array[3]);

    for (int i = BUFF_SIZE - 1 ; i>=0; i--){
        //_aa[i] = new 
        std::cout << &_array[i]  << std::endl;
    }

    free(_array);
    _array = nullptr;

    return 0;
}