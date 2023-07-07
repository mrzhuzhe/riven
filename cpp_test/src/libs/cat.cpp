#include <iostream>


extern "C" {
    void print_name(const char*);
}

void print_name(const char* type){
    printf("cat name is %s \n", type);
}
        