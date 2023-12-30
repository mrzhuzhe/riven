#include <iostream>
#include "testf.h"

int main(){

//  https://en.wikipedia.org/wiki/X86_calling_conventions
    printf("Intel ASM test go \n");  
    float *a, *b, *c, *d, *e, *f, *g;
    a = new float{0};
    b = new float{0};
    c = new float{0};
    d = new float{0};
    e = new float{0};
    f = new float{0};
    g = new float{0};
    
    testf(a, b, c ,d, e, f, g, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1);
    std::cout << "first param a: " << *a << std::endl;
    std::cout << "second param a: " << *b << std::endl;
    std::cout << "third param a: " << *c << std::endl;
    std::cout << "fouth param a: " << *d << std::endl;
    std::cout << "five param a: " << *e << std::endl;
    std::cout << "sex param a: " << *f << std::endl;
    std::cout << "seven param a: " << *g << std::endl;

    delete a;
    a = nullptr;
    delete b;
    b = nullptr;
    delete c;
    c = nullptr;
    delete d;
    d = nullptr;
    delete e;
    e = nullptr;
    delete f;
    f = nullptr;
    delete g;
    g = nullptr;
    return 0;
}