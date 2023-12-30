#include <iostream>
#include "test.h"

int main(){

//  https://en.wikipedia.org/wiki/X86_calling_conventions
    printf("Intel ASM test go \n");  
    char *a, *b, *c, *d, *e, *f, *g;
    a = new char[3]{"x"};
    b = new char[3]{"x"};
    c = new char[3]{"x"};
    d = new char[3]{"x"};
    e = new char[3]{"x"};
    f = new char[3]{"x"};
    g = new char[3]{"x"};
    
    test(a, b, c ,d, e, f, g);
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