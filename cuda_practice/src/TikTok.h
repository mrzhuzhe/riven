#include <iostream>
#include <thread>

auto t0 = std::chrono::steady_clock::now();    
auto t1 = std::chrono::steady_clock::now();
using double_ms = std::chrono::duration<double, std::milli>;

void Tik(){
    t0 = std::chrono::steady_clock::now();    
}

void Tok(std::string s){
    t1 = std::chrono::steady_clock::now();
    auto dt = t1 - t0;  
    double ms = std::chrono::duration_cast<double_ms>(dt).count();
    std::cout << s.c_str() << " time elapsed: " << ms << " ms" << std::endl;
}