//  g++ tbb.cpp  -o outputs/tbb.o -pthread -std=c++17 -ltbb
#include <iostream>
#include <tbb/task_group.h>
#include <tbb/parallel_invoke.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#include <string>
#include <thread>
#include <vector>
#include <cmath>

void download(std::string file){
    for (int i = 0; i < 10; i++){
        std::cout << "Downloading" << file << " (" << i * 10 << "%)... " << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(400));
    }
    std::cout << "Download complete: " << file << std::endl;
}

void interact() {
    std::string name;
    std::cin >> name;
    std::cout << "Hi, " << name << std::endl;
}

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
    

int main() {
    /*
    tbb::task_group tg;
    tg.run([&]{
        download("Hello.zip");
    });
    tg.run([&]{
        interact();
    });
    tg.wait();
    */
    /*
    tbb::parallel_invoke([&]{
        download("Hello.zip");
    },[&]{
        interact();
    });
    */
    std::string s = "Hello world";
    char ch = 'd';
    tbb::parallel_invoke([&]{
        for (size_t i =0; i<s.size()/2;i++){
            if (s[i] == ch)
                std::cout << "found!" << std::endl;
        }
    },
    [&] {
        for (size_t i = s.size()/2; i < s.size(); i++){
            if (s[i] == ch)
                std::cout << "found!" << std::endl;
        }
    });

    // parallel for loop
    size_t n = 1<<26;
    printf("%ld\n", n);
    std::vector<float> a(n);
    
    
    Tik();
    for (size_t i = 0; i<n;i++){
        a[i] = std::sin(i);
    }
    Tok("case native ");

    size_t maxt = 4;
    tbb::task_group tg;

    Tik();
    for (size_t t = 0; t < maxt; t++){
        auto beg = t * n / maxt;
        auto end = std::min(n, (t+1)*n/maxt);
        tg.run([&, beg, end]{
            for (size_t i = beg; i<end;i++){
                a[i] = std::sin(i);
            }
        });
    }
    tg.wait();
    Tok("case parallel_invoke ");

    Tik();
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
    [&](tbb::blocked_range<size_t> r){
        for (size_t i = r.begin(); i< r.end(); i++){
            a[i] = std::sin(i);
        }
        }
    );
    Tok("case parallel_for ");


    Tik();
    tbb::parallel_for((size_t)0, (size_t)n, [&](size_t i){
        a[i] = std::sin(i);
    });
    Tok("case parallel_for simper ");

    Tik();
    tbb::parallel_for_each(a.begin(), a.end(), [&](float &f){
        f = 32.f;
    });
    Tok("parallel_for_each");

    return 0;
}