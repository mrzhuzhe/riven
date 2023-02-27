//  g++ tbb.cpp  -o outputs/tbb.o -pthread -std=c++17 -ltbb
#include <iostream>

#include <tbb/task_group.h>
#include <tbb/parallel_invoke.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#include <tbb/blocked_range2d.h>
#include <tbb/blocked_range3d.h>
#include <tbb/parallel_reduce.h>

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

    size_t n1 = 1<<13;
    std::vector<float> a1((n1)*(n1));
    Tik();
    tbb::parallel_for(tbb::blocked_range2d<size_t>(0 , n1, 0, n1), 
    [&](tbb::blocked_range2d<size_t> r){
        for (size_t i = r.cols().begin(); i < r.cols().end(); i++){
            for (size_t j = r.rows().begin(); j < r.rows().end(); j++){
                a1[i*n1+j] = std::sin(i) * std::sin(j);
                //std::cout<< "1";
            }
        }
    });
    Tok("2d range");

    size_t n2 = 1<<6; // over 1 << 11 is out of memory
    std::vector<float> a2((n2)*(n2)*(n2));
    Tik();
    tbb::parallel_for(tbb::blocked_range3d<size_t>(0 , n2, 0, n2,  0, n2), 
    [&](tbb::blocked_range3d<size_t> r){
        for (size_t i = r.pages().begin(); i < r.pages().end(); i++){
            for (size_t j = r.cols().begin(); j < r.cols().end(); j++){
                for (size_t k = r.rows().begin(); k < r.rows().end(); k++){
                    a2[(i*n2 + j ) * n2 +k] = std::sin(i) * std::sin(j) * std::sin(k);
                }
            }
        }
    });
    Tok("3d range");

    // reduce and scan 
    Tik();
    size_t n3 = 1 << 26;
    float res = 0;
    size_t maxt2 = 4;
    tbb::task_group tg2;
    std::vector<float> tmp_res(maxt2);
    for (size_t t = 0; t <maxt2; t++) {
        size_t beg = t * n / maxt;
        size_t end = std::min(n, (t+1)*n/maxt);
        tg2.run([&, t, beg, end]{
            float local_res = 0;
            for (size_t i = beg; i < end; i++){
                local_res += std::sin(i);
            }
            tmp_res[t] = local_res;
        });
    }
    tg2.wait();
    for (size_t t = 0; t < maxt2; t++){
        res += tmp_res[t];
    }
    std::cout << res << std::endl;
    Tok("Reduce");

    Tik();
    size_t n4 = 1 << 26;
    float res1 = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, n4), (float)0,    
    [&](tbb::blocked_range<size_t> r, float local_res){
        for (size_t i = r.begin(); i < r.end(); i++){
            local_res += std::sin(i);
        }
        return local_res;
    }, [](float x, float y){
        return x + y;
    });
    std::cout << res1 << std::endl;
    Tok("parallel reduce");

    Tik();
    size_t n5 = 1 << 26;
    //float res1 = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, n4), (float)0,
    float res2 = tbb::parallel_deterministic_reduce(tbb::blocked_range<size_t>(0, n4), (float)0,
    [&](tbb::blocked_range<size_t> r, float local_res){
        for (size_t i = r.begin(); i < r.end(); i++){
            local_res += std::sin(i);
        }
        return local_res;
    }, [](float x, float y){
        return x + y;
    });
    std::cout << res2 << std::endl;
    Tok("parallel deterministic reduce");

    // scan
    Tik();
    size_t n6 = 1 << 26;
    std::vector<float> a3(n);
    float res3 = 0;
    for (size_t i = 0; i < n6; i++){
        res += std::sin(i);
        a3[i] = res;
    }
    std::cout << a[n6/2] << std::endl;
    std::cout << res3 << std::endl;
    Tok("scan native");


    return 0;
}