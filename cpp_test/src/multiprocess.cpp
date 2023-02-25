#include <iostream>
#include <thread>
#include <chrono>
#include <vector>

void download(std::string file){
    for (int i = 0; i < 10; i++){
        std::cout << "Downloading " << file << " (" << i * 10 << "%)..." << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(400));
    }
    std::cout << "Download complete: " << file << std::endl;
}

void interact(){
    std::string name;
    std::cin >> name;
    std::cout << "Hi, " << name << std::endl;
}

//std::vector<std::thread> pool;

class ThreadPool {
    std::vector<std::thread> m_pool;

public:
    void push_back(std::thread thr){
        m_pool.push_back(std::move(thr));
    }
    ~ThreadPool(){
        for (auto &t:m_pool) t.join();
    }
};

//ThreadPool tpool;
//  std::vector<std::jthread> tpool; // cpp 20


int main(){
    auto t0 = std::chrono::steady_clock::now();
    for (volatile int i = 0; i< 10000000; i++);
    //std::this_thread::sleep_for(std::chrono::milliseconds(4000));
    auto t1 = std::chrono::steady_clock::now();
    auto dt = t1 - t0;
    //int64_t ms = std::chrono::duration_cast<std::chrono::milliseconds>(dt).count();
    using double_ms = std::chrono::duration<double, std::milli>;
    

    double ms = std::chrono::duration_cast<double_ms>(dt).count();
    std::cout << "time elapsed: " << ms << " ms" << std::endl;

    //  must be with pthread  g++ multiprocess.cpp  -o outputs/multiprocess.o -pthread
    
    std::thread trd1([&] {
        download("hello.zip");
    });
    //trd1.detach();
    //pool.push_back(std::move(trd1));
    tpool.push_back(std::move(trd1));

    interact();

    //for (auto &t: pool) t.join();
    //trd1.join();

    return 0;
}