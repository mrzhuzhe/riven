#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <mutex>

int main() {
    std::vector<int> arr;   // not thread safe
    std::mutex mtx;
    std::mutex mtx2;

    std::thread t1([&]{
        for (int i = 0; i< 1000; i++){
            //mtx.lock();
            //mtx2.lock();
            //std::lock_guard<std::mutex> grd(mtx);
            //std::unique_lock<std::mutex> grd(mtx);
            arr.push_back(1);
            //mtx.unlock();
            //mtx2.unlock();
        }
    });
    std::thread t2([&]{
        for (int i = 0; i< 1000; i++){
            //mtx.lock();
            //mtx2.lock();
            //std::lock_guard<std::mutex> grd(mtx, 
            //std::defer_lock
            //);
            //std::unique_lock<std::mutex> grd(mtx);
            //grd.lock();
            arr.push_back(2);
            //grd.unlock();

            //mtx.unlock();
            //mtx2.unlock();
        }
    });
    t1.join();
    t2.join();

    return 0;
}