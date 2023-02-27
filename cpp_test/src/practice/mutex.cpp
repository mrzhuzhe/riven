#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <mutex>
#include <shared_mutex> // g++ mutex.cpp  -o outputs/mutex.o -pthread -std=c++17
#include <condition_variable>

class  MTVector {
    std::vector<int> m_arr;
    mutable std::shared_mutex m_mtx;
public:
     void push_back(int val) {
         m_mtx.lock();
         m_arr.push_back(val);
         m_mtx.unlock();
     }
     size_t size() const {
         m_mtx.lock_shared();
         size_t ret = m_arr.size();
         m_mtx.unlock_shared();
         return ret;
     }
};


int main() {
    //std::vector<int> arr;   // not thread safe
    MTVector arr;

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

    std::mutex mtx3;
    if (mtx3.try_lock())
        printf("succeed\n");
    else
        printf("fail\n");

    if (mtx3.try_lock())
        printf("succeed\n");
    else
        printf("fail\n");
    mtx3.unlock();

    std::mutex mtx4;
    std::thread t3([&]{
        std::unique_lock<std::mutex> grd(mtx, std::try_to_lock);
        if (grd.owns_lock())
            printf("t3 success\n");
        else 
            printf("t3 fail\n");
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    });

    std::thread t4([&]{
        std::unique_lock<std::mutex> grd(mtx, std::try_to_lock);
        if (grd.owns_lock())
            printf("t4 success\n");
        else 
            printf("t4 fail\n");
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    });

    t3.join();
    t4.join();

    std::condition_variable cv;
    std::mutex mtx5;
    bool ready = false;

    std::thread t5([&]{
        std::unique_lock lck(mtx5);
        cv.wait(lck, [&]{ return ready; });
        lck.unlock();
        std::cout << "t1 is awake" << std::endl;
    });

    //std::this_thread::sleep_for(std::chrono::milliseconds(400));
    std::cout << "notifying not ready" << std::endl;
    cv.notify_one();

    ready = true;
    std::cout << "notifying ready" << std::endl;
    //std::cout << "notifying..." << std::endl;
    cv.notify_one();
    t5.join();

    return 0;
}