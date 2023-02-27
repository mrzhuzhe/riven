#include <iostream>
#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#include <tbb/concurrent_vector.h>
//#include <tbb/parallel_pipeline.h>
#include <vector>
#include <cmath>
#include "TikTok.h"

struct Data {
    std::vector<float> arr;
    Data() {
        arr.resize(std::rand()%100*500+100000);
        for (int i = 0; i< arr.size(); i++){
            arr[i] = std::rand() * (1.f / (float)RAND_MAX);
        }
    }
    void step1(){
        for (int i = 0; i < arr.size(); i++){
            arr[i] += 3.14f;
        }
    }
    void step2(){
        std::vector<float> tmp(arr.size());
        for (int i = 1; i < arr.size(); i++){
            tmp[i] = arr[i-1] + arr[i] + arr[i+1];
        }
        std:swap(tmp, arr);
    }
    void step3(){
        for (int i = 0; i<arr.size(); i++){
            arr[i] = std::sqrt(std::abs(arr[i]));
        }
    }  
    void step4(){
        std::vector<float> tmp(arr.size());
        for (int i = 1; i < arr.size(); i++){
            tmp[i] = arr[i-1] - 2 * arr[i] + arr[i+1];
        }
        std::swap(tmp, arr);
    } 

};

int main(){
    size_t n = 1 << 13;
    std::vector<float> a(n*n);
    Tik();
    tbb::parallel_for((size_t)0, (size_t)n, [&](size_t i){
        tbb::this_task_arena::isolate([&]{
            tbb::parallel_for((size_t)0, (size_t)n, [&](size_t j){
                a[i*n+j] = std::sin(i) * std::sin(j);
            }, tbb::auto_partitioner{});
        });
        
    });
    Tok("nest parallel loog");
    
    //  concurrent container
    size_t n1 = 1 << 10;
    std::vector<float> a1;
    std::vector<float *> pa(n1);
    a1.reserve(n1);

    for (size_t i =0; i < n1; i++) {
        a1.push_back(std::sin(i));
        pa[i] = &a1.back();
    }

    /*
    for (size_t i = 0; i< n1; i++){
        std::cout << (&a1[i] == pa[i]) << " ";
    }
    std::cout << std::endl;
    */

    tbb::concurrent_vector<float> a2; // multithread safe
    std::vector<float *> pa1(n1);
    for (size_t i =0; i < n1; i++) {
        auto it = a2.push_back(std::sin(i));
        pa1[i] = &*it;;
    }
    for (size_t i = 0; i< n1; i++){
        std::cout << (&a2[i] == pa1[i]) << " ";
    }
    std::cout << std::endl;

    size_t n3 = 1 << 12;
    Tik();
    std::vector<Data> datas(n3);
    tbb::parallel_for_each(datas.begin(), datas.end(), [&](Data &data){
        data.step1();
        data.step2();
        data.step3();
        data.step4();
    });
    Tok("tbb parallel for");

    /*
    Tik();
    auto it = datas.begin();
    tbb::parallel_pipline(8, 
        tbb::make_filter<void, Data *>(
            tbb::filter_mode::serial_in_order, [&](tbb::flow_control &fc) -> Data * {
                if (it == datas.end()){
                    fc.stop();
                    return nullptr;
                }
                return &*it++;
        })
        , tbb::make_filter<Data *, Data *>(tbb::filter_mode::parallel, [&](Data &data) -> Data *){
            data->step1();
            return data;
        })
        , tbb::make_filter<Data *, Data *>(tbb::filter_mode::parallel, [&](Data &data) -> Data *){
            data->step2();
            return data;
        })
        , tbb::make_filter<Data *, Data *>(tbb::filter_mode::parallel, [&](Data &data) -> Data *){
            data->step3();
            return data;
        })
        , tbb::make_filter<Data *, Data *>(tbb::filter_mode::parallel, [&](Data &data) -> Data *){
            data->step4();
            return data;
        })
    );

    Tok("tbb parallel for");
    */

    return 0;
}