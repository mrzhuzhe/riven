#include <vector>
#include <iostream>
#include <algorithm>
//#include <numeric> // cpp 17

int sum3 = 0;
void func(int vi){
    sum3 += vi;
}

int main() {

    size_t nv = 4;
    int *v = (int *)malloc(nv * sizeof(int));
    v[0] = 4;
    v[1] = 3;
    v[2] = 2;
    v[3] = 1;
    int sum = 0;
    for (size_t i =0; i< nv; i++){
        sum += v[i];
    }
    printf("%d\n", sum);
    free(v);


    std::vector<int> v1(4);
    v1[0] = 4;
    v1[1] = 3;
    v1[2] = 2;
    v1[3] = 1;
    v1 = {5, 4, 3, 2, 1};

    int sum1 = 0;
    for (size_t i = 0; i< v1.size(); i++){
        sum1 += v1[i];
    }
    std::cout << sum1 << std::endl;

    int sum2 = 0;
    // range based for-loop
    for (int vi: v1){
        sum2 += vi;
    }
    std::cout << sum2 << std::endl;


    std::for_each(v1.begin(), v1.end(), func);
    std::cout << "sum3 " << sum3 << std::endl;
    
    // lambda phrase
    int sum4 = 0;
    //std::for_each(v1.begin(), v1.end(), [&](int vi){
    std::for_each(v1.begin(), v1.end(), [&](auto vi){
        sum4 += vi;
    });

    std::cout << "sum4 " << sum4 << std::endl;

    //int sum5 = std::reduce(v1.begin(), v1.end());
    //int sum5 = std::reduce(v1.begin(), v1.end(), 0, std::plus{});

    //std::cout << sum5 << std::endl;


    return 0;
}