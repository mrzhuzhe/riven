#include <string>
#include <iostream>
#include "sales_data.h"
#include <vector>
#include <algorithm>
#include <list>
#include <memory>

using namespace std;


list<int> lst = {1, 2, 3, 4};
list<int> lst2, lst3;


int main() {
    
    copy(lst.cbegin(), lst.cend(), front_inserter(lst2));
    copy(lst.cbegin(), lst.cend(), inserter(lst3, lst3.begin()));

    vector<int> vec = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    for (auto r_iter = vec.crbegin();
        r_iter != vec.crend();
        ++r_iter)
    cout << *r_iter << endl;
    //cout << lst2(1) << endl;




    unique_ptr<int[]> up(new int[10]);
    //up.release();

    for (size_t i =0; i != 10; ++i){
        up[i] = i;
        cout << " i: " << up[i]; 
    }
    cout << endl;

    up.release();
    

    shared_ptr<int> sp(new int[10], [](int *p){ delete[] p;});

    for (size_t i=0; i != 10; ++i){
        *(sp.get() + i) = i;
        cout << *(sp.get() + i) << "\n" ;
    }

    sp.reset();


    return 0;
}