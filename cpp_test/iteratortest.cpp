#include <string>
#include <iostream>
#include "sales_data.h"
#include <vector>
#include <algorithm>
#include <list>

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
    return 0;
}