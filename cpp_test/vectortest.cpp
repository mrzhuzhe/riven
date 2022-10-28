#include <string>
#include <iostream>
#include "sales_data.h"
#include <vector>
using std::vector;
using namespace std;

//vector<int> ivec;
vector<Sales_data> Sales_vec;
vector<vector<string>> file;
//vector<string> svec;


vector<string> articles = {"a", "an", "the"};
vector<string> v1{"a", "an", "the"};

vector<int> ivec(10,-1);
vector<string> svec(10, "hi!");

vector<int> v2;


int main() {

    for (int i=0; i !=100; ++i)
        v2.push_back(i);

    cout << v1[1] << endl;
    cout << ivec[1] << endl;
    cout << ivec.size() << endl;

    cout << v2.size() << endl;
    
    return 0;
}