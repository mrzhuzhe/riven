#include <string>
#include <iostream>
#include "sales_data.h"
#include <vector>
#include <algorithm>

using std::vector;
using namespace std;


char &get_val(string &str, string::size_type ix)
{
    return str[ix];
}

struct TestClass
{
    std::string isbn() const {
        return "123123";
    }
};

// lambda
auto f = []{ return "I m f()"; };

int main(int argc, char **argv) {
    
    string s("a value");
    cout << s << endl;
    get_val(s, 0) = 'A';
    cout << s << endl;
    
    TestClass testa;
    //std::string TestClass::isbn const { return "456456"; };
    
    cout << testa.isbn() << endl;

    cout << f() << endl;

    [](const string &a, const string &b){
        return a.size() < b.size();
    };
    
    vector<string> words = {"aa", "bbb", "cccc"};

    // lambda sort
    stable_sort(words.begin(), words.end(), [](const string &a, const string &b){
        return a.size() > b.size();
    });

    cout << words[0] << words[1] << words[2] << endl;
    
    return 0;
}