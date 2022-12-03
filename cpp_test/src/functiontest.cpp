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

string make_plural(size_t ctr, const string& word, const string& ending )
{
    return (ctr > 1) ? word + ending : word;
}

struct TestClass
{
    std::string isbn() const {
        return "123123";
    }
};

// lambda
auto f = []{ return "I m f()"; };

// function pointer 

string lengthCompare(const string &, const string &);

// must be init or cannot be invoke
string lengthCompare(const string &s1, const string &s2){
    return s1 + "" + s2;
};

string (*pf)(const string &, const string &); // params and return type must be same


string useBigger(const string &s1, const string &s2, string (*pf)(const string &, const string &)){
    return pf(s1, s2);
}


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
    

    vector<string>::size_type sz;
    auto wc = find_if(words.begin(), words.end(), [sz](const string &a){
        return a.size() >= sz;
    });
    auto count = words.end() - wc; 
    cout << count << " " << make_plural(count, "word", "s") << " of length " << sz << " or longer " << endl;
    


    pf = lengthCompare;
    cout << pf("aaaa", "bbbb") << endl;

    cout << useBigger("cccc", "dddd", pf) << endl;

    return 0;
}