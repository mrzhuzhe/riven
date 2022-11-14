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

string word;
vector<string> text;


string::size_type find_char(const string &s, char c, string::size_type &occurs)
{
    auto ret = s.size();
    occurs = 0;
    for (decltype(ret) i =0; i != s.size(); ++i){
        if (s[i] == c){
            if (ret == s.size())
                ret = i;
            ++occurs;
        }
    }
    return ret;
}

bool is_sentence(const string &s)
{
    string::size_type ctr =0;
    return find_char(s, '.', ctr) == s.size() - 1 && ctr == 1; 
}

int main(int argc, char **argv) {

    for (int i=0; i !=100; ++i)
        v2.push_back(i);

    /*
    while(cin >> word){
        text.push_back(word);
    }
    */


    vector<int> v{1, 2, 3, 4, 5, 6, 7, 8, 9};
    for (auto &i : v)
        i *= i;
    for (auto i: v)
        cout << i << " ";
    cout << endl;

    cout << v1[1] << endl;
    cout << ivec[1] << endl;
    cout << ivec.size() << endl;

    cout << "conpare" << endl;

    cout << (v2.size() < 99) << endl;

    string::size_type ctr = 0;
    
    find_char("Hello World", 'o', ctr);

    cout << ctr << endl;

    cout << is_sentence("asdasda.s") << endl;
    
    cout << argv[0] << argv[1] << argv[2] << endl;

    return 0;
}