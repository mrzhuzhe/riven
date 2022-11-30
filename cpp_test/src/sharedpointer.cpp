#include <iostream>
#include <vector>
#include <array>
#include <string>
#include <memory>
#include <list>

using std::vector;
using namespace std;

std::array<double, 6> _testArr;

shared_ptr<int> factory(int arg) 
{
    return make_shared<int>(arg);
}

shared_ptr<int>  use_factory(int arg)
{
    shared_ptr<int> p = factory(arg);
    return p;
} 

/*
vector<string> v1 {
    vector<string> v2 = { "a", "an", "the" };
    v1 = v2;
}
*/

class StrBlob {
    public:
        typedef std::vector<std::string>::size_type size_type;
        StrBlob();
        StrBlob(std::initializer_list<std::string> il);
        size_type size() const { return data->size(); }
        bool empty() const { return data->empty(); }
        void push_back(const std::string &t) { data->push_back(t); }
        void pop_back();
        std::string& front();
        std::string& back();
    private:
        std::shared_ptr<std::vector<std::string>> data;
        void check(size_type i, const std::string &msg) const;
};

StrBlob::StrBlob(): data(make_shared<vector<string>>()) {}
StrBlob::StrBlob(initializer_list<string> il): data(make_shared<vector<string>>(il)) {}

void StrBlob::check(size_type i, const string &msg) const
    {
        if (i >= data->size())
            throw out_of_range(msg);
    }

string& StrBlob::front()
{
    check(0, "front on empty StrBlob");
    return data->front(); 
}

string& StrBlob::back()
{
    check(0, "back on empty StrBlob");
    return data->back();
}

void StrBlob::pop_back()
{
    check(0, "pop_back on empty StrBlob");
    data->pop_back();
}

int main(int argc, char **argv) {
    _testArr[0] = 1;
    _testArr[7] = 2;
    _testArr[17] = 3;
    
    shared_ptr<string> p1;
    shared_ptr<list<int>> p2;

    if (p1 && p1->empty())
        *p1 = "hi";


    shared_ptr<int> p3 = make_shared<int>(42);
    shared_ptr<string> p4 = make_shared<string>(10, '9');
    shared_ptr<int> p5 = make_shared<int>();
    
    cout << *p3 << endl;
    cout << *p4 << endl;
    cout << *p5 << endl;

    auto p6 = make_shared<vector<string>>();
    //cout << *p6 << endl;

    auto p = make_shared<int>(42);
    auto q(p);

    cout << *q << " " << q.get() << " "<< q << " " << p << endl;

    auto r = make_shared<int>(42);
    r = q;
    cout << *r << " " << r.get() << endl;

    // compare to normal pointer
    int *aaa; 
    int bbb = 123;
    aaa = &bbb;
    cout << *aaa << " " << aaa << " " << bbb << " " << &bbb << endl;


    cout << *factory(456) << endl;

    cout << *use_factory(789) << endl;

    //  cout << *p1 << endl; //   Segmentation fault (core dumped)
    cout << _testArr[0] << _testArr[7] << _testArr[17] << endl;


    // manage memory directly 
    int *pi = new int;
    string *ps = new string;

    cout << *ps << *pi << endl;

    return 0;
}