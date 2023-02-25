
#include <iostream>
#include <cstring>
#include <vector>
#include <memory>
//#include <ofstram>

using std::vector;
using namespace std;

template <typename T> T compare(const T &v1, const T &v2)
{
    if (v1 < v2) return v2;
    if (v2 < v1) return v1;
    return v2;

}

template<unsigned N, unsigned M> int compareStr(const char (&p1)[N], const char(&p2)[M])
{
    return strcmp(p1, p2);
}


template <typename T> class  Blob2 {
public:
    typedef T value_type;
    typedef typename std::vector<T>::size_type size_type;
    Blob2() ;
    Blob2(std::initializer_list<T> il);
    size_type size() const {    return data->size();    }
    bool empty() const { return data->empty();   }
    void push_back(const T &t) {    return data->push_back();    }
    void push_back(const T &&t) {    return data->push_back(std::move(t));    }
    void pop_back();
    T& back();
    T& operator[](size_type i);

private:
    std::shared_ptr<std::vector<T>> data;
    void check(size_type i, const std::string &msg) const;
};

template <typename T> Blob2<T>::Blob2(): data(std::make_shared<std::vector<T>>()) {

}

template <typename T> Blob2<T>::Blob2(std::initializer_list<T> il): data(std::make_shared<std::vector<T>>(il)) {

}


template <typename T>  void Blob2<T>::check(size_type i, const std::string &msg) const {
    if (i >= data->size())
        throw std::out_of_range(msg);
}

template <typename T> T& Blob2<T>::back()
{
    check(0, "back on empty blob");
    return data->back();
}

template <typename T> T& Blob2<T>::operator[] (size_type i)
{
    check(i, "subscript out of range");
    return (*data)[i];
}

template <typename T> void Blob2<T>::pop_back()
{
    check(0, "pop back on empty Blob");
    data->pop_back();
}

//  pointer to template
template <typename T> class BlobPtr
{
    public:
        BlobPtr(): curr(0) {}
        BlobPtr(Blob2<T> &a, size_t sz = 0): wptr(a.data), curr(sz) {}
        T& operator*() const 
        {
            auto p = check(curr, "dereference past end");
            return (*p)[curr];
        }
        BlobPtr operator++();
        BlobPtr operator--();
    private:
        std::shared_ptr<std::vector<T>>
            check(std::size_t, const std::string&) const;
        std::weak_ptr<std::vector<T>> wptr;
        std::size_t curr;
};


template<typename T> using BlobRename = Blob2<T>;
using BlobIntRename = Blob2<int>;


template <typename T> ostream &print(ostream &os, const T &obj)
{
    return os << obj << endl;
}

/*
    function pointer 
*/
void g(int &&i, int& j)
{
    cout << i << " " << j << endl;
}

void f(int v1, int &v2)
{
    cout << v1 << " " << ++v2 << endl;
} 

template <typename F, typename T1, typename T2> void flip(F f, T1 t1, T2 t2)
{
    f(t2, t1);
}

template <typename F, typename T1, typename T2> void flip2(F f, T1 &&t1, T2 &&t2)
{
    //f(t2, t1);
    f(std::forward<T2>(t2), std::forward<T1>(t1));
}



#include <cstdlib>
#include <string>
#if defined(__GNU__) || defined(__clang__)
#include <cxxabi.h>
#endif

template <class T>
std::string cpp_type_name() {
    const char *name = typeid(T).name();
#if defined(__GNU__) || defined(__clang__)
    int status;
    char *p = abi::__cxa_demangle(name, 0, 0, &status);
    std::string s = p;
    std::free(p);
#else
    std::string s = name;
#endif    
    if (std::is_const_v<std::remove_reference_t<T>>)
        s += " const";
    if (std::is_volatile_v<std::remove_reference_t<T>>)
        s += " volatile";
    if (std::is_lvalue_reference_v<T>)
        s += " &";
    if (std::is_rvalue_reference_v<T>)
        s += " &&";    
    return s;
}


int main(int argc, char **argv) {
    vector<int> vec1{1, 2, 3}, vec2{4, 5, 6};
    
    cout << compare(1, 0) << endl;
    cout << compare<int>(1, 0) << endl;

    cout << compare<vector<int>>(vec1, vec2)[0] << endl;
    cout << compare(vec1, vec2)[1] << endl;

    cout << compareStr("hi", "mom") << endl;


    Blob2<int> ia;
    Blob2<int> ia2 = {0, 1, 2, 3, 4};
    
    //ia2.back();

    Blob2<string> name;
    Blob2<string> name2 = { "a", "an", "the" };
    Blob2<double> prices;

    BlobRename<int> ia3 = {   0, 2, 3, 4  };
    BlobIntRename ia4 = { 5, 4, 3 };

    print(cout, 42);
    //ofstram f("output");
    //print(f, 10);

    int i = 123;
    f(422, i);
    flip(f, i, 42);

    flip2(g, i, 42);    // error: can’t initialize int&& from an lvalue

    printf("%c\n", cpp_type_name<int>);


    return 0;
}
