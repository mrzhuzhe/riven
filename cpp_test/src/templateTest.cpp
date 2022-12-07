
#include <iostream>
#include <cstring>
#include <vector>
#include <memory>
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

template <typename T> Blob2<T>::Blob2() {

}

template <typename T> Blob2<T>::Blob2(std::initializer_list<T> il) {

}


template <typename T>  void Blob2<T>::check(size_type i, const std::string &msg) const {
    if (i >= data->size())
        throw std::out_of_range(msg);
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

    Blob2<string> name;

    Blob2<double> prices;

    return 0;
}
