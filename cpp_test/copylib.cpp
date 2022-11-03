#include <iostream>
#include <vector>
#include <map>
#include <functional>

using std::vector;
using namespace std;

class Sales_data {
    public:
        Sales_data(const Sales_data&);
    private:
        std::string bookNo;
        int units_sold = 0;
        double revenue = 0.0;
};  
  

Sales_data::Sales_data(const Sales_data &orig):
    bookNo(orig.bookNo),
    units_sold(orig.units_sold),
    revenue(orig.revenue)
    {}

int add(int i, int j) { return i +j; };
auto mod = [](int i, int j) { return i % j; };

struct divide {
    int operator()(int dominator, int divisor) {
        return dominator / divisor;
    }
};

map<string, function<int(int, int)>> binops = {
    { "+", add },
    { "-", minus<int>() },
    { "/", divide() },
    { "*", [](int i, int j){
        return i * j;
    }},
    { "%", mod }
};

/*
Sales_data& Sales_data::operator=(const Sales_data &rhs)
{
    bookNo = rhs.bookNo;
    Units_sold = rhs.units_sold;
    revenue = ths.revenue;
    return *this;
}
*/

class SmallInt {
    public:
        SmallInt(int i = 0): val(i)
        {
            if (i < 0 || i > 255)
                throw   out_of_range("Bad SmallInt value");
        }
        operator int() const { return val; }
    private:
        size_t val;
};

int main(int argc, char **argv) {


    //Sales_data *p = new Sales_data;

    
    /*
    Sales_data data1({
        bookNo: "123",
        units_sold: 0.0,
        revenue: 0.0
    });
    */
    //data1.bookNo = "123123";


    //data2 = data1;
    
    cout << binops["+"](10, 5) << endl;
    cout << binops["-"](10, 5) << endl;
    cout << binops["/"](10, 5) << endl;
    cout << binops["*"](10, 5) << endl;
    cout << binops["%"](10, 5) << endl;

    SmallInt si;
    si = 4.13;
    cout << si + 3 << endl;;

    return 0;
}