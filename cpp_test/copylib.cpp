#include <iostream>
#include <vector>

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




/*
Sales_data& Sales_data::operator=(const Sales_data &rhs)
{
    bookNo = rhs.bookNo;
    Units_sold = rhs.units_sold;
    revenue = ths.revenue;
    return *this;
}
*/

int main(int argc, char **argv) {


    //Sales_data *p = new Sales_data;

    
    Sales_data data1({
        bookNo: "123",
        units_sold: 0.0,
        revenue: 0.0
    });
    //data1.bookNo = "123123";


    //data2 = data1;
    

    return 0;
}