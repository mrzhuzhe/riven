#include <iostream>
#include <string>
#include "sales_data.h"

using namespace std;

int main() {
    Sales_data data1, data2;
    data1.bookNo = "123123\n";
    data1.units_sold = 12.5;
    data1.revenue = 25;
    std::cout << data1.bookNo << data1.units_sold << "\n";


    std::string s("Hello world!!!");
    for (auto &c : s)
        c = toupper(c);
    std::cout << s << std::endl;

    if (!s.empty())
    std::cout << s[0] << endl;

    const string hexdigits = "0123456789ABCDEF";
    cout << "Enter a series of numbers between 0 and 15"
        << " separated by spaceds. Hit ENTER when finished: "
        << endl;
    
    string result;
    string::size_type n;
    // unix ctrl + d windows ctrl + c
    while (cin >> n)
        if (n < hexdigits.size())
            result += hexdigits[n];
    cout << "Your hex number is: " << result << endl;


    return 0;
}