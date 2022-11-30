#include <iostream>
#include <vector>
#include <array>
#include <string>
#include <memory>
#include <list>

using std::vector;
using namespace std;

std::array<double, 6> _testArr;


int main(int argc, char **argv) {
    _testArr[0] = 1;
    _testArr[7] = 2;
    _testArr[17] = 3;
    
    shared_ptr<string> p1;
    shared_ptr<list<int>> p2;

    if (p1 && p1->empty())
        *p1 = "hi";


    shared_ptr<int> p3 = make_shared<int>(42);
    cout << p1 << endl;
    cout << *p3 << endl;

    //  cout << *p1 << endl; //   Segmentation fault (core dumped)
    cout << _testArr[0] << _testArr[7] << _testArr[17] << endl;
    return 0;
}