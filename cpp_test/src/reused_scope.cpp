#include <iostream>
using namespace std;

int reused = 42;

int main()
{

    int unique = 0;
    cout << reused << " " << unique << endl;
    int reused = 0;
    cout << reused << " " << unique << endl;
    // explicit global 
    cout << ::reused << " " << unique << endl;
    return 0;

}