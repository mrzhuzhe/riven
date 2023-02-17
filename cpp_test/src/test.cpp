#include <iostream>

using namespace std;
#include <algorithm>
#include <vector>
#include <set>
using std::vector;





using namespace std;
class Base
{
    public:
    virtual void show() = 0;
};

template<class A>
void f(A a)
{
    cout<<1<<endl;
}
void f(int a) { cout << 2<<endl; }

class Person
{
    char full_name[25];
    int age;
    public:
        void change(){

        }
        void value(){}
};

class Student: private Person{};
int a();

int &fun()
{
    static int x = 10;
    return x;
}

int main(void) {
    
    /*
    int a, b, c;
    cin>>a>>b>>c;
    cout<<a<<b<<c<<endl;
    */
    
    //int a = 1;
    //f<float>(a);
    //Student b;
    //cout << sizeof(Student);
    
    /*
    int t[] = {1,2,3,2,3,5,1,2,7,3,2,1,10,4,4,5};
    vector<int> v (t,t+15);
    */
    //int number = count(v.begin(), v.end(), 5);
    //cout << number << endl;
    //cout << sizeof(void *);

    /*
    int myints[] = {3, 4, 2, 1, 6, 5, 7, 9, 8, 0};
    vector<int> v(myints, myints+10);
    set<int> s1(v.begin(), v.end());
    s1.insert(v.begin(), v.end());
    s1.erase(s1.lower_bound(2), s1.upper_bound(7));
    for (set<int>::iterator i=s1.begin();i!=s1.end();i++){
        cout<<*i<< "";
    }
    */
    //a();
    /*
    v.empty();
    cout << v.isempty() << v.size();
    return 0;
    */
    //cout << true << "" << boolalpha << false ; return 0;

    //fun() = 30;
    //cout << fun();
    int array[] = { 1, 2,  3};
    //cout << -2*[array];
    for (int &vi : array){
        cout << vi << " ";
    }

    
    return 0;
};

/*
int a(){
    printf("123123");
};
*/