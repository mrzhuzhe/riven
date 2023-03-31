# include <iostream>
# include <vector>

using namespace std;
class Animal
{
public:
    virtual void eat() const { cout << "I eat like a generic Animal." << endl; }
    //virtual void eat();
/*
    /usr/bin/ld: CMakeFiles/vfn_test.o.dir/vfn_test.cpp.o: in function `Animal::~Animal()':
vfn_test.cpp:(.text._ZN6AnimalD2Ev[_ZN6AnimalD5Ev]+0xf): undefined reference to `vtable for Animal'
/usr/bin/ld: CMakeFiles/vfn_test.o.dir/vfn_test.cpp.o: in function `Animal::Animal()':
vfn_test.cpp:(.text._ZN6AnimalC2Ev[_ZN6AnimalC5Ev]+0xf): undefined reference to `vtable for Animal'
/usr/bin/ld: CMakeFiles/vfn_test.o.dir/vfn_test.cpp.o:(.data.rel.ro._ZTV11OtherAnimal[_ZTV11OtherAnimal]+0x10): undefined reference to `Animal::eat()'
/usr/bin/ld: CMakeFiles/vfn_test.o.dir/vfn_test.cpp.o:(.data.rel.ro._ZTV8GoldFish[_ZTV8GoldFish]+0x10): undefined reference to `Animal::eat()'
/usr/bin/ld: CMakeFiles/vfn_test.o.dir/vfn_test.cpp.o:(.data.rel.ro._ZTV4Fish[_ZTV4Fish]+0x10): undefined reference to `Animal::eat()'
/usr/bin/ld: CMakeFiles/vfn_test.o.dir/vfn_test.cpp.o:(.data.rel.ro._ZTV4Wolf[_ZTV4Wolf]+0x10): undefined reference to `Animal::eat()'
/usr/bin/ld: CMakeFiles/vfn_test.o.dir/vfn_test.cpp.o:(.data.rel.ro._ZTI11OtherAnimal[_ZTI11OtherAnimal]+0x10): undefined reference to `typeinfo for Animal'
/usr/bin/ld: CMakeFiles/vfn_test.o.dir/vfn_test.cpp.o:(.data.rel.ro._ZTI4Fish[_ZTI4Fish]+0x10): undefined reference to `typeinfo for Animal'
/usr/bin/ld: CMakeFiles/vfn_test.o.dir/vfn_test.cpp.o:(.data.rel.ro._ZTI4Wolf[_ZTI4Wolf]+0x10): undefined reference to `typeinfo for Animal'
collect2: error: ld returned 1 exit status
make[2]: *** [CMakeFiles/vfn_test.o.dir/build.make:97: vfn_test.o] Error 1
make[1]: *** [CMakeFiles/Makefile2:103: CMakeFiles/vfn_test.o.dir/all] Error 2
make: *** [Makefile:91: all] Error 2
*/
    virtual ~Animal() {}
};
 
class Wolf : public Animal
{
public:
    void eat() const { cout << "I eat like a wolf!" << endl; }
};
 
class Fish : public Animal
{
public:
    void eat() const { cout << "I eat like a fish!" << endl; }
};
 
class GoldFish : public Fish
{
public:
    void eat() const { cout << "I eat like a goldfish!" << endl; }
};
 
 
class OtherAnimal : public Animal
{
};
 
int main()
{
    std::vector<Animal*> animals;
    animals.push_back( new Animal() );
    animals.push_back( new Wolf() );
    animals.push_back( new Fish() );
    animals.push_back( new GoldFish() );
    animals.push_back( new OtherAnimal() );
 
    for( std::vector<Animal*>::const_iterator it = animals.begin();
       it != animals.end(); ++it) 
    {
        (*it)->eat();
        delete *it;
    }
 
   return 0;
}
